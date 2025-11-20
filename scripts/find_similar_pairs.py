#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch as t
import argparse
import sys
from tqdm import tqdm

# Add current directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Word2Vec, SGNS


def parse_args():
    parser = argparse.ArgumentParser(description='Find top similar pairs using toric similarity')
    parser.add_argument('--embeddings_file', type=str, required=True, help="path to embeddings text file")
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model .pt file")
    parser.add_argument('--data_dir', type=str, required=True, help="data directory (contains idx2word.dat)")
    parser.add_argument('--output_file', type=str, default=None, help="output file path")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--top_k', type=int, default=100, help="number of top pairs to find")
    parser.add_argument('--vocab_size', type=int, default=None, help="vocab size (auto-detect if not provided)")
    return parser.parse_args()


def load_embeddings(filepath):
    """Load embeddings from text file."""
    embeddings = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vec
    return embeddings


def toric_similarity(x, y, coord_weights):
    """
    Compute toric similarity: (coord_weights * cos(pi * (x - y))).sum()
    
    Args:
        x: numpy array of shape (E,)
        y: numpy array of shape (E,)
        coord_weights: torch tensor of shape (E,)
    
    Returns:
        scalar similarity value
    """
    x_t = t.from_numpy(x).float()
    y_t = t.from_numpy(y).float()
    coord_weights_t = coord_weights.float()
    
    # Add batch and context dimensions for compatibility with model function
    x_t = x_t.unsqueeze(0).unsqueeze(0)  # [1, 1, E]
    y_t = y_t.unsqueeze(0).unsqueeze(0)  # [1, 1, E]
    
    # Compute similarity: coord_weights * cos(pi * (x - y))
    diff = x_t - y_t
    cos_diff = t.cos(t.pi * diff)
    similarity = (coord_weights_t * cos_diff).sum(dim=-1)
    
    return similarity.item()


def find_top_similar_pairs(embeddings, coord_weights, top_k=100):
    """Find top K most similar pairs of embeddings."""
    words = list(embeddings.keys())
    n = len(words)
    
    print(f"Computing similarities for {n} words ({n*(n-1)//2} pairs)...")
    
    similarities = []
    
    # Compute all pairwise similarities (upper triangle only)
    for i in tqdm(range(n), desc="Computing similarities"):
        word1 = words[i]
        vec1 = embeddings[word1]
        for j in range(i + 1, n):
            word2 = words[j]
            vec2 = embeddings[word2]
            similarity = toric_similarity(vec1, vec2, coord_weights)
            similarities.append((similarity, word1, word2))
    
    # Sort by similarity (descending) and get top K
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_pairs = similarities[:top_k]
    
    return top_pairs


def main():
    args = parse_args()
    
    # Load vocabulary to get vocab_size if not provided
    if args.vocab_size is None:
        idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
        vocab_size = len(idx2word)
    else:
        vocab_size = args.vocab_size
    
    # Load model to get coord_weights
    print(f"Loading model from {args.model_path}...")
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim, torus=True)
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=20, torus=True)
    sgns.load_state_dict(t.load(args.model_path, map_location='cpu'))
    
    if not hasattr(sgns, 'coord_weights'):
        print("Error: Model does not have coord_weights. Make sure it's a toric model.")
        return
    
    coord_weights = sgns.coord_weights
    print(f"Loaded coord_weights: shape={coord_weights.shape}, mean={coord_weights.mean().item():.4f}")
    
    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings_file}...")
    embeddings = load_embeddings(args.embeddings_file)
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Find top similar pairs
    top_pairs = find_top_similar_pairs(embeddings, coord_weights, args.top_k)
    
    # Determine output file
    if args.output_file is None:
        output_dir = os.path.dirname(args.embeddings_file)
        output_file = os.path.join(output_dir, f'top_{args.top_k}_similar_pairs.txt')
    else:
        output_file = args.output_file
    
    # Save results
    print(f"\nTop {args.top_k} most similar pairs:")
    print("=" * 80)
    
    with open(output_file, 'w') as f:
        f.write(f"Top {args.top_k} Most Similar Pairs (by Toric Similarity)\n")
        f.write("=" * 80 + "\n\n")
        
        for rank, (similarity, word1, word2) in enumerate(top_pairs, 1):
            line = f"{rank:4d}. {word1:20s} ~ {word2:20s} : {similarity:10.6f}"
            print(line)
            f.write(line + "\n")
    
    print(f"\nSaved results to {output_file}")
    
    # Print some statistics
    similarities_only = [sim for sim, _, _ in top_pairs]
    print(f"\nStatistics:")
    print(f"  Max similarity: {max(similarities_only):.6f}")
    print(f"  Min similarity: {min(similarities_only):.6f}")
    print(f"  Mean similarity: {np.mean(similarities_only):.6f}")
    print(f"  Median similarity: {np.median(similarities_only):.6f}")


if __name__ == '__main__':
    main()

