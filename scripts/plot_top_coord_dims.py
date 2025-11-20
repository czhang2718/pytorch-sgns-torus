#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import argparse
import sys

# Add current directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Word2Vec, SGNS


def parse_args():
    parser = argparse.ArgumentParser(description='Plot 2D scatterplot using top 2 coordinate weight dimensions')
    parser.add_argument('--embeddings_file', type=str, required=True, help="path to embeddings text file (full vocab)")
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model .pt file")
    parser.add_argument('--data_dir', type=str, required=True, help="data directory (contains idx2word.dat, wc.dat)")
    parser.add_argument('--output_path', type=str, default=None, help="output path for plot")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--top_k', type=int, default=100, help="number of top words to plot")
    parser.add_argument('--word_range_start', type=int, default=None, help="start of word rank range (1-indexed)")
    parser.add_argument('--word_range_end', type=int, default=None, help="end of word rank range (1-indexed)")
    parser.add_argument('--zoom_limit', type=float, default=0.1, help="zoom limit around origin (default: 0.1)")
    parser.add_argument('--output_suffix', type=str, default='', help="suffix to add to output filename")
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


def get_top_words_by_frequency(data_dir, top_k):
    """Get top K words by frequency."""
    wc = pickle.load(open(os.path.join(data_dir, 'wc.dat'), 'rb'))
    idx2word = pickle.load(open(os.path.join(data_dir, 'idx2word.dat'), 'rb'))
    
    # Get word frequencies
    word_freqs = [(word, wc.get(word, 0)) for word in idx2word]
    word_freqs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K words
    top_words = [word for word, freq in word_freqs[:top_k]]
    return top_words


def main():
    args = parse_args()
    
    # Load vocabulary to get vocab_size
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    vocab_size = len(idx2word)
    
    # Load model to get coord_weights
    print(f"Loading model from {args.model_path}...")
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim, torus=True)
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=20, torus=True)
    sgns.load_state_dict(t.load(args.model_path, map_location='cpu'))
    
    if not hasattr(sgns, 'coord_weights'):
        print("Error: Model does not have coord_weights. Make sure it's a toric model.")
        return
    
    coord_weights = sgns.coord_weights.detach().cpu().numpy()
    print(f"Loaded coord_weights: shape={coord_weights.shape}, mean={coord_weights.mean():.4f}")
    
    # Find top 2 dimensions by coordinate weight
    top2_indices = np.argsort(coord_weights)[-2:][::-1]  # Get top 2, highest first
    top2_weights = coord_weights[top2_indices]
    
    print(f"\nTop 2 coordinate weight dimensions:")
    print(f"  Dimension {top2_indices[0]}: weight = {top2_weights[0]:.6f}")
    print(f"  Dimension {top2_indices[1]}: weight = {top2_weights[1]:.6f}")
    
    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings_file}...")
    embeddings = load_embeddings(args.embeddings_file)
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Get words by frequency range or top K
    if args.word_range_start is not None and args.word_range_end is not None:
        print(f"\nGetting words ranked {args.word_range_start}-{args.word_range_end} by frequency...")
        # Get more words than needed to account for missing ones
        top_words_all = get_top_words_by_frequency(args.data_dir, args.word_range_end)
        # Extract the range (1-indexed, so subtract 1 for 0-indexed)
        words_to_use = top_words_all[args.word_range_start-1:args.word_range_end]
        range_label = f"ranked {args.word_range_start}-{args.word_range_end}"
    else:
        print(f"\nGetting top {args.top_k} words by frequency...")
        words_to_use = get_top_words_by_frequency(args.data_dir, args.top_k)
        range_label = f"top {args.top_k}"
    
    # Filter to words that exist in embeddings
    available_words = [w for w in words_to_use if w in embeddings]
    print(f"Found {len(available_words)} words in embeddings ({range_label})")
    
    if len(available_words) == 0:
        print("Error: No words found in embeddings!")
        return
    
    # Extract coordinates for top 2 dimensions
    x_coords = []
    y_coords = []
    words_to_plot = []
    
    for word in available_words:
        vec = embeddings[word]
        x_coords.append(vec[top2_indices[0]])
        y_coords.append(vec[top2_indices[1]])
        words_to_plot.append(word)
    
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    # Filter points: keep only those within zoom_limit of origin in both dimensions
    mask = (np.abs(x_coords) <= args.zoom_limit) & (np.abs(y_coords) <= args.zoom_limit)
    x_coords_filtered = x_coords[mask]
    y_coords_filtered = y_coords[mask]
    words_filtered = [words_to_plot[i] for i in range(len(words_to_plot)) if mask[i]]
    
    print(f"\nFiltered: {len(words_to_plot)} -> {len(words_filtered)} words (within {args.zoom_limit} of origin)")
    
    # Create scatterplot
    print(f"\nCreating scatterplot with {len(words_filtered)} words...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot points with larger size and better spacing
    scatter = ax.scatter(x_coords_filtered, y_coords_filtered, s=100, alpha=0.7, 
                         c=range(len(words_filtered)), cmap='viridis', 
                         edgecolors='black', linewidths=1.0)
    
    # Add labels for all words with better positioning
    for i, word in enumerate(words_filtered):
        # Offset label slightly to avoid overlap
        offset_x = (x_coords_filtered.max() - x_coords_filtered.min()) * 0.02
        offset_y = (y_coords_filtered.max() - y_coords_filtered.min()) * 0.02
        ax.annotate(word, (x_coords_filtered[i], y_coords_filtered[i]), 
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=7, ha='left', va='bottom', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Center plot at (0, 0) with symmetric limits based on zoom_limit
    ax.set_xlim(-args.zoom_limit, args.zoom_limit)
    ax.set_ylim(-args.zoom_limit, args.zoom_limit)
    
    # Add grid lines at origin
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    ax.set_xlabel(f'Dimension {top2_indices[0]} (weight={top2_weights[0]:.6f})', fontsize=12)
    ax.set_ylabel(f'Dimension {top2_indices[1]} (weight={top2_weights[1]:.6f})', fontsize=12)
    title_range = range_label if args.word_range_start is not None else f"Top {len(words_to_plot)}"
    ax.set_title(f'{title_range} Words by Frequency (within {args.zoom_limit} of origin)\n'
                f'Plotted in Top 2 Coordinate Weight Dimensions', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Word Rank (by frequency)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Determine output path
    if args.output_path is None:
        output_dir = os.path.dirname(args.embeddings_file)
        if args.word_range_start is not None and args.word_range_end is not None:
            base_name = f'top_coord_dims_scatter_rank{args.word_range_start}_{args.word_range_end}'
        else:
            base_name = 'top_coord_dims_scatter'
        if args.output_suffix:
            base_name += f'_{args.output_suffix}'
        output_path = os.path.join(output_dir, f'{base_name}.png')
    else:
        output_path = args.output_path
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatterplot to {output_path}")
    
    # Print some statistics
    print(f"\nStatistics (filtered points):")
    print(f"  X-axis (dim {top2_indices[0]}): min={x_coords_filtered.min():.4f}, max={x_coords_filtered.max():.4f}, mean={x_coords_filtered.mean():.4f}")
    print(f"  Y-axis (dim {top2_indices[1]}): min={y_coords_filtered.min():.4f}, max={y_coords_filtered.max():.4f}, mean={y_coords_filtered.mean():.4f}")
    print(f"\nStatistics (all points before filtering):")
    print(f"  X-axis (dim {top2_indices[0]}): min={x_coords.min():.4f}, max={x_coords.max():.4f}, mean={x_coords.mean():.4f}")
    print(f"  Y-axis (dim {top2_indices[1]}): min={y_coords.min():.4f}, max={y_coords.max():.4f}, mean={y_coords.mean():.4f}")


if __name__ == '__main__':
    main()

