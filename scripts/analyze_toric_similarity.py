#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import sys

# Add current directory to path to import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Word2Vec, SGNS


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze embeddings using toric similarity function')
    parser.add_argument('--embeddings_file', type=str, required=True, help="path to embeddings text file")
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model .pt file")
    parser.add_argument('--data_dir', type=str, required=True, help="data directory (contains idx2word.dat)")
    parser.add_argument('--output_dir', type=str, default=None, help="output directory for plots")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
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
        x: numpy array of shape (E,) or (B, E)
        y: numpy array of shape (E,) or (B, E)
        coord_weights: numpy array of shape (E,)
    
    Returns:
        scalar similarity value
    """
    x_t = t.from_numpy(x).float()
    y_t = t.from_numpy(y).float()
    coord_weights_t = coord_weights.float()
    
    # Ensure same shape
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(0)
    
    # Add batch and context dimensions for compatibility with model function
    # x: [1, 1, E], y: [1, 1, E]
    x_t = x_t.unsqueeze(0).unsqueeze(0)  # [1, 1, E]
    y_t = y_t.unsqueeze(0).unsqueeze(0)  # [1, 1, E]
    
    # Compute similarity: coord_weights * cos(pi * (x - y))
    diff = x_t - y_t
    cos_diff = t.cos(t.pi * diff)
    similarity = (coord_weights_t * cos_diff).sum(dim=-1)
    
    return similarity.item()


def compute_pairwise_toric_similarities(embeddings, coord_weights):
    """Compute toric similarities between all pairs."""
    words = list(embeddings.keys())
    n = len(words)
    similarities = np.zeros((n, n))
    
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            similarities[i, j] = toric_similarity(
                embeddings[word1], 
                embeddings[word2], 
                coord_weights
            )
    
    return words, similarities


def test_linear_representation_toric(embeddings, coord_weights, word1, word2, word3, word4):
    """
    Test if word1 - word2 + word3 ≈ word4 using toric similarity.
    Returns the predicted vector and toric similarity with word4.
    """
    if not all(w in embeddings for w in [word1, word2, word3, word4]):
        return None, None, None
    
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    vec3 = embeddings[word3]
    vec4 = embeddings[word4]
    
    # Compute predicted vector (standard vector arithmetic)
    predicted = vec1 - vec2 + vec3
    
    # Compute toric similarity between predicted and actual
    similarity = toric_similarity(predicted, vec4, coord_weights)
    
    # Also compute standard L2 distance for comparison
    l2_distance = np.linalg.norm(predicted - vec4)
    
    return predicted, similarity, l2_distance


def plot_pairwise_toric_similarities(words, similarities, output_path):
    """Plot heatmap of pairwise toric similarities."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarities, cmap='coolwarm', aspect='auto')
    
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    ax.set_title('Pairwise Toric Similarities\n(coord_weights * cos(π * (x - y)))')
    
    # Add text annotations
    for i in range(len(words)):
        for j in range(len(words)):
            text = ax.text(j, i, f'{similarities[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Toric Similarity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved toric similarities heatmap to {output_path}")


def plot_2d_projection_toric(embeddings, coord_weights, output_path):
    """Plot 2D PCA projection of embeddings with toric similarity annotations."""
    words = list(embeddings.keys())
    vectors = np.array([embeddings[w] for w in words])
    
    # Apply PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    ax.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.6)
    
    # Add labels
    for i, word in enumerate(words):
        ax.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                   fontsize=12, ha='center', va='bottom')
    
    # Draw vectors for linear representation test
    if 'man' in embeddings and 'woman' in embeddings and 'King' in embeddings and 'Queen' in embeddings:
        man_idx = words.index('man')
        woman_idx = words.index('woman')
        king_idx = words.index('King')
        queen_idx = words.index('Queen')
        
        # Draw vectors
        ax.arrow(vectors_2d[king_idx, 0], vectors_2d[king_idx, 1],
                vectors_2d[man_idx, 0] - vectors_2d[king_idx, 0],
                vectors_2d[man_idx, 1] - vectors_2d[king_idx, 1],
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5, linestyle='--',
                label='King → Man')
        
        ax.arrow(vectors_2d[woman_idx, 0], vectors_2d[woman_idx, 1],
                vectors_2d[queen_idx, 0] - vectors_2d[woman_idx, 0],
                vectors_2d[queen_idx, 1] - vectors_2d[woman_idx, 1],
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5, linestyle='--',
                label='Woman → Queen')
        
        # Draw predicted vector (King - Man + Woman)
        predicted_vec = embeddings['King'] - embeddings['man'] + embeddings['woman']
        predicted_2d = pca.transform(predicted_vec.reshape(1, -1))[0]
        ax.scatter(predicted_2d[0], predicted_2d[1], s=200, marker='*', 
                  color='green', label='Predicted (King - Man + Woman)', zorder=5)
        ax.annotate('Predicted', (predicted_2d[0], predicted_2d[1]), 
                   fontsize=10, ha='center', va='top', color='green')
        
        # Compute and display toric similarity
        predicted_sim = toric_similarity(predicted_vec, embeddings['Queen'], coord_weights)
        ax.text(0.02, 0.98, f'Toric Sim (Predicted, Queen): {predicted_sim:.4f}',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('2D PCA Projection (Toric Embeddings)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D toric projection to {output_path}")


def plot_vector_comparison_toric(embeddings, coord_weights, output_path):
    """Plot comparison of actual vs predicted vectors using toric similarity."""
    if not all(w in embeddings for w in ['King', 'man', 'woman', 'Queen']):
        print("Warning: Missing required words for vector comparison plot")
        return
    
    predicted, similarity, l2_dist = test_linear_representation_toric(
        embeddings, coord_weights, 'King', 'man', 'woman', 'Queen'
    )
    
    actual = embeddings['Queen']
    
    # Plot first 50 dimensions
    dims_to_plot = min(50, len(actual))
    dims = np.arange(dims_to_plot)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot values
    ax1.plot(dims, actual[:dims_to_plot], 'o-', label='Actual Queen', alpha=0.7)
    ax1.plot(dims, predicted[:dims_to_plot], 's-', label='Predicted (King - Man + Woman)', alpha=0.7)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Vector Comparison (Toric Similarity: {similarity:.4f}, L2 Distance: {l2_dist:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot difference
    diff = actual[:dims_to_plot] - predicted[:dims_to_plot]
    ax2.bar(dims, diff, alpha=0.7)
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Difference (Actual - Predicted)')
    ax2.set_title('Difference Between Actual and Predicted Vectors')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot coord_weights to show which dimensions matter most
    coord_weights_np = coord_weights.detach().cpu().numpy()
    ax3.bar(dims, coord_weights_np[:dims_to_plot], alpha=0.7, color='purple')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Coordinate Weight')
    ax3.set_title('Coordinate Weights (Higher = More Important for Toric Similarity)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved toric vector comparison to {output_path}")


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
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.embeddings_file)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Compute pairwise toric similarities
    print("\nComputing pairwise toric similarities...")
    words, similarities = compute_pairwise_toric_similarities(embeddings, coord_weights)
    
    print("\nPairwise Toric Similarities:")
    print("=" * 70)
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i <= j:  # Only print upper triangle
                print(f"{word1:10s} ~ {word2:10s} = {similarities[i, j]:8.4f}")
    
    # Test linear representation hypothesis
    print("\n" + "=" * 70)
    print("Testing Linear Representation Hypothesis (Toric Similarity):")
    print("=" * 70)
    
    test_cases = [
        ('King', 'man', 'woman', 'Queen'),
    ]
    
    for word1, word2, word3, word4 in test_cases:
        predicted, similarity, l2_dist = test_linear_representation_toric(
            embeddings, coord_weights, word1, word2, word3, word4
        )
        
        if predicted is not None:
            print(f"\n{word1} - {word2} + {word3} ≈ {word4}")
            print(f"  Toric Similarity: {similarity:.6f}")
            print(f"  L2 Distance: {l2_dist:.6f}")
            print(f"  Prediction quality: {'✓ GOOD' if similarity > 0.5 else '✗ POOR'}")
        else:
            print(f"\nMissing words for: {word1} - {word2} + {word3} ≈ {word4}")
    
    # Create plots
    print("\n" + "=" * 70)
    print("Creating visualizations...")
    print("=" * 70)
    
    plot_pairwise_toric_similarities(words, similarities, 
                                     os.path.join(output_dir, 'toric_similarities.png'))
    
    plot_2d_projection_toric(embeddings, coord_weights,
                            os.path.join(output_dir, 'toric_pca_projection.png'))
    
    plot_vector_comparison_toric(embeddings, coord_weights,
                                os.path.join(output_dir, 'toric_vector_comparison.png'))
    
    print("\nToric similarity analysis complete!")


if __name__ == '__main__':
    main()

