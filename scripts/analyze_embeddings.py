#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze embeddings: dot products and linear representation hypothesis')
    parser.add_argument('--embeddings_file', type=str, required=True, help="path to embeddings text file")
    parser.add_argument('--output_dir', type=str, default=None, help="output directory for plots (default: same as embeddings file)")
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


def compute_pairwise_dot_products(embeddings):
    """Compute dot products between all pairs."""
    words = list(embeddings.keys())
    n = len(words)
    dot_products = np.zeros((n, n))
    
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            dot_products[i, j] = np.dot(embeddings[word1], embeddings[word2])
    
    return words, dot_products


def test_linear_representation(embeddings, word1, word2, word3, word4):
    """
    Test if word1 - word2 + word3 ≈ word4 (e.g., King - Man + Woman ≈ Queen)
    Returns the predicted vector and cosine similarity with word4.
    """
    if not all(w in embeddings for w in [word1, word2, word3, word4]):
        return None, None, None
    
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    vec3 = embeddings[word3]
    vec4 = embeddings[word4]
    
    # Compute predicted vector
    predicted = vec1 - vec2 + vec3
    
    # Compute cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarity = cosine_sim(predicted, vec4)
    
    # Compute L2 distance
    l2_distance = np.linalg.norm(predicted - vec4)
    
    return predicted, similarity, l2_distance


def plot_pairwise_dot_products(words, dot_products, output_path):
    """Plot heatmap of pairwise dot products."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dot_products, cmap='coolwarm', aspect='auto')
    
    ax.set_xticks(range(len(words)))
    ax.set_yticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    ax.set_title('Pairwise Dot Products')
    
    # Add text annotations
    for i in range(len(words)):
        for j in range(len(words)):
            text = ax.text(j, i, f'{dot_products[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved dot products heatmap to {output_path}")


def plot_2d_projection(embeddings, output_path):
    """Plot 2D PCA projection of embeddings."""
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
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5, linestyle='--')
        
        ax.arrow(vectors_2d[woman_idx, 0], vectors_2d[woman_idx, 1],
                vectors_2d[queen_idx, 0] - vectors_2d[woman_idx, 0],
                vectors_2d[queen_idx, 1] - vectors_2d[woman_idx, 1],
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5, linestyle='--')
        
        # Draw predicted vector (King - Man + Woman)
        predicted_vec = embeddings['King'] - embeddings['man'] + embeddings['woman']
        predicted_2d = pca.transform(predicted_vec.reshape(1, -1))[0]
        ax.scatter(predicted_2d[0], predicted_2d[1], s=200, marker='*', 
                  color='green', label='Predicted (King - Man + Woman)', zorder=5)
        ax.annotate('Predicted', (predicted_2d[0], predicted_2d[1]), 
                   fontsize=10, ha='center', va='top', color='green')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('2D PCA Projection of Embeddings')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D projection to {output_path}")


def plot_vector_comparison(embeddings, output_path):
    """Plot comparison of actual vs predicted vectors."""
    if not all(w in embeddings for w in ['King', 'man', 'woman', 'Queen']):
        print("Warning: Missing required words for vector comparison plot")
        return
    
    predicted, similarity, l2_dist = test_linear_representation(
        embeddings, 'King', 'man', 'woman', 'Queen'
    )
    
    actual = embeddings['Queen']
    
    # Plot first 50 dimensions
    dims_to_plot = min(50, len(actual))
    dims = np.arange(dims_to_plot)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot values
    ax1.plot(dims, actual[:dims_to_plot], 'o-', label='Actual Queen', alpha=0.7)
    ax1.plot(dims, predicted[:dims_to_plot], 's-', label='Predicted (King - Man + Woman)', alpha=0.7)
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Vector Comparison (Cosine Similarity: {similarity:.4f}, L2 Distance: {l2_dist:.4f})')
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
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved vector comparison to {output_path}")


def main():
    args = parse_args()
    
    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_file}...")
    embeddings = load_embeddings(args.embeddings_file)
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(args.embeddings_file)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Compute pairwise dot products
    print("\nComputing pairwise dot products...")
    words, dot_products = compute_pairwise_dot_products(embeddings)
    
    print("\nPairwise Dot Products:")
    print("=" * 60)
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            if i <= j:  # Only print upper triangle
                print(f"{word1:10s} · {word2:10s} = {dot_products[i, j]:8.4f}")
    
    # Test linear representation hypothesis
    print("\n" + "=" * 60)
    print("Testing Linear Representation Hypothesis:")
    print("=" * 60)
    
    test_cases = [
        ('King', 'man', 'woman', 'Queen'),
    ]
    
    for word1, word2, word3, word4 in test_cases:
        predicted, similarity, l2_dist = test_linear_representation(
            embeddings, word1, word2, word3, word4
        )
        
        if predicted is not None:
            print(f"\n{word1} - {word2} + {word3} ≈ {word4}")
            print(f"  Cosine Similarity: {similarity:.6f}")
            print(f"  L2 Distance: {l2_dist:.6f}")
            print(f"  Prediction quality: {'✓ GOOD' if similarity > 0.7 else '✗ POOR'}")
        else:
            print(f"\nMissing words for: {word1} - {word2} + {word3} ≈ {word4}")
    
    # Create plots
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    plot_pairwise_dot_products(words, dot_products, 
                               os.path.join(output_dir, 'dot_products.png'))
    
    plot_2d_projection(embeddings, 
                      os.path.join(output_dir, 'pca_projection.png'))
    
    plot_vector_comparison(embeddings, 
                         os.path.join(output_dir, 'vector_comparison.png'))
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

