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
    parser = argparse.ArgumentParser(description='Plot coordinate weights bar chart')
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model .pt file")
    parser.add_argument('--data_dir', type=str, required=True, help="data directory (contains idx2word.dat)")
    parser.add_argument('--output_path', type=str, default=None, help="output path for plot")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--top_k', type=int, default=None, help="plot only top K dimensions (default: all)")
    return parser.parse_args()


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
    print(f"Loaded coord_weights: shape={coord_weights.shape}, mean={coord_weights.mean():.4f}, max={coord_weights.max():.4f}, min={coord_weights.min():.4f}")
    
    # Sort by weight if we want top K
    if args.top_k is not None:
        sorted_indices = np.argsort(coord_weights)[::-1][:args.top_k]
        sorted_indices = np.sort(sorted_indices)  # Sort by dimension index for better visualization
        coord_weights_plot = coord_weights[sorted_indices]
        dims_plot = sorted_indices
        title_suffix = f" (Top {args.top_k} dimensions)"
    else:
        coord_weights_plot = coord_weights
        dims_plot = np.arange(len(coord_weights))
        title_suffix = ""
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(16, 6))
    
    bars = ax.bar(dims_plot, coord_weights_plot, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars by value (darker = higher weight)
    colors = plt.cm.viridis((coord_weights_plot - coord_weights_plot.min()) / 
                           (coord_weights_plot.max() - coord_weights_plot.min() + 1e-10))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Dimension Index', fontsize=12)
    ax.set_ylabel('Coordinate Weight', fontsize=12)
    ax.set_title(f'Coordinate Weights for Toric Similarity{title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Mean: {coord_weights.mean():.4f}\nMax: {coord_weights.max():.4f} (dim {coord_weights.argmax()})\nMin: {coord_weights.min():.4f} (dim {coord_weights.argmin()})'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Highlight top 2 dimensions
    top2_indices = np.argsort(coord_weights)[-2:][::-1]
    for idx in top2_indices:
        if idx in dims_plot:
            pos = np.where(dims_plot == idx)[0][0]
            bars[pos].set_edgecolor('red')
            bars[pos].set_linewidth(2)
            ax.text(dims_plot[pos], coord_weights_plot[pos] + coord_weights_plot.max() * 0.02,
                   f'#{np.argsort(coord_weights)[::-1].tolist().index(idx) + 1}',
                   ha='center', va='bottom', fontsize=8, color='red', weight='bold')
    
    plt.tight_layout()
    
    # Determine output path
    if args.output_path is None:
        output_dir = os.path.dirname(args.model_path)
        output_path = os.path.join(output_dir, 'coord_weights_bar.png')
    else:
        output_path = args.output_path
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved coordinate weights bar chart to {output_path}")
    
    # Print top 10 dimensions
    print(f"\nTop 10 coordinate weight dimensions:")
    top10_indices = np.argsort(coord_weights)[-10:][::-1]
    for i, idx in enumerate(top10_indices, 1):
        print(f"  {i:2d}. Dimension {idx:3d}: weight = {coord_weights[idx]:.6f}")


if __name__ == '__main__':
    main()

