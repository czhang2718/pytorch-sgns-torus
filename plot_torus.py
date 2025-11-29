# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import matplotlib
import numpy as np
import torch as t

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from model import Word2Vec, SGNS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--result_dir', type=str, default='./result/', help="result directory path")
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model .pt file (to extract coord_weights)")
    parser.add_argument('--e_dim', type=int, default=300, help="embedding dimension")
    parser.add_argument('--model', type=str, default='tsne', choices=['pca', 'tsne'], help="model for visualization")
    parser.add_argument('--top_k', type=int, default=1000, help="scatter top-k words (deprecated, use --num_alg_points)")
    parser.add_argument('--num_alg_points', type=int, default=None, help="number of top words to use for PCA/TSNE algorithm")
    parser.add_argument('--num_plot_points', type=int, default=None, help="number of top words to actually plot (must be <= num_alg_points)")
    parser.add_argument('--word_range_start', type=int, default=None, help="start of word rank range (1-indexed)")
    parser.add_argument('--word_range_end', type=int, default=None, help="end of word rank range (1-indexed)")
    parser.add_argument('--plot_name', type=str, default=None, help="custom name for the output plot file (without .png extension)")
    parser.add_argument('--perplexity', type=int, default=30, help="perplexity for t-SNE")
    return parser.parse_args()


def torus_projection(X, model=None,coord_weights=None):
    """
    Project each embedding x_i to [w*cos(pi*x_i), w*sin(pi*x_i)].
    If x_i is d-dimensional, the result is 2d-dimensional.
    
    Args:
        X: numpy array of shape (n_samples, d_dimensions)
        coord_weights: numpy array of shape (d_dimensions,) or None
                      If provided, weights each dimension by coord_weights
    
    Returns:
        numpy array of shape (n_samples, 2*d_dimensions)
    """
    # Use pi factor to match toric similarity function
    cos_X = np.cos(np.pi * X)
    sin_X = np.sin(np.pi * X)
    
    # Apply coord_weights if provided
    if coord_weights is not None:
        coord_weights = coord_weights.reshape(1, -1)  # Shape: (1, d)
        euclidean_coord_weights = np.maximum(coord_weights, 0) if model == 'tsne' else np.abs(coord_weights)
        cos_X = cos_X * np.sqrt(euclidean_coord_weights)
        sin_X = sin_X * np.sqrt(euclidean_coord_weights)
    
    # Concatenate cos and sin along the feature dimension
    X_torus = np.concatenate([cos_X, sin_X], axis=1)
    return X_torus


def plot(args):
    print("Loading word counts...")
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    print(f"Loaded {len(wc)} words from vocabulary")
    
    # Load vocab size for model initialization
    print("Loading vocabulary...")
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    vocab_size = len(idx2word)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load model to get coord_weights
    print(f"Loading model from {args.model_path}...")
    model = Word2Vec(vocab_size=vocab_size, embedding_size=args.e_dim, torus=True)
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=20, torus=True)
    sgns.load_state_dict(t.load(args.model_path, map_location='cpu'))
    
    if not hasattr(sgns, 'coord_weights'):
        print("Warning: Model does not have coord_weights. Using uniform weights.")
        coord_weights = None
    else:
        coord_weights = sgns.coord_weights.detach().cpu().numpy()
        print(f"Loaded coord_weights: shape={coord_weights.shape}, mean={coord_weights.mean():.4f}")
    
    # Determine number of words for algorithm and plotting
    if args.num_alg_points is not None:
        num_alg_points = args.num_alg_points
    else:
        num_alg_points = args.top_k
    
    if args.num_plot_points is not None:
        num_plot_points = min(args.num_plot_points, num_alg_points)
    else:
        num_plot_points = num_alg_points
    
    # Get words by frequency range or top K
    print("Selecting words...")
    if args.word_range_start is not None and args.word_range_end is not None:
        sorted_words = sorted(wc, key=wc.get, reverse=True)
        # Extract the range (1-indexed, so subtract 1 for 0-indexed)
        words = sorted_words[args.word_range_start-1:args.word_range_end]
        range_label = f"ranked {args.word_range_start}-{args.word_range_end}"
    else:
        sorted_words = sorted(wc, key=wc.get, reverse=True)
        words = sorted_words[:num_alg_points]
        range_label = f"top {num_alg_points}"
    
    num_words = len(words)
    print(f"Selected {num_words} words for algorithm ({range_label})")
    print(f"Will plot {num_plot_points} words")
    
    print("Loading embeddings...")
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    idx2vec = pickle.load(open(os.path.join(args.result_dir, 'idx2vec.dat'), 'rb'))
    print("Extracting embeddings for selected words...")
    X = np.array([idx2vec[word2idx[word]] for word in words])
    print(f"Loaded embeddings: shape {X.shape}")
    
    # Apply torus projection: x_i -> [w*cos(pi*x_i), w*sin(pi*x_i)]
    print("Applying torus projection with coord_weights (cos/sin transformation)...")
    X_torus = torus_projection(X, args.model, coord_weights=coord_weights)
    print(f"Original embedding dimension: {X.shape[1]}")
    print(f"Torus projection dimension: {X_torus.shape[1]}")
    
    # Apply PCA or t-SNE to the torus-projected embeddings
    print(f"Initializing {args.model.upper()} model...")
    if args.model == 'pca':
        model = PCA(n_components=2)
    elif args.model == 'tsne':
        # Use 'barnes_hut' method for faster computation, or remove method parameter for auto-selection
        # Perplexity should be less than the number of samples
        perplexity = args.perplexity if args.perplexity is not None else min(30, num_words - 1)
        print(f"  Using perplexity={perplexity}")
        model = TSNE(n_components=2, perplexity=perplexity, init='pca')
    
    print(f"Running {args.model.upper()} on torus-projected embeddings (this may take a while)...")
    X_reduced = model.fit_transform(X_torus)
    print(f"{args.model.upper()} completed! Reduced to shape {X_reduced.shape}")
    
    print("Creating plot...")
    plt.figure(figsize=(18, 18))
    
    # Only plot the first num_plot_points words
    for i in range(min(num_plot_points, len(X_reduced))):
        plt.text(X_reduced[i, 0], X_reduced[i, 1], words[i], 
                bbox=dict(facecolor='blue', alpha=0.08, edgecolor='none'),
                ha='center', va='center')
    
    # Set limits based on all points (for context) or just plotted points
    xlim_data = X_reduced[:num_plot_points, 0] if num_plot_points < len(X_reduced) else X_reduced[:, 0]
    ylim_data = X_reduced[:num_plot_points, 1] if num_plot_points < len(X_reduced) else X_reduced[:, 1]
    plt.xlim((np.min(xlim_data), np.max(xlim_data)))
    plt.ylim((np.min(ylim_data), np.max(ylim_data)))
    
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    
    if num_plot_points < num_words:
        title_text = f"{args.model.upper()} Visualization (Torus Projection w/ coord_weights)\nAlgorithm: {range_label} words, Plotted: top {num_plot_points} words"
    else:
        title_text = f"{args.model.upper()} Visualization (Torus Projection w/ coord_weights) of {range_label} words"
    plt.title(title_text)
    
    # Determine output filename
    if args.plot_name is not None:
        output_filename = args.plot_name if args.plot_name.endswith('.png') else f"{args.plot_name}.png"
    elif args.word_range_start is not None and args.word_range_end is not None:
        output_filename = f"{args.model}_torus_rank{args.word_range_start}_{args.word_range_end}.png"
    perplexity_tag = f"_perplexity{args.perplexity}" if args.model == 'tsne' else ""
    output_filename = f"{args.model}_torus_alg{num_words}_plot{num_plot_points}{perplexity_tag}.png"
    
    print(f"Saving figure...")
    plt.savefig(os.path.join(args.result_dir, output_filename))
    print(f"Saved figure to {os.path.join(args.result_dir, output_filename)}")
    print("Done!")


if __name__ == '__main__':
    # Use default font (works cross-platform)
    # For better Unicode support, you can install and use fonts like 'DejaVu Sans' or 'Liberation Sans'
    plot(parse_args())

