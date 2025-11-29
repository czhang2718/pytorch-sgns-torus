# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import matplotlib
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--result_dir', type=str, default='./result/', help="result directory path")
    parser.add_argument('--model', type=str, default='tsne', choices=['pca', 'tsne'], help="model for visualization")
    parser.add_argument('--top_k', type=int, default=1000, help="scatter top-k words")
    parser.add_argument('--word_range_start', type=int, default=None, help="start of word rank range (1-indexed)")
    parser.add_argument('--word_range_end', type=int, default=None, help="end of word rank range (1-indexed)")
    parser.add_argument('--plot_name', type=str, default=None, help="custom name for the output plot file (without .png extension)")
    return parser.parse_args()


def plot(args):
    print("Loading word counts...")
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    print(f"Loaded {len(wc)} words from vocabulary")
    
    # Get words by frequency range or top K
    print("Selecting words...")
    if args.word_range_start is not None and args.word_range_end is not None:
        sorted_words = sorted(wc, key=wc.get, reverse=True)
        # Extract the range (1-indexed, so subtract 1 for 0-indexed)
        words = sorted_words[args.word_range_start-1:args.word_range_end]
        range_label = f"ranked {args.word_range_start}-{args.word_range_end}"
    else:
        words = sorted(wc, key=wc.get, reverse=True)[:args.top_k]
        range_label = f"top {len(words)}"
    
    num_words = len(words)
    print(f"Selected {num_words} words ({range_label})")
    
    print("Loading embeddings...")
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    idx2vec = pickle.load(open(os.path.join(args.result_dir, 'idx2vec.dat'), 'rb'))
    print("Extracting embeddings for selected words...")
    X = np.array([idx2vec[word2idx[word]] for word in words])
    print(f"Loaded embeddings: shape {X.shape}")
    
    print(f"Initializing {args.model.upper()} model...")
    if args.model == 'pca':
        model = PCA(n_components=2)
    elif args.model == 'tsne':
        # Use 'barnes_hut' method for faster computation, or remove method parameter for auto-selection
        # Perplexity should be less than the number of samples
        perplexity = min(30, num_words - 1)
        print(f"  Using perplexity={perplexity}")
        model = TSNE(n_components=2, perplexity=perplexity, init='pca')
    
    print(f"Running {args.model.upper()} (this may take a while)...")
    X = model.fit_transform(X)
    print(f"{args.model.upper()} completed! Reduced to shape {X.shape}")
    
    print("Creating plot...")
    plt.figure(figsize=(18, 18))
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1], words[i], bbox=dict(facecolor='blue', alpha=0.1))
    plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    title_text = f"{args.model.upper()} Visualization of {range_label} words"
    plt.title(title_text)
    
    # Determine output filename
    if args.plot_name is not None:
        output_filename = args.plot_name if args.plot_name.endswith('.png') else f"{args.plot_name}.png"
    elif args.word_range_start is not None and args.word_range_end is not None:
        output_filename = f"{args.model}_rank{args.word_range_start}_{args.word_range_end}.png"
    else:
        output_filename = f"{args.model}.png"
    
    print(f"Saving figure...")
    plt.savefig(os.path.join(args.result_dir, output_filename))
    print(f"Saved figure to {os.path.join(args.result_dir, output_filename)}")
    print("Done!")


if __name__ == '__main__':
    # Use default font (works cross-platform)
    # For better Unicode support, you can install and use fonts like 'DejaVu Sans' or 'Liberation Sans'
    plot(parse_args())
