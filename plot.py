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
    return parser.parse_args()


def plot(args):
    wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
    words = sorted(wc, key=wc.get, reverse=True)[:args.top_k]
    if args.model == 'pca':
        model = PCA(n_components=2)
    elif args.model == 'tsne':
        # Use 'barnes_hut' method for faster computation, or remove method parameter for auto-selection
        model = TSNE(n_components=2, perplexity=min(30, args.top_k - 1), init='pca')
    word2idx = pickle.load(open(os.path.join(args.data_dir, 'word2idx.dat'), 'rb'))
    idx2vec = pickle.load(open(os.path.join(args.result_dir, 'idx2vec.dat'), 'rb'))
    X = np.array([idx2vec[word2idx[word]] for word in words])
    X = model.fit_transform(X)
    plt.figure(figsize=(18, 18))
    for i in range(len(X)):
        plt.text(X[i, 0], X[i, 1], words[i], bbox=dict(facecolor='blue', alpha=0.1))
    plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))
    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)
    plt.title(f"TSNE Visualization of {args.model} on {args.result_dir}")
    plt.savefig(os.path.join(args.result_dir, args.model) + '.png')
    print("saved figure to", os.path.join(args.result_dir, args.model) + '.png')


if __name__ == '__main__':
    # Use default font (works cross-platform)
    # For better Unicode support, you can install and use fonts like 'DejaVu Sans' or 'Liberation Sans'
    plot(parse_args())
