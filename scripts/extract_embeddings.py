#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Extract embeddings to readable text file')
    parser.add_argument('--data_dir', type=str, required=True, help="data directory path (contains idx2word.dat)")
    parser.add_argument('--result_dir', type=str, required=True, help="result directory path (contains idx2vec.dat)")
    parser.add_argument('--output', type=str, default=None, help="output file path (default: result_dir/embeddings.txt)")
    parser.add_argument('--limit', type=int, default=1000, help="limit number of embeddings to save (0 = save all)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load vocabulary mapping
    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))
    
    # Load embeddings
    idx2vec = pickle.load(open(os.path.join(args.result_dir, 'idx2vec.dat'), 'rb'))
    
    # Determine output path
    if args.output is None:
        output_path = os.path.join(args.result_dir, 'embeddings.txt')
    else:
        output_path = args.output
    
    # Write embeddings to text file
    limit = args.limit if args.limit > 0 else len(idx2word)
    num_to_save = min(limit, len(idx2word))
    print(f"Writing {num_to_save} embeddings to {output_path}...")
    with open(output_path, 'w') as f:
        for idx in range(num_to_save):
            word = idx2word[idx]
            vec_str = ' '.join([str(x) for x in idx2vec[idx]])
            f.write(f"{word} {vec_str}\n")
    
    print(f"Done! Saved {num_to_save} embeddings to {output_path}")
    print(f"Format: <word> <dim1> <dim2> ... <dimN>")


if __name__ == '__main__':
    main()

