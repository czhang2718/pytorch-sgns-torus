# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.bin', help="corpus path for building vocab (.bin file)")
    parser.add_argument('--corpus', type=str, default='./data/corpus.bin', help="corpus path (.bin file)")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    parser.add_argument('--chunk_size', type=int, default=10000000, help="chunk size for processing binary files (in tokens)")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir
        # Create data directory if it doesn't exist
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[max(i - self.window, 0): i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        # Read binary file as uint16 token IDs
        print(f"Reading binary file: {filepath}")
        tokens = np.fromfile(filepath, dtype=np.uint16)
        print(f"Total tokens: {len(tokens):,}")
        print("Counting token frequencies...")
        
        # Count token frequencies
        self.wc = {self.unk: 1}
        unique_tokens, counts = np.unique(tokens, return_counts=True)
        for token_id, count in zip(unique_tokens, counts):
            self.wc[str(token_id)] = int(count)
        
        print(f"Unique tokens: {len(unique_tokens):,}")
        print("")
        
        # Build vocabulary: keep top max_vocab most frequent tokens
        sorted_items = sorted(self.wc.items(), key=lambda x: x[1], reverse=True)
        vocab_items = [self.unk] + [word for word, _ in sorted_items[:max_vocab - 1]]
        
        self.idx2word = vocab_items
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath, chunk_size=10000000):
        print("converting corpus...")
        # Read binary file and process in chunks
        print(f"Reading binary file: {filepath}")
        tokens = np.fromfile(filepath, dtype=np.uint16)
        print(f"Total tokens: {len(tokens):,}")
        
        # Process in chunks to manage memory
        num_chunks = (len(tokens) + chunk_size - 1) // chunk_size
        print(f"Processing in {num_chunks} chunks...")
        
        data = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Convert token IDs to strings and filter by vocabulary
            sent = []
            for token_id in chunk_tokens:
                token_str = str(token_id)
                if token_str in self.vocab:
                    sent.append(token_str)
                else:
                    sent.append(self.unk)
            
            # Generate skip-gram pairs
            for i in range(len(sent)):
                iword, owords = self.skipgram(sent, i)
                data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
            
            print(f"Processed chunk {chunk_idx + 1}/{num_chunks} ({end_idx:,}/{len(tokens):,} tokens, {len(data):,} pairs)", end='\r')
        
        print("")
        pickle.dump(data, open(os.path.join(self.data_dir, 'train.dat'), 'wb'))
        print(f"conversion done: {len(data):,} skip-gram pairs created")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus, chunk_size=args.chunk_size)

