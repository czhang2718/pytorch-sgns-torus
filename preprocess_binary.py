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
    parser.add_argument('--max_vocab', type=int, default=200000, help="maximum number of vocab")
    parser.add_argument('--chunk_size', type=int, default=10000000, help="chunk size for processing binary files (in tokens)")
    parser.add_argument('--max_chunks', type=int, default=None, help="maximum number of chunks to process (for testing, e.g., 90 for 1/10 of 904 chunks)")
    parser.add_argument('--vocab_chunk_size', type=int, default=50000000, help="chunk size for reading file when building vocab (in tokens)")
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

    def build(self, filepath, max_vocab=20000, vocab_chunk_size=50000000, max_chunks=None):
        print("building vocab...")
        # Read binary file in chunks to avoid loading everything into memory
        print(f"Reading binary file: {filepath}")
        
        # Get file size to calculate number of tokens
        file_size = os.path.getsize(filepath)
        num_tokens = file_size // 2  # uint16 = 2 bytes
        print(f"Total tokens: {num_tokens:,}")
        
        # Calculate how many chunks to process for vocab building
        num_chunks = (num_tokens + vocab_chunk_size - 1) // vocab_chunk_size
        
        # If max_chunks is set, limit vocab building to the same portion of the file
        if max_chunks is not None:
            # Calculate how many vocab chunks correspond to max_chunks conversion chunks
            # This ensures vocab is built from the same data that will be converted
            convert_chunk_size = 10000000  # default from convert method
            max_tokens_for_conversion = max_chunks * convert_chunk_size
            max_vocab_chunks = (max_tokens_for_conversion + vocab_chunk_size - 1) // vocab_chunk_size
            num_chunks = min(num_chunks, max_vocab_chunks)
            print(f"Limiting vocab building to {num_chunks} chunks (matching --max_chunks={max_chunks})...")
        
        print("Counting token frequencies...")
        
        # Count token frequencies by processing in chunks
        self.wc = {self.unk: 1}
        tokens_processed = 0
        
        with open(filepath, 'rb') as f:
            for chunk_idx in range(num_chunks):
                # Read chunk_size tokens (each token is 2 bytes)
                bytes_to_read = min(vocab_chunk_size * 2, file_size - f.tell())
                if bytes_to_read <= 0:
                    break
                    
                chunk_bytes = f.read(bytes_to_read)
                chunk_tokens = np.frombuffer(chunk_bytes, dtype=np.uint16)
                
                # Count frequencies in this chunk
                unique_tokens, counts = np.unique(chunk_tokens, return_counts=True)
                for token_id, count in zip(unique_tokens, counts):
                    token_str = str(token_id)
                    if token_str in self.wc:
                        self.wc[token_str] += int(count)
                    else:
                        self.wc[token_str] = int(count)
                
                tokens_processed += len(chunk_tokens)
                if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
                    print(f"Processed {tokens_processed:,}/{num_tokens:,} tokens for vocab ({chunk_idx + 1}/{num_chunks} chunks)", end='\r')
        
        print("")
        print(f"Unique tokens: {len(self.wc):,}")
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

    def convert(self, filepath, chunk_size=10000000, max_chunks=None):
        print("converting corpus...")
        # Read binary file and process in chunks without loading everything into memory
        print(f"Reading binary file: {filepath}")
        
        # Get file size to calculate number of tokens
        file_size = os.path.getsize(filepath)
        num_tokens = file_size // 2  # uint16 = 2 bytes
        print(f"Total tokens: {num_tokens:,}")
        
        # Process in chunks to manage memory
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        if max_chunks is not None:
            num_chunks = min(num_chunks, max_chunks)
            print(f"Limiting to {max_chunks} chunks for testing...")
        print(f"Processing {num_chunks} chunks...")
        
        output_path = os.path.join(self.data_dir, 'train.dat')
        temp_dir = os.path.join(self.data_dir, 'temp_chunks')
        os.makedirs(temp_dir, exist_ok=True)
        
        total_pairs = 0
        chunk_files = []
        
        # Open file and read/write in chunks
        with open(filepath, 'rb') as f:
            for chunk_idx in range(num_chunks):
                # Read chunk_size tokens (each token is 2 bytes)
                bytes_to_read = min(chunk_size * 2, file_size - f.tell())
                if bytes_to_read <= 0:
                    break
                    
                chunk_bytes = f.read(bytes_to_read)
                chunk_tokens = np.frombuffer(chunk_bytes, dtype=np.uint16)
                
                # Convert token IDs to strings and filter by vocabulary
                sent = []
                for token_id in chunk_tokens:
                    token_str = str(token_id)
                    if token_str in self.vocab:
                        sent.append(token_str)
                    else:
                        sent.append(self.unk)
                
                # Generate skip-gram pairs for this chunk
                chunk_pairs = []
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    chunk_pairs.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
                
                # Write chunk to temporary file
                chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:06d}.pkl')
                with open(chunk_file, 'wb') as chunk_f:
                    pickle.dump(chunk_pairs, chunk_f)
                chunk_files.append(chunk_file)
                total_pairs += len(chunk_pairs)
                
                tokens_processed = min((chunk_idx + 1) * chunk_size, num_tokens)
                print(f"Processed chunk {chunk_idx + 1}/{num_chunks} ({tokens_processed:,}/{num_tokens:,} tokens, {total_pairs:,} pairs)", end='\r')
        
        print("")
        print(f"Combining {len(chunk_files)} chunk files...")
        
        # Combine chunks in batches to avoid loading everything into memory
        batch_size = 50  # Combine 50 chunks at a time
        combined_chunks = []
        
        for batch_start in range(0, len(chunk_files), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_files))
            batch_files = chunk_files[batch_start:batch_end]
            
            # Combine this batch
            batch_data = []
            for chunk_file in batch_files:
                with open(chunk_file, 'rb') as chunk_f:
                    chunk_data = pickle.load(chunk_f)
                    batch_data.extend(chunk_data)
                os.remove(chunk_file)
            
            # Write batch to intermediate file
            if len(chunk_files) <= batch_size:
                # If all chunks fit in one batch, write directly to output
                with open(output_path, 'wb') as out_f:
                    pickle.dump(batch_data, out_f)
            else:
                # Write to intermediate combined chunk
                combined_chunk_file = os.path.join(temp_dir, f'combined_batch_{batch_start // batch_size:06d}.pkl')
                with open(combined_chunk_file, 'wb') as out_f:
                    pickle.dump(batch_data, out_f)
                combined_chunks.append(combined_chunk_file)
            
            print(f"Combined batch {batch_start // batch_size + 1}/{(len(chunk_files) + batch_size - 1) // batch_size}", end='\r')
        
        # If we have multiple batches, combine them recursively
        while len(combined_chunks) > 1:
            new_combined = []
            for batch_start in range(0, len(combined_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(combined_chunks))
                batch_files = combined_chunks[batch_start:batch_end]
                
                batch_data = []
                for chunk_file in batch_files:
                    with open(chunk_file, 'rb') as chunk_f:
                        chunk_data = pickle.load(chunk_f)
                        batch_data.extend(chunk_data)
                    os.remove(chunk_file)
                
                if len(combined_chunks) <= batch_size:
                    with open(output_path, 'wb') as out_f:
                        pickle.dump(batch_data, out_f)
                else:
                    combined_chunk_file = os.path.join(temp_dir, f'combined_level2_{batch_start // batch_size:06d}.pkl')
                    with open(combined_chunk_file, 'wb') as out_f:
                        pickle.dump(batch_data, out_f)
                    new_combined.append(combined_chunk_file)
            
            combined_chunks = new_combined
        
        # If there's one final combined chunk, rename it to output
        if len(combined_chunks) == 1:
            os.rename(combined_chunks[0], output_path)
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass  # Directory might not be empty if something went wrong
        
        print(f"conversion done: {total_pairs:,} skip-gram pairs created")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab, vocab_chunk_size=args.vocab_chunk_size, max_chunks=args.max_chunks)
    preprocess.convert(args.corpus, chunk_size=args.chunk_size, max_chunks=args.max_chunks)

