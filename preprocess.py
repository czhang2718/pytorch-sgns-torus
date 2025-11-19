# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
import gc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='/cache/openwebtext/val.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='/cache/openwebtext/val.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    parser.add_argument('--max_lines', type=int, default=None, help="maximum number of lines to process (for testing/partial processing)")
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
        step = 0
        self.wc = {self.unk: 1}
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = line.split()
                for word in sent:
                    self.wc[word] = self.wc.get(word, 0) + 1
        print("")
        self.idx2word = [self.unk] + sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.data_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.data_dir, 'word2idx.dat'), 'wb'))
        print("build done")

    def convert(self, filepath, chunk_size=1000000, max_lines=None):
        print("converting corpus...")
        if max_lines:
            print(f"Processing up to {max_lines:,} lines")
        step = 0
        chunk_data = []
        chunk_idx = 0
        total_pairs = 0
        temp_dir = os.path.join(self.data_dir, 'temp_chunks')
        os.makedirs(temp_dir, exist_ok=True)
        chunk_files = []
        
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if max_lines and step > max_lines:
                    break
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                sent = []
                for word in line.split():
                    if word in self.vocab:
                        sent.append(word)
                    else:
                        sent.append(self.unk)
                for i in range(len(sent)):
                    iword, owords = self.skipgram(sent, i)
                    chunk_data.append((self.word2idx[iword], [self.word2idx[oword] for oword in owords]))
                    total_pairs += 1
                
                # Write chunk to file periodically to avoid memory buildup
                if len(chunk_data) >= chunk_size:
                    chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:06d}.pkl')
                    pickle.dump(chunk_data, open(chunk_file, 'wb'))
                    chunk_files.append(chunk_file)
                    chunk_data = []
                    chunk_idx += 1
        
        # Write remaining data
        if chunk_data:
            chunk_file = os.path.join(temp_dir, f'chunk_{chunk_idx:06d}.pkl')
            pickle.dump(chunk_data, open(chunk_file, 'wb'))
            chunk_files.append(chunk_file)
        
        print("")
        print(f"Combining {len(chunk_files)} chunk files...")
        
        # Combine chunks in smaller batches to avoid memory issues
        # Use a much smaller batch size to reduce memory footprint
        # With 6548 chunks, we need to be very conservative with memory
        batch_size = 5
        combined_chunks = chunk_files
        merge_round = 0
        
        while len(combined_chunks) > 1:
            merge_round += 1
            print(f"  Merge round {merge_round}: combining {len(combined_chunks)} chunks...", end='\r')
            new_combined = []
            
            for batch_start in range(0, len(combined_chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(combined_chunks))
                batch_files = combined_chunks[batch_start:batch_end]
                
                # Process chunks one at a time and write incrementally to reduce memory
                batch_data = []
                for chunk_file in batch_files:
                    with open(chunk_file, 'rb') as chunk_f:
                        chunk_data = pickle.load(chunk_f)
                        batch_data.extend(chunk_data)
                        del chunk_data  # Explicitly delete to free memory
                    # Delete chunk file immediately after loading to free disk space
                    os.remove(chunk_file)
                
                # If this is the final batch and we're done, write to output
                if len(combined_chunks) <= batch_size and len(new_combined) == 0:
                    output_path = os.path.join(self.data_dir, 'train.dat')
                    with open(output_path, 'wb') as out_f:
                        pickle.dump(batch_data, out_f)
                    del batch_data  # Clear reference
                    gc.collect()  # Force garbage collection
                    break
                else:
                    # Write combined batch to new chunk file
                    combined_chunk_file = os.path.join(temp_dir, f'combined_r{merge_round}_{batch_start // batch_size:06d}.pkl')
                    with open(combined_chunk_file, 'wb') as out_f:
                        pickle.dump(batch_data, out_f)
                    new_combined.append(combined_chunk_file)
                    del batch_data  # Clear reference to help GC
                    gc.collect()  # Force garbage collection after each batch
            
            combined_chunks = new_combined
            gc.collect()  # Force garbage collection after each merge round
        
        # If there's one final combined chunk, rename it to output
        if len(combined_chunks) == 1:
            output_path = os.path.join(self.data_dir, 'train.dat')
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(combined_chunks[0], output_path)
        
        print("")  # New line after progress indicator
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            try:
                # Remove any remaining files
                for f in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, f))
                os.rmdir(temp_dir)
            except OSError:
                pass
        
        print(f"conversion done: {total_pairs:,} skip-gram pairs created")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window, unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.corpus, max_lines=args.max_lines)
