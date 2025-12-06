"""
Count (word, context) co-occurrences from train.dat
Outputs wc_pairs.dat: dict mapping (w, c) tuples to counts
"""
import pickle
from collections import Counter
import sys

DATA_DIR = "/cache/openwebtext/train"

def main():
    print("Loading train.dat...", flush=True)
    with open(f"{DATA_DIR}/train.dat", "rb") as f:
        train = pickle.load(f)
    
    print(f"Loaded {len(train):,} samples", flush=True)
    
    # Count (word, context) pairs
    pair_counts = Counter()
    
    total = len(train)
    report_every = 10_000_000
    
    for i, (word, context_list) in enumerate(train):
        for ctx in context_list:
            if ctx != 0:  # skip padding (0 is typically <UNK> or padding)
                pair_counts[(word, ctx)] += 1
        
        if (i + 1) % report_every == 0:
            print(f"  Processed {i+1:,}/{total:,} ({100*(i+1)/total:.1f}%) - {len(pair_counts):,} unique pairs", flush=True)
    
    print(f"\nDone! {len(pair_counts):,} unique (word, context) pairs", flush=True)
    
    # Save
    out_path = f"{DATA_DIR}/wc_pairs.dat"
    print(f"Saving to {out_path}...", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(dict(pair_counts), f)
    
    print("Saved!", flush=True)
    
    # Show some stats
    print("\nTop 10 most common pairs:")
    with open(f"{DATA_DIR}/idx2word.dat", "rb") as f:
        idx2word = pickle.load(f)
    
    for (w, c), count in pair_counts.most_common(10):
        w_str = idx2word[w] if w < len(idx2word) else f"[{w}]"
        c_str = idx2word[c] if c < len(idx2word) else f"[{c}]"
        print(f"  ({w_str}, {c_str}): {count:,}")

if __name__ == "__main__":
    main()




