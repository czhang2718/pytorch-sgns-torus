#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import os

data_dir = '/cache/openwebtext/train'
filepath = os.path.join(data_dir, 'train.dat')
idx2word_path = os.path.join(data_dir, 'idx2word.dat')

print(f"Loading idx2word mapping...")
with open(idx2word_path, 'rb') as f:
    idx2word = pickle.load(f)

print(f"Loading {filepath}...")
print("(Still requires full file load - pickle limitation)\n")

try:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Loaded {len(data):,} entries\n")
    
    
    print("First 100 entries:")
    print("-" * 80)
    for i in range(min(100, len(data))):
        iword, owords = data[i]
        input_word = idx2word[iword] if iword < len(idx2word) else f"<UNK:{iword}>"
        context_words = [idx2word[idx] if idx < len(idx2word) else f"<UNK:{idx}>" for idx in owords]
        
        print(f"Entry {i}:")
        print(f"  Input word: '{input_word}' (index: {iword})")
        print(f"  Context words: {context_words}")
        print(f"  Number of context words: {len(owords)}")
        print()
        
except KeyboardInterrupt:
    print("\n⚠️  Cancelled")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

