#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import sys
import os

def inspect_train_dat(filepath, num_samples=5):
    """Inspect train.dat - shows first entries and counts.
    
    Note: Pickle format requires full file load. For 26GB files, this takes
    time and memory. If it's too slow, check preprocessing logs instead.
    """
    file_size = os.path.getsize(filepath)
    file_size_gb = file_size / (1024**3)
    
    print(f"File: {filepath}")
    print(f"Size: {file_size_gb:.2f} GB")
    print("\n⚠️  Pickle limitation: Must load entire file (no streaming)")
    print("   For 26GB file, expect 2-5 minutes and 30-50GB RAM usage.")
    print("   Press Ctrl+C to cancel.\n")
    
    try:
        print("Loading...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        total_entries = len(data)
        print(f"\n✓ Loaded {total_entries:,} entries\n")
        
        # Show first few
        print(f"First {num_samples} entries:")
        print("-" * 80)
        for i in range(min(num_samples, len(data))):
            iword, owords = data[i]
            print(f"Entry {i}:")
            print(f"  Input word index: {iword}")
            print(f"  Context word indices: {owords}")
            print(f"  Number of context words: {len(owords)}")
            print()
        
        # Estimate total context pairs
        sample_size = min(10000, total_entries)
        avg_context = sum(len(owords) for _, owords in data[:sample_size]) / sample_size
        total_context_pairs = int(avg_context * total_entries)
        
        print(f"Total skip-gram pairs: {total_entries:,}")
        print(f"Average context words per entry: {avg_context:.2f}")
        print(f"Estimated total context pairs: {total_context_pairs:,}")
        
        return total_entries, total_context_pairs
        
    except MemoryError:
        print("\n❌ Out of memory - file too large.")
        print("\n💡 Alternative: The preprocessing script prints the count!")
        print("   Look for output like: 'conversion done: X skip-gram pairs created'")
        return None, None
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled - loading takes too long.")
        print("\n💡 Tip: Check preprocessing output for the total count.")
        return None, None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None, None

if __name__ == '__main__':
    filepath = '/cache/openwebtext/train/train.dat'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    inspect_train_dat(filepath)
