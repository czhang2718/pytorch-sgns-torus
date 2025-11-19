# -*- coding: utf-8 -*-
"""
Convert text files to tokenized binary format (.bin files).
Uses GPT-2 tokenizer to encode text into uint16 token IDs.
"""

import os
import numpy as np
import argparse
from transformers import GPT2Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True, help="Input .txt file path")
    parser.add_argument('--output_bin', type=str, required=True, help="Output .bin file path")
    parser.add_argument('--tokenizer', type=str, default='gpt2', help="Tokenizer name (default: gpt2)")
    parser.add_argument('--chunk_size', type=int, default=10000, help="Number of lines to process before writing (default: 10000)")
    parser.add_argument('--max_lines', type=int, default=None, help="Maximum number of lines to process (for testing)")
    return parser.parse_args()


def convert_text_to_bin(input_txt, output_bin, tokenizer_name='gpt2', chunk_size=10000, max_lines=None):
    """
    Convert text file to binary tokenized format.
    
    Args:
        input_txt: Path to input .txt file (plain text, one line per document)
        output_bin: Path to output .bin file (uint16 token IDs)
        tokenizer_name: Name of the tokenizer to use for encoding
        chunk_size: Number of lines to accumulate before writing to disk
        max_lines: Maximum number of lines to process (None for all)
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    
    print(f"Reading text file: {input_txt}")
    file_size = os.path.getsize(input_txt)
    print(f"Input file size: {file_size / (1024**3):.2f} GB")
    
    print(f"Tokenizing and writing to: {output_bin}")
    
    lines_processed = 0
    tokens_written = 0
    chunk_tokens = []
    
    with open(input_txt, 'r', encoding='utf-8') as f_in, open(output_bin, 'wb') as f_out:
        for line in f_in:
            if max_lines and lines_processed >= max_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Tokenize the line
            token_ids = tokenizer.encode(line, add_special_tokens=False)
            
            # Add token IDs to chunk
            chunk_tokens.extend(token_ids)
            
            lines_processed += 1
            
            # Write chunk when it reaches chunk_size lines
            if lines_processed % chunk_size == 0:
                # Convert to uint16 array and write
                token_array = np.array(chunk_tokens, dtype=np.uint16)
                f_out.write(token_array.tobytes())
                tokens_written += len(chunk_tokens)
                chunk_tokens = []
                
                print(f"Processed {lines_processed:,} lines ({tokens_written:,} tokens)", end='\r')
        
        # Write remaining tokens
        if chunk_tokens:
            token_array = np.array(chunk_tokens, dtype=np.uint16)
            f_out.write(token_array.tobytes())
            tokens_written += len(chunk_tokens)
    
    print(f"\nConversion complete!")
    print(f"  Lines processed: {lines_processed:,}")
    print(f"  Tokens written: {tokens_written:,}")
    print(f"  Output file size: {os.path.getsize(output_bin) / (1024**3):.2f} GB")


if __name__ == '__main__':
    args = parse_args()
    convert_text_to_bin(args.input_txt, args.output_bin, args.tokenizer, args.chunk_size, args.max_lines)





