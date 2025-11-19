# -*- coding: utf-8 -*-
"""
Convert OpenWebText .bin files (tokenized binary) to text format.
Requires transformers library to decode GPT-2 BPE tokens.
"""

import os
import numpy as np
import argparse
from transformers import GPT2Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_bin', type=str, required=True, help="Input .bin file path")
    parser.add_argument('--output_txt', type=str, required=True, help="Output .txt file path")
    parser.add_argument('--tokenizer', type=str, default='gpt2', help="Tokenizer name (default: gpt2)")
    parser.add_argument('--max_lines', type=int, default=None, help="Maximum number of lines to process (for testing)")
    return parser.parse_args()


def convert_bin_to_text(input_bin, output_txt, tokenizer_name='gpt2', max_lines=None):
    """
    Convert binary tokenized file to text format.
    
    Args:
        input_bin: Path to input .bin file (uint16 token IDs)
        output_txt: Path to output .txt file (plain text)
        tokenizer_name: Name of the tokenizer to use for decoding
        max_lines: Maximum number of lines to process (None for all)
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    
    print(f"Reading binary file: {input_bin}")
    # Get file size to calculate number of tokens
    file_size = os.path.getsize(input_bin)
    num_tokens = file_size // 2  # uint16 = 2 bytes
    print(f"Total tokens: {num_tokens:,}")
    print(f"Decoding tokens to text...")
    
    # Decode tokens in chunks to avoid memory issues
    # Use a reasonable chunk size that balances memory and efficiency
    chunk_size = 1000000  # Process 1M tokens at a time
    lines_written = 0
    tokens_processed = 0
    
    with open(input_bin, 'rb') as f_in, open(output_txt, 'w', encoding='utf-8') as f_out:
        while tokens_processed < num_tokens:
            if max_lines and lines_written >= max_lines:
                break
            
            # Read chunk_size tokens (each token is 2 bytes)
            bytes_to_read = min(chunk_size * 2, file_size - f_in.tell())
            if bytes_to_read <= 0:
                break
                
            chunk_bytes = f_in.read(bytes_to_read)
            chunk_tokens = np.frombuffer(chunk_bytes, dtype=np.uint16)
            
            # Decode chunk
            text = tokenizer.decode(chunk_tokens.tolist(), skip_special_tokens=True)
            
            # Split by newlines if present, otherwise write as single line
            text_lines = text.split('\n')
            for line in text_lines:
                if max_lines and lines_written >= max_lines:
                    break
                line = line.strip()
                if line:  # Only write non-empty lines
                    f_out.write(line + '\n')
                    lines_written += 1
            
            tokens_processed += len(chunk_tokens)
            if (tokens_processed // chunk_size) % 100 == 0 or tokens_processed >= num_tokens:
                print(f"Processed {tokens_processed:,}/{num_tokens:,} tokens ({lines_written} lines)", end='\r')
    
    print(f"\nConversion complete! Wrote {lines_written} lines to {output_txt}")


if __name__ == '__main__':
    args = parse_args()
    convert_bin_to_text(args.input_bin, args.output_txt, args.tokenizer, args.max_lines)

