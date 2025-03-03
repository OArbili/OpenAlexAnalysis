#!/usr/bin/env python
# run_sliding_window.py
# Script to run the sliding window approach for increased stability

import os
import sys
import argparse
from subfield_gnn import set_seed
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run subfield collaboration model with sliding window approach')
    parser.add_argument('--data', type=str, default="../data/all_articles_2023_2024.jsonl",
                       help='Path to the data file')
    parser.add_argument('--window-size', type=int, default=10000,
                       help='Size of sliding window (number of articles)')
    parser.add_argument('--stride', type=int, default=5000,
                       help='Stride between windows (number of articles)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with smaller settings')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    print("="*80)
    print("Running Subfield Collaboration Analysis with Sliding Window Approach")
    print("="*80)
    print(f"Data path: {args.data}")
    print(f"Window size: {args.window_size} articles")
    print(f"Stride: {args.stride} articles")
    print(f"Training epochs: {args.epochs}")
    print(f"Random seed: {args.seed}")
    print(f"Debug mode: {'Yes' if args.debug else 'No'}")
    print("="*80)
    
    # Build command to run the model
    cmd = ["python", "run_model.py"]
    
    if args.debug:
        cmd.append("--debug")
    
    cmd.extend(["--window-size", str(args.window_size),
               "--stride", str(args.stride),
               "--epochs", str(args.epochs),
               "--seed", str(args.seed)])
    
    # Run the command
    print("Starting analysis...")
    subprocess.run(cmd)
    
    print("\nAnalysis complete!")
    print("If the results are satisfactory, you can use 'redraw_networks.py' to customize visualizations.")

if __name__ == "__main__":
    main()