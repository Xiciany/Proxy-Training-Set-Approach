# -*- coding: utf-8 -*-
"""
Knowledge Graph Inference for Feature Derivation (Batch Processing).

This script reads all CSV files from the Input folder, applies inference rules
to generate new features, and saves the enhanced datasets to the Output folder.

Usage:
    Simply run this script. It will process all CSV files in ./Input
    and save results to ./Output with the same filenames (or with "_enhanced" suffix).

Folder structure:
    ./Input/         -- place your target datasets here
    ./Output/        -- enhanced datasets will be saved here (created if not exists)
"""

import os
import pandas as pd
import numpy as np
from feature_extraction_utils import (
    compute_follower_following_ratio,
    compute_avg_tweets_per_day,
    compute_description_length,
    compute_has_any_url,
    compute_special_char_ratio,
    compute_english_ratio,
    derive_features
)

# ==================== Configuration ====================
INPUT_DIR = "../Input"      # Folder containing input CSV files
OUTPUT_DIR = "../Output"    # Folder to save enhanced CSV files
BASELINE_FILE = "../data/Twibot-22（part）.csv"  # Path to baseline feature set

# ==================== Predefined Inference Rules ====================
INFERENCE_RULES = [
    {
        'func': compute_follower_following_ratio,
        'kwargs': {
            'follower_col': 'followers_count',
            'following_col': 'following_count',
            'new_col': 'follower_following_ratio'
        }
    },
    {
        'func': compute_avg_tweets_per_day,
        'kwargs': {
            'tweet_col': 'tweet_count',
            'age_col': 'user_age_days',
            'new_col': 'avg_tweets_per_day'
        }
    },
    {
        'func': compute_description_length,
        'kwargs': {
            'desc_col': 'description',
            'new_col': 'desc_length'
        }
    },
    {
        'func': compute_has_any_url,
        'kwargs': {
            'url_cols': ['has_url', 'desc_has_url'],
            'new_col': 'has_any_url'
        }
    },
    {
        'func': compute_special_char_ratio,
        'kwargs': {
            'text_col': 'description',
            'new_col': 'special_char_ratio'
        }
    },
    {
        'func': compute_english_ratio,
        'kwargs': {
            'text_col': 'description',
            'new_col': 'english_ratio'
        }
    }
]

# ==================== Main Function ====================
def main():
    # Create output directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load baseline columns for reference (optional)
    if os.path.exists(BASELINE_FILE):
        baseline_df = pd.read_csv(BASELINE_FILE, nrows=0)
        baseline_cols = set(baseline_df.columns)
        print(f"Loaded baseline columns ({len(baseline_cols)}) from {BASELINE_FILE}")
    else:
        baseline_cols = set()
        print(f"Warning: Baseline file not found at {BASELINE_FILE}. Proceeding without baseline reference.")

    # Get list of CSV files in Input directory
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{INPUT_DIR}'. Exiting.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process.")

    for filename in csv_files:
        input_path = os.path.join(INPUT_DIR, filename)
        print(f"\nProcessing: {filename}")

        # Read the CSV
        df = pd.read_csv(input_path, encoding='utf-8-sig')
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

        # Apply inference rules
        applied = []
        for rule in INFERENCE_RULES:
            func = rule['func']
            kwargs = rule['kwargs']
            new_col = kwargs.get('new_col')

            # Skip if new column already exists
            if new_col is not None and new_col in df.columns:
                print(f"  - Column '{new_col}' already exists. Skipping.")
                continue

            # Check if required source columns exist
            required_source_cols = []
            for v in kwargs.values():
                if isinstance(v, str) and v in df.columns:
                    required_source_cols.append(v)
                elif isinstance(v, list):
                    required_source_cols.extend([c for c in v if isinstance(c, str) and c in df.columns])
            if not required_source_cols:
                print(f"  - Rule for '{new_col}' has no available source columns. Skipping.")
                continue

            # Apply the rule
            try:
                df = func(df, **kwargs)
                applied.append(new_col)
                print(f"  - Derived '{new_col}' successfully.")
            except Exception as e:
                print(f"  - Error applying rule for '{new_col}': {e}")

        # Determine output filename (use original name or add suffix)
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_enhanced{ext}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Save enhanced dataframe
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved enhanced file to: {output_path} (columns: {len(df.columns)})")

    print("\nAll files processed successfully.")

if __name__ == "__main__":
    main()