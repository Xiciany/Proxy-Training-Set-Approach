import pandas as pd
import os

# ========== Path Configuration ==========
# The script assumes the data file is in "../data" relative to the script's location.
# If you run this script from a different location, adjust DATA_DIR accordingly.
DATA_DIR = "../data"
FILE_NAME = "Twibot-22（part）.csv"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)


def main():
    print("=" * 70)
    print("Feature Summary of Twibot-22 (part) Dataset")
    print("=" * 70)

    # Read CSV (with proper encoding for Chinese parentheses)
    df = pd.read_csv(FILE_PATH, encoding='utf-8-sig')

    print(f"Total rows    : {len(df):,}")
    print(f"Total columns : {len(df.columns)}")
    print("\n" + "=" * 70)
    print("Column Information (dtype, non-null counts, missing values)")
    print("-" * 70)

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        nulls = df[col].isnull().sum()
        print(f"{col:35s} | dtype={str(dtype):12s} | non-null={non_null:>8,} | nulls={nulls:>6}")

    print("\n" + "=" * 70)
    print("Sample Data (first 5 rows)")
    print("-" * 70)
    # Display all columns without truncation
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(df.head(5))

    print("\n" + "=" * 70)
    print("Basic Statistics for Numeric Columns")
    print("-" * 70)
    # Show summary statistics for numeric columns
    print(df.describe())

    print("\n" + "=" * 70)
    print("Output completed. No files were saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()