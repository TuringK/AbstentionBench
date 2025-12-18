"""
Split sample_pairs.csv into train and validation sets.
Stratified split to maintain balance of should_abstain labels.

Usage:
    python split_data.py --input data/sample_pairs.csv --output_dir data/ --val_ratio 0.2
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val sets")
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench/PromptTuning/data/sample_pairs.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/parscratch/users/acb20df/private/TuringProj/AbstentionBench/PromptTuning/data/",
        help="Directory to save train.csv and val.csv",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total samples: {len(df)}")
    
    # Check class distribution
    print(f"\nClass distribution (should_abstain):")
    print(df["should_abstain"].value_counts())
    
    # Stratified split
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=df["should_abstain"],  # Maintain class balance
    )
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"  - should_abstain=True:  {train_df['should_abstain'].sum()}")
    print(f"  - should_abstain=False: {(~train_df['should_abstain']).sum()}")
    
    print(f"\nVal set: {len(val_df)} samples")
    print(f"  - should_abstain=True:  {val_df['should_abstain'].sum()}")
    print(f"  - should_abstain=False: {(~val_df['should_abstain']).sum()}")
    
    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\nSaved:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")


if __name__ == "__main__":
    main()