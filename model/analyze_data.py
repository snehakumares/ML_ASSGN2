import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_mobile_price_data():
    print("=" * 80)
    print("MOBILE PRICE CLASSIFICATION - DATA ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n1. LOADING DATA")
    print("-" * 80)

    # Try both paths to work from project root or model directory
    import os
    if os.path.exists('data/train.csv'):
        train_df = pd.read_csv('data/train.csv')
    elif os.path.exists('../data/train.csv'):
        train_df = pd.read_csv('../data/train.csv')
    else:
        raise FileNotFoundError("train.csv not found. Please ensure it's in the data/ directory")

    print(f"Dataset shape: {train_df.shape}")
    print(f"Features: {train_df.shape[1] - 1} (excluding target)")
    print(f"Instances: {train_df.shape[0]}")

    # Basic information
    print("\n2. DATASET OVERVIEW")
    print("-" * 80)
    print(train_df.info())

    # Check for missing values
    print("\n3. MISSING VALUES CHECK")
    print("-" * 80)
    missing = train_df.isnull().sum()

    if missing.sum() == 0:
        print("No missing values found")
    else:
        print("Missing values per column:")
        print(missing[missing > 0])

    # Check for duplicates
    print("\n4. DUPLICATE ROWS CHECK")
    print("-" * 80)
    duplicates = train_df.duplicated().sum()

    print(f"Duplicate rows: {duplicates}")
    if duplicates == 0:
        print("No duplicate rows found")

    # Data types
    print("\n5. DATA TYPES")
    print("-" * 80)
    print(train_df.dtypes)

    # Statistical summary
    print("\n6. STATISTICAL SUMMARY")
    print("-" * 80)
    print(train_df.describe())

    # Target variable analysis
    print("\n7. TARGET VARIABLE ANALYSIS (price_range)")
    print("-" * 80)
    print("\nClass Distribution:")
    print(train_df['price_range'].value_counts().sort_index())
    print("\nClass Percentages:")
    print(train_df['price_range'].value_counts(normalize=True).sort_index() * 100)

    # Check for class balance
    class_counts = train_df['price_range'].value_counts()
    if class_counts.max() / class_counts.min() < 1.5:
        print("\nClasses are well balanced")
    else:
        print("\nClasses are imbalanced")

    # Correlation analysis
    print("\n8. CORRELATION ANALYSIS")
    print("-" * 80)

    # Calculate correlation with target
    correlations = train_df.corr()['price_range'].sort_values(ascending=False)
    print("\nTop 10 features correlated with price_range:")
    print(correlations.head(11)[1:])  # Exclude price_range itself

    print("\nBottom 5 features (least correlated):")
    print(correlations.tail(5))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return train_df


if __name__ == "__main__":
    train_df = analyze_mobile_price_data()
