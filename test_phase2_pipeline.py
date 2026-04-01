#!/usr/bin/env python
"""
Comprehensive test script for Phase 2: Synthetic Data Pipeline.

This script tests:
1. Data generation
2. Preprocessing
3. Feature engineering
4. Data loading
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generator import generate_synthetic_dataset
from src.preprocess import preprocess_dataset
from src.feature_engineer import engineer_dataset
from src.data_loader import load_training_data


def main():
    """Run complete Phase 2 pipeline."""
    print("=" * 70)
    print("PHASE 2: SYNTHETIC DATA PIPELINE TEST")
    print("=" * 70)
    
    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic sensor data...")
    print("-" * 70)
    try:
        data = generate_synthetic_dataset(
            n_samples=10000,
            noise_level=0.05,
            drift_factor=0.0,
        )
        print("✓ Data generation successful")
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False
    
    # Step 2: Preprocess data
    print("\n[2/4] Preprocessing data...")
    print("-" * 70)
    try:
        train, val, test = preprocess_dataset(
            raw_data_path="data/raw/synthetic_data.csv",
            output_dir="data/processed/",
        )
        print("✓ Preprocessing successful")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False
    
    # Step 3: Engineer features
    print("\n[3/4] Engineering features...")
    print("-" * 70)
    try:
        train_feat, val_feat, test_feat = engineer_dataset(
            train_path="data/processed/train.csv",
            val_path="data/processed/val.csv",
            test_path="data/processed/test.csv",
            output_dir="data/processed/",
        )
        print("✓ Feature engineering successful")
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False
    
    # Step 4: Load and test data
    print("\n[4/4] Testing data loading...")
    print("-" * 70)
    try:
        loader = load_training_data(
            data_path="data/processed/train_features.csv",
            batch_size=32,
        )
        
        # Test batch generation
        batch_count = 0
        for X, y in loader.batch_generator(batch_size=32, shuffle=True, num_epochs=1):
            batch_count += 1
            if batch_count == 1:
                print(f"  Batch shape: {X.shape}")
                print(f"  Features: {loader.get_feature_names()[:5]}... ({len(loader.get_feature_names())} total)")
        
        print(f"✓ Data loading successful ({batch_count} batches)")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 2 PIPELINE COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  ✓ data/raw/synthetic_data.csv")
    print("  ✓ data/processed/train.csv")
    print("  ✓ data/processed/val.csv")
    print("  ✓ data/processed/test.csv")
    print("  ✓ data/processed/train_features.csv")
    print("  ✓ data/processed/val_features.csv")
    print("  ✓ data/processed/test_features.csv")
    print("\nNext steps:")
    print("  - Phase 3: Exploratory analysis (EDA notebook)")
    print("  - Phase 4: Model training")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
