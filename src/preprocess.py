"""
Data preprocessing module.

Handles:
- Missing value imputation
- Outlier detection and removal
- Feature normalization
- Train/Val/Test splitting
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from pathlib import Path


class DataPreprocessor:
    """Preprocess raw sensor data."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.feature_means = None
        self.feature_stds = None
    
    def preprocess(
        self,
        data: pd.DataFrame,
        handle_missing: str = "forward_fill",
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Preprocess raw data.
        
        Args:
            data: Raw DataFrame with columns: timestamp, heart_rate, spo2, steps
            handle_missing: How to handle NaN ('forward_fill', 'drop', 'mean')
            outlier_method: Outlier detection method ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
        
        Returns:
            Cleaned and preprocessed DataFrame
        """
        data = data.copy()
        
        print("Starting preprocessing...")
        print(f"  Initial shape: {data.shape}")
        
        # Handle missing values
        data = self._handle_missing_values(data, method=handle_missing)
        print(f"  After missing value handling: {data.shape}")
        
        # Detect and remove outliers
        data = self._remove_outliers(
            data,
            method=outlier_method,
            threshold=outlier_threshold,
        )
        print(f"  After outlier removal: {data.shape}")
        
        return data
    
    def normalize(
        self,
        data: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame:
        """
        Normalize features using StandardScaler.
        
        Args:
            data: DataFrame with sensor columns
            fit: If True, fit scaler on this data (training data)
        
        Returns:
            Normalized DataFrame
        """
        data = data.copy()
        feature_cols = ['heart_rate', 'spo2', 'steps']
        
        if fit:
            # Fit scaler on training data
            self.scaler.fit(data[feature_cols])
            self.feature_means = self.scaler.mean_
            self.feature_stds = self.scaler.scale_
            print("✓ Scaler fitted on training data")
        
        # Transform
        data[feature_cols] = self.scaler.transform(data[feature_cols])
        
        return data
    
    def split_data(
        self,
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test with temporal ordering.
        
        Args:
            data: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(data)
        
        # Compute split indices
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        # Split (temporal ordering preserved)
        train_data = data.iloc[:train_idx].reset_index(drop=True)
        val_data = data.iloc[train_idx:val_idx].reset_index(drop=True)
        test_data = data.iloc[val_idx:].reset_index(drop=True)
        
        print(f"✓ Data split:")
        print(f"  Train: {len(train_data)} samples ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_data)} samples ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_data)} samples ({test_ratio*100:.0f}%)")
        
        return train_data, val_data, test_data
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        method: str = "forward_fill",
    ) -> pd.DataFrame:
        """Handle missing values in dataset."""
        data = data.copy()
        
        if method == "forward_fill":
            data = data.fillna(method="ffill").fillna(method="bfill")
        elif method == "drop":
            data = data.dropna()
        elif method == "mean":
            data = data.fillna(data.mean())
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data
    
    def _remove_outliers(
        self,
        data: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """Detect and remove outliers."""
        data = data.copy()
        initial_len = len(data)
        
        feature_cols = ['heart_rate', 'spo2', 'steps']
        
        if method == "iqr":
            # Interquartile Range method
            for col in feature_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        elif method == "zscore":
            # Z-score method
            for col in feature_cols:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data = data[z_scores < threshold]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        removed = initial_len - len(data)
        if removed > 0:
            print(f"  Removed {removed} outliers ({removed/initial_len*100:.1f}%)")
        
        return data


def preprocess_dataset(
    raw_data_path: str = "data/raw/synthetic_data.csv",
    output_dir: str = "data/processed/",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete preprocessing pipeline.
    
    Args:
        raw_data_path: Path to raw CSV
        output_dir: Where to save processed splits
        train_ratio: Train/val/test split ratios
        val_ratio: ...
        test_ratio: ...
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load raw data
    print(f"Loading data from {raw_data_path}...")
    data = pd.read_csv(raw_data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess(data)
    
    # Split before normalization
    train, val, test = preprocessor.split_data(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    
    # Normalize (fit on training data only)
    print("\nNormalizing data...")
    train = preprocessor.normalize(train, fit=True)
    val = preprocessor.normalize(val, fit=False)
    test = preprocessor.normalize(test, fit=False)
    
    # Save processed splits
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_path = f"{output_dir}/train.csv"
    val_path = f"{output_dir}/val.csv"
    test_path = f"{output_dir}/test.csv"
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    
    print(f"\n✓ Processed data saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    
    # Log statistics
    print(f"\nNormalization statistics (from training set):")
    print(f"  HR mean: {preprocessor.feature_means[0]:.2f}, std: {preprocessor.feature_stds[0]:.2f}")
    print(f"  SpO2 mean: {preprocessor.feature_means[1]:.2f}, std: {preprocessor.feature_stds[1]:.2f}")
    print(f"  Steps mean: {preprocessor.feature_means[2]:.2f}, std: {preprocessor.feature_stds[2]:.2f}")
    
    return train, val, test


if __name__ == "__main__":
    # Example usage
    train, val, test = preprocess_dataset()
    print("\nTrain data sample:")
    print(train.head())
