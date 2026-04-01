"""
Feature engineering module.

Extracts time-series features from raw sensor data:
- Rolling window statistics (mean, std, min, max)
- Heart Rate Variability (HRV) metrics
- Trend features (slopes)
- Lag features (previous values)
- Temporal context (hour of day)
- Derived metrics (HR normalized by activity)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path


class FeatureEngineer:
    """Extract features from preprocessed sensor data."""
    
    def __init__(self, window_sizes: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            window_sizes: List of window sizes (in samples) for rolling aggregations
                         Default: [10, 20] (5-10 min windows at 30-sec sampling)
        """
        if window_sizes is None:
            window_sizes = [10, 20]  # 5 and 10 minute windows
        self.window_sizes = window_sizes
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        label_col: str = None,
    ) -> pd.DataFrame:
        """
        Engineer all features from preprocessed data.
        
        Args:
            data: DataFrame with normalized columns: timestamp, heart_rate, spo2, steps
            label_col: Optional column name with stress/fatigue labels
        
        Returns:
            DataFrame with engineered features
        """
        features = data[['timestamp']].copy()
        
        print("Engineering features...")
        
        # 1. Rolling statistics
        features = self._add_rolling_statistics(features, data)
        
        # 2. Trend features
        features = self._add_trend_features(features, data)
        
        # 3. Lag features
        features = self._add_lag_features(features, data)
        
        # 4. Heart Rate Variability (HRV)
        features = self._add_hrv_features(features, data)
        
        # 5. Temporal context
        features = self._add_temporal_features(features, data)
        
        # 6. Derived metrics
        features = self._add_derived_features(features, data)
        
        # 7. Optional: Add labels if provided
        if label_col and label_col in data.columns:
            features[label_col] = data[label_col].values
        
        # Drop rows with NaN (from rolling windows)
        features = features.dropna().reset_index(drop=True)
        
        print(f"✓ Engineered {features.shape[1] - 1} features from {len(features)} samples")
        
        return features
    
    def _add_rolling_statistics(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add rolling window statistics."""
        for window in self.window_sizes:
            # Heart Rate statistics
            features[f'hr_mean_w{window}'] = data['heart_rate'].rolling(window).mean()
            features[f'hr_std_w{window}'] = data['heart_rate'].rolling(window).std()
            features[f'hr_min_w{window}'] = data['heart_rate'].rolling(window).min()
            features[f'hr_max_w{window}'] = data['heart_rate'].rolling(window).max()
            features[f'hr_range_w{window}'] = (
                features[f'hr_max_w{window}'] - features[f'hr_min_w{window}']
            )
            
            # SpO2 statistics
            features[f'spo2_mean_w{window}'] = data['spo2'].rolling(window).mean()
            features[f'spo2_std_w{window}'] = data['spo2'].rolling(window).std()
            
            # Steps statistics
            features[f'steps_sum_w{window}'] = data['steps'].rolling(window).sum()
            features[f'steps_mean_w{window}'] = data['steps'].rolling(window).mean()
        
        return features
    
    def _add_trend_features(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add trend features (slopes and derivatives)."""
        # HR trend (slope over last window)
        for window in self.window_sizes:
            hr_rolled = data['heart_rate'].rolling(window)
            # Simple slope: (last - first) / window
            features[f'hr_trend_w{window}'] = (
                data['heart_rate'].diff(window-1) / (window-1)
            )
            
            # SpO2 trend
            features[f'spo2_trend_w{window}'] = (
                data['spo2'].diff(window-1) / (window-1)
            )
        
        return features
    
    def _add_lag_features(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add lagged features (previous values)."""
        # Lag-1 features (previous sample)
        features['hr_lag1'] = data['heart_rate'].shift(1)
        features['spo2_lag1'] = data['spo2'].shift(1)
        features['steps_lag1'] = data['steps'].shift(1)
        
        # Lag-2 features
        features['hr_lag2'] = data['heart_rate'].shift(2)
        features['steps_lag2'] = data['steps'].shift(2)
        
        return features
    
    def _add_hrv_features(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add Heart Rate Variability features.
        
        Note: These are computed on normalized data, so they represent 
        deviations from the mean distribution.
        """
        # RMSSD: Root Mean Square of Successive Differences
        # Approximated using rolling std of HR
        features['hr_rmssd_w10'] = data['heart_rate'].rolling(10).std()
        
        # NN50: Number of successive HR differences > 50ms equivalent
        # Approximated as variance metric
        for window in self.window_sizes:
            features[f'hr_variability_w{window}'] = (
                data['heart_rate'].rolling(window).apply(
                    lambda x: np.sum(np.abs(np.diff(x)) > 0.1) if len(x) > 1 else 0
                )
            )
        
        return features
    
    def _add_temporal_features(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add temporal context features."""
        # Hour of day (sine/cosine encoding for circularity)
        hour = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60.0
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # Day of week
        features['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Is weekend
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        return features
    
    def _add_derived_features(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add domain-specific derived metrics."""
        # HR normalized by activity (stress indicator)
        # Avoid division by zero
        features['hr_activity_ratio_w10'] = (
            features['hr_mean_w10'] / (features['steps_mean_w10'] + 0.1)
        )
        features['hr_activity_ratio_w20'] = (
            features['hr_mean_w20'] / (features['steps_mean_w20'] + 0.1)
        )
        
        # HR variability indicator
        features['hr_stability_w10'] = 1.0 / (features['hr_std_w10'] + 0.01)
        
        # SpO2 deviation from (normalized) baseline of 0
        features['spo2_deviation_w10'] = np.abs(features['spo2_mean_w10'])
        
        # Combined stress metric: high HR + low O2 + low activity
        features['composite_stress_w10'] = (
            features[f'hr_std_w10'] * 
            features['spo2_deviation_w10'] * 
            np.clip(1.0 - features['steps_mean_w10'], 0, 1)
        )
        
        return features


def engineer_dataset(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    test_path: str = "data/processed/test.csv",
    output_dir: str = "data/processed/",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Engineer features for train/val/test splits.
    
    Args:
        train_path: Path to normalized training data
        val_path: Path to normalized validation data
        test_path: Path to normalized test data
        output_dir: Where to save engineered features
    
    Returns:
        Tuple of (train_features, val_features, test_features)
    """
    engineer = FeatureEngineer(window_sizes=[10, 20])
    
    print("Loading preprocessed data...")
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    # Convert timestamps
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    val['timestamp'] = pd.to_datetime(val['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    # Engineer features
    print("\nEngineering features for train set...")
    train_features = engineer.engineer_features(train)
    
    print("Engineering features for val set...")
    val_features = engineer.engineer_features(val)
    
    print("Engineering features for test set...")
    test_features = engineer.engineer_features(test)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_feat_path = f"{output_dir}/train_features.csv"
    val_feat_path = f"{output_dir}/val_features.csv"
    test_feat_path = f"{output_dir}/test_features.csv"
    
    train_features.to_csv(train_feat_path, index=False)
    val_features.to_csv(val_feat_path, index=False)
    test_features.to_csv(test_feat_path, index=False)
    
    print(f"\n✓ Engineered features saved:")
    print(f"  Train: {train_feat_path} ({train_features.shape})")
    print(f"  Val:   {val_feat_path} ({val_features.shape})")
    print(f"  Test:  {test_feat_path} ({test_features.shape})")
    
    return train_features, val_features, test_features


if __name__ == "__main__":
    # Example usage
    train_feat, val_feat, test_feat = engineer_dataset()
    print("\nFeature list:")
    print(list(train_feat.columns))
    print(f"\nTrain features sample:")
    print(train_feat.head())
