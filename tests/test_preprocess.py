"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocess import DataPreprocessor


class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_handle_missing_values_forward_fill(self, sample_sensor_data):
        """Test forward fill for missing values."""
        # Add some NaN values
        data = sample_sensor_data.copy()
        data.loc[5:7, 'heart_rate'] = np.nan
        data.loc[10, 'spo2'] = np.nan
        
        preprocessor = DataPreprocessor()
        cleaned = preprocessor._handle_missing_values(data, method='forward_fill')
        
        # Check no NaN remaining
        assert cleaned.isna().sum().sum() == 0
    
    def test_handle_missing_values_drop(self, sample_sensor_data):
        """Test dropping rows with missing values."""
        data = sample_sensor_data.copy()
        data.loc[5:7, 'heart_rate'] = np.nan
        
        preprocessor = DataPreprocessor()
        cleaned = preprocessor._handle_missing_values(data, method='drop')
        
        # Should have fewer rows
        assert len(cleaned) < len(data)
        assert cleaned.isna().sum().sum() == 0
    
    def test_remove_outliers_iqr(self, sample_sensor_data):
        """Test IQR outlier detection."""
        data = sample_sensor_data.copy()
        # Add obvious outliers
        data.loc[5, 'heart_rate'] = 200  # Very high
        data.loc[10, 'spo2'] = 70  # Very low
        
        preprocessor = DataPreprocessor()
        cleaned = preprocessor._remove_outliers(data, method='iqr', threshold=1.5)
        
        # Should have fewer rows (outliers removed)
        assert len(cleaned) < len(data)
        
        # Remaining values in reasonable range
        assert cleaned['heart_rate'].min() >= 50
        assert cleaned['heart_rate'].max() <= 150
        assert cleaned['spo2'].min() >= 94
        assert cleaned['spo2'].max() <= 100
    
    def test_preprocess(self, sample_sensor_data):
        """Test full preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        cleaned = preprocessor.preprocess(sample_sensor_data)
        
        # Check output shape and types
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0
        assert 'heart_rate' in cleaned.columns
        assert 'spo2' in cleaned.columns
        assert 'steps' in cleaned.columns
        
        # Check no NaN values
        assert cleaned.isna().sum().sum() == 0
    
    def test_normalize_fit(self, sample_sensor_data):
        """Test normalization with fit."""
        preprocessor = DataPreprocessor()
        
        normalized = preprocessor.normalize(sample_sensor_data.copy(), fit=True)
        
        # Check that scaler was fit
        assert preprocessor.feature_means is not None
        assert preprocessor.feature_stds is not None
        
        # Normalized features should be centered around 0
        assert abs(normalized['heart_rate'].mean()) < 0.5
        assert abs(normalized['spo2'].mean()) < 0.5
        assert abs(normalized['steps'].mean()) < 0.5
        
        # Normalized features should have std close to 1
        assert abs(normalized['heart_rate'].std() - 1.0) < 0.1
    
    def test_normalize_without_fit(self, sample_sensor_data):
        """Test normalization without fit (should use existing scaler)."""
        preprocessor = DataPreprocessor()
        
        # Fit on first set of data
        train_data = sample_sensor_data.iloc[:50].copy()
        preprocessor.normalize(train_data, fit=True)
        
        # Transform second set without fit
        test_data = sample_sensor_data.iloc[50:].copy()
        normalized = preprocessor.normalize(test_data, fit=False)
        
        # Should not raise error
        assert isinstance(normalized, pd.DataFrame)
        assert len(normalized) == len(test_data)
    
    def test_split_data(self, sample_sensor_data):
        """Test train/val/test split."""
        preprocessor = DataPreprocessor()
        
        train, val, test = preprocessor.split_data(
            sample_sensor_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        
        # Check sizes roughly correct
        assert len(train) + len(val) + len(test) == len(sample_sensor_data)
        assert len(train) > len(val)
        assert len(val) >= len(test)
        
        # Check temporal ordering (train should come before val/test)
        assert train['timestamp'].max() <= val['timestamp'].min()
        assert val['timestamp'].max() <= test['timestamp'].min()
    
    def test_split_data_default_ratios(self, sample_sensor_data):
        """Test split with default ratios."""
        preprocessor = DataPreprocessor()
        train, val, test = preprocessor.split_data(sample_sensor_data)
        
        # Default is 70/15/15
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02


class TestPreprocessingIntegration:
    """Integration tests for preprocessing."""
    
    def test_full_pipeline(self, sample_sensor_data):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        
        # Preprocess
        data = preprocessor.preprocess(sample_sensor_data)
        
        # Split
        train, val, test = preprocessor.split_data(data)
        
        # Normalize (fit on train only)
        train = preprocessor.normalize(train, fit=True)
        val = preprocessor.normalize(val, fit=False)
        test = preprocessor.normalize(test, fit=False)
        
        # Check all are valid
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            assert len(split_data) > 0
            assert split_data.isna().sum().sum() == 0
            print(f"✓ {split_name}: {len(split_data)} samples")
    
    def test_no_data_leakage(self, sample_sensor_data):
        """Test that normalization stats don't leak from val/test to train."""
        preprocessor = DataPreprocessor()
        
        data = preprocessor.preprocess(sample_sensor_data)
        train, val, test = preprocessor.split_data(data)
        
        # Fit on train only
        train = preprocessor.normalize(train, fit=True)
        train_stats = (preprocessor.feature_means, preprocessor.feature_stds)
        
        # Transform val/test
        val = preprocessor.normalize(val, fit=False)
        test = preprocessor.normalize(test, fit=False)
        
        # Stats should not have changed
        assert np.allclose(preprocessor.feature_means, train_stats[0])
        assert np.allclose(preprocessor.feature_stds, train_stats[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
