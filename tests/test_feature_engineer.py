"""
Unit tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(window_sizes=[5, 10])
        assert engineer.window_sizes == [5, 10]
    
    def test_default_window_sizes(self):
        """Test default window sizes."""
        engineer = FeatureEngineer()
        assert engineer.window_sizes == [10, 20]
    
    def test_rolling_statistics(self, sample_sensor_data):
        """Test rolling window statistics."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Check rolling stat columns exist
        assert len(features) > 0
        assert 'hr_mean_w10' in features.columns
        assert 'hr_std_w10' in features.columns
        assert 'hr_min_w10' in features.columns
        assert 'hr_max_w10' in features.columns
        
        # Check values are reasonable (where they exist)
        if not features['hr_mean_w10'].isna().all():
            assert features['hr_mean_w10'].min() > -50  # Normalized data
            assert features['hr_max_w10'].max() >= features['hr_mean_w10'].max()
    
    def test_trend_features(self, sample_sensor_data):
        """Test trend feature generation."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Check trend columns exist
        assert 'hr_trend_w10' in features.columns
        assert 'spo2_trend_w10' in features.columns
        # Just verify columns have valid data
        assert not features['hr_trend_w10'].isna().all()
    
    def test_lag_features(self, sample_sensor_data):
        """Test lagged feature generation."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Check lag columns exist
        assert 'hr_lag1' in features.columns
        assert 'hr_lag2' in features.columns
        assert 'steps_lag1' in features.columns
    
    def test_temporal_features(self, sample_sensor_data):
        """Test temporal context features."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Check temporal columns exist
        assert 'hour_sin' in features.columns
        assert 'hour_cos' in features.columns
        assert 'day_of_week' in features.columns
        assert 'is_weekend' in features.columns
        
        # Check sine/cosine bounds
        assert features['hour_sin'].min() >= -1.0
        assert features['hour_sin'].max() <= 1.0
        assert features['hour_cos'].min() >= -1.0
        assert features['hour_cos'].max() <= 1.0
    
    def test_derived_features(self, sample_sensor_data):
        """Test derived feature generation."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Check derived columns exist
        assert 'hr_activity_ratio_w10' in features.columns
        assert 'hr_stability_w10' in features.columns
        assert 'spo2_deviation_w10' in features.columns
        assert 'composite_stress_w10' in features.columns
    
    def test_output_shape(self, sample_sensor_data):
        """Test output feature matrix shape."""
        engineer = FeatureEngineer(window_sizes=[10, 20])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Should have fewer rows due to NaN from rolling windows
        assert len(features) < len(sample_sensor_data)
        
        # Should have many features
        assert features.shape[1] > 15  # At least 15 features
        
        # No NaN values allowed
        assert features.isna().sum().sum() == 0
    
    def test_no_nan_values(self, sample_sensor_data):
        """Test that engineered features have no NaN."""
        engineer = FeatureEngineer(window_sizes=[10, 20])
        features = engineer.engineer_features(sample_sensor_data)
        
        # All numeric columns should have no NaN
        numeric_cols = features.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            assert features[col].isna().sum() == 0, f"NaN found in {col}"
    
    def test_multiple_window_sizes(self, sample_sensor_data):
        """Test with multiple window sizes."""
        engineer = FeatureEngineer(window_sizes=[5, 10, 20])
        features = engineer.engineer_features(sample_sensor_data)
        
        # Check that features for all windows are present
        for window in [5, 10, 20]:
            assert f'hr_mean_w{window}' in features.columns
            assert f'hr_std_w{window}' in features.columns
    
    def test_timestamp_preserved(self, sample_sensor_data):
        """Test that timestamp is preserved in output."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        assert 'timestamp' in features.columns
        assert len(features['timestamp'].unique()) > 1


class TestFeatureEngineerEdgeCases:
    """Test edge cases in feature engineering."""
    
    def test_constant_values(self):
        """Test with constant input values."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=50, freq='30s'),
            'heart_rate': np.ones(50) * 75.0,
            'spo2': np.ones(50) * 98.0,
            'steps': np.zeros(50),
        })
        
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(data)
        
        # Should still work without error
        assert len(features) > 0
        
        # Standard deviations should be very small (near zero)
        assert features['hr_std_w10'].max() < 0.01
        assert features['spo2_std_w10'].max() < 0.01
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2026-01-01', periods=15, freq='30s'),
            'heart_rate': np.random.normal(75, 10, 15),
            'spo2': np.random.normal(98, 1, 15),
            'steps': np.random.randint(0, 150, 15),
        })
        
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(data)
        
        # Should handle gracefully
        assert len(features) >= 0
    
    def test_with_labels(self, sample_sensor_data):
        """Test feature engineering with labels."""
        # Add fake labels
        data = sample_sensor_data.copy()
        data['stress'] = np.random.randint(0, 2, len(data))
        
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(data, label_col='stress')
        
        # Should include label column
        assert 'stress' in features.columns


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""
    
    def test_engineered_features_correlate_with_activity(self, sample_sensor_data):
        """Test that engineered features make sense."""
        engineer = FeatureEngineer(window_sizes=[10])
        features = engineer.engineer_features(sample_sensor_data)
        
        # HR activity ratio should be positive
        assert (features['hr_activity_ratio_w10'] >= 0).all()
        
        # HR stability should be positive
        assert (features['hr_stability_w10'] >= 0).all()
        
        # Stress metric should be non-negative
        assert (features['composite_stress_w10'] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
