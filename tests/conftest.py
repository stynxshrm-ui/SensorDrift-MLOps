"""
Pytest configuration and fixtures
"""
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'timestamp': pd.date_range('2026-01-01', periods=n_samples, freq='30s'),
        'heart_rate': np.random.normal(75, 10, n_samples),
        'spo2': np.random.normal(98, 1, n_samples),
        'steps': np.random.randint(0, 150, n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_features():
    """Generate sample engineered features for testing"""
    np.random.seed(42)
    n_samples = 100
    
    features = {
        'hr_mean': np.random.normal(75, 10, n_samples),
        'hr_std': np.random.uniform(5, 15, n_samples),
        'spo2_mean': np.random.normal(98, 1, n_samples),
        'steps_mean': np.random.uniform(20, 100, n_samples),
        'hr_trend': np.random.normal(0, 2, n_samples),
    }
    
    return pd.DataFrame(features)
