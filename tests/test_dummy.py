"""
Dummy test to verify pytest setup
"""
import pytest


def test_import_src():
    """Test that src package can be imported"""
    import src
    assert src.__version__ == "0.1.0"


def test_sample_fixture(sample_sensor_data):
    """Test that sample_sensor_data fixture works"""
    assert len(sample_sensor_data) == 100
    assert list(sample_sensor_data.columns) == ['timestamp', 'heart_rate', 'spo2', 'steps']


def test_simple_math():
    """Basic sanity test"""
    assert 2 + 2 == 4
    assert 10 - 3 == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
