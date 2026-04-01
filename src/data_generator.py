"""
Synthetic wearable sensor data generator.

Generates realistic HR, SpO2, and steps data with:
- Circadian rhythm patterns
- Activity-dependent HR variations
- Realistic noise and variations
- Optional drift injection for testing drift detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional
from pathlib import Path


class SyntheticSensorDataGenerator:
    """Generate synthetic wearable sensor data."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize generator with random seed for reproducibility.
        
        Args:
            random_seed: Random state seed
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_dataset(
        self,
        n_samples: int = 10000,
        sampling_interval_seconds: int = 30,
        noise_level: float = 0.05,
        drift_factor: float = 0.0,
        start_date: str = "2026-01-01",
    ) -> pd.DataFrame:
        """
        Generate synthetic sensor dataset.
        
        Args:
            n_samples: Number of 30-second samples to generate
            sampling_interval_seconds: Time between samples (default 30 sec)
            noise_level: Gaussian noise standard deviation (0-1 scale)
            drift_factor: Gradual drift in distributions (0-1 scale)
            start_date: Start date for timestamps (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: timestamp, heart_rate, spo2, steps
        """
        # Create timestamps
        start_time = pd.to_datetime(start_date)
        timestamps = [
            start_time + timedelta(seconds=sampling_interval_seconds * i)
            for i in range(n_samples)
        ]
        
        # Extract hour of day for circadian patterns
        hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
        
        # Generate base signals with circadian rhythm
        hr_base = self._generate_heart_rate(hours, n_samples, noise_level, drift_factor)
        spo2_base = self._generate_spo2(hours, n_samples, noise_level, drift_factor)
        steps = self._generate_steps(n_samples, noise_level, drift_factor)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'heart_rate': hr_base,
            'spo2': spo2_base,
            'steps': steps,
        })
        
        return data
    
    def _generate_heart_rate(
        self,
        hours: np.ndarray,
        n_samples: int,
        noise_level: float,
        drift_factor: float,
    ) -> np.ndarray:
        """
        Generate heart rate with circadian rhythm and activity correlation.
        
        HR baseline varies by time of day:
        - Night (22-6): 60-75 bpm (resting)
        - Morning (6-10): 70-85 bpm (waking)
        - Day (10-16): 75-90 bpm (active)
        - Evening (16-22): 70-85 bpm (cooling down)
        """
        # Circadian rhythm (sine wave with 24-hour period)
        circadian = 10 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)  # Peaks at midday
        
        # Base HR varies by time
        base_hr = 75 + circadian
        
        # Add random variations (activity, breathing, etc.)
        activity_variation = np.random.normal(0, 5, n_samples)
        
        # Add noise
        noise = np.random.normal(0, 3 * noise_level, n_samples)
        
        # Apply optional drift
        drift = np.linspace(0, 10 * drift_factor, n_samples)
        
        hr = base_hr + activity_variation + noise + drift
        
        # Clamp to realistic range
        hr = np.clip(hr, 50, 150)
        
        return np.round(hr, 1)
    
    def _generate_spo2(
        self,
        hours: np.ndarray,
        n_samples: int,
        noise_level: float,
        drift_factor: float,
    ) -> np.ndarray:
        """
        Generate SpO2 (blood oxygen saturation).
        
        Healthy resting SpO2: 95-100%
        During activity: 94-98%
        """
        # SpO2 is relatively stable; add subtle circadian and activity effects
        base_spo2 = 98.0
        
        # Slight decrease during active periods (lower during day)
        circadian = -1.5 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
        
        # Random variations
        variation = np.random.normal(0, 0.8, n_samples)
        
        # Add noise
        noise = np.random.normal(0, 0.3 * noise_level, n_samples)
        
        # Apply optional drift (SpO2 degradation over time)
        drift = np.linspace(0, -2 * drift_factor, n_samples)
        
        spo2 = base_spo2 + circadian + variation + noise + drift
        
        # Clamp to realistic range
        spo2 = np.clip(spo2, 94.0, 100.0)
        
        return np.round(spo2, 1)
    
    def _generate_steps(
        self,
        n_samples: int,
        noise_level: float,
        drift_factor: float,
    ) -> np.ndarray:
        """
        Generate step count (activity level).
        
        Typical range: 0-150 steps per 30-sec window
        """
        # Activity has bursts (periods of activity, periods of rest)
        # Use exponential distribution for activity spikes
        base_activity = np.random.exponential(scale=15, size=n_samples)
        
        # Add noise/variation
        noise = np.random.normal(0, 5 * noise_level, n_samples)
        
        # Apply optional drift (activity decrease over time, e.g., fatigue)
        drift = np.linspace(0, -20 * drift_factor, n_samples)
        
        steps = base_activity + noise + drift
        
        # Clamp to realistic range
        steps = np.clip(steps, 0, 150)
        
        return np.round(steps, 0).astype(int)
    
    def save_dataset(
        self,
        data: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Save dataset to CSV.
        
        Args:
            data: DataFrame to save
            output_path: Path to save (default: data/raw/synthetic_data.csv)
        
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = "data/raw/synthetic_data.csv"
        
        # Create parent directories
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        print(f"✓ Synthetic data saved to {output_path}")
        print(f"  Samples: {len(data)}")
        print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"  HR: {data['heart_rate'].min():.1f} - {data['heart_rate'].max():.1f} bpm")
        print(f"  SpO2: {data['spo2'].min():.1f} - {data['spo2'].max():.1f} %")
        print(f"  Steps: {data['steps'].min()} - {data['steps'].max()} steps")
        
        return output_path


def generate_synthetic_dataset(
    n_samples: int = 10000,
    noise_level: float = 0.05,
    drift_factor: float = 0.0,
    output_path: Optional[str] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Convenience function to generate and save synthetic dataset.
    
    Args:
        n_samples: Number of samples
        noise_level: Noise level (0-1)
        drift_factor: Drift level (0-1)
        output_path: Where to save CSV
        random_seed: Random seed
    
    Returns:
        Generated DataFrame
    """
    generator = SyntheticSensorDataGenerator(random_seed=random_seed)
    data = generator.generate_dataset(
        n_samples=n_samples,
        noise_level=noise_level,
        drift_factor=drift_factor,
    )
    generator.save_dataset(data, output_path)
    return data


if __name__ == "__main__":
    # Example usage
    print("Generating synthetic sensor data...")
    data = generate_synthetic_dataset(
        n_samples=10000,
        noise_level=0.05,
        drift_factor=0.0,
    )
    print("\nDataset statistics:")
    print(data.describe())
