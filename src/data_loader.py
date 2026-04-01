"""
Data loading and batching module.

Provides interfaces for:
- Training with batches (shuffled)
- Inference streaming (sequential)
- Data normalization on the fly
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional
from pathlib import Path


class SensorDataLoader:
    """Load and batch sensor data for training and inference."""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV file with features
        """
        self.data = pd.read_csv(data_path)
        self.n_samples = len(self.data)
        self.feature_cols = self._identify_feature_cols()
        
        print(f"✓ Loaded {self.n_samples} samples with {len(self.feature_cols)} features")
    
    def _identify_feature_cols(self) -> List[str]:
        """Identify feature columns (exclude timestamp, labels)."""
        exclude = {'timestamp', 'label', 'stress', 'fatigue'}
        cols = [col for col in self.data.columns if col not in exclude]
        return cols
    
    def batch_generator(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_epochs: int = 1,
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Generate batches for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_epochs: Number of epochs to iterate
        
        Yields:
            Tuple of (features_batch, labels_batch) or (features_batch, None)
        """
        indices = np.arange(self.n_samples)
        
        for epoch in range(num_epochs):
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, self.n_samples, batch_size):
                end_idx = min(start_idx + batch_size, self.n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Extract features
                X = self.data.iloc[batch_indices][self.feature_cols].values
                
                # Extract labels if available
                label_cols = [col for col in self.data.columns if col in {'label', 'stress', 'fatigue'}]
                if label_cols:
                    y = self.data.iloc[batch_indices][label_cols[0]].values
                else:
                    y = None
                
                yield X, y
    
    def stream_batch(
        self,
        start_idx: int = 0,
        batch_size: int = 32,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Stream batches sequentially (for inference).
        
        Args:
            start_idx: Starting index
            batch_size: Batch size
        
        Yields:
            Tuple of (batch_idx, features_batch)
        """
        for batch_idx in range(start_idx, self.n_samples, batch_size):
            end_idx = min(batch_idx + batch_size, self.n_samples)
            
            X = self.data.iloc[batch_idx:end_idx][self.feature_cols].values
            
            yield batch_idx, X
    
    def get_all(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get all data at once.
        
        Returns:
            Tuple of (features, labels) or (features, None)
        """
        X = self.data[self.feature_cols].values
        
        label_cols = [col for col in self.data.columns if col in {'label', 'stress', 'fatigue'}]
        if label_cols:
            y = self.data[label_cols[0]].values
        else:
            y = None
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get feature column names."""
        return self.feature_cols
    
    def get_sample_at(self, idx: int) -> Tuple[np.ndarray, Optional[float]]:
        """
        Get single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, label) or (features, None)
        """
        features = self.data.iloc[idx][self.feature_cols].values
        
        label_cols = [col for col in self.data.columns if col in {'label', 'stress', 'fatigue'}]
        label = self.data.iloc[idx][label_cols[0]] if label_cols else None
        
        return features, label


class StreamingDataLoader:
    """Real-time streaming data loader (for API inference)."""
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize streaming loader.
        
        Args:
            feature_names: List of expected feature names
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.buffer = []
    
    def add_sample(self, features_dict: dict) -> None:
        """
        Add a new sample to the buffer.
        
        Args:
            features_dict: Dict with feature_name -> value pairs
        """
        # Extract features in correct order
        features = np.array([features_dict.get(name, 0.0) for name in self.feature_names])
        self.buffer.append(features)
    
    def get_batch(self, batch_size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get accumulated batch.
        
        Args:
            batch_size: If specified, return batch only if buffer has this many samples
        
        Returns:
            Batch as (batch_size, n_features) array or None if not ready
        """
        if batch_size is not None and len(self.buffer) < batch_size:
            return None
        
        if len(self.buffer) == 0:
            return None
        
        batch = np.array(self.buffer)
        self.buffer = []
        
        return batch
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer = []


def load_training_data(
    data_path: str = "data/processed/train_features.csv",
    batch_size: int = 32,
) -> SensorDataLoader:
    """
    Load training data.
    
    Args:
        data_path: Path to training features CSV
        batch_size: Batch size for iteration
    
    Returns:
        SensorDataLoader instance
    """
    loader = SensorDataLoader(data_path)
    return loader


if __name__ == "__main__":
    # Example usage
    print("Loading training data...")
    train_loader = load_training_data()
    
    print("\nIterating batches (first 2):")
    for i, (X, y) in enumerate(train_loader.batch_generator(batch_size=32)):
        print(f"  Batch {i}: shape {X.shape}")
        if i >= 1:
            break
    
    print("\nGetting all data:")
    X_all, y_all = train_loader.get_all()
    print(f"  Shape: {X_all.shape}")
    print(f"  Labels: {y_all}")
    
    print("\nFeature names:")
    print(f"  {train_loader.get_feature_names()}")
