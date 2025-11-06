"""Offline dataset loader for behavioral cloning pretraining.

This module loads telemetry data for offline training.
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Iterator, Tuple
from torch.utils.data import Dataset, DataLoader


class OfflineDataset(Dataset):
    """PyTorch dataset for offline RL training from telemetry logs."""
    
    def __init__(
        self,
        data_path: Path,
        max_samples: int = None
    ):
        """Initialize offline dataset.
        
        Args:
            data_path: Path to telemetry data directory or file
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_path = Path(data_path)
        self.samples = []
        
        # Load data
        self._load_data(max_samples)
    
    def _load_data(self, max_samples: int = None):
        """Load data from telemetry files."""
        if self.data_path.is_file():
            # Single JSONL file
            self._load_jsonl_file(self.data_path, max_samples)
        elif self.data_path.is_dir():
            # Directory of JSONL files
            jsonl_files = list(self.data_path.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                self._load_jsonl_file(jsonl_file, max_samples)
                if max_samples and len(self.samples) >= max_samples:
                    break
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        print(f"Loaded {len(self.samples)} samples from {self.data_path}")
    
    def _load_jsonl_file(self, file_path: Path, max_samples: int = None):
        """Load data from a JSONL file.
        
        Args:
            file_path: Path to JSONL file
            max_samples: Maximum samples to load
        """
        with open(file_path, "r") as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                
                try:
                    entry = json.loads(line)
                    
                    # Extract observation, action, reward
                    sample = {
                        "observation": np.array(entry["observation"], dtype=np.float32),
                        "action": np.array(entry["action"], dtype=np.float32),
                        "reward": float(entry.get("reward", 0.0)),
                        "done": bool(entry.get("done", False))
                    }
                    
                    self.samples.append(sample)
                except Exception as e:
                    print(f"Failed to parse line: {e}")
                    continue
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary
        """
        return self.samples[idx]
    
    def get_loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Get PyTorch DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
