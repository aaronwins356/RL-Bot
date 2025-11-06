"""Checkpoint management for model saving and loading.

This module handles atomic checkpointing with best model promotion.
"""
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging

# Import SafeJSONEncoder from logging module
from core.infra.logging import SafeJSONEncoder

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manager for model checkpoints with atomic saves and best model tracking."""
    
    def __init__(
        self,
        save_dir: Path,
        keep_best_n: int = 5,
        keep_latest: bool = True
    ):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory for checkpoints
            keep_best_n: Number of best checkpoints to keep
            keep_latest: Whether to always keep latest checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.keep_latest = keep_latest
        
        # Track checkpoints
        self.checkpoints = []
        self.best_metric = float('-inf')
        self.best_checkpoint_path = None
        
        # Load existing checkpoint metadata
        self._load_metadata()
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        step: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> Path:
        """Save model checkpoint atomically.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            step: Training step
            metrics: Current metrics
            metadata: Additional metadata
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        # Save to temp file first (atomic write)
        checkpoint_name = f"checkpoint_step_{step}.pt"
        temp_path = self.save_dir / f".{checkpoint_name}.tmp"
        final_path = self.save_dir / checkpoint_name
        
        torch.save(checkpoint, temp_path)
        shutil.move(str(temp_path), str(final_path))
        
        # Track checkpoint
        self.checkpoints.append({
            "path": final_path,
            "step": step,
            "metrics": metrics,
            "is_best": is_best
        })
        
        # Save as best if applicable
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            shutil.copy(str(final_path), str(best_path))
            self.best_checkpoint_path = best_path
            self.best_metric = metrics.get("eval_score", float('-inf'))
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        return final_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            device: Device to load to
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return {
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
            "metadata": checkpoint.get("metadata", {})
        }
    
    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load best checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            device: Device to load to
            
        Returns:
            Checkpoint metadata or None if no best checkpoint exists
        """
        if not self.best_checkpoint_path or not self.best_checkpoint_path.exists():
            return None
        
        return self.load_checkpoint(self.best_checkpoint_path, model, optimizer, device)
    
    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            device: Device to load to
            
        Returns:
            Checkpoint metadata or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        
        # Sort by step and get latest
        latest = max(self.checkpoints, key=lambda x: x["step"])
        return self.load_checkpoint(latest["path"], model, optimizer, device)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping best N and optionally latest."""
        if len(self.checkpoints) <= self.keep_best_n:
            return
        
        # Sort by eval score
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x["metrics"].get("eval_score", float('-inf')),
            reverse=True
        )
        
        # Keep best N
        to_keep = set(c["path"] for c in sorted_checkpoints[:self.keep_best_n])
        
        # Keep latest if specified
        if self.keep_latest and self.checkpoints:
            latest = max(self.checkpoints, key=lambda x: x["step"])
            to_keep.add(latest["path"])
        
        # Remove old checkpoints
        for checkpoint in self.checkpoints[:]:
            if checkpoint["path"] not in to_keep:
                try:
                    checkpoint["path"].unlink()
                    self.checkpoints.remove(checkpoint)
                except FileNotFoundError:
                    pass
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        metadata = {
            "checkpoints": [
                {
                    "path": str(c["path"]),
                    "step": c["step"],
                    "metrics": c["metrics"],
                    "is_best": c.get("is_best", False)
                }
                for c in self.checkpoints
            ],
            "best_metric": self.best_metric,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None
        }
        
        metadata_path = self.save_dir / "metadata.json"
        try:
            with open(metadata_path, "w", encoding='utf-8') as f:
                # Use SafeJSONEncoder to handle NumPy types
                json.dump(metadata, f, indent=2, cls=SafeJSONEncoder)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint metadata: {e}")
    
    def _load_metadata(self):
        """Load checkpoint metadata."""
        metadata_path = self.save_dir / "metadata.json"
        if not metadata_path.exists():
            return
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            self.checkpoints = [
                {
                    "path": Path(c["path"]),
                    "step": c["step"],
                    "metrics": c["metrics"],
                    "is_best": c.get("is_best", False)
                }
                for c in metadata.get("checkpoints", [])
                if Path(c["path"]).exists()
            ]
            
            self.best_metric = metadata.get("best_metric", float('-inf'))
            if metadata.get("best_checkpoint"):
                self.best_checkpoint_path = Path(metadata["best_checkpoint"])
        except Exception as e:
            print(f"Failed to load checkpoint metadata: {e}")
    
    def get_latest_path(self) -> Optional[Path]:
        """Get path to latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x["step"])
        return latest["path"]
