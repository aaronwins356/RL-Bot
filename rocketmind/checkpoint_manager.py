"""
Smart Checkpoint Management System for RocketMind.
Implements top-K model retention, automatic cleanup, and recovery.
"""

import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import heapq
import warnings


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_path: str
    timestep: int
    mean_reward: float
    win_rate: float
    episode_count: int
    created_at: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Smart checkpoint management with top-K retention.
    Inspired by Coconut's stability guard and best model tracking.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/rocketmind",
        keep_top_k: int = 3,
        save_interval: int = 500,
        metric: str = "mean_reward",
        auto_cleanup: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_top_k: Number of best checkpoints to keep
            save_interval: Save checkpoint every N updates
            metric: Metric to use for ranking ('mean_reward', 'win_rate', etc.)
            auto_cleanup: Automatically delete old checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_top_k = keep_top_k
        self.save_interval = save_interval
        self.metric = metric
        self.auto_cleanup = auto_cleanup
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        
        # Load existing metadata
        self.checkpoints: List[CheckpointMetadata] = []
        self._load_metadata()
        
        # Best checkpoint tracking
        self.best_checkpoint: Optional[CheckpointMetadata] = None
        self._update_best_checkpoint()
        
        # Last good checkpoint for rollback
        self.last_good_checkpoint: Optional[CheckpointMetadata] = None
    
    def _load_metadata(self):
        """Load checkpoint metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = [
                        CheckpointMetadata.from_dict(cp) for cp in data.get('checkpoints', [])
                    ]
            except Exception as e:
                warnings.warn(f"Failed to load checkpoint metadata: {e}")
                self.checkpoints = []
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        data = {
            'checkpoints': [cp.to_dict() for cp in self.checkpoints],
            'best_checkpoint': self.best_checkpoint.to_dict() if self.best_checkpoint else None,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint metadata: {e}")
    
    def _update_best_checkpoint(self):
        """Update best checkpoint based on metric."""
        if not self.checkpoints:
            return
        
        # Find checkpoint with best metric
        best = max(
            self.checkpoints,
            key=lambda cp: getattr(cp, self.metric, float('-inf'))
        )
        
        if self.best_checkpoint is None or \
           getattr(best, self.metric) > getattr(self.best_checkpoint, self.metric):
            self.best_checkpoint = best
            
            # Copy to best_model.pt
            best_path = Path(best.checkpoint_path)
            if best_path.exists():
                best_model_path = self.checkpoint_dir / "best_model.pt"
                shutil.copy(best_path, best_model_path)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        timestep: int,
        metrics: Dict[str, float],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        force: bool = False
    ) -> Optional[str]:
        """
        Save a checkpoint if it meets criteria.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            timestep: Current training timestep
            metrics: Training metrics (mean_reward, win_rate, etc.)
            model_config: Model configuration
            training_config: Training configuration
            scheduler: Optional learning rate scheduler
            force: Force save even if not at save_interval
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        # Check if should save
        if not force and timestep % self.save_interval != 0:
            return None
        
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_{timestep}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestep': timestep,
            'metrics': metrics,
            'model_config': model_config,
            'training_config': training_config,
        }
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
        except Exception as e:
            warnings.warn(f"Failed to save checkpoint: {e}")
            return None
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_path=str(checkpoint_path),
            timestep=timestep,
            mean_reward=metrics.get('mean_reward', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            episode_count=metrics.get('episode_count', 0),
            created_at=datetime.now().isoformat(),
            model_config=model_config,
            training_config=training_config
        )
        
        # Add to checkpoints list
        self.checkpoints.append(metadata)
        
        # Update best checkpoint
        self._update_best_checkpoint()
        
        # Update last good checkpoint
        self.last_good_checkpoint = metadata
        
        # Perform cleanup if auto_cleanup is enabled
        if self.auto_cleanup:
            self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """
        Remove old checkpoints, keeping only top-K by metric.
        Always keeps the most recent checkpoint as well.
        """
        if len(self.checkpoints) <= self.keep_top_k:
            return
        
        # Sort by metric (descending)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda cp: getattr(cp, self.metric, float('-inf')),
            reverse=True
        )
        
        # Keep top-K
        checkpoints_to_keep = set(sorted_checkpoints[:self.keep_top_k])
        
        # Also keep the most recent checkpoint
        most_recent = max(self.checkpoints, key=lambda cp: cp.timestep)
        checkpoints_to_keep.add(most_recent)
        
        # Delete old checkpoints
        for checkpoint in self.checkpoints[:]:
            if checkpoint not in checkpoints_to_keep:
                checkpoint_path = Path(checkpoint.checkpoint_path)
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                        self.checkpoints.remove(checkpoint)
                    except Exception as e:
                        warnings.warn(f"Failed to delete checkpoint {checkpoint_path}: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file, or None to load latest
            load_best: Load the best checkpoint instead
            
        Returns:
            Checkpoint data dictionary
        """
        # Determine which checkpoint to load
        if load_best and self.best_checkpoint:
            path = Path(self.best_checkpoint.checkpoint_path)
        elif checkpoint_path:
            path = Path(checkpoint_path)
        elif self.checkpoints:
            # Load most recent checkpoint
            latest = max(self.checkpoints, key=lambda cp: cp.timestep)
            path = Path(latest.checkpoint_path)
        else:
            warnings.warn("No checkpoints available to load")
            return None
        
        # Load checkpoint
        if not path.exists():
            warnings.warn(f"Checkpoint not found: {path}")
            return None
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            return checkpoint
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint: {e}")
            return None
    
    def rollback_to_last_good(self) -> Optional[Dict[str, Any]]:
        """
        Rollback to last good checkpoint (for recovery from divergence).
        Implements Coconut's stability guard rollback.
        
        Returns:
            Last good checkpoint data, or None if not available
        """
        if not self.last_good_checkpoint:
            warnings.warn("No last good checkpoint available for rollback")
            return None
        
        return self.load_checkpoint(self.last_good_checkpoint.checkpoint_path)
    
    def get_checkpoint_info(self) -> List[Dict[str, Any]]:
        """Get information about all checkpoints."""
        return [cp.to_dict() for cp in self.checkpoints]
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if self.best_checkpoint:
            return self.best_checkpoint.checkpoint_path
        return None
    
    def export_to_rlbot(
        self,
        checkpoint_path: Optional[str] = None,
        output_dir: str = "rlbot_models"
    ) -> str:
        """
        Export a checkpoint for RLBot deployment.
        
        Args:
            checkpoint_path: Path to checkpoint (uses best if None)
            output_dir: Directory to export to
            
        Returns:
            Path to exported model
        """
        # Load checkpoint
        if checkpoint_path is None and self.best_checkpoint:
            checkpoint_path = self.best_checkpoint.checkpoint_path
        
        if not checkpoint_path:
            raise ValueError("No checkpoint specified and no best checkpoint available")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint
        src = Path(checkpoint_path)
        dst = output_path / "model.pt"
        shutil.copy(src, dst)
        
        # Save config
        checkpoint = self.load_checkpoint(checkpoint_path)
        if checkpoint:
            config_path = output_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'model_config': checkpoint.get('model_config', {}),
                    'training_config': checkpoint.get('training_config', {}),
                    'metrics': checkpoint.get('metrics', {}),
                    'timestep': checkpoint.get('timestep', 0)
                }, f, indent=2)
        
        return str(dst)


def setup_wandb_integration(
    project_name: str = "rocketmind",
    config: Optional[Dict[str, Any]] = None,
    resume: bool = False,
    run_id: Optional[str] = None
):
    """
    Set up Weights & Biases integration.
    
    Args:
        project_name: W&B project name
        config: Configuration to log
        resume: Resume previous run
        run_id: Run ID to resume (if resume=True)
        
    Returns:
        W&B run object
    """
    try:
        import wandb
        
        if resume and run_id:
            run = wandb.init(
                project=project_name,
                id=run_id,
                resume="must",
                config=config
            )
        else:
            run = wandb.init(
                project=project_name,
                config=config
            )
        
        return run
    except ImportError:
        warnings.warn(
            "wandb is not installed. Install with: pip install wandb\n"
            "W&B integration disabled."
        )
        return None


def log_to_wandb(metrics: Dict[str, float], step: int):
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
