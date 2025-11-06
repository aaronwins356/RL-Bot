"""Checkpoint export utilities for RLBot deployment.

This module provides utilities to export trained models for direct use in RLBot
without requiring the training framework.
"""
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointExporter:
    """Exports trained checkpoints for RLBot deployment."""
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint exporter.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        
        if not self.checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    def export_for_rlbot(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        export_format: str = "torchscript"
    ) -> Path:
        """Export checkpoint for RLBot use.
        
        Args:
            checkpoint_path: Path to checkpoint file
            output_dir: Output directory for exported model
            export_format: Export format ("torchscript", "onnx", or "raw")
            
        Returns:
            Path to exported model
        """
        checkpoint_path = Path(checkpoint_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting checkpoint from {checkpoint_path}")
        logger.info(f"Export format: {export_format}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if export_format == "torchscript":
            return self._export_torchscript(checkpoint, output_dir)
        elif export_format == "onnx":
            return self._export_onnx(checkpoint, output_dir)
        elif export_format == "raw":
            return self._export_raw(checkpoint_path, output_dir)
        else:
            raise ValueError(f"Unknown export format: {export_format}")
    
    def _export_torchscript(self, checkpoint: Dict[str, Any], output_dir: Path) -> Path:
        """Export as TorchScript for fast inference.
        
        Args:
            checkpoint: Loaded checkpoint
            output_dir: Output directory
            
        Returns:
            Path to exported model
        """
        from core.models.nets import ActorCriticNet
        
        # Reconstruct model
        # Note: This assumes standard architecture - may need config
        model = ActorCriticNet(
            input_size=173,  # Standard obs size
            hidden_sizes=[512, 512, 256],
            action_categoricals=5,
            action_bernoullis=3,
            activation='relu',
            use_lstm=False
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create example input for tracing
        example_input = torch.randn(1, 173)
        
        # Trace model
        traced_model = torch.jit.trace(model, example_input)
        
        # Save traced model
        output_path = output_dir / "model_traced.pt"
        torch.jit.save(traced_model, str(output_path))
        
        logger.info(f"TorchScript model exported to {output_path}")
        
        # Save metadata
        metadata = {
            'format': 'torchscript',
            'input_size': 173,
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }
        
        metadata_path = output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def _export_onnx(self, checkpoint: Dict[str, Any], output_dir: Path) -> Path:
        """Export as ONNX for cross-platform inference.
        
        Args:
            checkpoint: Loaded checkpoint
            output_dir: Output directory
            
        Returns:
            Path to exported model
        """
        from core.models.nets import ActorCriticNet
        
        # Reconstruct model
        model = ActorCriticNet(
            input_size=173,
            hidden_sizes=[512, 512, 256],
            action_categoricals=5,
            action_bernoullis=3,
            activation='relu',
            use_lstm=False
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, 173)
        
        # Export to ONNX
        output_path = output_dir / "model.onnx"
        
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['observation'],
            output_names=['action_logits', 'value'],
            dynamic_axes={
                'observation': {0: 'batch_size'},
                'action_logits': {0: 'batch_size'},
                'value': {0: 'batch_size'}
            }
        )
        
        logger.info(f"ONNX model exported to {output_path}")
        
        # Save metadata
        metadata = {
            'format': 'onnx',
            'input_size': 173,
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }
        
        metadata_path = output_dir / "export_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def _export_raw(self, checkpoint_path: Path, output_dir: Path) -> Path:
        """Export raw checkpoint with config.
        
        Args:
            checkpoint_path: Path to checkpoint
            output_dir: Output directory
            
        Returns:
            Path to exported checkpoint
        """
        output_path = output_dir / "model.pt"
        shutil.copy(checkpoint_path, output_path)
        
        logger.info(f"Raw checkpoint copied to {output_path}")
        
        return output_path
    
    def create_rlbot_package(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        bot_name: str = "ExportedBot"
    ):
        """Create a complete RLBot package with exported model.
        
        Args:
            checkpoint_path: Path to checkpoint
            output_dir: Output directory for package
            bot_name: Name for the bot
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating RLBot package: {bot_name}")
        
        # Export model
        model_path = self.export_for_rlbot(
            checkpoint_path,
            output_dir / "models",
            export_format="torchscript"
        )
        
        # Create bot.cfg
        bot_cfg = f"""[Bot Parameters]
Name = {bot_name}
Description = Exported trained RL bot
Fun Fact = Trained with PPO and curriculum learning

[Locations]
python_file = bot.py

[Details]
developer = RL-Bot Training System
language = python
"""
        
        with open(output_dir / "bot.cfg", 'w') as f:
            f.write(bot_cfg)
        
        # Create simple bot.py wrapper
        bot_py = """# Exported RL-Bot
from pathlib import Path
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

class ExportedBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.model = None
        
    def initialize_agent(self):
        # Load exported model
        model_path = Path(__file__).parent / "models" / "model_traced.pt"
        self.model = torch.jit.load(str(model_path))
        self.model.eval()
        
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Get observation from packet
        obs = self.build_observation(packet)
        
        # Run inference
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_dist, value = self.model(obs_tensor)
            
        # Convert to controller state
        controller = self.action_to_controller(action_dist)
        
        return controller
    
    def build_observation(self, packet):
        # Placeholder - implement proper observation encoding
        return [0.0] * 173
    
    def action_to_controller(self, action_dist):
        # Placeholder - implement proper action decoding
        return SimpleControllerState()
"""
        
        with open(output_dir / "bot.py", 'w') as f:
            f.write(bot_py)
        
        # Create README
        readme = f"""# {bot_name}

Exported RL-Bot trained with curriculum learning.

## Usage

1. Copy this directory to your RLBot bots folder
2. Add the bot to your match configuration
3. Run the match

## Model Info

- Checkpoint: {checkpoint_path.name}
- Export format: TorchScript
- Inference: PyTorch CPU

## Requirements

- Python 3.7+
- PyTorch
- RLBot framework
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(readme)
        
        logger.info(f"RLBot package created at {output_dir}")
        logger.info("Package contents:")
        logger.info("  - bot.cfg")
        logger.info("  - bot.py")
        logger.info("  - models/model_traced.pt")
        logger.info("  - models/export_metadata.json")
        logger.info("  - README.md")
