"""Discord webhook integration for training notifications.

This module sends training progress updates to Discord channels via webhooks.
"""
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DiscordWebhook:
    """Discord webhook client for training notifications."""
    
    def __init__(self, webhook_url: Optional[str] = None, enabled: bool = True):
        """Initialize Discord webhook.
        
        Args:
            webhook_url: Discord webhook URL
            enabled: Whether webhook notifications are enabled
        """
        self.webhook_url = webhook_url
        self.enabled = enabled and webhook_url is not None
        
        if self.enabled:
            logger.info("Discord webhook notifications enabled")
        else:
            logger.info("Discord webhook notifications disabled")
    
    def send_message(self, content: str, embeds: Optional[list] = None) -> bool:
        """Send message to Discord webhook.
        
        Args:
            content: Message content
            embeds: Optional list of embed dictionaries
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        try:
            import requests
            
            payload = {
                "content": content,
                "username": "RL-Bot Training"
            }
            
            if embeds:
                payload["embeds"] = embeds
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.debug("Discord webhook message sent successfully")
                return True
            else:
                logger.warning(f"Discord webhook failed with status {response.status_code}")
                return False
                
        except ImportError:
            logger.warning("requests library not available, cannot send Discord webhook")
            return False
        except Exception as e:
            logger.error(f"Failed to send Discord webhook: {e}")
            return False
    
    def send_training_start(self, config: Dict[str, Any]):
        """Send training start notification.
        
        Args:
            config: Training configuration
        """
        if not self.enabled:
            return
        
        embed = {
            "title": "🚀 Training Started",
            "color": 3447003,  # Blue
            "fields": [
                {
                    "name": "Algorithm",
                    "value": config.get('algorithm', 'PPO'),
                    "inline": True
                },
                {
                    "name": "Total Timesteps",
                    "value": f"{config.get('total_timesteps', 0):,}",
                    "inline": True
                },
                {
                    "name": "Batch Size",
                    "value": str(config.get('batch_size', 4096)),
                    "inline": True
                },
                {
                    "name": "Learning Rate",
                    "value": str(config.get('learning_rate', 3e-4)),
                    "inline": True
                },
                {
                    "name": "Device",
                    "value": config.get('device', 'cpu'),
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.send_message("", embeds=[embed])
    
    def send_training_progress(
        self,
        timestep: int,
        total_timesteps: int,
        elo: float,
        metrics: Dict[str, Any]
    ):
        """Send training progress update.
        
        Args:
            timestep: Current timestep
            total_timesteps: Total timesteps
            elo: Current Elo rating
            metrics: Additional metrics
        """
        if not self.enabled:
            return
        
        progress_pct = (timestep / total_timesteps) * 100
        
        embed = {
            "title": "📊 Training Progress",
            "color": 15844367,  # Gold
            "fields": [
                {
                    "name": "Progress",
                    "value": f"{timestep:,} / {total_timesteps:,} ({progress_pct:.1f}%)",
                    "inline": False
                },
                {
                    "name": "Current Elo",
                    "value": f"{elo:.0f}",
                    "inline": True
                },
                {
                    "name": "Win Rate",
                    "value": f"{metrics.get('win_rate', 0.0):.1%}",
                    "inline": True
                },
                {
                    "name": "Avg Reward",
                    "value": f"{metrics.get('avg_reward', 0.0):.2f}",
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add curriculum stage if available
        if 'curriculum_stage' in metrics:
            embed['fields'].append({
                "name": "Curriculum Stage",
                "value": str(metrics['curriculum_stage']),
                "inline": True
            })
        
        self.send_message("", embeds=[embed])
    
    def send_evaluation_results(
        self,
        timestep: int,
        elo_before: float,
        elo_after: float,
        results: Dict[str, Any]
    ):
        """Send evaluation results.
        
        Args:
            timestep: Current timestep
            elo_before: Elo before evaluation
            elo_after: Elo after evaluation
            results: Evaluation results
        """
        if not self.enabled:
            return
        
        elo_change = elo_after - elo_before
        elo_emoji = "📈" if elo_change > 0 else "📉" if elo_change < 0 else "➡️"
        
        embed = {
            "title": f"{elo_emoji} Evaluation Complete",
            "color": 5763719 if elo_change > 0 else 15158332,  # Green if up, red if down
            "fields": [
                {
                    "name": "Timestep",
                    "value": f"{timestep:,}",
                    "inline": True
                },
                {
                    "name": "Elo Change",
                    "value": f"{elo_before:.0f} → {elo_after:.0f} ({elo_change:+.0f})",
                    "inline": False
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add results by opponent
        for opponent, result in results.items():
            wins = result.get('wins', 0)
            losses = result.get('losses', 0)
            win_rate = result.get('win_rate', 0.0)
            
            embed['fields'].append({
                "name": f"vs {opponent}",
                "value": f"{wins}W-{losses}L ({win_rate:.1%})",
                "inline": True
            })
        
        self.send_message("", embeds=[embed])
    
    def send_checkpoint_saved(
        self,
        timestep: int,
        elo: float,
        is_best: bool = False
    ):
        """Send checkpoint saved notification.
        
        Args:
            timestep: Timestep of checkpoint
            elo: Elo rating at checkpoint
            is_best: Whether this is the best checkpoint
        """
        if not self.enabled:
            return
        
        title = "⭐ Best Checkpoint Saved!" if is_best else "💾 Checkpoint Saved"
        color = 3066993 if is_best else 10070709  # Green for best, gray for regular
        
        embed = {
            "title": title,
            "color": color,
            "fields": [
                {
                    "name": "Timestep",
                    "value": f"{timestep:,}",
                    "inline": True
                },
                {
                    "name": "Elo",
                    "value": f"{elo:.0f}",
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.send_message("", embeds=[embed])
    
    def send_training_complete(
        self,
        final_timestep: int,
        final_elo: float,
        best_elo: float,
        total_time: float
    ):
        """Send training completion notification.
        
        Args:
            final_timestep: Final timestep reached
            final_elo: Final Elo rating
            best_elo: Best Elo achieved
            total_time: Total training time in seconds
        """
        if not self.enabled:
            return
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        embed = {
            "title": "🏁 Training Complete!",
            "color": 3066993,  # Green
            "fields": [
                {
                    "name": "Final Timestep",
                    "value": f"{final_timestep:,}",
                    "inline": True
                },
                {
                    "name": "Final Elo",
                    "value": f"{final_elo:.0f}",
                    "inline": True
                },
                {
                    "name": "Best Elo",
                    "value": f"{best_elo:.0f}",
                    "inline": True
                },
                {
                    "name": "Training Time",
                    "value": f"{hours}h {minutes}m",
                    "inline": True
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.send_message("🎉 Training completed successfully!", embeds=[embed])
    
    def send_error(self, error_message: str, timestep: int = 0):
        """Send error notification.
        
        Args:
            error_message: Error message
            timestep: Timestep when error occurred
        """
        if not self.enabled:
            return
        
        embed = {
            "title": "❌ Training Error",
            "color": 15158332,  # Red
            "fields": [
                {
                    "name": "Timestep",
                    "value": f"{timestep:,}",
                    "inline": True
                },
                {
                    "name": "Error",
                    "value": error_message[:1000],  # Limit length
                    "inline": False
                }
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.send_message("⚠️ Training encountered an error", embeds=[embed])
