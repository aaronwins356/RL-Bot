# RL-Bot Training System - Enhanced Features Guide

This document describes the comprehensive enhancements made to the RL-Bot training system to produce pro-level bots capable of reaching 1600+ Elo.

## 🎯 Overview

The training system has been significantly enhanced with:
- **9-stage progressive curriculum** from basics to pro-level play
- **Performance-based transitions** that adapt to bot's skill level
- **Advanced evaluation metrics** for deep performance insights
- **SAC algorithm** as an alternative to PPO
- **Comprehensive reward shaping** for all key Rocket League skills
- **Discord integration** for real-time training notifications
- **Export utilities** for deploying trained bots directly in RLBot

## 🚀 Quick Start

### Basic Training

```bash
# Simple training with defaults
python scripts/train.py --config configs/base.yaml

# Training with debug mode (1k steps, verbose logging)
python scripts/train.py --debug --timesteps 1000

# Training with specific device
python scripts/train.py --device cuda

# Training with custom log directory
python scripts/train.py --logdir logs/my_experiment
```

### Advanced Training

```bash
# Aerial-focused curriculum
python scripts/train.py --aerial-curriculum

# Force specific curriculum stage (for testing)
python scripts/train.py --curriculum-stage 3

# Offline pretraining before RL
python scripts/train.py --offline-pretrain

# With Discord notifications
python scripts/train.py --discord-webhook "https://discord.com/api/webhooks/..."

# Export checkpoint after training
python scripts/train.py --export-checkpoint exported_models/my_bot --export-format torchscript
```

## 📚 9-Stage Curriculum

The training curriculum progressively teaches the bot from basic mechanics to pro-level play:

### Stage 0: Basic 1v1 Ground Play (0-500k timesteps)
- Basic ball control
- Ground movement
- Simple shooting
- **Opponent**: Basic script (70% speed)

### Stage 1: Boost Control & Management (500k-1.5M)
- Boost pad collection
- Boost conservation
- Efficient boost usage
- **Focus**: `--explore-boost` style training
- **Opponent**: Rule policy (90% speed)

### Stage 2: Kickoff Micro-Strategy (1.5M-2.5M)
- Kickoff positioning
- Fast kickoffs
- Boost stealing on kickoff
- **Opponent**: Rule policy (100% speed)

### Stage 3: Aerial Basics & Defense (2.5M-4M)
- Basic aerial touches
- Aerial defense
- Jump timing
- **Opponent**: Self-play

### Stage 4: Advanced Aerial Play (4M-5.5M)
- Aerial shots
- Redirect aerials
- Fast aerial mechanics
- **Game Mode**: 2v2
- **Opponent**: Self-play

### Stage 5: 2v2 Rotation Focus (5.5M-7M)
- Team positioning
- Rotation mechanics
- Passing plays
- **Focus**: Heavy rotation penalties (0.6 weight)
- **Opponent**: Self-play

### Stage 6: 1v2 Defense Training (7M-8.5M)
- Outnumbered defense
- Shadow defense
- Buying time
- **Game Mode**: 1v2
- **Opponent**: Checkpoints (110% speed)

### Stage 7: 3v3 Team Play (8.5M-10M)
- 3v3 positioning
- Team coordination
- Advanced rotations
- **Game Mode**: 3v3
- **Opponent**: Self-play

### Stage 8: Pro-Level 3v3 Chaos (10M+)
- High-speed gameplay
- Complex team plays
- Pro-level mechanics
- **Opponent**: Checkpoints (115% speed)
- **Focus**: Maximum difficulty

## 🎮 Performance-Based Transitions

Stages automatically transition when the bot meets ALL criteria:
- **Win rate** ≥ 60% against current opponent
- **Elo rating** ≥ 1400
- **Games played** ≥ 100 in current stage
- **Minimum timesteps** ≥ 100k in current stage

Configure thresholds in `configs/base.yaml`:
```yaml
training:
  curriculum:
    use_performance_transitions: true
    transition_win_rate: 0.6
    transition_elo: 1400
    transition_min_games: 100
    transition_min_timesteps: 100000
```

## 📊 Advanced Evaluation Metrics

### Tracked Metrics

1. **Expected Value of State**
   - Average value prediction across sampled states
   - Measures bot's ability to evaluate positions

2. **Strategy Score**
   - Composite metric: `0.5 * expected_value + 0.5 * entropy`
   - Balances position evaluation with action diversity

3. **Action Entropy**
   - Measures strategy diversity and exploration
   - Higher = more varied playstyle

4. **Curriculum Stage Elos**
   - Separate Elo tracking per curriculum stage
   - Visualizes progress through training phases

5. **Checkpoint Comparisons**
   - Head-to-head matches between model versions
   - Tracks improvement over time

### Visualization

Four new plot types automatically generated:
- `expected_value.png` - Value predictions over time
- `action_entropy.png` - Strategy diversity trends
- `strategy_score.png` - Overall strategy quality
- `curriculum_elos.png` - Elo per curriculum stage

All metrics exported to CSV:
- `eval_summary.csv` - High-level results
- `game_by_game.csv` - Detailed match records
- `advanced_metrics.csv` - All tracked metrics

## 🏆 Reward Shaping

Comprehensive reward shaping encourages all key skills:

### Reward Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Goal Scored | 10.0 | Scoring a goal |
| Goal Conceded | -10.0 | Conceding a goal |
| Touch Ball | 0.1 | Making contact with ball |
| Boost Pickup | 0.05 | Collecting boost pads |
| Boost Usage | -0.01 | Using boost (conservation) |
| Demolition | 2.0 | Demolishing opponent |
| Demolished | -1.0 | Getting demolished |
| Goal Proximity | 0.5 | Being near ball/goal |
| Ball Velocity | 0.3 | Hitting ball toward goal |
| Position Reward | 0.2 | Good field positioning |
| Rotation Reward | 0.1 | Proper rotation |
| Aerial Reward | 0.5 | Aerial touches |
| Shot on Goal | 1.0 | Taking shots |
| Save | 2.0 | Making saves |
| Pass | 0.5 | Passing to teammates |

### Curriculum-Adaptive Rewards

Reward weights automatically adjust per curriculum stage:
- Stage 0-1: Emphasize ball touches and boost management
- Stage 2-3: Focus on velocity toward goal
- Stage 4-5: Prioritize positioning and rotation
- Stage 6-7: Emphasize aerials and shots
- Stage 8-9: Team play and advanced mechanics

## 🤖 SAC Algorithm

Soft Actor-Critic available as alternative to PPO:

```yaml
training:
  algorithm: "sac"  # or "ppo"
  
  # SAC-specific hyperparameters
  tau: 0.005  # Target network soft update
  alpha: 0.2  # Entropy temperature
  auto_entropy_tuning: true  # Automatic temperature adjustment
```

SAC advantages:
- Maximum entropy framework for exploration
- Twin Q-networks for stability
- Off-policy learning (more sample efficient)
- Better for continuous action spaces

## 📱 Discord Notifications

Real-time training updates sent to Discord:

### Setup

1. Create Discord webhook in your server
2. Enable in config or CLI:

```bash
python scripts/train.py --discord-webhook "https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"
```

Or in `configs/base.yaml`:
```yaml
notifications:
  discord:
    enabled: true
    webhook_url: "https://discord.com/api/webhooks/..."
    notify_on_start: true
    notify_on_checkpoint: true
    notify_on_evaluation: true
    notify_on_complete: true
    notify_on_error: true
```

### Notification Types

- 🚀 **Training Start** - Config summary
- 📊 **Progress Updates** - Timestep, Elo, win rate
- 📈 **Evaluation Results** - Detailed per-opponent results
- 💾 **Checkpoint Saved** - Regular and best checkpoints
- 🏁 **Training Complete** - Final stats and time
- ❌ **Errors** - Automatic error reporting

## 📦 Checkpoint Export

Export trained models for direct RLBot use:

### Export Formats

1. **TorchScript** (recommended)
   - Fast inference
   - No Python overhead
   - Easy deployment

2. **ONNX**
   - Cross-platform
   - Works with other frameworks
   - Good for production

3. **Raw**
   - Original PyTorch checkpoint
   - Full training state
   - Best for fine-tuning

### Usage

```bash
# Export after training
python scripts/train.py --export-checkpoint exported_models/my_bot --export-format torchscript

# Or export existing checkpoint
python -c "
from core.infra.export import CheckpointExporter
from pathlib import Path

exporter = CheckpointExporter(Path('checkpoints'))
exporter.create_rlbot_package(
    checkpoint_path=Path('checkpoints/best_model.pt'),
    output_dir=Path('exported_bots/my_bot'),
    bot_name='MyProBot'
)
"
```

### RLBot Package Contents

- `bot.cfg` - RLBot configuration
- `bot.py` - Bot wrapper code
- `models/model_traced.pt` - Exported model
- `models/export_metadata.json` - Model info
- `README.md` - Usage instructions

## 🐛 Debug Mode

Enhanced debug mode for development:

```bash
python scripts/train.py --debug --timesteps 1000
```

Features:
- Limits training to 1k steps (or custom with `--timesteps`)
- Verbose logging every 10 steps
- Detailed action/reward/state logging
- Training state inspection
- No production overhead

Additional debug options:
```bash
# Limit ticks in debug mode
python scripts/train.py --debug --debug-ticks 100

# Debug specific curriculum stage
python scripts/train.py --debug --curriculum-stage 3 --timesteps 500
```

## 📈 Evaluation

Enhanced evaluation script with new metrics:

```bash
# Basic evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Generate plots
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --plot

# Custom opponents and games
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --opponents rule_policy baseline_ml nexto \
  --num-games 20 \
  --k-factor 32
```

## ⚙️ Configuration

Complete example config in `configs/base.yaml`:

```yaml
training:
  algorithm: "ppo"
  total_timesteps: 10000000
  batch_size: 4096
  learning_rate: 3.0e-4
  
  curriculum:
    use_performance_transitions: true
    transition_win_rate: 0.6
    transition_elo: 1400
    
  offline:
    enabled: false
    dataset_path: "data/telemetry_logs"
    pretrain_epochs: 10

network:
  hidden_sizes: [512, 512, 256]
  activation: "relu"

notifications:
  discord:
    enabled: false
    webhook_url: null

export:
  enabled: false
  format: "torchscript"
  output_dir: "exported_models"
```

## 🎯 Expected Performance

With this enhanced training system, bots should achieve:

- **Stage 3**: 1000-1200 Elo (basic mechanics)
- **Stage 5**: 1300-1500 Elo (intermediate play)
- **Stage 7**: 1500-1700 Elo (advanced play)
- **Stage 9**: 1600-1800+ Elo (pro-level)

Training times (on typical hardware):
- Full 10M timesteps: 20-40 hours
- Stage 0-5 (5.5M): 12-24 hours
- Stage 0-3 (4M): 8-16 hours

## 📝 Tips & Best Practices

1. **Start with defaults** - Base config is well-tuned
2. **Use aerial curriculum** for aerial-focused training
3. **Enable Discord** for long training runs
4. **Monitor metrics** - Check plots every 1M steps
5. **Export checkpoints** - Save best models regularly
6. **Test stages individually** with `--curriculum-stage`
7. **Use debug mode** when developing features
8. **Track Elo per stage** to identify weak points

## 🔧 Troubleshooting

### CUDA Issues
```bash
# System doesn't have CUDA
python scripts/train.py --device cpu  # Automatically falls back
```

### Low Performance
- Reduce batch_size in config
- Use fewer hidden layers
- Lower total_timesteps for testing

### Curriculum Not Progressing
- Check performance metrics in logs
- Lower transition thresholds
- Verify opponent difficulty

### Export Failures
- Ensure checkpoint exists
- Check PyTorch version compatibility
- Use raw format if others fail

## 📖 Further Reading

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [SAC Paper](https://arxiv.org/abs/1801.01290)
- [Curriculum Learning](https://arxiv.org/abs/2003.04960)
- [RLBot Documentation](https://rlbot.org/)

## 🤝 Contributing

When adding new features:
1. Add comprehensive docstrings
2. Update this README
3. Add tests for new functionality
4. Ensure backward compatibility
5. Update base.yaml with new config options

## 📄 License

Same as parent RL-Bot project.
