# Quick Start Guide - Optimized Training

## What's New in v2.0.0

This version includes a completely refactored training system that fixes the reward plateau issue (-7.7 to -7.3). The bot should now learn recognizable Rocket League skills and achieve positive average rewards.

## Key Improvements

✅ **Dynamic PPO Features** - Adaptive learning algorithms
✅ **Reward Normalization** - Stable learning signals  
✅ **Learning Rate Scheduling** - Better convergence
✅ **Progressive Reward Shaping** - Start simple, add complexity
✅ **Advanced Mechanics** - Bumping, flicks, aerial saves, wall shots, recoveries
✅ **Automatic Curriculum** - Performance-based stage transitions

## Quick Start

### Windows (Recommended)

```powershell
# Basic training with optimized settings
.\train.ps1

# Custom configuration
.\train.ps1 -Timesteps 5000000 -Device cuda

# Debug mode (1000 steps)
.\train.ps1 -Debug
```

### Linux/Mac

```bash
# Make script executable
chmod +x train.sh

# Basic training
./train.sh

# Or use Python directly
python scripts/train.py --config configs/training_optimized.yaml
```

## What to Expect

### Training Progress

**First 100k steps** (0-20 minutes):
- Average reward: -5.0 to 0.0
- Agent learns to touch ball, basic movement
- Action entropy decreases gradually

**100k-500k steps** (20 minutes - 2 hours):
- Average reward: 0.0 to +2.0
- Agent learns positioning, simple shots
- First goals scored

**500k-1M steps** (2-4 hours):
- Average reward: +2.0 to +5.0
- Agent learns aerial basics, boost management
- Consistent goal scoring

**1M+ steps** (4+ hours):
- Average reward: +5.0 to +10.0
- Advanced mechanics emerge
- Win rate vs. bots improves

### Key Metrics

Monitor these in TensorBoard (`tensorboard --logdir logs/latest_run`):

- **Episode Reward**: Should trend upward
- **Explained Variance**: Should be > 0.5 (good value function)
- **Policy Loss**: Should decrease
- **Entropy**: Should gradually decrease
- **Clip Fraction**: Should stay 0.1-0.3

## Configuration Files

### `configs/training_optimized.yaml`
Main training configuration with all advanced PPO features enabled.

**Key settings**:
- Dynamic GAE lambda: Adapts based on learning progress
- Entropy annealing: Gradual exploration reduction  
- Clip range decay: Improved stability over time
- Learning rate scheduling: Cosine annealing with warmup
- Reward normalization: Running mean/std tracking

### `configs/rewards_optimized.yaml`
Improved reward function with progressive complexity.

**Key changes**:
- Sparse rewards: 10x larger (100 for goals vs 10 before)
- Dense rewards: Reduced noise (50-80% smaller)
- Progressive schedules: Enable complex rewards gradually
- Better penalties: Stronger deterrents for bad habits

## Troubleshooting

### Issue: Rewards still negative after 500k steps

**Check**:
1. Config file is `training_optimized.yaml` (not `base.yaml`)
2. Reward normalization is enabled (check logs for "Normalization wrapper applied")
3. CUDA is working if using GPU (check "CUDA initialized successfully")

**Fix**:
```powershell
# Verify config
.\train.ps1 -Config configs/training_optimized.yaml

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Training crashes / Out of Memory

**Solutions**:
- Reduce batch size: Change `batch_size: 16384` to `8192` in config
- Reduce num_envs: Change `num_envs: 8` to `4` in config  
- Disable AMP: Set `use_amp: false` in optimizations

### Issue: Rewards exploding (>100)

**Check**:
- Reward clipping is enabled in `configs/rewards_optimized.yaml`
- Normalization wrapper is applied

## Advanced Usage

### Custom Reward Function

Edit `configs/rewards_optimized.yaml`:

```yaml
sparse:
  goal_scored: 100.0  # Increase for stronger signal
  
penalties:
  whiff: -0.5  # Increase penalty to discourage behavior
```

### Learning Rate Tuning

Edit `configs/training_optimized.yaml`:

```yaml
training:
  learning_rate: 3.0e-4  # Reduce if training unstable
  
  lr_scheduler:
    min_lr: 1.0e-5  # Lower bound
```

### Curriculum Stages

Force specific stage:

```powershell
.\train.ps1 -CurriculumStage 0  # Force 1v1
.\train.ps1 -CurriculumStage 1  # Force 1v2  
.\train.ps1 -CurriculumStage 2  # Force 2v2
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/latest_run
```

View at http://localhost:6006

### Log Files

Training logs are saved to:
- `logs/latest_run/train.log` - Detailed training log
- `logs/latest_run/events.*` - TensorBoard events
- `logs/latest_run/checkpoints/` - Model checkpoints

### Checkpoints

Checkpoints are saved:
- Every 50k steps (configurable)
- Every 250k steps (automatic)
- Best model (highest ELO)

Load checkpoint:

```powershell
.\train.ps1 -Checkpoint logs/latest_run/checkpoints/best_checkpoint.pth
```

## Performance Tips

### RTX 3060 Optimization

The default config is optimized for RTX 3060:
- Mixed precision (AMP) enabled
- 8 parallel environments
- Batch size 16384
- Expected: ~2000-3000 steps/second

### CPU Training

If using CPU:

```yaml
inference:
  device: "cpu"

training:
  num_envs: 4  # Reduce from 8
  batch_size: 8192  # Reduce from 16384
  optimizations:
    use_amp: false  # Disable AMP
```

## Next Steps

1. **Run basic training**: `.\train.ps1`
2. **Monitor TensorBoard**: Watch episode rewards increase
3. **Evaluate bot**: `python scripts/evaluate.py`
4. **Export model**: Use `--export-checkpoint` flag

## Support

For detailed technical information, see [TRAINING_REFACTORING.md](TRAINING_REFACTORING.md).

For issues:
1. Check log files in `logs/latest_run/`
2. Verify config files are UTF-8 encoded
3. Ensure Python 3.10+ and PyTorch 2.0+ installed

---

**Happy Training! 🚀**
