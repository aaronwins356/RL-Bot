# RL-Bot Quickstart Guide

## Overview
This guide will help you get started with training and optimizing the RL-Bot in under 10 minutes.

## Prerequisites

### System Requirements
- Python 3.9+
- 8GB+ RAM
- GPU with CUDA 12.x (recommended for best performance)
- Linux/Windows/macOS

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/aaronwins356/RL-Bot.git
cd RL-Bot
```

2. **Install dependencies**:
```bash
# With CUDA (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Quick Training

### 1. Run Performance Diagnostics (Optional)

Get baseline performance metrics:
```bash
python scripts/diagnose_performance.py --test all
```

This will show:
- Environment speed (ticks/sec)
- Model inference speed (ms)
- GPU utilization (%)
- Memory usage

### 2. Start Training

**Basic training** (default settings):
```bash
python scripts/train.py --config configs/base.yaml
```

**Optimized training** (recommended):
```bash
python scripts/train.py \
  --config configs/base.yaml \
  --timesteps 5000000 \
  --device cuda \
  --logdir logs/my_first_run
```

**Debug mode** (fast test with 1000 steps):
```bash
python scripts/train.py --debug --debug-ticks 1000
```

### 3. Monitor Training

**With TensorBoard**:
```bash
# In another terminal
tensorboard --logdir logs/
# Open http://localhost:6006
```

**Console output**:
```
[OK] Training speed: 15.3 timesteps/sec
[OK] GPU utilization: 45.2%, Memory: 3024/8192 MB
Episode 1000, Reward: -5.2, Elo: 1520
```

## Configuration

### Enable All Optimizations

Edit `configs/base.yaml`:

```yaml
training:
  # More environments = faster training
  num_envs: 8
  
  # Larger batch = better GPU utilization
  batch_size: 32768
  
  # Performance optimizations
  optimizations:
    use_subproc_vec_env: true  # True multiprocessing (Linux)
    use_amp: true               # Mixed precision training
    use_torch_compile: false    # PyTorch 2.0+ only
    use_pinned_memory: true     # Faster GPU transfers
    batch_inference: true       # Batch observations
    action_repeat: 1            # 1 = no repeat
```

### Adjust for Your Hardware

**Low-end GPU** (4GB VRAM):
```yaml
training:
  num_envs: 4
  batch_size: 16384
  optimizations:
    use_amp: true  # Reduce memory usage
```

**High-end GPU** (16GB+ VRAM):
```yaml
training:
  num_envs: 16
  batch_size: 65536
  optimizations:
    use_amp: true
    use_torch_compile: true
```

**CPU only**:
```yaml
training:
  num_envs: 4
  batch_size: 8192
  optimizations:
    use_subproc_vec_env: true  # Still helps on CPU
    use_amp: false  # AMP requires GPU
    force_dummy_vec_env: false
```

## Evaluation

### Evaluate a Trained Model

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --opponents rule_policy baseline_ml \
  --num-games 10
```

**Output**:
```
Playing against rule_policy...
  Game 1/10: WIN (3-2) - Elo: 1516
  Game 2/10: WIN (4-1) - Elo: 1532
  ...
  Summary: 7-3-0 (Win rate: 70.0%)

Final Elo Rating: 1543
```

## Performance Tips

### Maximize Training Speed

1. **Use GPU**: 2-3× faster than CPU
   ```bash
   --device cuda
   ```

2. **Increase environments**: More parallel experience collection
   ```yaml
   num_envs: 16  # If you have enough RAM/GPU memory
   ```

3. **Enable SubprocVecEnv**: True multiprocessing on Linux
   ```yaml
   use_subproc_vec_env: true
   ```

4. **Enable AMP**: Mixed precision for GPU speedup
   ```yaml
   use_amp: true
   ```

5. **Larger batch size**: Better GPU utilization
   ```yaml
   batch_size: 65536  # If GPU memory allows
   ```

### Maximize Learning Quality

1. **Use curriculum learning**: Progressive difficulty
   ```bash
   --aerial-curriculum
   ```

2. **Longer training**: More timesteps = better performance
   ```bash
   --timesteps 10000000
   ```

3. **Evaluation**: Regular evaluation to track progress
   ```yaml
   logging:
     eval_interval: 20000
     eval_num_games: 25
   ```

4. **Self-play**: Learn from past versions
   ```yaml
   training:
     selfplay:
       enabled: true
   ```

## Monitoring

### TensorBoard Metrics

Key metrics to watch:

1. **Performance**:
   - `training_speed_timesteps_per_sec`: Should be ≥15 ticks/sec
   - `gpu_utilization_percent`: Target 40-80%

2. **Learning**:
   - `policy_loss`: Should decrease over time
   - `value_loss`: Should decrease over time
   - `entropy`: Should gradually decrease
   - `reward_mean`: Should increase over time

3. **Evaluation**:
   - `eval_elo`: Should increase over time (target >1600)
   - `eval_win_rate`: Should increase (target >70%)

### Common Issues

**Low GPU utilization (<20%)**:
- Increase `num_envs`
- Increase `batch_size`
- Enable `batch_inference`

**Out of memory**:
- Reduce `num_envs`
- Reduce `batch_size`
- Enable `use_amp`

**Training too slow**:
- Enable `use_subproc_vec_env` (Linux)
- Enable `use_amp`
- Increase `batch_size`
- Use GPU instead of CPU

**NaN losses**:
- Reduce `learning_rate`
- Increase `clip_range`
- Check observation normalization

## Next Steps

### 1. Experiment with Hyperparameters

Try different settings in `configs/base.yaml`:
- Learning rate: `1e-4` to `1e-3`
- Batch size: `16384` to `65536`
- Number of environments: `4` to `16`

### 2. Use Advanced Features

**LSTM for temporal modeling**:
```yaml
network:
  use_lstm: true
  lstm_hidden_size: 256
```

**Custom reward shaping**:
Edit `configs/rewards.yaml` to tune reward weights.

**Offline pretraining**:
```bash
--offline-pretrain
```

### 3. Evaluate Progress

Run regular evaluations:
```bash
# Every 100k steps
python scripts/evaluate.py \
  --checkpoint logs/my_run/checkpoints/checkpoint_100000.pt \
  --opponents rule_policy \
  --num-games 25
```

### 4. Deploy to RLBot

Once trained, deploy your bot:
```bash
python scripts/train.py \
  --export-checkpoint exported_models/ \
  --export-format torchscript
```

## Useful Commands

### Training
```bash
# Basic training
python scripts/train.py

# Custom config
python scripts/train.py --config my_config.yaml

# Resume training
python scripts/train.py --checkpoint logs/latest_run/checkpoints/latest_model.pt

# Debug mode
python scripts/train.py --debug --debug-ticks 1000
```

### Diagnostics
```bash
# Full diagnostics
python scripts/diagnose_performance.py --test all

# Environment speed only
python scripts/diagnose_performance.py --test env --num-envs 8

# GPU utilization
python scripts/diagnose_performance.py --test gpu
```

### Evaluation
```bash
# Evaluate checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Multiple opponents
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --opponents rule_policy baseline_ml nexto \
  --num-games 20
```

### Monitoring
```bash
# TensorBoard
tensorboard --logdir logs/

# Watch logs
tail -f logs/latest_run/train.log
```

## Getting Help

### Documentation
- `README.md` - Main documentation
- `docs/IMPLEMENTATION_GUIDE.md` - Detailed implementation guide
- `docs/PHASE_0_BASELINE.md` - Performance baseline
- `docs/PHASE_1_OPTIMIZATIONS.md` - Optimization details

### Common Questions

**Q: How long does training take?**
A: ~12-18 hours for 10M steps on GPU, ~48-72 hours on CPU.

**Q: What performance should I expect?**
A: With optimizations: 15-25 ticks/sec, Elo ~1550-1650 after 10M steps.

**Q: Can I train on CPU?**
A: Yes, but it will be 2-3× slower. Use fewer environments and smaller batch size.

**Q: How do I know if training is working?**
A: Check that:
1. Loss decreases over time
2. Elo rating increases
3. No NaN errors
4. GPU utilization >30%

**Q: When should I stop training?**
A: When Elo plateaus or exceeds your target (e.g., >1600).

## Quick Reference

### Expected Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Training Speed | ≥25 ticks/sec | 15-20 |
| GPU Utilization | 40-80% | 35-60% |
| Elo (10M steps) | >1600 | 1550-1650 |
| Win Rate vs Rule | >70% | 65-75% |
| Training Time (GPU) | 12-18h | ~15h |

### Hardware Recommendations

| Hardware | num_envs | batch_size | Expected Speed |
|----------|----------|------------|----------------|
| CPU only | 4 | 8192 | 4-6 ticks/sec |
| GPU 4GB | 4 | 16384 | 10-15 ticks/sec |
| GPU 8GB | 8 | 32768 | 15-20 ticks/sec |
| GPU 16GB+ | 16 | 65536 | 20-30 ticks/sec |

---

**Ready to start?** Run `python scripts/train.py --config configs/base.yaml` and watch your bot learn!
