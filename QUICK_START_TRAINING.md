# RL-Bot Training System - Quick Start Guide

## Overview

This guide provides quick instructions for using the newly implemented training system upgrades.

## New Features

1. **League-Based Self-Play** - Automatically manages 8-12 agents for diverse training
2. **Enhanced Evaluation** - Comprehensive Elo tracking with 200 games per evaluation
3. **Adaptive PPO** - Decay schedules for clip range and entropy
4. **Dense Rewards** - Ball touches, shots, goals with automatic normalization
5. **Windows Fix** - torch.compile works on Windows with eager backend

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install optuna  # For hyperparameter tuning (optional)
```

### 2. Configure Training

Edit `configs/config_optimized.yaml` to enable features:

```yaml
training:
  # Enable league-based self-play
  league:
    enabled: true
    min_population: 8
    max_population: 12
  
  # Enable enhanced evaluation
  evaluation:
    enabled: true
    eval_interval: 25000  # Every 25k steps
    games_per_opponent: 100  # 200 total games
  
  # Enable reward shaping
  reward_shaping:
    enabled: true
    normalize_final_reward: true
```

### 3. Run Training

```bash
python scripts/train.py --config configs/config_optimized.yaml
```

### 4. Monitor Progress

```bash
# TensorBoard
tensorboard --logdir logs/latest_run/tensorboard

# Or Weights & Biases (if enabled)
# View at https://wandb.ai/your-username/rlbot-league
```

## Key Files

### Training Outputs

- `logs/latest_run/checkpoints/` - Model checkpoints every 25k steps
- `logs/latest_run/league/league_state.json` - League population and stats
- `logs/evaluation/elo_history.json` - Elo progression over time
- `logs/latest_run/tensorboard/` - TensorBoard logs

### Best Checkpoint

The best checkpoint is automatically saved at:
- `logs/latest_run/checkpoints/best_checkpoint.pth`

## Hyperparameter Tuning (Optional)

Run automated hyperparameter search:

```bash
python scripts/tune.py --trials 50 --timesteps 300000 --output tuning_results.json
```

This will:
- Search over LR, entropy, batch size, etc.
- Run 50 trials @ 300k steps each
- Save results to `tuning_results.json`
- Generate optimization plot

## Evaluation

After training, evaluate the model:

```bash
python scripts/evaluate.py --checkpoint logs/latest_run/checkpoints/best_checkpoint.pth
```

## Configuration Reference

### PPO Hyperparameters

```yaml
training:
  batch_size: 8192          # Batch size
  learning_rate: 1.5e-4     # Initial LR
  clip_range: 0.2           # Initial clip (decays to 0.1)
  ent_coef: 0.02            # Initial entropy (decays to 0.005)
  gamma: 0.99               # Discount factor
  gae_lambda: 0.97          # GAE lambda
  num_envs: 16              # Parallel environments
```

### League Configuration

```yaml
training:
  league:
    enabled: true
    min_population: 8       # Min agents
    max_population: 12      # Max agents
    max_past_agents: 6      # Past checkpoints
    max_exploiters: 3       # Exploiter agents
    diversity_temperature: 1.0  # Matchmaking diversity
    age_decay_rate: 0.1     # Older agent decay
```

### Evaluation Configuration

```yaml
training:
  evaluation:
    enabled: true
    eval_interval: 25000        # Steps between evals
    games_per_opponent: 100     # Games per opponent
    elo_ema_alpha: 0.3          # EMA smoothing
    early_stop_patience: 8      # Evals without improvement
    early_stop_max_std: 15.0    # Confidence threshold
```

### Curriculum Configuration

```yaml
training:
  curriculum:
    enabled: true
    stages: ["1v1", "1v2", "2v2"]
    progression_rules:
      - stage: "1v1"
        base_elo: 1400
        promotion_elo_delta: 100    # Promote if Elo > 1500
        promotion_win_rate: 0.55     # AND win rate > 55%
```

## Performance Tuning

### GPU Optimization

```yaml
training:
  optimizations:
    use_amp: true               # Mixed precision
    use_torch_compile: true     # PyTorch 2.0+ compile
    compile_mode: "reduce-overhead"
```

### CPU/Memory Optimization

```yaml
training:
  num_envs: 16                  # Adjust for your CPU
  optimizations:
    use_subproc_vec_env: true   # True for Linux
    force_dummy_vec_env: false  # True for Windows
```

## Troubleshooting

### Windows Triton Error

If you see "Cannot find a working triton installation":
- This is fixed automatically
- Model will use `backend="eager"` on Windows
- You'll see: `[Windows] Triton not supported, using eager backend`

### Low GPU Utilization

Try increasing:
- `num_envs` (more parallel environments)
- `batch_size` (larger batches)
- Enable `use_amp` and `use_torch_compile`

### High Memory Usage

Try decreasing:
- `num_envs` (fewer parallel environments)
- `batch_size` (smaller batches)
- `buffer_size` in telemetry config

### Slow Training

Check:
- GPU utilization (should be >40%)
- `num_envs` (should be 8-16)
- `use_amp` and `use_torch_compile` enabled
- Timesteps/sec in logs (should be >15)

## Success Metrics

Monitor these during training:

1. **Elo Progression**
   - Target: 1550 @ 300k steps
   - Target: 1700 @ 1M steps

2. **Training Speed**
   - Target: ≥15 timesteps/sec
   - Check: TensorBoard "performance/timesteps_per_sec"

3. **GPU Utilization**
   - Target: ≥40%
   - Check: TensorBoard "performance/gpu_utilization"

4. **Entropy Decay**
   - Should decrease smoothly from 0.02 to 0.005
   - Check: TensorBoard "train/entropy_coef"

5. **KL Divergence**
   - Should stay in 0.01-0.03 range
   - Check: TensorBoard "train/kl_divergence"

## Advanced Usage

### Custom Reward Weights

Edit `configs/config_optimized.yaml`:

```yaml
training:
  reward_shaping:
    ball_touch_reward: 0.05
    shot_on_goal_reward: 0.10
    goal_reward: 1.0
    own_goal_penalty: -1.0
```

### Custom Curriculum

```yaml
training:
  curriculum:
    stages: ["1v1", "1v2", "2v2"]
    progression_rules:
      - stage: "1v1"
        base_elo: 1400
        promotion_elo_delta: 150  # Harder to promote
```

### Weights & Biases Integration

```yaml
logging:
  wandb:
    enabled: true
    project: "rlbot-league"
    entity: "your-username"
```

## Next Steps

1. ✅ Run dry test: Verify setup works
2. ✅ Train to 150k: Check learning curves
3. ✅ Analyze metrics: Verify convergence
4. ✅ Continue to 1M: Full training run
5. ✅ Evaluate: Test final Elo

## Support

See full documentation in:
- `TRAINING_IMPLEMENTATION_REPORT.md` - Complete implementation details
- `configs/config_optimized.yaml` - All configuration options

## Tips

- Start with shorter runs (150k steps) to verify setup
- Monitor Elo and entropy closely in first 50k steps
- Save checkpoints frequently (every 25k)
- Use TensorBoard for real-time monitoring
- League composition will auto-update every 25k steps
- Early stopping will trigger after 8 evals without improvement

Good luck reaching Elo 1700! 🚀
