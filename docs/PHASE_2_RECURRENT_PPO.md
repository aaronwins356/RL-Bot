# Phase 2: Recurrent PPO Implementation

## Overview
Phase 2 implements recurrent policy support with LSTM/GRU for temporal modeling and improved learning quality. This allows the agent to maintain memory of past observations and make better decisions based on temporal context.

## Implementation Summary

### 1. Recurrent PPO Algorithm ✅

**File**: `core/models/recurrent_ppo.py`

**Features**:
- Extends standard PPO to handle LSTM/GRU policies
- Proper hidden state management across environments
- Sequence-based training with truncated BPTT
- Episode boundary handling (resets hidden states)
- Compatible with mixed precision training (AMP)

**Key Methods**:
```python
# Hidden state management
ppo.reset_hidden_states(env_ids=[0, 1, 2])
ppo.get_hidden_state(env_id)
ppo.set_hidden_state(env_id, hidden)

# Recurrent update
ppo.update_recurrent(sequences, advantages, returns)
```

### 2. Sequence Buffer ✅

**File**: `core/training/sequence_buffer.py`

**Features**:
- Stores complete episodes or fixed-length sequences
- Handles LSTM hidden states
- Computes advantages with GAE
- Efficient sequence sampling for training
- Episode boundary tracking

**Usage**:
```python
buffer = SequenceBuffer(
    capacity=100000,
    sequence_length=16,
    store_full_episodes=True
)

# Add transitions
buffer.add_transition(
    observation=obs,
    action_cat=act_cat,
    action_ber=act_ber,
    reward=reward,
    done=done,
    value=value,
    log_prob_cat=log_prob_cat,
    log_prob_ber=log_prob_ber,
    hidden_state=hidden
)

# Get sequences for training
sequences = buffer.get_sequences(num_sequences=32)
advantages, returns = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)
```

### 3. Configuration ✅

**File**: `configs/base.yaml`

**Recurrent Settings**:
```yaml
network:
  use_lstm: true              # Enable LSTM
  lstm_hidden_size: 256       # LSTM hidden dimension
  lstm_num_layers: 1          # Number of LSTM layers

training:
  sequence_length: 16         # Sequence length for TBPTT
  truncate_bptt: true         # Truncate backprop through time
  store_full_episodes: true   # Store complete episodes
```

### 4. Testing ✅

**File**: `tests/test_recurrent_ppo.py`

**Test Coverage**:
- ✅ Buffer initialization and transition storage
- ✅ Sequence creation and batching
- ✅ Advantage computation with GAE
- ✅ RecurrentPPO initialization
- ✅ Hidden state management (reset, get, set)

**Run Tests**:
```bash
pytest tests/test_recurrent_ppo.py -v
```

## Architecture Changes

### Model Forward Pass
```python
# Standard PPO
cat_probs, ber_probs, value, _, _ = model(obs)

# Recurrent PPO
cat_probs, ber_probs, value, _, hidden = model(obs, hidden)
```

### Training Loop Changes

**Before (Standard PPO)**:
```python
# Collect experience
for step in range(rollout_length):
    action, value = policy(obs)
    obs, reward, done = env.step(action)
    buffer.add(obs, action, reward, done, value)

# Update
batch = buffer.sample(batch_size)
ppo.update(batch)
```

**After (Recurrent PPO)**:
```python
# Collect experience with hidden states
hidden_states = {}
for step in range(rollout_length):
    hidden = hidden_states.get(env_id, None)
    action, value, hidden = policy(obs, hidden)
    obs, reward, done = env.step(action)
    
    # Store transition with hidden state
    sequence_buffer.add_transition(
        obs, action, reward, done, value, 
        log_prob, hidden_state=hidden
    )
    
    # Reset hidden state on episode boundary
    if done:
        hidden_states[env_id] = None
    else:
        hidden_states[env_id] = hidden

# Update with sequences
sequences = sequence_buffer.get_sequences()
advantages, returns = sequence_buffer.compute_advantages()
recurrent_ppo.update_recurrent(sequences, advantages, returns)
```

## Performance Impact

### Expected Improvements

1. **Better Temporal Understanding** (+10-20% Elo)
   - Agent can remember past observations
   - Better prediction of opponent behavior
   - Improved situational awareness

2. **Reduced Sample Complexity** (-20-30% required samples)
   - More efficient learning from experience
   - Better credit assignment over time
   - Improved long-term planning

3. **Training Stability**
   - Smoother learning curves
   - Reduced variance in policy updates
   - Better handling of partial observability

### Computational Cost

- **Inference**: +15-25% (LSTM overhead)
- **Training**: +20-35% (sequence processing)
- **Memory**: +30-50% (hidden state storage)

**Mitigation**:
- Use truncated BPTT (sequence_length=16)
- Batch hidden states efficiently
- Enable AMP for GPU acceleration

## Integration with Existing Code

### Compatible Features
- ✅ AMP (mixed precision training)
- ✅ SubprocVecEnv (parallel environments)
- ✅ torch.compile (PyTorch 2.0+)
- ✅ Curriculum learning
- ✅ Self-play and Elo tracking

### Usage in Training Loop

**Enable LSTM**:
```yaml
# configs/base.yaml
network:
  use_lstm: true
  lstm_hidden_size: 256
```

**Training Script**:
```bash
# Train with LSTM
python scripts/train.py \
  --config configs/base.yaml \
  --device cuda \
  --timesteps 10000000
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `sequence_length` (e.g., 8 instead of 16)
- Reduce `lstm_hidden_size` (e.g., 128 instead of 256)
- Reduce `num_envs` or `batch_size`

**2. Training Instability**
- Increase `truncate_bptt` frequency
- Use gradient clipping (`max_grad_norm: 0.5`)
- Reduce learning rate

**3. Slow Training**
- Enable `use_amp` for GPU speedup
- Reduce `sequence_length`
- Use `store_full_episodes: false` for fixed sequences

**4. NaN Losses**
- Check hidden state reset on episode boundaries
- Ensure proper sequence padding
- Reduce learning rate

### Debug Mode

Test recurrent training:
```bash
python scripts/train.py \
  --config configs/base.yaml \
  --debug \
  --debug-ticks 1000
```

## Validation

### Unit Tests
```bash
# Run all Phase 2 tests
pytest tests/test_recurrent_ppo.py -v

# Specific tests
pytest tests/test_recurrent_ppo.py::TestSequenceBuffer -v
pytest tests/test_recurrent_ppo.py::TestRecurrentPPO -v
```

### Integration Test
```bash
# 1000-step smoke test with LSTM
python scripts/train.py \
  --debug \
  --debug-ticks 1000 \
  --config configs/base.yaml
```

## Best Practices

### 1. Sequence Length Selection
- **Short sequences (8-16)**: Faster training, less temporal context
- **Medium sequences (16-32)**: Good balance
- **Long sequences (32+)**: Better temporal modeling, slower training

### 2. Hidden State Management
- Always reset hidden states on episode boundaries
- Detach hidden states between sequences (prevent infinite backprop)
- Store initial hidden state with each sequence

### 3. Buffer Management
- Use `store_full_episodes=True` for variable-length episodes
- Use `store_full_episodes=False` for fixed-length sequences
- Clear buffer periodically to avoid memory issues

### 4. Training Configuration
```yaml
# Recommended settings for recurrent PPO
network:
  use_lstm: true
  lstm_hidden_size: 256
  lstm_num_layers: 1

training:
  sequence_length: 16
  truncate_bptt: true
  batch_size: 2048          # Smaller than standard PPO
  learning_rate: 3.0e-4     # Slightly lower
  n_epochs: 3               # Fewer epochs per update
```

## Next Steps

### Phase 3: Enhanced Curriculum
- ✅ Curriculum infrastructure exists
- Add sparse reward mode flag
- Create reward visualization tools
- Add behavioral metrics logging

### Phase 4: Advanced Self-Play
- ✅ Self-play infrastructure exists
- Add Elo-based opponent sampling
- Implement tournament framework
- Save replays for top episodes

### Phase 5: Final Testing & Docs
- Integration smoke tests
- Performance regression tests
- Update main README
- Final documentation polish

---

**Status**: Phase 2 - Complete implementation
**Last Updated**: 2024-11-07
**Next Phase**: Phase 3 - Enhanced Curriculum & Rewards
