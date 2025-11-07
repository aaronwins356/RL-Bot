# Phase 1: Training Speed Optimizations

## Overview
Phase 1 focuses on implementing critical performance optimizations to achieve ≥2× speedup in training throughput.

## Implementation Summary

### 1. Environment Vectorization ✅
**Status**: Implemented with configuration flags

**Changes**:
- Enhanced `create_vectorized_env()` to support optimization config
- Added `use_subproc_vec_env` flag (default: true for Linux)
- Added `force_dummy_vec_env` flag for debugging
- OS-aware selection (SubprocVecEnv on Linux, DummyVecEnv on Windows)

**Configuration** (`configs/base.yaml`):
```yaml
training:
  optimizations:
    use_subproc_vec_env: true  # True multiprocessing on Linux
    force_dummy_vec_env: false  # Force DummyVecEnv for debugging
```

**Expected Impact**: 
- SubprocVecEnv provides true multiprocessing vs sequential execution
- Should improve env step throughput by 2-3× on multi-core systems

### 2. Automatic Mixed Precision (AMP) ✅
**Status**: Already implemented, now configurable

**Changes**:
- AMP already functional in `core/models/ppo.py` and `train_loop.py`
- Added configuration flags for control
- Added dtype selection (float16 vs bfloat16)

**Configuration**:
```yaml
training:
  optimizations:
    use_amp: true  # Enable AMP
    amp_dtype: "float16"  # "float16" or "bfloat16"
```

**Expected Impact**:
- 1.5-2× speedup on CUDA GPUs
- Reduced memory usage (allows larger batch sizes)
- Minimal accuracy impact with proper gradient scaling

### 3. PyTorch Compile ✅
**Status**: Implemented with safety fallback

**Changes**:
- Added `torch.compile()` support in `_create_model()`
- Configurable compile mode (default, reduce-overhead, max-autotune)
- Safe fallback if compilation fails

**Configuration**:
```yaml
training:
  optimizations:
    use_torch_compile: false  # Disabled by default (PyTorch 2.0+ required)
    compile_mode: "default"  # "default", "reduce-overhead", "max-autotune"
```

**Expected Impact**:
- 10-30% speedup for inference on PyTorch 2.0+
- Graph optimizations and kernel fusion
- May cause issues on some systems (hence disabled by default)

### 4. Memory Optimizations ✅
**Status**: Configuration added, implementation pending

**Changes**:
- Added pinned memory flag for faster CPU-GPU transfers
- Added num_workers configuration for data loading

**Configuration**:
```yaml
training:
  optimizations:
    use_pinned_memory: true  # Faster CPU-GPU transfers
    num_workers: 4  # DataLoader workers
```

**Expected Impact**:
- Reduced data transfer overhead
- Better GPU utilization
- Async data loading

### 5. Batch Inference ✅
**Status**: Configuration added, implementation pending

**Changes**:
- Added batch inference flags
- Configurable inference batch size

**Configuration**:
```yaml
training:
  optimizations:
    batch_inference: true  # Batch observations for faster inference
    inference_batch_size: 8  # Should match num_envs
```

**Expected Impact**:
- Reduced per-sample inference overhead
- Better GPU utilization
- 1.5-2× inference speedup

### 6. Action Repeat ✅
**Status**: Configuration added, implementation pending

**Changes**:
- Added action repeat configuration
- Allows repeating actions N times for faster training

**Configuration**:
```yaml
training:
  optimizations:
    action_repeat: 1  # 1 = no repeat, 2+ = repeat N times
```

**Expected Impact**:
- Faster training with less frequent policy updates
- Useful for environments where actions can be repeated
- Tradeoff: may reduce control precision

## Verification and Logging

Enhanced configuration summary now prints optimization status:
```
Optimizations:
  - SubprocVecEnv: True
  - AMP (Mixed Precision): True
  - PyTorch Compile: False
  - Pinned Memory: True
  - Batch Inference: True
  - Action Repeat: 1
```

## Performance Targets

### Current Baseline
- Training speed: 6-9 ticks/sec
- GPU utilization: 7-10%

### Phase 1 Targets
- Training speed: ≥15 ticks/sec (2× baseline)
- GPU utilization: >30%
- No stability regression

### Expected Improvements by Feature
1. **SubprocVecEnv**: +100-200% (2-3× speedup on multi-core)
2. **AMP**: +50-100% (1.5-2× speedup on GPU)
3. **torch.compile**: +10-30% (additional speedup)
4. **Pinned Memory**: +5-15% (reduced transfer overhead)
5. **Batch Inference**: +50-100% (better GPU utilization)

**Total Expected**: 3-5× baseline (if all optimizations stack)

## Testing

### Unit Tests Needed
- [ ] Test SubprocVecEnv creation with opt_config
- [ ] Test torch.compile with different modes
- [ ] Test AMP with gradient scaling
- [ ] Test configuration loading

### Integration Tests
- [ ] 1000-step smoke test with all optimizations
- [ ] Performance regression test
- [ ] Stability test (no NaN/crashes)

## Benchmarking Protocol

### Before Phase 1
```bash
python scripts/diagnose_performance.py --test all > logs/phase1_before.txt
```

### After Phase 1
```bash
python scripts/diagnose_performance.py --test all > logs/phase1_after.txt
```

### Compare Results
- Training speed (ticks/sec)
- GPU utilization (%)
- Inference time (ms)
- Memory usage (MB)

## Risk Mitigation

### Feature Flags
All optimizations are controlled by configuration flags:
- Can be disabled individually if causing issues
- Safe fallback mechanisms in place

### Rollback Procedure
1. Set problematic optimization flag to `false` in `configs/base.yaml`
2. Restart training
3. If issues persist, revert git commit

### Known Issues and Workarounds

**torch.compile**:
- May fail on older PyTorch versions → Fallback to uncompiled model
- May cause CUDA errors on some GPUs → Set `use_torch_compile: false`

**SubprocVecEnv**:
- Windows compatibility issues → Falls back to DummyVecEnv
- Multiprocessing overhead on small envs → Use `force_dummy_vec_env: true`

**AMP**:
- Requires CUDA device → Automatically disabled on CPU
- May cause gradient underflow → Gradient scaling handles this

## Next Steps (Phase 2)

### LSTM/GRU Support
- [ ] Add recurrent architecture support
- [ ] Implement hidden state management
- [ ] Update PPO for recurrent policies

### Training Stability
- [ ] Add NaN detection
- [ ] Implement gradient clipping
- [ ] Add training stability gates

---

**Status**: Phase 1 - Core optimizations implemented
**Last Updated**: 2025-11-07
**Next Phase**: Phase 2 - Recurrent PPO and stability improvements
