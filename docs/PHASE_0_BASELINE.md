# Phase 0: Baseline Metrics Documentation

## Overview
This document records the baseline performance metrics before Phase 1-5 optimizations.

## Baseline Metrics (From Problem Statement)

### Training Performance
- **Training Speed**: 6-9 ticks/sec
- **GPU Utilization**: 7-10%
- **Number of Environments**: 8 (DummyVecEnv)
- **Vectorization**: DummyVecEnv (single-process)

### Model Architecture
- **Policy Size**: ~1.2M parameters (MLP)
- **Architecture**: MLP with hidden sizes [1024, 512, 256]
- **Activation**: ELU
- **Recurrent**: No LSTM/GRU

### Training Configuration
- **Algorithm**: PPO
- **Batch Size**: 32768
- **Learning Rate**: 8e-4
- **Gamma**: 0.995
- **GAE Lambda**: 0.95
- **Number of Environments**: 8

### Performance Metrics
- **Average Reward**: ~-7.4
- **Elo Rating**: ~1490 vs RulePolicy
- **Curriculum**: 5-stage (1v1 → 3v3)

## Performance Bottlenecks Identified

### 1. Environment Vectorization
- **Current**: DummyVecEnv (single-process, sequential)
- **Impact**: Limited parallelism, ~6-9 ticks/sec
- **Solution**: SubprocVecEnv for true multiprocessing

### 2. GPU Utilization
- **Current**: 7-10% GPU utilization
- **Impact**: Underutilized GPU, slow training
- **Causes**:
  - Small batch sizes relative to GPU capacity
  - CPU-bound environment steps
  - No batch inference optimization
  - No mixed precision training

### 3. Model Inference
- **Current**: Single observation inference
- **Impact**: CPU/GPU transfer overhead
- **Solution**: Batch inference, pinned memory

### 4. Training Loop
- **Current**: Standard PPO without optimizations
- **Impact**: Moderate training speed
- **Missing**:
  - Mixed precision (AMP)
  - PyTorch compile
  - Action repeat
  - Efficient buffer management

### 5. Memory Architecture
- **Current**: No LSTM/GRU for temporal modeling
- **Impact**: Limited temporal understanding
- **Solution**: Recurrent PPO with hidden state management

## Target Metrics (Phase 1-5 Goals)

### Performance Targets
- **Training Speed**: ≥25 ticks/sec (3.3× speedup)
- **GPU Utilization**: 40-80%
- **Training Time**: <12h for 10M steps (vs ~30h baseline)

### Quality Targets
- **Elo Rating**: >1600 vs RulePolicy
- **Win Rate**: >70% vs RulePolicy
- **Training Stability**: No NaN crashes, smooth convergence

### Infrastructure Targets
- **Reproducibility**: Config-driven experiments, seeded runs
- **Monitoring**: TensorBoard/W&B integration
- **Testing**: Unit and integration tests
- **Documentation**: Complete guides for training and evaluation

## Diagnostic Tools Created

### 1. Performance Diagnostics Script
**Location**: `scripts/diagnose_performance.py`

**Features**:
- Environment speed benchmark (ticks/sec)
- Model inference speed (ms/inference)
- GPU utilization monitoring
- Memory usage tracking
- Baseline comparison

**Usage**:
```bash
# Run all diagnostics
python scripts/diagnose_performance.py --test all

# Test environment speed only
python scripts/diagnose_performance.py --test env --num-steps 1000 --num-envs 8

# Test inference speed
python scripts/diagnose_performance.py --test inference

# Test GPU utilization
python scripts/diagnose_performance.py --test gpu
```

### 2. Existing Tools
- `core/infra/performance.py`: PerformanceMonitor class
- `core/infra/profiler.py`: FrameProfiler for inference timing
- Training loop includes performance logging

## Next Steps (Phase 1)

### Priority 1: Vectorization
- [ ] Implement SubprocVecEnv support
- [ ] Add environment pooling
- [ ] Benchmark improvement

### Priority 2: GPU Optimization
- [ ] Implement AMP (mixed precision)
- [ ] Add batch inference
- [ ] Enable PyTorch compile
- [ ] Use pinned memory

### Priority 3: Training Speed
- [ ] Action repeat mechanism
- [ ] Optimize buffer operations
- [ ] Reduce CPU-GPU transfers

### Success Criteria for Phase 1
- Achieve ≥15 ticks/sec (2× baseline)
- GPU utilization >30%
- No stability regression
- All tests passing

## Measurement Protocol

### Before Each Phase
1. Run full diagnostics: `python scripts/diagnose_performance.py --test all`
2. Record metrics in this document
3. Save results to `logs/phase_N_before.json`

### After Each Phase
1. Run full diagnostics again
2. Compare to baseline and previous phase
3. Document improvements and issues
4. Save results to `logs/phase_N_after.json`

### Key Metrics to Track
- **Training speed** (ticks/sec)
- **GPU utilization** (%)
- **Memory usage** (MB)
- **Inference time** (ms)
- **Training stability** (NaN count, crashes)
- **Elo rating** (vs RulePolicy)
- **Win rate** (%)

## Risk Notes

### Rollback Procedures
1. **If performance degrades**: Revert to previous commit
2. **If stability issues**: Disable optimization flags
3. **If crashes**: Check logs, reduce batch size
4. **If NaN issues**: Reduce learning rate, add gradient clipping

### Feature Flags
All optimizations will use feature flags in config:
```yaml
optimizations:
  use_subproc_vec_env: true
  use_amp: true
  use_torch_compile: true
  use_pinned_memory: true
  action_repeat: 1
```

### Testing Strategy
- Unit tests for each optimization
- Integration test: 1000-step smoke test
- Performance regression tests
- Stability tests (long runs without crashes)

## Baseline Results (To Be Filled)

### Environment Speed
- Total steps: TBD
- Total time: TBD
- Steps/sec: TBD
- Ticks/sec: TBD

### Model Inference
- Mean inference time: TBD ms
- P95 inference time: TBD ms
- P99 inference time: TBD ms
- Inferences/sec: TBD

### GPU Utilization (if CUDA available)
- Mean GPU utilization: TBD%
- Max GPU utilization: TBD%
- Mean memory utilization: TBD%

---

**Status**: Phase 0 - Diagnostics and baseline documentation
**Last Updated**: 2025-11-07
**Next Phase**: Phase 1 - Speed Optimizations
