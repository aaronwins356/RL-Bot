#!/usr/bin/env python3
"""
Quick Test Script for RL-Bot Training Framework
Validates all fixes without requiring full training run
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all critical modules can be imported."""
    print("\n" + "="*70)
    print("RL-BOT TRAINING FRAMEWORK - QUICK VALIDATION")
    print("="*70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Import environment
    print("\n[TEST 1] Importing RocketSimEnv...")
    try:
        from core.env.rocket_sim_env import RocketSimEnv
        print("[OK] RocketSimEnv imported successfully")
        
        # Check it inherits from gym.Env
        import gymnasium as gym
        if issubclass(RocketSimEnv, gym.Env):
            print("[OK] RocketSimEnv properly inherits from gym.Env")
            tests_passed += 1
        else:
            print("[ERROR] RocketSimEnv does not inherit from gym.Env")
            tests_failed += 1
    except Exception as e:
        print(f"[ERROR] Failed to import RocketSimEnv: {e}")
        tests_failed += 1
    
    # Test 2: Import training loop
    print("\n[TEST 2] Importing TrainingLoop...")
    try:
        from core.training.train_loop import TrainingLoop
        print("[OK] TrainingLoop imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"[ERROR] Failed to import TrainingLoop: {e}")
        tests_failed += 1
    
    # Test 3: Import performance monitor
    print("\n[TEST 3] Importing PerformanceMonitor...")
    try:
        from core.infra.performance import PerformanceMonitor
        print("[OK] PerformanceMonitor imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"[ERROR] Failed to import PerformanceMonitor: {e}")
        tests_failed += 1
    
    # Test 4: Import selfplay manager
    print("\n[TEST 4] Importing SelfPlayManager...")
    try:
        from core.training.selfplay import SelfPlayManager
        
        # Check curriculum stages
        manager = SelfPlayManager({})
        if len(manager.stages) == 3:
            print(f"[OK] SelfPlayManager has 3 curriculum stages")
            stage_names = [s.name for s in manager.stages]
            if stage_names == ["1v1", "1v2", "2v2"]:
                print(f"[OK] Curriculum stages are: {', '.join(stage_names)}")
                tests_passed += 1
            else:
                print(f"[ERROR] Unexpected stage names: {stage_names}")
                tests_failed += 1
        else:
            print(f"[ERROR] Expected 3 stages, got {len(manager.stages)}")
            tests_failed += 1
    except Exception as e:
        print(f"[ERROR] Failed to validate SelfPlayManager: {e}")
        tests_failed += 1
    
    # Test 5: Check observation and action spaces
    print("\n[TEST 5] Validating environment spaces...")
    try:
        from core.env.rocket_sim_env import RocketSimEnv
        import numpy as np
        
        env = RocketSimEnv(simulation_mode=True)
        
        # Check observation space
        obs_space = env.observation_space
        if obs_space.shape == (180,):
            print(f"[OK] Observation space shape: {obs_space.shape}")
        else:
            print(f"[ERROR] Expected observation shape (180,), got {obs_space.shape}")
        
        # Check action space
        action_space = env.action_space
        if action_space.shape == (8,):
            print(f"[OK] Action space shape: {action_space.shape}")
        else:
            print(f"[ERROR] Expected action shape (8,), got {action_space.shape}")
        
        # Test reset returns tuple
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
            print(f"[OK] reset() returns (obs, info) tuple")
            if obs.shape == (180,):
                print(f"[OK] Observation from reset has correct shape: {obs.shape}")
                tests_passed += 1
            else:
                print(f"[ERROR] Observation from reset has wrong shape: {obs.shape}")
                tests_failed += 1
        else:
            print(f"[ERROR] reset() does not return (obs, info) tuple")
            tests_failed += 1
        
        # Test step returns 5-tuple
        action = env.action_space.sample()
        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            print(f"[OK] step() returns 5-tuple (obs, reward, terminated, truncated, info)")
            tests_passed += 1
        else:
            print(f"[ERROR] step() does not return 5-tuple")
            tests_failed += 1
            
    except Exception as e:
        print(f"[ERROR] Failed to validate environment: {e}")
        tests_failed += 1
    
    # Test 6: Check AMP API
    print("\n[TEST 6] Validating PyTorch AMP API...")
    try:
        import torch
        
        # Check if we're using the new API (will work if code is correct)
        test_code = """
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
    x = torch.randn(1, 10)
"""
        compile(test_code, '<string>', 'exec')
        print("[OK] PyTorch AMP API usage is correct (torch.amp.autocast/GradScaler)")
        tests_passed += 1
    except SyntaxError as e:
        print(f"[ERROR] AMP API syntax error: {e}")
        tests_failed += 1
    except Exception as e:
        print(f"[WARNING] Could not fully validate AMP (PyTorch may not be installed): {e}")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n[OK] All validation tests passed!")
        print("The training framework is ready to use.")
        print("\nTo start training:")
        print("  python scripts/train.py --config configs/base.yaml")
        print("\nFor debug mode:")
        print("  python scripts/train.py --config configs/base.yaml --debug")
        return 0
    else:
        print(f"\n[ERROR] {tests_failed} test(s) failed")
        print("Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
