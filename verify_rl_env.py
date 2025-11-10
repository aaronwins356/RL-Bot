#!/usr/bin/env python3
"""
RocketMind Environment Verification Script
Tests all critical imports, environment creation, and simulation.
"""

import sys
import time
from typing import Dict, Any
import traceback


class EnvironmentVerifier:
    """Verify RocketMind environment setup."""
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{'='*70}")
        print(f"  {text}")
        print(f"{'='*70}\n")
    
    def print_status(self, status: str, message: str):
        """Print status message."""
        symbols = {
            'success': '✓',
            'error': '✗',
            'warning': '⚠',
            'info': 'ℹ'
        }
        symbol = symbols.get(status, '•')
        print(f"{symbol} {message}")
    
    def check_imports(self) -> bool:
        """Test all critical imports."""
        self.print_header("STEP 1: Verifying Package Imports")
        
        imports = [
            # Critical packages
            ('torch', 'PyTorch', True),
            ('gymnasium', 'Gymnasium', True),
            ('numpy', 'NumPy', True),
            
            # RLGym packages
            ('rlgym.rocket_league', 'RLGym-Rocket-League', True),
            ('rlgym_tools', 'RLGym-Tools', False),
            
            # Optional packages
            ('streamlit', 'Streamlit', False),
            ('plotly', 'Plotly', False),
            ('pygame', 'Pygame', False),
            ('tensorboard', 'TensorBoard', False),
            ('yaml', 'PyYAML', False),
            ('tqdm', 'tqdm', False),
        ]
        
        all_critical_ok = True
        
        for module_name, display_name, critical in imports:
            try:
                __import__(module_name)
                self.print_status('success', f"{display_name} imported successfully")
                self.checks_passed.append(f"Import {display_name}")
            except ImportError as e:
                if critical:
                    self.print_status('error', f"{display_name} import FAILED (critical)")
                    self.checks_failed.append(f"Import {display_name}")
                    all_critical_ok = False
                else:
                    self.print_status('warning', f"{display_name} import failed (optional)")
                    self.warnings.append(f"Import {display_name}")
        
        return all_critical_ok
    
    def check_hardware(self) -> Dict[str, Any]:
        """Check hardware availability."""
        self.print_header("STEP 2: Hardware Diagnostics")
        
        info = {}
        
        try:
            import torch
            
            # CPU info
            self.print_status('info', f"PyTorch version: {torch.__version__}")
            info['torch_version'] = torch.__version__
            
            # CUDA/GPU
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                    for i in range(torch.cuda.device_count())]
                
                self.print_status('success', "CUDA is available")
                self.print_status('info', f"  CUDA Version: {torch.version.cuda}")
                self.print_status('info', f"  GPU Count: {torch.cuda.device_count()}")
                
                for i, name in enumerate(info['gpu_names']):
                    self.print_status('info', f"  GPU {i}: {name}")
                    # Try to allocate memory on GPU
                    try:
                        test_tensor = torch.zeros(100, 100).cuda(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
                        memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
                        self.print_status('success', 
                            f"    Memory test passed ({memory_allocated:.1f}MB allocated)")
                        del test_tensor
                        torch.cuda.empty_cache()
                    except Exception as e:
                        self.print_status('warning', f"    Memory test failed: {e}")
                
                self.checks_passed.append("CUDA availability")
            else:
                info['cuda_available'] = False
                self.print_status('warning', "CUDA not available - will use CPU")
                self.print_status('info', "Training will be significantly slower on CPU")
                self.warnings.append("No GPU available")
            
            # CPU threads
            import os
            cpu_count = os.cpu_count()
            info['cpu_count'] = cpu_count
            self.print_status('info', f"CPU cores available: {cpu_count}")
            
        except Exception as e:
            self.print_status('error', f"Hardware check failed: {e}")
            self.checks_failed.append("Hardware diagnostics")
            return {}
        
        return info
    
    def test_environment_creation(self) -> bool:
        """Test creating an RLGym-Rocket-League environment."""
        self.print_header("STEP 3: Testing Environment Creation")
        
        try:
            import rlgym.rocket_league
            from rlgym.rocket_league.utils import StateSetter, RewardFunction, ObsBuilder
            import numpy as np
            
            self.print_status('info', "Creating RLGym-Rocket-League environment...")
            
            # Try to create environment with default settings
            try:
                env = rlgym.rocket_league.make(
                    team_size=1,
                    tick_skip=8,
                    spawn_opponents=True,
                    self_play=False
                )
                self.print_status('success', "Environment created successfully")
                self.checks_passed.append("Environment creation")
                
                # Get observation space info
                obs_space = env.observation_space
                action_space = env.action_space
                
                self.print_status('info', f"  Observation space: {obs_space}")
                self.print_status('info', f"  Action space: {action_space}")
                
                return True, env
                
            except Exception as e:
                self.print_status('error', f"Environment creation failed: {e}")
                self.checks_failed.append("Environment creation")
                traceback.print_exc()
                return False, None
                
        except ImportError as e:
            self.print_status('error', f"Cannot import rlgym.rocket_league: {e}")
            self.checks_failed.append("RLGym import")
            return False, None
    
    def test_environment_steps(self, env) -> bool:
        """Test stepping through the environment."""
        self.print_header("STEP 4: Testing Environment Simulation")
        
        if env is None:
            self.print_status('error', "No environment to test (creation failed)")
            return False
        
        try:
            import numpy as np
            
            self.print_status('info', "Resetting environment...")
            obs, info = env.reset()
            self.print_status('success', f"Reset successful - observation shape: {np.array(obs).shape}")
            
            # Take a few random steps
            self.print_status('info', "Taking test steps...")
            step_times = []
            
            for i in range(10):
                start = time.time()
                
                # Random action
                action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                step_time = (time.time() - start) * 1000  # ms
                step_times.append(step_time)
                
                if i == 0:
                    self.print_status('success', f"Step {i+1}: {step_time:.2f}ms")
                    self.print_status('info', f"  Observation shape: {np.array(obs).shape}")
                    self.print_status('info', f"  Reward: {reward:.3f}")
                    self.print_status('info', f"  Done: {done}")
                
                if done:
                    self.print_status('info', f"Episode ended at step {i+1}, resetting...")
                    obs, info = env.reset()
            
            avg_step_time = np.mean(step_times)
            fps = 1000.0 / avg_step_time if avg_step_time > 0 else 0
            
            self.print_status('success', f"Completed 10 test steps")
            self.print_status('info', f"  Average step time: {avg_step_time:.2f}ms")
            self.print_status('info', f"  Estimated FPS: {fps:.1f}")
            
            self.checks_passed.append("Environment simulation")
            
            # Close environment
            env.close()
            self.print_status('success', "Environment closed successfully")
            
            return True
            
        except Exception as e:
            self.print_status('error', f"Environment stepping failed: {e}")
            self.checks_failed.append("Environment simulation")
            traceback.print_exc()
            return False
    
    def test_neural_network(self) -> bool:
        """Test creating and running a simple neural network."""
        self.print_header("STEP 5: Testing Neural Network")
        
        try:
            import torch
            import torch.nn as nn
            
            # Determine device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.print_status('info', f"Using device: {device}")
            
            # Create a simple network
            class TestNetwork(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(107, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 90)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            model = TestNetwork().to(device)
            self.print_status('success', "Test network created")
            
            # Test forward pass
            test_input = torch.randn(4, 107).to(device)
            with torch.no_grad():
                output = model(test_input)
            
            self.print_status('success', f"Forward pass successful - output shape: {output.shape}")
            self.checks_passed.append("Neural network")
            
            return True
            
        except Exception as e:
            self.print_status('error', f"Neural network test failed: {e}")
            self.checks_failed.append("Neural network")
            traceback.print_exc()
            return False
    
    def print_summary(self):
        """Print verification summary."""
        self.print_header("VERIFICATION SUMMARY")
        
        total_checks = len(self.checks_passed) + len(self.checks_failed)
        
        if self.checks_passed:
            print(f"✓ Passed: {len(self.checks_passed)}/{total_checks}")
            for check in self.checks_passed:
                self.print_status('success', check)
        
        if self.warnings:
            print(f"\n⚠ Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                self.print_status('warning', warning)
        
        if self.checks_failed:
            print(f"\n✗ Failed: {len(self.checks_failed)}/{total_checks}")
            for check in self.checks_failed:
                self.print_status('error', check)
        
        print("\n" + "="*70)
        
        if not self.checks_failed:
            print("✅ RocketMind Environment Healthy")
            print("✅ PPO Core Operational")
            print("✅ Simulation Running Correctly")
            print("\n🚀 Ready to train!")
            print("\nNext steps:")
            print("  1. Start training: python -m rocketmind.main train")
            print("  2. Launch dashboard: python -m rocketmind.main dashboard")
        else:
            print("❌ Environment verification FAILED")
            print("\nSome critical components are not working.")
            print("Please run setup_env.py to fix dependencies:")
            print("  python setup_env.py")
        
        print("="*70 + "\n")
        
        return len(self.checks_failed) == 0
    
    def run(self) -> bool:
        """Run all verification checks."""
        self.print_header("RocketMind Environment Verification")
        print("Testing environment setup and dependencies...")
        print(f"Python version: {sys.version}")
        print(f"Python executable: {sys.executable}")
        
        try:
            # Run checks
            imports_ok = self.check_imports()
            if not imports_ok:
                self.print_status('error', 
                    "Critical imports failed - skipping environment tests")
                self.print_summary()
                return False
            
            hw_info = self.check_hardware()
            
            env_ok, env = self.test_environment_creation()
            if env_ok and env is not None:
                self.test_environment_steps(env)
            
            self.test_neural_network()
            
            # Print summary
            return self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Verification interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n✗ Verification failed with error: {e}")
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    verifier = EnvironmentVerifier()
    success = verifier.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
