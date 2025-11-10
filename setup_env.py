#!/usr/bin/env python3
"""
RocketMind Environment Setup and Repair Script
Automatically detects and fixes missing or broken dependencies.
"""

import subprocess
import sys
import os
from typing import List, Tuple
import importlib.util


class EnvironmentSetup:
    """Automated environment setup and dependency repair."""
    
    def __init__(self):
        self.issues = []
        self.fixes = []
        
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
    
    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run command and capture output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
    
    def uninstall_package(self, package: str):
        """Uninstall a package."""
        self.print_status('info', f"Uninstalling {package}...")
        success, output = self.run_command(
            [sys.executable, "-m", "pip", "uninstall", "-y", package],
            check=False
        )
        if success:
            self.print_status('success', f"Uninstalled {package}")
        return success
    
    def install_package(self, package: str):
        """Install a package."""
        self.print_status('info', f"Installing {package}...")
        success, output = self.run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            check=False
        )
        if success:
            self.print_status('success', f"Installed {package}")
            self.fixes.append(f"Installed {package}")
        else:
            self.print_status('error', f"Failed to install {package}")
            self.issues.append(f"Failed to install {package}")
        return success
    
    def check_package(self, package_name: str, import_name: str = None) -> bool:
        """Check if a package is installed and importable."""
        if import_name is None:
            import_name = package_name
        
        spec = importlib.util.find_spec(import_name)
        return spec is not None
    
    def uninstall_old_packages(self):
        """Remove old or conflicting packages."""
        self.print_header("STEP 1: Removing Old/Broken Packages")
        
        old_packages = [
            'rlgym',
            'rlgym-sim'
        ]
        
        for package in old_packages:
            # Check if installed
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True
            )
            if result.returncode == 0:
                self.print_status('warning', f"Found old package: {package}")
                self.uninstall_package(package)
            else:
                self.print_status('info', f"{package} not installed (OK)")
    
    def install_core_dependencies(self):
        """Install core dependencies."""
        self.print_header("STEP 2: Installing Core Dependencies")
        
        dependencies = [
            "torch>=2.2",
            "gymnasium>=0.29",
            "numpy>=1.24.0,<2.0.0",  # Specify numpy version for compatibility
        ]
        
        for dep in dependencies:
            self.install_package(dep)
    
    def install_rlgym_packages(self):
        """Install rlgym_rocket_league and related packages."""
        self.print_header("STEP 3: Installing rlgym_rocket_league")
        
        # Install the unified package
        packages = [
            "rlgym_rocket_league>=2.0.1",
            "rlgym-tools>=2.3.13",
        ]
        
        for package in packages:
            self.install_package(package)
        
        # Create compatibility shims
        self.create_compatibility_shims()
    
    def install_rlbot_packages(self):
        """Install RLBot and RocketSim."""
        self.print_header("STEP 4: Installing RLBot and RocketSim")
        
        packages = [
            "rlbot>=1.67",
            "rocketsim>=2.1",
        ]
        
        for package in packages:
            # These might fail on some systems, so we don't treat as critical
            success = self.install_package(package)
            if not success:
                self.print_status('warning', 
                    f"{package} installation failed - RLBot integration may not work")
    
    def install_dashboard_packages(self):
        """Install Streamlit and visualization packages."""
        self.print_header("STEP 5: Installing Dashboard Packages")
        
        packages = [
            "streamlit>=1.30",
            "plotly>=5.17.0",
            "pygame>=2.5.0",
        ]
        
        for package in packages:
            self.install_package(package)
    
    def install_optional_packages(self):
        """Install optional packages for enhanced functionality."""
        self.print_header("STEP 6: Installing Optional Packages")
        
        optional = [
            "wandb>=0.16.0",
            "tensorboard>=2.14.0",
            "pyyaml>=6.0.1",
            "tqdm>=4.65.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "pandas>=2.0.0",
        ]
        
        for package in optional:
            self.install_package(package)
    
    def check_cuda(self):
        """Check CUDA/GPU availability."""
        self.print_header("STEP 7: Checking CUDA/GPU Availability")
        
        try:
            import torch
            if torch.cuda.is_available():
                self.print_status('success', f"CUDA is available!")
                self.print_status('info', f"  CUDA Version: {torch.version.cuda}")
                self.print_status('info', f"  GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    self.print_status('info', f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                self.print_status('warning', "CUDA not available - will use CPU")
                self.print_status('info', "Training will be slower on CPU")
        except ImportError:
            self.print_status('error', "PyTorch not installed - cannot check CUDA")
    
    def verify_imports(self):
        """Verify critical imports."""
        self.print_header("STEP 8: Verifying Installations")
        
        critical_imports = [
            ('torch', 'PyTorch'),
            ('gymnasium', 'Gymnasium'),
            ('numpy', 'NumPy'),
            ('rlgym_rocket_league', 'RLGym-Rocket-League'),
        ]
        
        optional_imports = [
            ('rlgym', 'RLGym (legacy compatibility)'),
            ('rlgym_sim', 'RLGym-Sim (legacy compatibility)'),
            ('rlgym_tools', 'RLGym-Tools'),
            ('streamlit', 'Streamlit'),
            ('plotly', 'Plotly'),
            ('tensorboard', 'TensorBoard'),
        ]
        
        all_ok = True
        
        # Check critical imports
        for module, name in critical_imports:
            if self.check_package(module, module):
                self.print_status('success', f"{name} is installed")
            else:
                self.print_status('error', f"{name} is NOT installed")
                all_ok = False
                self.issues.append(f"{name} not installed")
        
        # Check optional imports
        for module, name in optional_imports:
            if self.check_package(module, module):
                self.print_status('success', f"{name} is installed")
            else:
                self.print_status('warning', f"{name} is NOT installed (optional)")
        
        return all_ok
    
    def create_compatibility_shims(self):
        """Create compatibility shims for legacy rlgym imports."""
        self.print_status('info', "Creating compatibility shims...")
        
        from pathlib import Path
        
        # Create rlgym shim
        rlgym_shim = Path("rlgym/__init__.py")
        rlgym_shim.parent.mkdir(exist_ok=True)
        
        rlgym_shim_content = '''"""
Compatibility shim for legacy 'rlgym' imports.
Redirects to 'rlgym_rocket_league' automatically.
"""
import sys, importlib, warnings
warnings.warn("Using legacy rlgym compatibility shim. Please update to rlgym_rocket_league.", DeprecationWarning, stacklevel=2)
try:
    _rlgym_rl = importlib.import_module("rlgym_rocket_league")
    sys.modules["rlgym"] = _rlgym_rl
    globals().update(vars(_rlgym_rl))
except ImportError as e:
    raise ImportError("rlgym_rocket_league not installed. Install with: pip install rlgym_rocket_league>=2.0.1") from e
'''
        rlgym_shim.write_text(rlgym_shim_content)
        self.print_status('success', "Created rlgym compatibility shim")
        
        # Create rlgym_sim shim
        rlgym_sim_shim = Path("rlgym_sim/__init__.py")
        rlgym_sim_shim.parent.mkdir(exist_ok=True)
        
        rlgym_sim_shim_content = '''"""
Compatibility shim for legacy 'rlgym_sim' imports.
Redirects to 'rlgym_rocket_league' automatically.
"""
import sys, importlib, types, warnings
warnings.warn("Using legacy rlgym_sim compatibility shim. Please update to rlgym_rocket_league.", DeprecationWarning, stacklevel=2)
try:
    _rlgym_rl = importlib.import_module("rlgym_rocket_league")
    sys.modules["rlgym_sim"] = _rlgym_rl
    globals().update(vars(_rlgym_rl))
    envs = types.SimpleNamespace()
    if hasattr(_rlgym_rl, 'make'):
        envs.RLGymSimEnv = _rlgym_rl.make
    sys.modules['rlgym_sim.envs'] = envs
except ImportError as e:
    raise ImportError("rlgym_rocket_league not installed. Install with: pip install rlgym_rocket_league>=2.0.1") from e
'''
        rlgym_sim_shim.write_text(rlgym_sim_shim_content)
        self.print_status('success', "Created rlgym_sim compatibility shim")
    
    def print_summary(self):
        """Print setup summary."""
        self.print_header("SETUP SUMMARY")
        
        print(f"Fixes Applied: {len(self.fixes)}")
        for fix in self.fixes:
            self.print_status('success', fix)
        
        if self.issues:
            print(f"\nIssues Found: {len(self.issues)}")
            for issue in self.issues:
                self.print_status('warning', issue)
        else:
            print("\n✓ No issues found!")
        
        print("\n" + "="*70)
        if not self.issues:
            print("✅ RocketMind Environment Setup Complete!")
            print("✅ All dependencies installed successfully")
            print("\nNext steps:")
            print("  1. Run 'python verify_rl_env.py' to test the environment")
            print("  2. Start training with 'python -m rocketmind.main train'")
            print("  3. Launch dashboard with 'python -m rocketmind.main dashboard'")
        else:
            print("⚠ Setup completed with some warnings")
            print("Some optional packages may not be installed.")
            print("Core functionality should still work.")
        print("="*70 + "\n")
    
    def run(self):
        """Run the full setup process."""
        self.print_header("RocketMind Environment Setup")
        print("This script will automatically install and repair dependencies.")
        print("Python version:", sys.version)
        print("Python executable:", sys.executable)
        
        try:
            # Run setup steps
            self.uninstall_old_packages()
            self.install_core_dependencies()
            self.install_rlgym_packages()
            self.install_rlbot_packages()
            self.install_dashboard_packages()
            self.install_optional_packages()
            self.check_cuda()
            self.verify_imports()
            
            # Print summary
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n✗ Setup failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    setup = EnvironmentSetup()
    setup.run()


if __name__ == "__main__":
    main()
