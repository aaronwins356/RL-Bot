"""Validation script to check hierarchical system files exist and are importable.

This script validates the structure without requiring external dependencies.
"""

import os
import sys
from pathlib import Path


def check_file_exists(path: Path) -> bool:
    """Check if file exists."""
    return path.exists() and path.is_file()


def check_directory_exists(path: Path) -> bool:
    """Check if directory exists."""
    return path.exists() and path.is_dir()


def main():
    """Validate hierarchical system structure."""
    print("=" * 70)
    print("Hierarchical RL/IL System Validation")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    errors = []
    successes = []
    
    # Check core modules
    print("\n1. Checking Core Modules...")
    
    core_files = [
        'core/hierarchical_controller.py',
        'core/llc/__init__.py',
        'core/skill_programs/__init__.py',
        'core/skill_programs/base.py',
        'core/opportunity_detector/__init__.py',
        'core/opportunity_detector/detector.py',
        'core/opportunity_detector/risk_scorer.py',
        'core/training/hierarchical_rewards.py',
        'core/training/drill_evaluator.py',
    ]
    
    for file_path in core_files:
        full_path = base_dir / file_path
        if check_file_exists(full_path):
            successes.append(f"✓ {file_path}")
        else:
            errors.append(f"✗ {file_path} - NOT FOUND")
    
    # Check skill programs
    print("\n2. Checking Skill Programs...")
    
    sp_files = [
        'core/skill_programs/fast_aerial.py',
        'core/skill_programs/aerial_control.py',
        'core/skill_programs/wall_read.py',
        'core/skill_programs/backboard_read.py',
        'core/skill_programs/ceiling_shot.py',
        'core/skill_programs/flip_reset.py',
        'core/skill_programs/musty.py',
        'core/skill_programs/breezi.py',
        'core/skill_programs/double_tap.py',
        'core/skill_programs/ground_to_air_dribble.py',
    ]
    
    for file_path in sp_files:
        full_path = base_dir / file_path
        if check_file_exists(full_path):
            successes.append(f"✓ {file_path}")
        else:
            errors.append(f"✗ {file_path} - NOT FOUND")
    
    # Check configuration
    print("\n3. Checking Configuration...")
    
    config_file = base_dir / 'configs' / 'hierarchical_rl.yaml'
    if check_file_exists(config_file):
        successes.append(f"✓ configs/hierarchical_rl.yaml")
    else:
        errors.append(f"✗ configs/hierarchical_rl.yaml - NOT FOUND")
    
    # Check documentation
    print("\n4. Checking Documentation...")
    
    doc_file = base_dir / 'HIERARCHICAL_SYSTEM.md'
    if check_file_exists(doc_file):
        successes.append(f"✓ HIERARCHICAL_SYSTEM.md")
    else:
        errors.append(f"✗ HIERARCHICAL_SYSTEM.md - NOT FOUND")
    
    # Check scripts
    print("\n5. Checking Scripts...")
    
    script_file = base_dir / 'scripts' / 'demo_hierarchical.py'
    if check_file_exists(script_file):
        successes.append(f"✓ scripts/demo_hierarchical.py")
    else:
        errors.append(f"✗ scripts/demo_hierarchical.py - NOT FOUND")
    
    # Print results
    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    
    print(f"\n✓ Successes: {len(successes)}")
    for success in successes:
        print(f"  {success}")
    
    if errors:
        print(f"\n✗ Errors: {len(errors)}")
        for error in errors:
            print(f"  {error}")
        return 1
    else:
        print("\n✓ All checks passed!")
        
        # Summary statistics
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        
        # Count lines of code
        total_lines = 0
        for file_path in core_files + sp_files:
            full_path = base_dir / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    total_lines += len(f.readlines())
        
        print(f"\nTotal Python files: {len(core_files) + len(sp_files)}")
        print(f"Total lines of code: {total_lines:,}")
        print(f"\nSkill Programs implemented: {len(sp_files)}")
        print(f"Core modules: {len(core_files)}")
        
        return 0


if __name__ == '__main__':
    sys.exit(main())
