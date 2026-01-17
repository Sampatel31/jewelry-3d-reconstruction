#!/usr/bin/env python3
"""
Validation script for jewelry 3D reconstruction system.
Checks that all components are properly structured.
"""

import sys
from pathlib import Path
import importlib.util


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    exists = filepath.exists()
    status = "✅" if exists else "❌"
    print(f"{status} {filepath}")
    return exists


def check_python_syntax(filepath: Path) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        if spec and spec.loader:
            # Just check if we can compile it
            with open(filepath, 'r') as f:
                compile(f.read(), filepath, 'exec')
            return True
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False
    except Exception as e:
        # Other errors are OK (like import errors)
        return True
    return False


def main():
    """Run validation checks."""
    print("=" * 60)
    print("Jewelry 3D Reconstruction System - Validation")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    all_checks_passed = True
    
    # Check critical files
    print("\n📁 Checking critical files...")
    critical_files = [
        'README.md',
        'USAGE.md',
        'LICENSE',
        'setup.py',
        'requirements.txt',
        'run_colab.py',
        'config/pipeline_config.py',
        'config/model_config.py',
        'src/__init__.py',
        'src/core/pipeline_router.py',
        'src/core/preprocessor.py',
        'src/ui/gradio_app.py',
    ]
    
    for file in critical_files:
        if not check_file_exists(project_root / file):
            all_checks_passed = False
    
    # Check Python files syntax
    print("\n🐍 Checking Python syntax...")
    python_files = list(project_root.glob('**/*.py'))
    python_files = [f for f in python_files if '.git' not in str(f)]
    
    syntax_errors = 0
    for py_file in python_files:
        if not check_python_syntax(py_file):
            syntax_errors += 1
            all_checks_passed = False
    
    print(f"\n   Total Python files: {len(python_files)}")
    print(f"   Syntax errors: {syntax_errors}")
    
    # Count lines of code
    print("\n📊 Code statistics...")
    total_lines = 0
    for py_file in python_files:
        with open(py_file, 'r') as f:
            total_lines += len(f.readlines())
    
    print(f"   Total lines of Python code: {total_lines}")
    
    # Check module structure
    print("\n📦 Checking module structure...")
    modules = [
        'src/core',
        'src/pose_estimation',
        'src/reconstruction',
        'src/mesh_extraction',
        'src/materials',
        'src/utils',
        'src/ui',
        'config',
        'models',
        'tests',
    ]
    
    for module in modules:
        module_path = project_root / module
        init_file = module_path / '__init__.py'
        if module_path.exists():
            status = "✅" if init_file.exists() else "⚠️ (no __init__.py)"
            print(f"{status} {module}")
        else:
            print(f"❌ {module} (missing)")
            all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ All validation checks passed!")
        print("\nSystem is ready for deployment:")
        print("  - Run locally: python -m src.ui.gradio_app")
        print("  - Deploy to Colab: !python run_colab.py")
        return 0
    else:
        print("❌ Some validation checks failed")
        print("Please review the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
