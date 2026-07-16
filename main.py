"""
Automated Bug Localization - Main Entry Point
Phase 1: Project Initialization Verification

This script verifies that the project setup is complete and working correctly.
"""

import sys
import os


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[FAIL] Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"[OK] Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def check_virtual_environment():
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    if in_venv:
        print(f"[OK] Virtual environment active: {sys.prefix}")
        return True
    else:
        print("[WARNING] Not running in a virtual environment!")
        print("  It's recommended to activate the virtual environment:")
        print("  Windows: venv\\Scripts\\activate")
        print("  macOS/Linux: source venv/bin/activate")
        return False


def check_project_structure():
    """Verify that the project folder structure exists."""
    required_dirs = ['src', 'data', 'data/raw', 'data/processed', 'tests', 'notebooks']
    required_files = ['requirements.txt', 'README.md']
    
    all_ok = True
    
    print("\nChecking project structure:")
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"  [OK] {directory}/")
        else:
            print(f"  [FAIL] {directory}/ - MISSING")
            all_ok = False
    
    for file in required_files:
        if os.path.isfile(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [FAIL] {file} - MISSING")
            all_ok = False
    
    return all_ok


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'transformers',
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'streamlit',
        'tqdm'
    ]
    
    print("\nChecking dependencies:")
    all_ok = True
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif package == 'torch':
                import torch
                version = torch.__version__
            elif package == 'transformers':
                import transformers
                version = transformers.__version__
            elif package == 'pandas':
                import pandas
                version = pandas.__version__
            elif package == 'numpy':
                import numpy
                version = numpy.__version__
            elif package == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
            elif package == 'streamlit':
                import streamlit
                version = streamlit.__version__
            elif package == 'tqdm':
                import tqdm
                version = tqdm.__version__
            
            print(f"  [OK] {package:15s} - version {version}")
        except ImportError:
            print(f"  [FAIL] {package:15s} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def main():
    """Main verification function."""
    print("=" * 60)
    print("Automated Bug Localization - Phase 1 Verification")
    print("=" * 60)
    
    results = []
    
    # Check Python version
    results.append(("Python Version", check_python_version()))
    
    # Check virtual environment
    results.append(("Virtual Environment", check_virtual_environment()))
    
    # Check project structure
    results.append(("Project Structure", check_project_structure()))
    
    # Check dependencies
    results.append(("Dependencies", check_dependencies()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{check_name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] Phase 1 Complete! Project is ready for Phase 2.")
        print("\nNext steps:")
        print("  1. Review the project structure")
        print("  2. Read README.md to understand the project")
        print("  3. Proceed to Phase 2: Dataset Preparation")
        return 0
    else:
        print("\n[ERROR] Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Create missing directories")
        return 1


if __name__ == "__main__":
    exit(main())