#!/usr/bin/env python3
"""
Comprehensive workflow verification script.
Checks all dependencies, file paths, and ensures the pipeline will work.
"""

import sys
from pathlib import Path
import subprocess

def check_python_package(package_name):
    """Check if a Python package is installed"""
    try:
        __import__(package_name)
        return True, "✅ Installed"
    except ImportError:
        return False, "❌ Missing"

def check_file_exists(file_path):
    """Check if a file exists"""
    path = Path(file_path)
    if path.exists():
        return True, f"✅ Found ({path})"
    else:
        return False, f"❌ Missing ({path})"

def check_directory(dir_path):
    """Check if directory exists and count files"""
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        file_count = len(list(path.rglob('*')))
        return True, f"✅ Exists ({file_count} items)"
    else:
        return False, f"❌ Missing ({path})"

def run_dry_run(script_path, args):
    """Try importing a script to check for syntax/import errors"""
    try:
        # Just try to import/parse the file
        with open(script_path, 'r') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        return True, "✅ No syntax errors"
    except Exception as e:
        return False, f"❌ Error: {str(e)[:100]}"

def main():
    workspace = Path("/Users/kirubeso.r/Documents/ArtPark")

    print("=" * 80)
    print("🔍 ASHA WORKER ANALYSIS PIPELINE - VERIFICATION")
    print("=" * 80)
    print()

    # Check Python version
    print("📍 PYTHON VERSION")
    print(f"   {sys.version}")
    print()

    # Check required packages
    print("📦 REQUIRED PACKAGES")
    packages = [
        "fastdup",
        "torch",
        "torchvision",
        "PIL",
        "pandas",
        "numpy",
        "tqdm",
        "sklearn",
        "geopandas",
        "plotly"
    ]

    all_packages_ok = True
    for pkg in packages:
        status, msg = check_python_package(pkg)
        print(f"   {pkg:20s} {msg}")
        if not status:
            all_packages_ok = False
    print()

    # Check critical files
    print("📄 CRITICAL FILES")
    files_to_check = [
        workspace / "visualization_utils.py",
        workspace / "src/prismh/core/preprocess.py",
        workspace / "scripts/download_continuous_200k.py",
        workspace / "scripts/generate_worker_report.py",
        workspace / "scripts/check_download_progress.py"
    ]

    all_files_ok = True
    for file_path in files_to_check:
        status, msg = check_file_exists(file_path)
        print(f"   {file_path.name:40s} {msg}")
        if not status:
            all_files_ok = False
    print()

    # Check data directories
    print("📁 DATA DIRECTORIES")
    dirs_to_check = [
        (workspace / "jsons", "JSON source files"),
        (workspace / "report_images", "Downloaded images (may be empty if not started)"),
    ]

    for dir_path, description in dirs_to_check:
        status, msg = check_directory(dir_path)
        print(f"   {description:40s} {msg}")
    print()

    # Check script syntax
    print("🐍 SCRIPT SYNTAX CHECK")
    scripts_to_check = [
        workspace / "scripts/download_continuous_200k.py",
        workspace / "scripts/generate_worker_report.py",
        workspace / "src/prismh/core/preprocess.py"
    ]

    all_scripts_ok = True
    for script_path in scripts_to_check:
        status, msg = run_dry_run(script_path, [])
        print(f"   {script_path.name:40s} {msg}")
        if not status:
            all_scripts_ok = False
    print()

    # Test imports specifically
    print("🔗 IMPORT VERIFICATION")
    import_tests = [
        ("VisualizationUtils", "src.prismh.utils.visualization_utils", "VisualizationUtils"),
        ("FastDup preprocessing", "src.prismh.core.preprocess", "ImagePreprocessor"),
    ]

    all_imports_ok = True
    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(workspace / "src"))

    for name, module_path, class_name in import_tests:
        try:
            module = __import__(module_path.replace('/', '.'), fromlist=[class_name])
            getattr(module, class_name)
            print(f"   {name:40s} ✅ Import successful")
        except Exception as e:
            print(f"   {name:40s} ❌ Import failed: {str(e)[:50]}")
            all_imports_ok = False
    print()

    # Overall status
    print("=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)

    if all_packages_ok and all_files_ok and all_scripts_ok and all_imports_ok:
        print("✅ ALL CHECKS PASSED")
        print()
        print("The pipeline is ready to run! Once images are downloaded:")
        print()
        print("1. Preprocess images:")
        print("   python src/prismh/core/preprocess.py \\")
        print("     --input-dir report_images \\")
        print("     --output-dir report_preprocess_results")
        print()
        print("2. Generate worker report:")
        print("   python scripts/generate_worker_report.py \\")
        print("     --metadata report_images/downloaded_metadata.json \\")
        print("     --preprocess-results report_preprocess_results \\")
        print("     --output worker_analysis_report")
        print()
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        if not all_packages_ok:
            print("⚠️  Install missing packages:")
            print("   conda activate prism-h")
            print("   pip install <missing_package>")
        if not all_files_ok:
            print("⚠️  Some critical files are missing - check file paths")
        if not all_scripts_ok:
            print("⚠️  Some scripts have syntax errors - review error messages")
        if not all_imports_ok:
            print("⚠️  Import errors detected - check module paths")

    print("=" * 80)

if __name__ == "__main__":
    main()
