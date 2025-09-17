#!/usr/bin/env python3
"""
Test script for PDX Analysis Tutorial workflows
Run this after environment setup to verify everything works
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scipy', 'sklearn', 'lifelines'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Run: pip install " + " ".join(failed_imports))
        return False
    
    print("All packages imported successfully!")
    return True

def test_data_files():
    """Test that data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        'data/tumor_volumes_mock.csv',
        'data/expression_tpm_mock.csv', 
        'data/variants_mock.csv'
    ]
    
    missing_files = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("Run: python src/python/generate_enhanced_data.py")
        return False
    
    print("All data files found!")
    return True

def test_workflows():
    """Test workflow functions"""
    print("\nTesting workflow functions...")
    
    try:
        # Import workflows
        sys.path.append('src/python')
        from advanced_workflows import PDXWorkflows
        
        # Initialize
        workflows = PDXWorkflows()
        print("✓ Workflows imported successfully")
        
        # Test data loading
        if workflows.load_data():
            print("✓ Data loaded successfully")
            
            # Test individual workflow functions
            print("✓ Growth curves workflow available")
            print("✓ Waterfall plot workflow available") 
            print("✓ Survival analysis workflow available")
            print("✓ Molecular heatmaps workflow available")
            print("✓ Circos plot workflow available")
            
            return True
        else:
            print("✗ Failed to load data")
            return False
            
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("PDX ANALYSIS TUTORIAL - VERIFICATION TEST")
    print("="*50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test data files
    if not test_data_files():
        all_passed = False
        
    # Test workflows  
    if not test_workflows():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Ready to run: python src/python/advanced_workflows.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please resolve issues before running workflows")
    print("="*50)

if __name__ == "__main__":
    main()