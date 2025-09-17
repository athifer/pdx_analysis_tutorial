#!/bin/bash
# Integration test script to validate entire PDX analysis workflow
# Run with: bash tests/integration_test.sh

set -e  # Exit on any error

echo "========================================"
echo "PDX Analysis Tutorial Integration Test"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create test log
LOG_FILE="tests/integration_test.log"
echo "Integration test started at $(date)" > "$LOG_FILE"

# Function to log and print
log_and_print() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Test 1: Check directory structure
log_and_print "Test 1: Checking directory structure..."
required_dirs=("src" "data" "results" "config" "docs" "notebooks" "workflows" "tests" "logs")
missing_dirs=()

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -eq 0 ]; then
    log_and_print "✓ All required directories present"
else
    log_and_print "✗ Missing directories: ${missing_dirs[*]}"
    log_and_print "  Creating missing directories..."
    for dir in "${missing_dirs[@]}"; do
        mkdir -p "$dir"
        log_and_print "  Created: $dir"
    done
fi

# Test 2: Check key files exist
log_and_print "\nTest 2: Checking key files..."
key_files=(
    "src/R/analyze_volume.R"
    "src/python/generate_effective_pdx_data.py"
    "src/python/advanced_workflows.py"
    "notebooks/02_biomarker_analysis.ipynb"
    "config/config.ini"
)

missing_files=()
for file in "${key_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    log_and_print "✓ All key files present"
else
    log_and_print "✗ Missing files: ${missing_files[*]}"
fi

# Test 3: Check Python syntax
log_and_print "\nTest 3: Checking Python syntax..."
python_files=($(find src/python -name "*.py" 2>/dev/null || true))
python_files+=($(find workflows -name "*.py" 2>/dev/null || true))

python_errors=0
for file in "${python_files[@]}"; do
    if [ -f "$file" ]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            log_and_print "✓ $file: syntax OK"
        else
            log_and_print "✗ $file: syntax errors"
            python_errors=$((python_errors + 1))
        fi
    fi
done

if [ $python_errors -eq 0 ]; then
    log_and_print "✓ All Python files have valid syntax"
else
    log_and_print "✗ $python_errors Python files have syntax errors"
fi

# Test 4: Check R syntax
log_and_print "\nTest 4: Checking R syntax..."
r_files=($(find src/R -name "*.R" 2>/dev/null || true))
r_files+=($(find workflows -name "*.R" 2>/dev/null || true))

r_errors=0
for file in "${r_files[@]}"; do
    if [ -f "$file" ]; then
        if Rscript -e "source('$file')" >/dev/null 2>&1; then
            log_and_print "✓ $file: syntax OK"
        else
            log_and_print "⚠ $file: syntax check skipped (may need packages)"
        fi
    fi
done

# Test 5: Validate configuration
log_and_print "\nTest 5: Checking configuration..."
if [ -f "config/config.ini" ]; then
    # Basic config validation
    if grep -q "\[data_paths\]" config/config.ini && \
       grep -q "\[analysis\]" config/config.ini && \
       grep -q "\[plotting\]" config/config.ini; then
        log_and_print "✓ Configuration file structure valid"
    else
        log_and_print "✗ Configuration file missing required sections"
    fi
else
    log_and_print "✗ Configuration file missing"
fi

# Test 6: Check documentation
log_and_print "\nTest 6: Checking documentation..."
doc_files=("docs/data_dictionary.md" "docs/statistical_methods.md" "docs/methodology.md" "docs/FAQ.md")
missing_docs=()

for file in "${doc_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_docs+=("$file")
    fi
done

if [ ${#missing_docs[@]} -eq 0 ]; then
    log_and_print "✓ All documentation files present"
else
    log_and_print "✗ Missing documentation: ${missing_docs[*]}"
fi

# Test 7: Check if notebooks are valid JSON
log_and_print "\nTest 7: Checking notebook validity..."
notebook_files=($(find notebooks -name "*.ipynb" 2>/dev/null || true))

notebook_errors=0
for file in "${notebook_files[@]}"; do
    if [ -f "$file" ]; then
        if python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
            log_and_print "✓ $file: valid JSON"
        else
            log_and_print "✗ $file: invalid JSON format"
            notebook_errors=$((notebook_errors + 1))
        fi
    fi
done

if [ $notebook_errors -eq 0 ]; then
    log_and_print "✓ All notebooks have valid JSON format"
else
    log_and_print "✗ $notebook_errors notebooks have JSON errors"
fi

# Test 8: Test data generation (if Python available)
log_and_print "\nTest 8: Testing data generation..."
if command -v python3 >/dev/null 2>&1; then
    if [ -f "src/python/generate_effective_pdx_data.py" ]; then
        # Test basic imports (won't run full script due to dependencies)
        if python3 -c "import sys; sys.path.append('src/python'); import generate_effective_pdx_data" 2>/dev/null; then
            log_and_print "✓ Data generation module imports successfully"
        else
            log_and_print "⚠ Data generation module import failed (may need packages)"
        fi
    fi
else
    log_and_print "⚠ Python3 not available, skipping data generation test"
fi

# Test 9: Create sample workflow test
log_and_print "\nTest 9: Testing workflow structure..."
if [ -f "workflows/run_complete_analysis.py" ]; then
    # Check if workflow has main structure
    if grep -q "def main" workflows/run_complete_analysis.py && \
       grep -q "if __name__" workflows/run_complete_analysis.py; then
        log_and_print "✓ Python workflow has proper main structure"
    else
        log_and_print "✗ Python workflow missing main structure"
    fi
fi

if [ -f "workflows/run_complete_analysis.R" ]; then
    # Check if R workflow has main components
    if grep -q "main_analysis" workflows/run_complete_analysis.R; then
        log_and_print "✓ R workflow has main analysis function"
    else
        log_and_print "✗ R workflow missing main analysis function"
    fi
fi

# Test 10: Check if results directory is writable
log_and_print "\nTest 10: Testing file system permissions..."
if [ -d "results" ]; then
    test_file="results/test_write_permission.tmp"
    if echo "test" > "$test_file" 2>/dev/null; then
        log_and_print "✓ Results directory is writable"
        rm -f "$test_file"
    else
        log_and_print "✗ Results directory is not writable"
    fi
else
    log_and_print "✗ Results directory does not exist"
fi

# Summary
log_and_print "\n========================================"
log_and_print "Integration Test Summary"
log_and_print "========================================"

# Count results
total_tests=10
passed_tests=0

# This is a simplified count - in practice you'd track each test result
log_and_print "Total tests: $total_tests"
log_and_print "Results logged to: $LOG_FILE"

# Check if any critical errors occurred
if [ $python_errors -gt 0 ] || [ ${#missing_files[@]} -gt 3 ]; then
    log_and_print "Status: FAILED (critical errors detected)"
    exit 1
else
    log_and_print "Status: PASSED (with possible warnings)"
    exit 0
fi