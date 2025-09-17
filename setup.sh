#!/bin/bash
# Setup script for PDX Analysis Tutorial
# Run with: bash setup.sh

set -e  # Exit on any error

echo "========================================"
echo "PDX Analysis Tutorial Setup"
echo "========================================"

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Starting PDX Analysis Tutorial setup..."

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p data results logs config docs notebooks workflows tests

# Check for Python
if command -v python3 >/dev/null 2>&1; then
    print_status "Python3 found: $(python3 --version)"
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    print_status "Python found: $(python --version)"
    PYTHON_CMD="python"
else
    print_warning "Python not found. Please install Python 3.7+ to use Python components."
    PYTHON_CMD=""
fi

# Check for R
if command -v R >/dev/null 2>&1; then
    print_status "R found: $(R --version | head -1)"
    R_AVAILABLE=true
elif command -v Rscript >/dev/null 2>&1; then
    print_status "Rscript found"
    R_AVAILABLE=true
else
    print_warning "R not found. Please install R to use R components."
    R_AVAILABLE=false
fi

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Check if conda is available
    if command -v conda >/dev/null 2>&1; then
        print_status "Conda found. Creating conda environment..."
        if conda env list | grep -q "pdx_analysis"; then
            print_warning "Conda environment 'pdx_analysis' already exists. Updating..."
            conda env update -f environment.yml
        else
            conda env create -f environment.yml
        fi
        print_status "Conda environment created. Activate with: conda activate pdx_analysis"
        
    # Check if virtual environment is preferred
    elif command -v python3 >/dev/null 2>&1; then
        print_status "Setting up Python virtual environment..."
        
        if [ ! -d "venv" ]; then
            python3 -m venv venv
            print_status "Virtual environment created"
        fi
        
        # Activate and install requirements
        source venv/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install requirements
        if [ -f "requirements.txt" ]; then
            print_status "Installing Python packages..."
            pip install -r requirements.txt
            print_status "Python packages installed"
        fi
        
        print_status "Virtual environment ready. Activate with: source venv/bin/activate"
        
    else
        print_warning "Cannot set up Python environment. Please install Python or Conda."
    fi
}

# Setup R environment
setup_r() {
    if [ "$R_AVAILABLE" = true ]; then
        print_status "Setting up R environment..."
        
        # Check if renv is available
        if Rscript -e "if (!require('renv', quietly=TRUE)) quit(status=1)" >/dev/null 2>&1; then
            print_status "renv found. Setting up R environment..."
            Rscript -e "
            if (!file.exists('renv.lock')) {
                message('No renv.lock found, initializing renv...')
                renv::init()
            } else {
                message('Restoring R environment from renv.lock...')
                renv::restore()
            }
            "
        else
            print_status "Installing required R packages manually..."
            Rscript -e "
            packages <- c('ggplot2', 'dplyr', 'lme4', 'broom', 'configr', 'logger', 
                         'scales', 'RColorBrewer', 'gridExtra', 'knitr', 'rmarkdown', 'testthat')
            for (pkg in packages) {
                if (!require(pkg, character.only=TRUE, quietly=TRUE)) {
                    message(paste('Installing', pkg, '...'))
                    install.packages(pkg, repos='https://cran.r-project.org')
                } else {
                    message(paste(pkg, 'already installed'))
                }
            }
            message('R package installation complete')
            "
        fi
        print_status "R environment setup complete"
    else
        print_warning "R not available. Skipping R environment setup."
    fi
}

# Generate sample data
generate_sample_data() {
    print_status "Generating sample data..."
    
    if [ -n "$PYTHON_CMD" ] && [ -f "src/python/generate_effective_pdx_data.py" ]; then
        # Try to generate data (may fail if packages not installed)
        if $PYTHON_CMD src/python/generate_effective_pdx_data.py >/dev/null 2>&1; then
            print_status "Sample data generated successfully"
        else
            print_warning "Could not generate effective data (packages may not be installed yet)"
            print_status "Creating basic mock data files..."
            
            # Create basic CSV files if they don't exist
            if [ ! -f "data/tumor_volumes_mock.csv" ]; then
                echo "Model,Arm,Day,Volume_mm3" > data/tumor_volumes_mock.csv
                echo "PDX1,control,0,100" >> data/tumor_volumes_mock.csv
                echo "PDX1,control,7,150" >> data/tumor_volumes_mock.csv
                echo "PDX2,treatment,0,110" >> data/tumor_volumes_mock.csv
                echo "PDX2,treatment,7,120" >> data/tumor_volumes_mock.csv
            fi
            
            if [ ! -f "data/expression_tpm_mock.csv" ]; then
                echo "Gene,PDX1,PDX2" > data/expression_tpm_mock.csv
                echo "GENE1,5.2,3.1" >> data/expression_tpm_mock.csv
                echo "GENE2,2.8,6.7" >> data/expression_tpm_mock.csv
            fi
            
            print_status "Basic mock data created"
        fi
    else
        print_warning "Cannot generate sample data. Python or data generation script not available."
    fi
}

# Set up git hooks (if git repo)
setup_git_hooks() {
    if [ -d ".git" ]; then
        print_status "Setting up git hooks..."
        
        # Create pre-commit hook for testing
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook to run basic tests

echo "Running pre-commit tests..."

# Check Python syntax
if ls src/python/*.py >/dev/null 2>&1; then
    for file in src/python/*.py; do
        if ! python3 -m py_compile "$file" 2>/dev/null; then
            echo "Python syntax error in $file"
            exit 1
        fi
    done
fi

# Check R syntax (basic)
if ls src/R/*.R >/dev/null 2>&1; then
    for file in src/R/*.R; do
        if ! Rscript -e "parse('$file')" >/dev/null 2>&1; then
            echo "R syntax error in $file"
            exit 1
        fi
    done
fi

echo "Pre-commit tests passed"
EOF
        
        chmod +x .git/hooks/pre-commit
        print_status "Git hooks installed"
    fi
}

# Main setup workflow
main() {
    print_status "Environment detection complete"
    
    # Ask user what to set up
    echo ""
    echo "What would you like to set up?"
    echo "1) Python environment only"
    echo "2) R environment only"
    echo "3) Both Python and R environments"
    echo "4) Full setup (environments + data + git hooks)"
    echo "5) Skip environment setup"
    
    read -p "Choose option (1-5): " choice
    
    case $choice in
        1)
            if [ -n "$PYTHON_CMD" ]; then
                setup_python
            else
                print_error "Python not available"
                exit 1
            fi
            ;;
        2)
            setup_r
            ;;
        3)
            if [ -n "$PYTHON_CMD" ]; then
                setup_python
            fi
            setup_r
            ;;
        4)
            if [ -n "$PYTHON_CMD" ]; then
                setup_python
            fi
            setup_r
            generate_sample_data
            setup_git_hooks
            ;;
        5)
            print_status "Skipping environment setup"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
    
    print_status "Setup complete!"
    
    # Print next steps
    echo ""
    echo "========================================"
    echo "Next Steps:"
    echo "========================================"
    
    if [ -n "$PYTHON_CMD" ] && [ "$choice" != "2" ] && [ "$choice" != "5" ]; then
        if command -v conda >/dev/null 2>&1; then
            echo "1. Activate conda environment: conda activate pdx_analysis"
        else
            echo "1. Activate virtual environment: source venv/bin/activate"
        fi
    fi
    
    echo "2. Read the README.md for detailed instructions"
    echo "3. Run integration tests: bash tests/integration_test.sh"
    echo "4. Start with the tutorial: notebooks/02_biomarker_analysis.ipynb"
    
    if [ "$choice" = "4" ]; then
        echo "5. Generate effective data: python src/python/generate_effective_pdx_data.py"
        echo "6. Run complete analysis: python src/python/advanced_workflows.py"
    fi
    
    echo ""
    echo "For help: see docs/FAQ.md or README.md"
}

# Run main setup
main