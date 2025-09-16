#!/bin/bash
# Enhanced setup script for PDX Analysis Tutorial
# Handles dependency conflicts and environment issues

set -e

echo "========================================"
echo "PDX Analysis Tutorial - Enhanced Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_step "Detecting environment and dependency conflicts..."

# Check for existing problematic packages
problematic_packages=(
    "open-webui"
    "rtfde"
)

conflicts_detected=false
for pkg in "${problematic_packages[@]}"; do
    if pip show "$pkg" >/dev/null 2>&1; then
        print_warning "Detected $pkg which may cause dependency conflicts"
        conflicts_detected=true
    fi
done

if [ "$conflicts_detected" = true ]; then
    echo ""
    print_warning "Dependency conflicts detected!"
    echo "This tutorial requires specific package versions that conflict with your system packages."
    echo ""
    echo "Recommended solutions:"
    echo "1. Use a virtual environment (recommended)"
    echo "2. Use conda environment"
    echo "3. Use Docker (isolated environment)"
    echo ""
    
    read -p "Choose setup method: [1] Virtual Env, [2] Conda, [3] Docker, [4] Skip: " choice
    
    case $choice in
        1)
            setup_method="venv"
            ;;
        2)
            setup_method="conda"
            ;;
        3)
            setup_method="docker"
            ;;
        4)
            setup_method="skip"
            ;;
        *)
            print_error "Invalid choice. Using virtual environment as default."
            setup_method="venv"
            ;;
    esac
else
    echo ""
    print_status "No major conflicts detected. Proceeding with installation."
    read -p "Choose setup method: [1] Current Python, [2] Virtual Env, [3] Conda: " choice
    
    case $choice in
        1)
            setup_method="current"
            ;;
        2)
            setup_method="venv"
            ;;
        3)
            setup_method="conda"
            ;;
        *)
            setup_method="current"
            ;;
    esac
fi

# Setup based on chosen method
case $setup_method in
    "current")
        print_step "Installing packages in current Python environment..."
        
        # Create minimal requirements to avoid conflicts
        cat > requirements_minimal.txt << EOF
# Minimal requirements for PDX analysis
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0
EOF
        
        pip install --upgrade pip
        pip install -r requirements_minimal.txt
        
        print_status "Packages installed in current environment"
        print_warning "Note: This may still have conflicts with existing packages"
        ;;
        
    "venv")
        print_step "Creating virtual environment..."
        
        # Remove existing venv if it exists
        if [ -d "pdx_env" ]; then
            print_warning "Removing existing virtual environment..."
            rm -rf pdx_env
        fi
        
        # Create virtual environment
        python3 -m venv pdx_env
        
        # Activate virtual environment
        source pdx_env/bin/activate
        
        # Upgrade pip
        pip install --upgrade pip
        
        # Install minimal requirements
        cat > requirements_minimal.txt << EOF
# Minimal requirements for PDX analysis
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
jupyter>=1.0.0
ipython>=7.0.0
EOF
        
        pip install -r requirements_minimal.txt
        
        print_status "Virtual environment created successfully!"
        print_status "To activate: source pdx_env/bin/activate"
        print_status "To deactivate: deactivate"
        ;;
        
    "conda")
        print_step "Setting up conda environment..."
        
        if ! command -v conda >/dev/null 2>&1; then
            print_error "Conda not found. Please install Anaconda or Miniconda first."
            exit 1
        fi
        
        # Remove existing environment
        conda env remove -n pdx_analysis -y 2>/dev/null || true
        
        # Create conda environment with minimal packages
        conda create -n pdx_analysis python=3.9 -y
        
        # Activate and install packages
        eval "$(conda shell.bash hook)"
        conda activate pdx_analysis
        
        conda install -c conda-forge pandas numpy matplotlib seaborn scipy scikit-learn jupyter -y
        
        print_status "Conda environment created successfully!"
        print_status "To activate: conda activate pdx_analysis"
        print_status "To deactivate: conda deactivate"
        ;;
        
    "docker")
        print_step "Setting up Docker environment..."
        
        if ! command -v docker >/dev/null 2>&1; then
            print_error "Docker not found. Please install Docker first."
            exit 1
        fi
        
        print_status "Building Docker image..."
        docker build -t pdx_analysis .
        
        print_status "Docker image built successfully!"
        print_status "To run: docker-compose up"
        print_status "Access Jupyter at: http://localhost:8888"
        print_status "Token: pdx_analysis_token"
        ;;
        
    "skip")
        print_warning "Skipping environment setup."
        ;;
esac

# Test installation
if [ "$setup_method" != "skip" ] && [ "$setup_method" != "docker" ]; then
    print_step "Testing installation..."
    
    # Test Python packages
    if [ "$setup_method" = "venv" ]; then
        source pdx_env/bin/activate
    elif [ "$setup_method" = "conda" ]; then
        eval "$(conda shell.bash hook)"
        conda activate pdx_analysis 2>/dev/null || true
    fi
    
    python3 -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
print('✅ All required packages imported successfully!')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Matplotlib: {plt.matplotlib.__version__}')
print(f'Seaborn: {sns.__version__}')
"
    
    # Test Jupyter
    if python3 -m jupyter --version >/dev/null 2>&1; then
        print_status "Jupyter installation verified"
    else
        print_warning "Jupyter verification failed"
    fi
fi

# Generate data if possible
if [ "$setup_method" != "skip" ]; then
    print_step "Generating sample data..."
    
    if [ -f "src/python/generate_enhanced_data.py" ]; then
        if python3 src/python/generate_enhanced_data.py --models 2 --timepoints 4 >/dev/null 2>&1; then
            print_status "Sample data generated successfully"
        else
            print_warning "Could not generate sample data (packages may need to be activated)"
        fi
    fi
fi

# Final instructions
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"

case $setup_method in
    "current")
        echo "Packages installed in current Python environment"
        echo ""
        echo "Next steps:"
        echo "1. Run: python3 -m jupyter notebook"
        echo "2. Open: notebooks/01_data_exploration.ipynb"
        ;;
    "venv")
        echo "Virtual environment created: pdx_env"
        echo ""
        echo "Next steps:"
        echo "1. Activate: source pdx_env/bin/activate"
        echo "2. Run: jupyter notebook"
        echo "3. Open: notebooks/01_data_exploration.ipynb"
        echo "4. When done: deactivate"
        ;;
    "conda")
        echo "Conda environment created: pdx_analysis"
        echo ""
        echo "Next steps:"
        echo "1. Activate: conda activate pdx_analysis"
        echo "2. Run: jupyter notebook"
        echo "3. Open: notebooks/01_data_exploration.ipynb"
        echo "4. When done: conda deactivate"
        ;;
    "docker")
        echo "Docker environment ready"
        echo ""
        echo "Next steps:"
        echo "1. Run: docker-compose up"
        echo "2. Open: http://localhost:8888"
        echo "3. Token: pdx_analysis_token"
        echo "4. Open: notebooks/01_data_exploration.ipynb"
        ;;
    "skip")
        echo "Environment setup skipped"
        echo ""
        echo "Manual installation:"
        echo "pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter"
        ;;
esac

if [ "$setup_method" != "skip" ]; then
    echo ""
    echo "📖 Documentation: README.md"
    echo "🧪 Test setup: bash tests/integration_test.sh"
    echo "❓ Troubleshooting: docs/FAQ.md"
fi