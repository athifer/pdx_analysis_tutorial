# PDX Analysis Tutorial

This repository provides a comprehensive tutorial for analyzing Patient-Derived Xenograft (PDX) data, including tumor growth, gene expression, and variant data. The package includes advanced visualization workflows and statistical analysis tools for preclinical oncology research using R and Python.

## ⚡ Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/athifer/pdx_analysis_tutorial.git
cd pdx_analysis_tutorial

# 2. Create conda environment with essential packages
conda create -n pdx_analysis python=3.9 pandas numpy matplotlib seaborn scipy scikit-learn jupyter -y
conda activate pdx_analysis

# 3. Install additional packages for advanced workflows
pip install lifelines
# Note: You may see a deprecation warning about 'autograd-gamma' - this is normal and safe to ignore

# 4. Generate enhanced dataset (20 models: 10+10 design)
python src/python/generate_enhanced_data.py

# 5. Run all advanced workflows
python src/python/advanced_workflows.py
```

**Alternative Setup**: Use the provided setup script
```bash
bash setup.sh  # Choose option 1 for Python environment
conda activate pdx_analysis
```

## 🔥 What's New in This Version

### 📊 **Enhanced Dataset**
- **Doubled Sample Size**: Now 20 models (10+10) for statistical power
- **Expanded Gene Panel**: 1000 genes with realistic expression patterns
- **Stronger Differential Expression**: Enhanced treatment effects for DE detection
- **Improved Realism**: Cancer-type specific biology and technical variation

### 🎨 **Advanced Visualization Suite**
- **Growth Curve Workflows**: Multi-panel analysis with statistical testing
- **Waterfall Plots**: Drug response classification and ranking
- **Kaplan-Meier Analysis**: Survival curves with log-rank testing
- **Molecular Heatmaps**: Expression-response correlation visualization
- **Volcano Plots**: Differential gene expression analysis (treatment vs control)
- **Circos Plots**: Genome-wide variant distribution mapping

### 🔧 **Streamlined Package**
- **Reduced Complexity**: Removed Docker and non-essential components
- **Simplified Setup**: Single conda command for environment creation
- **Focused Documentation**: Clear workflows without unnecessary complexity
- **Publication Ready**: All visualizations at 300 DPI with professional styling

## 🎯 Analysis Workflows

This tutorial includes comprehensive workflows for:

### 📈 **Growth Curve Analysis**
- Individual mouse tumor trajectories
- Treatment arm comparisons with statistics
- Growth rate analysis and final volume comparisons
- Statistical testing (Mann-Whitney U tests)

### 🌊 **Waterfall Plots**
- Drug response classification across PDX models
- Response distribution analysis (Responder/Stable/Progressor)
- Treatment arm efficacy comparison

### ⏱️ **Survival Analysis**
- Kaplan-Meier survival curves
- Time-to-progression analysis
- Log-rank statistical testing
- Median survival time calculations

### 🔥 **Molecular Heatmaps**
- Gene expression vs drug response correlations
- Standardized expression visualization
- Response-associated gene identification
- Treatment arm molecular signatures

### 🌋 **Volcano Plots**
- Differential gene expression visualization with **FDR correction**
- Fold change vs statistical significance mapping (q-values)
- **Benjamini-Hochberg multiple testing correction**
- Raw p-value vs FDR-corrected significance comparison
- Top differentially expressed gene annotation
- Publication-ready statistical methodology

### 🧬 **Circos Plots**
- Genome-wide variant visualization
- Chromosomal distribution analysis
- Structural variant connections
- Multi-omics integration

## 📚 Learning Objectives

By completing this tutorial, you will learn to:
- Analyze tumor growth kinetics using advanced statistical models
- Create publication-quality visualizations for PDX studies
- Perform comprehensive biomarker discovery workflows
- Integrate multi-omics data (genomics, transcriptomics, phenomics)
- Apply survival analysis methods to preclinical data
- Generate circular genome plots for variant visualization

## 🆕 Key Features & Improvements

### Enhanced Statistical Power
- **20 PDX Models**: 10 control + 10 treatment for robust comparisons
- **1000 Genes**: Increased from 500 for better differential expression detection
- **Stronger Treatment Effects**: Enhanced signal-to-noise ratio in mock data
- **Realistic Biology**: Cancer-type specific growth patterns and batch effects

### Publication-Quality Visualizations
- **High Resolution**: All plots generated at 300 DPI for publication
- **Professional Styling**: Consistent color schemes and typography
- **Multi-Panel Layouts**: Comprehensive information in single figures
- **Statistical Annotations**: P-values, confidence intervals, and effect sizes

### Comprehensive Workflows
- **Growth Analysis**: Individual trajectories + group comparisons
- **Response Classification**: Standardized criteria (Responder/Stable/Progressor)
- **Survival Methods**: Time-to-event analysis with censoring
- **Multi-Omics Integration**: Combined genomic and transcriptomic visualization

### Streamlined Package
- **Reduced Size**: Removed Docker and non-essential components
- **Focused Content**: Core analysis functions without bloat
- **Easy Setup**: Single environment creation command
- **Clear Documentation**: Step-by-step instructions with troubleshooting

## 🔧 Prerequisites

### Software Requirements
- **Python** (≥ 3.9) - Core analysis environment
- **Conda** (recommended) - Package and environment management
- **Git** - Version control and repository cloning

### Required Python Packages
- **Data Science Stack**: `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **Survival Analysis**: `lifelines` (for Kaplan-Meier curves)
- **Interactive Analysis**: `jupyter`, `ipython`

### Knowledge Requirements
- Basic Python programming (functions, data structures)
- Understanding of data visualization concepts
- Familiarity with statistical testing and p-values
- Basic knowledge of cancer biology terminology

### Estimated Time
- **Environment Setup**: 10-15 minutes
- **Complete Tutorial**: 2-3 hours
- **Individual Workflows**: 15-30 minutes each

## 🚀 Tutorial Steps

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/athifer/pdx_analysis_tutorial.git
cd pdx_analysis_tutorial

# Setup environment (choose option 1 for Python only)
bash setup.sh
```

### Step 2: Activate Environment
```bash
# Activate conda environment
conda activate pdx_analysis

# Verify packages are installed
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn; print('✓ Core packages ready')"
python -c "import lifelines; print('✓ Survival analysis ready')" || pip install lifelines
```

### Step 3: Generate Enhanced Dataset (20 models)
```bash
# Generate 10+10 samples with strong differential expression
python src/python/generate_enhanced_data.py
```

### Step 4: Run Individual Workflows

#### Basic Analysis
```bash
# Data preprocessing
python scripts/preprocessing.py

# Variant analysis
python src/python/variant_analysis.py
```

#### Advanced Workflows
```bash
# Run all advanced visualizations
python src/python/advanced_workflows.py

# Or run individual components:
# Growth curves only
python -c "from src.python.advanced_workflows import PDXWorkflows; w=PDXWorkflows(); w.load_data(); w.growth_curves_analysis()"

# Waterfall plots only  
python -c "from src.python.advanced_workflows import PDXWorkflows; w=PDXWorkflows(); w.load_data(); w.waterfall_plot()"

# Survival analysis only
python -c "from src.python.advanced_workflows import PDXWorkflows; w=PDXWorkflows(); w.load_data(); w.survival_analysis()"

# Volcano plots only (differential expression)
python -c "from src.python.advanced_workflows import PDXWorkflows; w=PDXWorkflows(); w.load_data(); w.volcano_plot()"
```

### Step 5: Interactive Analysis
```bash
# Launch Jupyter for interactive analysis
jupyter notebook notebooks/03_integrated_analysis.ipynb
```

# You should see (pdx_env) prefix in your terminal prompt
# If you don't see the prefix, activation failed - check troubleshooting section
```

**⚠️ Critical**: The virtual environment MUST be activated before running any notebooks or Python scripts. Without activation, you'll get "bad interpreter" errors.

### Alternative Manual Setup
```bash
# If enhanced setup fails, use manual setup
python3 -m venv pdx_env
## 📁 Repository Structure

```
pdx_analysis_tutorial/
├── 📁 data/                    # Mock datasets (20 PDX models: 10+10 design)
│   ├── expression_tpm_mock.csv      # Gene expression matrix (1000 genes)
│   ├── tumor_volumes_mock.csv       # Tumor growth measurements  
│   └── variants_mock.csv            # Genomic variant calls
├── 📁 src/                     # Analysis source code
│   ├── python/
│   │   ├── advanced_workflows.py       # 🆕 Complete visualization suite
│   │   ├── generate_enhanced_data.py   # Enhanced data generation
│   │   ├── variant_analysis.py         # Variant-response analysis
│   │   ├── preprocessing.py            # Data preprocessing
│   │   └── plotting.py                 # Plotting utilities
│   └── R/
│       └── analyze_volume.R             # R-based growth analysis
├── 📁 notebooks/               # Jupyter notebooks for interactive analysis
│   ├── 01_data_exploration.ipynb       # Data overview and QC
│   ├── 02_biomarker_analysis.ipynb     # Expression analysis
│   └── 03_integrated_analysis.ipynb    # Multi-omics integration
├── 📁 scripts/                 # Standalone analysis scripts
│   └── preprocessing.py                # Data preprocessing
├── 📁 results/                 # Output directory for plots and results
├── 📁 tests/                   # Unit tests for analysis functions
├── environment.yml             # Conda environment specification
├── requirements.txt            # Python package requirements
└── setup.sh                   # Environment setup script
```

## 🎨 Visualization Gallery

This tutorial generates the following publication-ready visualizations:

### Growth Curve Analysis
- **Individual trajectories**: Per-mouse tumor growth over time
- **Mean curves with error bars**: Treatment comparisons with SEM
- **Growth rate analysis**: Statistical comparison of exponential growth rates  
- **Final volume comparison**: Endpoint efficacy analysis

### Waterfall Plots
- **Response classification**: Responder/Stable/Progressor categories
- **Ranked response**: Models ordered by treatment effect
- **Response distribution**: Pie chart summary of efficacy

### Survival Analysis  
- **Kaplan-Meier curves**: Time-to-progression analysis
- **Log-rank testing**: Statistical significance testing
- **Median survival**: Treatment arm comparisons

### Molecular Heatmaps
- **Expression-response correlation**: Gene signatures of drug response
- **Standardized heatmaps**: Z-scored expression visualization
- **Response annotation**: Models colored by treatment outcome

### Volcano Plots
- **Differential expression**: Treatment vs control gene expression comparison
- **Multiple testing correction**: Benjamini-Hochberg FDR correction implemented
- **Statistical rigor**: Q-values (FDR) vs raw p-values comparison
- **Effect size visualization**: Log2 fold change thresholds for biological significance
- **Publication standards**: Follows genomics best practices for significance testing
- **Gene annotation**: Top FDR-significant genes labeled for follow-up

### Circos Plots
- **Genome-wide variants**: Chromosomal distribution visualization
- **Structural connections**: Inter-chromosomal variant relationships
- **Multi-omics integration**: Combined genomic and transcriptomic data

## 🔬 Data Details

### Enhanced Mock Dataset (20 Models)
- **Sample size**: 10 control + 10 treatment models
- **Statistical power**: Designed for differential expression detection
- **Cancer types**: NSCLC, BRCA, CRC, PDAC representation
- **Time points**: 10 measurements per model (every 3 days)
- **Gene expression**: 1000 genes with realistic expression patterns
- **Variants**: ~50 variants across oncogenes and tumor suppressors

### Data Generation Features
- **Biological realism**: Cancer-type specific growth patterns
- **Treatment effects**: Differential response by cancer type
- **Technical variation**: Batch effects and measurement noise
- **Correlations**: Expression-response associations
- **Missing data**: Realistic dropout patterns

## 📊 Statistical Methods

### Growth Analysis
- **Exponential growth modeling**: Log-linear regression
- **Statistical testing**: Mann-Whitney U tests
- **Effect size**: Growth rate differences
- **Confidence intervals**: Bootstrap-based estimates

### Survival Analysis  
- **Kaplan-Meier estimation**: Non-parametric survival curves
- **Log-rank test**: Treatment comparison significance
- **Hazard ratios**: Risk quantification
- **Censoring handling**: Right-censored observations

### Expression Analysis
- **Correlation analysis**: Pearson correlation coefficients
- **Multiple testing**: P-value adjustment methods
- **Standardization**: Z-score normalization
- **Dimensionality reduction**: PCA for visualization

### Response Analysis
- **Response criteria**: RECIST-inspired thresholds
- **Classification**: Responder/Stable/Progressor categories
- **Biomarker discovery**: Expression-response associations
- **Cross-validation**: Model performance assessment
- **results/**: Example output figures generated by the analysis scripts.
	- `heatmap_top40.png`: Heatmap of the top 40 variable genes (example output).
	- `mean_growth_curves.png`: Mean tumor growth curves by treatment arm (example output).
	- `spider_plot.png`: Spider plot summarizing response metrics (example output).
- **scripts/**: Analysis scripts and notebooks.
	- `analyze_volume.R`: R script for analyzing tumor volume data using linear mixed models and the DRAP package.
	- `preprocessing.py`: Python helpers for loading and normalizing tumor volume data.
	- `download_real_data.sh`: Example shell script for downloading real PDX annotation data from public repositories.
	- `biomarker_analysis.ipynb`: (Empty) Jupyter notebook for biomarker analysis (template for further work).

## Getting Started

1. Clone the repository and install required R and Python packages as needed (see comments in scripts).
2. Use the mock data in `data/` to test the analysis scripts.
3. Run the scripts in `scripts/` to reproduce example results in `results/`.

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `Error: package 'DRAP' not found`
**Solution**: Install DRAP from GitHub: `devtools::install_github('SCBIT-YYLab/DRAP')`

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
**Solution**: Install Python dependencies: `pip install -r requirements.txt`

**Issue**: Memory error during analysis
**Solution**: Reduce dataset size or increase system memory allocation

**Issue**: Plots not displaying correctly
**Solution**: Check graphics device settings and ensure proper package installation

## 🚨 Troubleshooting

### ⚠️ MOST COMMON ISSUE: Virtual Environment Not Activated

**Problem**: Jupyter "bad interpreter" error or package not found
```
-bash: /usr/local/bin/jupyter: /usr/local/opt/python/bin/python3.7: bad interpreter: No such file or directory
```
OR
```
ModuleNotFoundError: No module named 'pandas'
```

**Root Cause**: Virtual environment is not activated

**Solution**: ALWAYS activate the virtual environment first:
## 🔧 Troubleshooting

### Environment Activation Issues

**Problem**: `conda activate` command not recognized
```bash
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'
```

**Solution**: Initialize conda for your shell
```bash
# Initialize conda (one-time setup)
conda init bash  # or zsh, depending on your shell
source ~/.bashrc  # or restart terminal

# Now you can activate
conda activate pdx_analysis
```

**Alternative**: Use full conda path or direct Python execution
```bash
# Option 1: Use full path
/path/to/conda/envs/pdx_analysis/bin/python src/python/advanced_workflows.py

# Option 2: Use conda run
conda run -n pdx_analysis python src/python/advanced_workflows.py
```

### Package Installation Issues
If you encounter dependency conflicts:

```bash
# Option 1: Use conda (recommended)
conda env create -f environment.yml
conda activate pdx_analysis

# Option 2: Use pip with virtual environment
python -m venv pdx_env
source pdx_env/bin/activate
pip install -r requirements.txt

# Option 3: Install individual packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn lifelines jupyter
```

### Expected Warnings During Installation

**Lifelines Package Warning**: When installing `lifelines`, you may see a deprecation warning:
```
DEPRECATION: Building 'autograd-gamma' using the legacy setup.py bdist_wheel mechanism...
```
**This is normal and safe to ignore.** The warning occurs because a dependency (`autograd-gamma`) uses older packaging standards. The installation will complete successfully and lifelines will work perfectly for survival analysis.

### Verification Commands
Test your setup:
```bash
# Activate environment
conda activate pdx_analysis

# Test core packages
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn; print('✓ Core packages ready')"

# Test lifelines for survival analysis
python -c "import lifelines; print('✓ Survival analysis ready')" || pip install lifelines

# Test workflow loading
python -c "from src.python.advanced_workflows import PDXWorkflows; print('✓ Workflows ready')"
```

## 📚 Resources and References

### PDX Research Background
- **PDX Models**: Patient-derived xenografts in cancer research
- **Biomarker Discovery**: Multi-omics approaches in oncology
- **Preclinical Analysis**: Statistical methods for animal studies

### Data Analysis Methods
- **Growth Modeling**: Exponential and linear mixed-effects models
- **Survival Analysis**: Kaplan-Meier and Cox proportional hazards
- **Multi-omics Integration**: Systems biology approaches

### Visualization Techniques
- **Waterfall Plots**: Drug response visualization standards
- **Circos Plots**: Genomic data circular representation
- **Heatmaps**: Expression data visualization best practices

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Bug reports and feature requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏗️ Acknowledgments

- Mock data generation inspired by real PDX studies
- Visualization methods adapted from cancer research best practices
- Statistical approaches based on preclinical analysis standards

---

**Ready to start?** Run `bash setup.sh` and begin your PDX analysis journey! 🚀

**Problem**: Missing packages after installation
**Solution**: Verify environment activation:
```bash
# Check if virtual environment is activated
which python
python --version

# Activate environment if needed
source pdx_env/bin/activate  # or conda activate pdx_analysis
```

**Problem**: Permission errors during installation
**Solution**: Use user installation or virtual environment:
```bash
# User installation
pip install --user package_name

# Or use virtual environment (recommended)
python3 -m venv pdx_env && source pdx_env/bin/activate
```

### Getting Help
- Check the [documentation](docs/) for detailed explanations
- Review the [FAQ](docs/FAQ.md) for common questions
- Open an [issue](https://github.com/athifer/pdx_analysis_tutorial/issues) for bugs or questions

## 📖 References

### PDX Research Background
- **Hidalgo et al. (2014)** - Patient-derived xenograft models: an emerging platform for translational cancer research. *Cancer Discovery*
- **Gao et al. (2015)** - High-throughput screening using patient-derived tumor xenografts to predict clinical trial drug response. *Nature Medicine*

### Statistical Methods
- **Bates et al. (2015)** - Fitting Linear Mixed-Effects Models Using lme4. *Journal of Statistical Software*
- **Love et al. (2014)** - Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. *Genome Biology*

### PDX Databases
- [PDX Finder](https://www.pdxfinder.org/) - Comprehensive PDX model database
- [PDMR](https://pdmr.cancer.gov/) - Patient-Derived Models Repository
- [EurOPDX](https://www.europdx.eu/) - European PDX consortium

## Notes
- The provided data are mock examples for tutorial purposes only.
- For real data, see the instructions in `download_real_data.sh` and update file paths as needed.
- The `biomarker_analysis.ipynb` notebook is a template for further biomarker discovery analyses.

## License
See `LICENSE` for details.
