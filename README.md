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

# 4. Generate effective PDX dataset (15+15 design)
python src/python/generate_effective_pdx_data.py

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

### Step 3: Generate Realistic PDX Dataset (30 models)
```bash
# Generate effective PDX study (15+15 samples)
python src/python/generate_effective_pdx_data.py
```

### Step 4: Run Analysis Workflows

#### Complete Analysis Suite
```bash
# Run all workflows with effective study data
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd

# Load effective study data (15+15 samples)
w = PDXWorkflows('data/', 'results/')
w.expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w.tumor_data = pd.read_csv('data/tumor_volumes_effective.csv') 
w.variant_data = pd.read_csv('data/variants_effective.csv')

# Run all analyses
w.growth_curves_analysis()
w.waterfall_plot() 
w.survival_analysis()
w.molecular_heatmaps()
w.volcano_plot()  # Expected: ~924 FDR-significant genes (4.6%)
w.circos_plot()

print('\\nAnalysis complete! Check results/ directory')
"
```

#### Individual Analysis Components
```bash
# Volcano plot only (differential expression with FDR correction)
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.expression_data = expression_data
w.volcano_plot()
"

# Growth curves only
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.tumor_data = tumor_data
w.growth_curves_analysis()
"

# All molecular analyses
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
w = PDXWorkflows('data/', 'results/')
w.expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w.tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w.variant_data = pd.read_csv('data/variants_effective.csv')
w.molecular_heatmaps()
w.volcano_plot()
w.circos_plot()
"
```

### Step 5: Individual Workflow Components

#### Basic Analysis
#### Advanced Workflows
```bash
# All analysis is now integrated in the main workflow
# Individual components as needed
# Growth curves only
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.tumor_data = tumor_data
w.growth_curves_analysis()
"

# Waterfall plots only  
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.tumor_data = tumor_data
w.waterfall_plot()
"

# Survival analysis only
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.tumor_data = tumor_data
w.survival_analysis()
"

# Volcano plots only (will show ~924 FDR genes - realistic for effective studies)
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.expression_data = expression_data
w.volcano_plot()
"
### Step 6: Interactive Analysis
```bash
# Launch Jupyter for interactive analysis
jupyter notebook notebooks/03_integrated_analysis.ipynb
```

## 📊 Expected Results Summary

| Analysis Type | Expected Outcome | Interpretation |
|---------------|------------------|----------------|
| **Volcano Plot** | ~924 FDR genes (4.6%) | Realistic for well-powered studies |
| **Growth Curves** | Significant treatment effect | Clear tumor growth inhibition |
| **Survival Analysis** | Extended progression-free survival | Treatment efficacy demonstration |
| **Molecular Heatmaps** | Expression-response correlations | Biomarker identification |

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
├── 📁 data/                    # Realistic PDX study datasets
│   ├── expression_tpm_effective.csv    # Gene expression: 20K genes × 30 samples (15+15)
│   ├── tumor_volumes_effective.csv     # Tumor growth measurements over time
│   ├── variants_effective.csv          # Genomic variants across cancer genes
│   └── metadata_effective.csv          # Treatment assignments and sample info
├── 📁 src/                     # Analysis source code
│   ├── python/
│   │   ├── advanced_workflows.py           # 🆕 Complete visualization suite with FDR correction
│   │   ├── generate_effective_pdx_data.py  # Realistic PDX study data generation
│   │   └── advanced_workflows.py           # Complete analysis pipeline
│   └── R/
│       └── analyze_volume.R                 # R-based growth analysis
├── 📁 notebooks/               # Jupyter notebooks for interactive analysis
│   ├── 01_data_exploration.ipynb           # Data overview and QC
│   ├── 02_biomarker_analysis.ipynb         # Expression analysis
│   └── 03_integrated_analysis.ipynb        # Multi-omics integration
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

### Realistic PDX Study Dataset (30 Models)
- **Study design**: 15 control + 15 treatment models (well-powered study)
- **Gene expression**: 20,000 genes (realistic RNA-seq scale)
- **Statistical power**: 80%+ for moderate effects (publication-ready)
- **Expected FDR genes**: ~924 (4.6% - matches real effective PDX studies)
- **Effect sizes**: 1.5-3x fold changes (realistic for targeted therapy)
- **Time points**: 5 measurements per model (0, 7, 14, 21, 28 days)
- **Variants**: 750 variants across cancer genes (realistic burden)

### Enhanced Study Features
- **Lower technical noise**: Improved experimental protocols
- **Larger sample size**: Adequate statistical power for discovery
- **Realistic effect distribution**: Based on published PDX studies
- **Gene categorization**: Oncogenes, tumor suppressors, immune genes, drug targets
- **Biological realism**: Cancer-type specific patterns and correlations
- **Publication-ready**: Meets standards for high-impact journals

### Data Generation Features
- **Treatment effects**: Differential response modeling
- **Technical variation**: Realistic batch effects and measurement noise
- **Missing data patterns**: Biologically motivated dropout
- **Correlations**: Expression-response associations
- **Variant-expression links**: Multi-omics integration

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

## 📊 Statistical Methods

### Differential Expression Analysis 🆕
- **Multiple testing correction**: Benjamini-Hochberg FDR correction
- **Statistical testing**: Welch's t-test for group comparisons
- **Effect size**: Log2 fold change (treatment vs control)
- **Significance thresholds**: |log2FC| > 1.0, FDR < 0.05
- **Publication standards**: Follows genomics best practices

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
- **Multiple testing**: FDR correction for genomics data
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

## 🚀 Quick Start Summary

### Generate and Analyze Realistic PDX Study
```bash
# 1. Generate effective study data (15+15 samples)
python src/python/generate_effective_pdx_data.py

# 2. Run complete analysis suite
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
w = PDXWorkflows('data/', 'results/')
w.expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w.tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w.volcano_plot()  # Expected: ~924 FDR genes (4.6%)
"

# 3. Check results
ls results/volcano_plot.png  # Publication-ready volcano plot
```

### Expected Results
- **FDR-significant genes**: ~924 (4.6%)
- **Effect sizes**: 1.5-3x fold changes
- **Statistical power**: Publication-ready discovery rate

## �️ Troubleshooting

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
```

### Verification Commands
Test your setup:
```bash
# Activate environment
conda activate pdx_analysis

# Test core packages
python -c "import pandas, numpy, matplotlib, seaborn, scipy, sklearn; print('✅ Core packages ready')"

# Test lifelines for survival analysis
python -c "import lifelines; print('✅ Survival analysis ready')"

# Test workflow loading
python -c "from src.python.advanced_workflows import PDXWorkflows; print('✅ Workflows ready')"
```

## 📚 Additional Resources

### Documentation Files 🆕
- **[VOLCANO_PLOT_SUMMARY.md](VOLCANO_PLOT_SUMMARY.md)**: Complete volcano plot implementation guide
- **[MULTIPLE_TESTING_CORRECTION_EXPLANATION.md](MULTIPLE_TESTING_CORRECTION_EXPLANATION.md)**: FDR correction methodology
- **[PDX_STUDY_EFFECTIVENESS_ANALYSIS.md](PDX_STUDY_EFFECTIVENESS_ANALYSIS.md)**: Study design best practices

### Key References
- **Benjamini & Hochberg (1995)** - Controlling the false discovery rate: a practical and powerful approach to multiple testing
- **Gao et al. (2015)** - High-throughput screening using patient-derived tumor xenografts to predict clinical trial drug response
- **Hidalgo et al. (2014)** - Patient-derived xenograft models: an emerging platform for translational cancer research

## 🤝 Contributing

We welcome contributions! This tutorial demonstrates:
- ✅ **Statistical rigor**: Proper multiple testing correction
- ✅ **Realistic expectations**: Effective study design with proper statistical power  
- ✅ **Publication-ready methods**: Following genomics best practices
- ✅ **Educational value**: Understanding PDX study design challenges

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🎯 Ready to start?** Run the setup commands above and explore effective PDX study design!
