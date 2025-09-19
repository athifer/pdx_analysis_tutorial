# PDX Analysis Tutorial

A streamlined tutorial for analyzing Patient-Derived Xenograft (PDX) data with Python. Generate publication-quality visualizations including growth curves, survival analysis, volcano plots, and molecular heatmaps.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/athifer/pdx_analysis_tutorial.git
cd pdx_analysis_tutorial

# 2. Create environment with all packages
conda create -n pdx_analysis python=3.9 pandas numpy matplotlib seaborn scipy scikit-learn jupyter lifelines -y
conda activate pdx_analysis

# 3. Generate mock PDX data (30 samples: 15 control + 15 treatment)
python src/python/generate_realistic_pdx_data.py

# 4. Run all analysis workflows
python src/python/advanced_workflows.py
```

That's it! Your `results/` folder now contains publication-ready plots.

## � What You Get

**6 Publication-Ready Visualizations:**
- **Growth Curves**: Treatment vs control with statistics
- **Waterfall Plot**: Drug response classification  
- **Survival Analysis**: Kaplan-Meier curves with log-rank test
- **Volcano Plot**: Differential gene expression (FDR corrected)
- **Molecular Heatmap**: Expression vs response correlation
- **Spider Plot**: Individual tumor trajectories

**Features:**
- Clinical color scheme (gray/blue) optimized for mobile viewing
- 300 DPI publication quality
- Statistical testing with multiple comparison correction
- 30 PDX models (15+15) for robust analysis
- 20,000 genes for comprehensive transcriptomics

## 🔬 Analysis Methods

- **Statistical Testing**: Mann-Whitney U, log-rank test, FDR correction
- **Survival Analysis**: Time-to-progression with censoring
- **Gene Expression**: Differential expression with fold change
- **Response Classification**: RECIST-like criteria (Responder/Stable/Progressor)
- **Growth Modeling**: Exponential and linear growth analysis

## 📁 Project Structure

```
data/                    # Mock PDX datasets
scripts/                 # Core analysis scripts  
├── generate_realistic_pdx_data.py  # Data generation
└── advanced_workflows.py           # Main analysis pipeline
results/                 # Generated plots and outputs
```

## 🛠️ Customization

Want to analyze your own data? Replace the CSV files in `data/` with your datasets:
- `tumor_volumes_mock.csv` - Tumor volume measurements
- `expression_tpm_mock.csv` - Gene expression (TPM values)  
- `variants_mock.csv` - Genomic variants

Then run: `python src/python/advanced_workflows.py`

## 🎯 Learning Objectives

By completing this tutorial, you will:
- Generate publication-quality PDX visualizations
- Apply statistical methods for preclinical data
- Perform survival analysis on tumor progression
- Create volcano plots with proper FDR correction
- Integrate multi-omics data for biomarker discovery

---

**Time Required**: ~10 minutes setup + analysis
**Output**: 6 publication-ready figures in `results/`
**Dependencies**: Python 3.9+, conda (recommended)

