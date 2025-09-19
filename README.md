# PDX Analysis Tutorial

A streamlined tutorial for analyzing Patient-Derived Xenograft (PDX) data with Python. Generate publication-quality visualizations including growth curves, volcano plots, and molecular heatmaps.

## 🚀 Quick Start (~20-30 minutes)

```bash
# 1. Clone and setup (~2 minutes)
git clone https://github.com/athifer/pdx_analysis_tutorial.git
cd pdx_analysis_tutorial

# 2. Create environment with core packages (~5-10 minutes)
conda create -n pdx_analysis python=3.9 pandas numpy matplotlib seaborn scipy scikit-learn jupyter -y

# 3. IMPORTANT: Activate the environment!
conda activate pdx_analysis

# 4. Generate mock PDX data (~3-5 minutes)
python src/python/generate_realistic_pdx_data.py

# 5. Run all analysis workflows (~5-10 minutes)
python src/python/advanced_workflows.py

# 6. Optional: Explore with interactive notebooks
jupyter notebook notebooks/
```

That's it! Your `results/` folder now contains publication-ready plots.

> **Note**: The conda environment creation (step 2) takes the longest time as it downloads and installs all packages. Subsequent runs are much faster since the environment is already set up.

## 📊 What You Get

**5 Publication-Ready Visualizations:**
- **Growth Curves**: Treatment vs control with statistics
- **Waterfall Plot**: Drug response classification  
- **Molecular Heatmap**: Expression vs response correlation
- **Volcano Plot**: Differential gene expression (FDR corrected)
- **Circos Plot**: Genome-wide variant visualization

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

- **Statistical Testing**: Mann-Whitney U, FDR correction
- **Gene Expression**: Differential expression with fold change
- **Response Classification**: RECIST-like criteria (Responder/Stable/Progressor)
- **Growth Modeling**: Exponential and linear growth analysis

## 📁 Project Structure

```
data/                    # Mock PDX datasets
src/python/              # Core analysis scripts  
├── generate_realistic_pdx_data.py  # Data generation
└── advanced_workflows.py           # Main analysis pipeline
notebooks/               # Interactive Jupyter notebooks
├── 01_data_exploration.ipynb       # Data loading and QC
├── 02_biomarker_analysis.ipynb     # Biomarker discovery
└── 03_integrated_analysis.ipynb    # Multi-omics integration
results/                 # Generated plots and outputs
```

## 📓 Interactive Notebooks

For step-by-step learning and customization, use the Jupyter notebooks:

```bash
# After setting up environment and generating data:
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Available notebooks:**
- **01_data_exploration.ipynb** - Data loading, quality control, and initial exploration
- **02_biomarker_analysis.ipynb** - Differential expression and biomarker discovery  
- **03_integrated_analysis.ipynb** - Multi-omics integration and predictive modeling

These notebooks provide interactive analysis with detailed explanations, perfect for learning and adapting to your own data.

## 🛠️ Customization

Want to analyze your own data? Replace the CSV files in `data/` with your datasets:
- `tumor_volumes_realistic.csv` - Tumor volume measurements
- `expression_tpm_realistic.csv` - Gene expression (TPM values)  
- `variants_realistic.csv` - Genomic variants

Then run: `python src/python/advanced_workflows.py`

## 🎯 Learning Objectives

By completing this tutorial, you will:
- Generate publication-quality PDX visualizations
- Apply statistical methods for preclinical data
- Create volcano plots with proper FDR correction
- Integrate multi-omics data for biomarker discovery

---

**Time Required**: 
- **First-time setup**: ~20-30 minutes (conda environment creation is the longest step)
- **Subsequent runs**: ~5-10 minutes (data generation + analysis only)
- **Individual plots**: ~1-2 minutes each

**Output**: 5 publication-ready figures in `results/`
**Dependencies**: Python 3.9+, conda (recommended)

## 🛠️ Common Issues

### "ModuleNotFoundError: No module named 'pandas'"
**Solution**: You forgot to activate the conda environment!
```bash
conda activate pdx_analysis  # Run this first!
python src/python/generate_realistic_pdx_data.py  # Now this will work
```

### Environment activation not working?
**Solution**: Initialize conda for your shell (one-time setup):
```bash
conda init bash  # or 'zsh' for Mac default shell
source ~/.bashrc  # or restart terminal
conda activate pdx_analysis
```
conda activate pdx_analysis
```

