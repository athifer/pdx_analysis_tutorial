# PDX Analysis Tutorial - Quick Reference Guide

## 🚀 **Features Overview**

### 🌋 **Volcano Plots with FDR Correction**
- **What**: Differential gene expression analysis with multiple testing correction
- **Method**: Benjamini-Hochberg FDR correction (publication-ready)
- **Output**: High-resolution volcano plots with significance thresholds

### 📊 **Realistic Study Design**
- **Sample size**: 15+15 PDX models (well-powered study)
- **Gene scale**: 20,000 genes (realistic RNA-seq)
- **Discovery rate**: ~4.6% FDR-significant genes

---

## ⚡ **Quick Commands**

### Generate Data
```bash
# Generate realistic PDX study data (15+15 samples)
python src/python/generate_effective_pdx_data.py
```

### Run Analysis
```bash
# Complete analysis suite
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
w = PDXWorkflows('data/', 'results/')
w.expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w.tumor_data = pd.read_csv('data/tumor_volumes_effective.csv')
w.variant_data = pd.read_csv('data/variants_effective.csv')
w.volcano_plot()
w.growth_curves_analysis()
w.molecular_heatmaps()
"

# Volcano plot only
python -c "
from src.python.advanced_workflows import PDXWorkflows
import pandas as pd
expression_data = pd.read_csv('data/expression_tpm_effective.csv')
w = PDXWorkflows('data/', 'results/')
w.expression_data = expression_data
w.volcano_plot()
"
```

---

## 📈 **Expected Results**

| Analysis | Expected Outcome | Interpretation |
|----------|------------------|----------------|
| **Volcano Plot** | ~924 FDR genes (4.6%) | Publication-ready discovery rate |
| **Effect Sizes** | 1.5-3x fold changes | Realistic for targeted therapy |
| **Raw p<0.05** | ~1,866 genes (9.3%) | Shows importance of FDR correction |

---

## 📁 **Key Files**

### Data Files
- `data/expression_tpm_effective.csv` - Expression data (20K genes × 30 samples)
- `data/tumor_volumes_effective.csv` - Tumor growth measurements 
- `data/variants_effective.csv` - Genomic variants across cancer genes
- `data/metadata_effective.csv` - Treatment assignments

### Analysis Scripts  
- `src/python/advanced_workflows.py` - Main analysis suite with volcano plots
- `src/python/generate_effective_pdx_data.py` - Generate realistic study data

### Documentation
- `README.md` - Complete tutorial and setup guide
- `QUICK_REFERENCE.md` - Quick command reference (this file)

### Results
- `results/volcano_plot.png` - Latest volcano plot (300 DPI, publication-ready)
- `results/` - All visualization outputs

---

## 🔬 **Statistical Methods**

### Multiple Testing Correction
- **Method**: Benjamini-Hochberg procedure
- **Controls**: False Discovery Rate (FDR)
- **Threshold**: FDR < 0.05
- **Why**: Essential when testing 20,000 genes simultaneously

### Significance Criteria
- **Fold Change**: |log2FC| > 1.0 (2-fold change)
- **Statistical**: FDR < 0.05 (corrected p-value)
- **Combined**: Both criteria must be met

---

## 🎯 **Use Cases**

### Educational
- **Demonstrate**: Publication-ready PDX analysis methodology
- **Show**: Impact of proper study design and sample size
- **Learn**: Multiple testing correction in genomics

### Research
- **Template**: Publication-ready volcano plot code
- **Methods**: Proper statistical methodology for PDX analysis
- **Realistic**: Achievable discovery rates for well-designed studies

---

## 🛠️ **Troubleshooting**

### Common Issues
1. **Environment**: Always activate conda/virtual environment first
2. **Data loading**: Use full paths to data files
3. **Memory**: 20K genes require adequate RAM

### Quick Fixes
```bash
# Activate environment
conda activate pdx_analysis

# Test installation
python -c "from src.python.advanced_workflows import PDXWorkflows; print('✅ Ready')"

# Regenerate data if needed
python src/python/generate_effective_pdx_data.py
```

---

## 📚 **Next Steps**

1. **Run the analysis** to see publication-ready results
2. **Explore documentation** for deeper understanding  
3. **Modify parameters** in data generation scripts
4. **Apply to real data** using the same methodology

---

*This tutorial provides a complete, publication-ready PDX analysis workflow with proper statistical rigor and realistic expectations for effective studies.*