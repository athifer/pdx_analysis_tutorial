# Frequently Asked Questions (FAQ)

## General Questions

### Q: What are Patient-Derived Xenografts (PDX)?
A: PDX models are created by implanting patient tumor tissue into immunocompromised mice. They maintain the original tumor's genetic and histological characteristics better than cell line models, making them valuable for drug testing and biomarker discovery.

### Q: Why use PDX models instead of cell lines?
A: PDX models preserve:
- Tumor heterogeneity and architecture
- Original genetic alterations
- Drug response patterns
- Stromal components (partially)

### Q: How long does a typical PDX study take?
A: Timeline varies by tumor type:
- Fast-growing tumors: 4-8 weeks
- Slow-growing tumors: 8-16 weeks
- Plus 2-4 weeks for data analysis

## Technical Questions

### Q: What statistical power do I need for PDX studies?
A: Generally aim for:
- **80% power** to detect clinically meaningful effects
- **8-10 mice per group** for tumor growth studies
- **Effect size >30%** for growth inhibition studies

### Q: How do I handle missing data in tumor volume measurements?
A: Options include:
- **Linear interpolation** for occasional missing points
- **Mixed-effects models** which handle missing data naturally
- **Exclude animals** with >20% missing measurements

### Q: What's the difference between TGI and growth rate analysis?
A: 
- **TGI** (Tumor Growth Inhibition): Single endpoint comparing final vs. initial volumes
- **Growth rate analysis**: Uses all timepoints, more powerful and informative

### Q: How many genes should I include in expression analysis?
A: Depends on your goals:
- **Focused panel**: 50-500 genes for targeted analysis
- **Whole transcriptome**: >20,000 genes for discovery

## Data Analysis Questions

### Q: Should I log-transform tumor volume data?
A: **Yes**, because:
- Tumor growth is exponential
- Log transformation normalizes variance
- Enables linear modeling approaches

### Q: What's the minimum fold-change for calling genes differentially expressed?
A: Common thresholds:
- **2-fold change** (|log2FC| > 1) for discovery
- **1.5-fold change** for validation studies
- **Consider biological significance**, not just statistical

### Q: How do I interpret negative TGI values?
A: Negative TGI indicates treatment accelerated growth compared to control. This could be due to:
- Ineffective treatment
- Growth-promoting side effects
- Measurement error or small sample size

### Q: What if my control group doesn't grow as expected?
A: Check for:
- **Model viability issues**: Poor engraftment or health
- **Environmental factors**: Housing, handling stress
- **Measurement errors**: Caliper calibration
- **Consider excluding** non-growing controls

## Software and Setup Questions

### Q: I'm getting package installation errors. What should I do?
A: Try these solutions:
1. Update R/Python to latest versions
2. Install packages from different repositories
3. Use conda/mamba for Python packages
4. Check system dependencies (compilers, libraries)

### Q: How much computational power do I need?
A: Minimum requirements:
- **RAM**: 8GB for basic analysis, 16GB+ for large datasets
- **Storage**: 10GB for tutorial data, scale up for real studies
- **Processing**: Modern multi-core CPU recommended

### Q: Can I run this analysis in the cloud?
A: Yes, several options:
- **RStudio Cloud**: For R-based analysis
- **Google Colab**: For Python notebooks
- **AWS/Azure**: For large-scale analysis
- **Institutional clusters**: Check with your IT department

## Interpretation Questions

### Q: What constitutes a "good" response in PDX studies?
A: Response criteria:
- **Complete response**: Tumor disappearance
- **Partial response**: >50% volume reduction
- **Stable disease**: <20% growth
- **Progressive disease**: >20% growth

### Q: How do I validate biomarkers discovered in PDX studies?
A: Validation pathway:
1. **Independent PDX cohort**: Test in new models
2. **Clinical samples**: Analyze patient tumor samples
3. **Functional studies**: Mechanistic validation
4. **Prospective validation**: Clinical trials

### Q: What are common reasons for PDX study failure?
A: Common issues:
- **Poor engraftment**: <70% take rate
- **High variability**: Inadequate standardization
- **Small effect sizes**: Underpowered studies
- **Technical artifacts**: Measurement or processing errors

## Need More Help?

- Check our detailed [documentation](/)
- Review the [statistical methods](statistical_methods.md)
- Read the [methodology guide](methodology.md)
- Open an issue on [GitHub](https://github.com/athifer/pdx_analysis_tutorial/issues)