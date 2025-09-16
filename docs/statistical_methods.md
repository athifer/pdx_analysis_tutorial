# Statistical Methods

This document explains the statistical approaches used in the PDX analysis tutorial.

## Tumor Growth Analysis

### Linear Mixed-Effects Models

**Model Formula**: `log(Volume) ~ Arm * Day + (1|Model)`

**Components**:
- **Fixed Effects**: 
  - `Arm`: Treatment vs. control group effect
  - `Day`: Time effect (linear growth rate)
  - `Arm * Day`: Interaction term (difference in growth rates between groups)
- **Random Effects**: 
  - `(1|Model)`: Random intercept for each PDX model (accounts for baseline differences)

**Assumptions**:
- Log-normal distribution of tumor volumes
- Linear growth on log scale
- Independent observations within models
- Homoscedasticity of residuals

**Interpretation**:
- **Intercept**: Baseline log volume for control group
- **ArmTreatment**: Difference in baseline between treatment and control
- **Day**: Growth rate (log volume/day) for control group
- **ArmTreatment:Day**: Difference in growth rates between groups

### Tumor Growth Inhibition (TGI)

**Formula**: `TGI = 100 × (1 - (T_end - T_start) / (C_end - C_start))`

Where:
- T_end, T_start: Final and initial volumes for treatment group
- C_end, C_start: Final and initial volumes for control group

**Interpretation**:
- TGI = 0%: No inhibition
- TGI = 100%: Complete growth inhibition
- TGI > 100%: Tumor regression

## Gene Expression Analysis

### Differential Expression

**Method**: DESeq2 negative binomial model

**Model**: `~ condition` (treatment vs. control)

**Key Statistics**:
- **log2FoldChange**: Effect size (treatment vs. control)
- **lfcSE**: Standard error of log2 fold change
- **padj**: Benjamini-Hochberg adjusted p-value

**Significance Thresholds**:
- |log2FoldChange| > 1 (2-fold change)
- padj < 0.05 (5% FDR)

### Principal Component Analysis (PCA)

**Purpose**: Dimensionality reduction and quality control

**Interpretation**:
- PC1, PC2: Major sources of variation
- Clustering by treatment indicates differential expression
- Outliers may indicate technical issues

## Biomarker Analysis

### Correlation Analysis

**Method**: Spearman rank correlation

**Applications**:
- Gene expression vs. growth rate correlation
- Variant burden vs. treatment response

**Significance**: p < 0.05 with multiple testing correction

### Pathway Enrichment

**Method**: Gene Set Enrichment Analysis (GSEA)

**Database**: Hallmark gene sets from MSigDB

**Statistics**:
- **NES**: Normalized Enrichment Score
- **FDR**: False Discovery Rate
- **Leading Edge**: Core enriched genes

## Multiple Testing Correction

**Methods Used**:
- **Benjamini-Hochberg**: Controls False Discovery Rate
- **Bonferroni**: Conservative family-wise error rate control

**When to Use**:
- BH: Exploratory analysis, gene expression
- Bonferroni: Confirmatory analysis, small number of tests