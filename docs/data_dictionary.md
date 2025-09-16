# Data Dictionary

This document describes all variables and data formats used in the PDX analysis tutorial.

## Tumor Volume Data (`tumor_volumes_mock.csv`)

| Variable | Type | Description | Units | Example Values |
|----------|------|-------------|-------|----------------|
| Model | String | PDX model identifier | - | PDX1, PDX2, PDX3 |
| Arm | String | Treatment arm designation | - | control, treatment |
| Day | Integer | Days since treatment start | days | 0, 4, 8, 12, 16, 20, 24, 28 |
| Volume_mm3 | Float | Tumor volume measurement | mm³ | 127.54, 151.57, 171.15 |

### Data Collection Notes
- Volume measurements taken twice weekly (every 4 days)
- Baseline measurements (Day 0) represent pre-treatment volumes
- Missing values indicate measurement failures or animal dropout

## Gene Expression Data (`expression_tpm_mock.csv`)

| Variable | Type | Description | Units | Range |
|----------|------|-------------|-------|-------|
| Row names | String | Gene identifiers | - | GENE1, GENE2, ..., GENE500 |
| Column names | String | PDX model identifiers | - | PDX1, PDX2, ..., PDX6 |
| Values | Float | Transcripts Per Million (TPM) | TPM | 0.1 - 50.0 |

### Data Processing Notes
- TPM normalization applied to raw RNA-seq counts
- Log2 transformation recommended for downstream analysis
- Values < 0.1 TPM considered not expressed

## Variant Data (`variants_mock.csv`)

| Variable | Type | Description | Units | Example Values |
|----------|------|-------------|-------|----------------|
| Model | String | PDX model identifier | - | PDX1, PDX2, PDX3 |
| Gene | String | Gene name with variant | - | GENE429, GENE201 |
| Chr | String | Chromosome location | - | chr1, chr2, ..., chrX |
| Pos | Integer | Genomic position | bp | 670673, 322257 |
| Ref | String | Reference allele | - | A, T, G, C |
| Alt | String | Alternative allele | - | A, T, G, C |
| VAF | Float | Variant Allele Frequency | fraction | 0.001 - 0.999 |

### Variant Calling Notes
- Only high-confidence variants included (QUAL > 30)
- VAF represents tumor purity-adjusted frequency
- Germline variants filtered out