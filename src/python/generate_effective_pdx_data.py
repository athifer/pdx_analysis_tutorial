"""
Generate "effective PDX study" data with larger effect sizes
to demonstrate what successful differential expression looks like
"""

import pandas as pd
import numpy as np
import os

def generate_effective_pdx_data():
    """
    Generate data for an effective PDX study scenario:
    - n=15 per group (realistic for well-funded study)
    - Larger effect sizes (2-4 fold changes)
    - Lower technical noise
    - Focused on targetable pathways
    """
    
    np.random.seed(456)  # Different seed for more realistic results
    
    # Study parameters
    n_control = 15
    n_treatment = 15
    n_genes = 20000
    
    print("Generating Realistic Effective PDX Study Data...")
    print(f"Study design: {n_control} control + {n_treatment} treatment PDX models")
    print(f"Genes analyzed: {n_genes:,}")
    print("Target: 3-5% FDR-significant genes (realistic for effective study)")
    
    # Enhanced differential expression parameters (more realistic)
    de_fraction = 0.05  # 5% of genes are DE (more realistic than 12%)
    n_de_genes = int(n_genes * de_fraction)
    
    # Create gene categories with more realistic DE rates
    gene_categories = []
    for i in range(n_genes):
        if i < 2000:  # First 2000 are oncogenes/tumor suppressors
            if np.random.random() < 0.12:  # 12% DE rate for cancer genes (reduced from 30%)
                gene_categories.append('cancer_gene')
            else:
                gene_categories.append('background')
        elif i < 5000:  # Next 3000 are immune/pathway genes
            if np.random.random() < 0.06:  # 6% DE rate (reduced from 15%)
                gene_categories.append('pathway_gene')
            else:
                gene_categories.append('background')
        else:  # Remaining are background
            if np.random.random() < 0.03:  # 3% DE rate (reduced from 8%)
                gene_categories.append('background_de')
            else:
                gene_categories.append('background')
    
    # Generate expression data with realistic distributions
    # PDX tumors typically have higher baseline expression variance
    base_expression = 8.0  # Log2 TPM baseline
    
    # Control samples (good quality study with lower technical variance)
    control_noise = 0.45  # Slightly increased from 0.4 for more realism
    control_data = np.random.normal(base_expression, control_noise, (n_genes, n_control))
    
    # Treatment samples
    treatment_data = np.random.normal(base_expression, control_noise, (n_genes, n_treatment))
    
    # Add realistic differential expression with more moderate effect sizes
    de_gene_count = 0
    upregulated_genes = []
    downregulated_genes = []
    
    for i, category in enumerate(gene_categories):
        if category != 'background':
            # More realistic effect sizes
            if category == 'cancer_gene':
                # Cancer genes: moderate to large effects (1.5-3 fold, not 2-5 fold)
                effect_size = np.random.uniform(1.0, 2.5)  # log2 fold change
            elif category == 'pathway_gene':
                # Pathway genes: small to moderate effects (1.2-2 fold)
                effect_size = np.random.uniform(0.8, 1.8)
            else:
                # Background DE: small effects (1.1-1.5 fold)
                effect_size = np.random.uniform(0.6, 1.2)
            
            # 60% upregulated, 40% downregulated
            if np.random.random() < 0.6:
                treatment_data[i, :] += effect_size
                upregulated_genes.append(f'GENE{i+1}')
            else:
                treatment_data[i, :] -= effect_size
                downregulated_genes.append(f'GENE{i+1}')
            
            de_gene_count += 1
    
    print(f"Target DE genes: {de_gene_count:,} ({de_gene_count/n_genes*100:.1f}%)")
    print(f"  - Upregulated: {len(upregulated_genes):,}")
    print(f"  - Downregulated: {len(downregulated_genes):,}")
    
    # Create expression matrix
    all_data = np.hstack([control_data, treatment_data])
    
    # Create model names
    control_models = [f'PDX_C{i+1:02d}' for i in range(n_control)]
    treatment_models = [f'PDX_T{i+1:02d}' for i in range(n_treatment)]
    all_models = control_models + treatment_models
    
    # Create gene names
    gene_names = [f'GENE{i+1}' for i in range(n_genes)]
    
    # Create expression dataframe
    expression_df = pd.DataFrame(
        all_data,
        index=gene_names,
        columns=all_models
    )
    expression_df.index.name = 'Gene'
    
    # Reset index to make Gene a column
    expression_df = expression_df.reset_index()
    
    # Generate enhanced tumor volume data (stronger treatment effect)
    volume_data = []
    
    for i, model in enumerate(all_models):
        arm = 'control' if model.startswith('PDX_C') else 'treatment'
        
        # Initial volume (similar between arms)
        initial_volume = np.random.normal(150, 30)
        initial_volume = max(50, initial_volume)  # Minimum 50 mm3
        
        # Growth parameters
        if arm == 'control':
            growth_rate = np.random.normal(0.15, 0.03)  # Control growth
        else:
            # Treatment effect: significant tumor shrinkage/stabilization
            growth_rate = np.random.normal(-0.05, 0.04)  # Negative growth (shrinkage)
        
        # Generate time series (0, 7, 14, 21, 28 days)
        for day in [0, 7, 14, 21, 28]:
            if day == 0:
                volume = initial_volume
            else:
                # Exponential growth/shrinkage with noise
                volume = initial_volume * np.exp(growth_rate * day / 7)
                volume *= np.random.normal(1.0, 0.1)  # Add measurement noise
                volume = max(10, volume)  # Minimum measurable volume
            
            volume_data.append({
                'Model': model,
                'Day': day,
                'Volume_mm3': round(volume, 1),
                'Arm': arm
            })
    
    volume_df = pd.DataFrame(volume_data)
    
    # Generate enhanced variant data (more realistic)
    n_variants_per_model = 25  # Fewer, more realistic variants
    variant_data = []
    
    # Common cancer genes for realistic variants
    cancer_genes = ['TP53', 'KRAS', 'PIK3CA', 'PTEN', 'EGFR', 'MYC', 'RB1', 'BRCA1', 'BRCA2', 'APC',
                   'BRAF', 'ERBB2', 'CDKN2A', 'VHL', 'MLH1', 'MSH2', 'ATM', 'CHEK2', 'PALB2', 'CDH1']
    
    for model in all_models:
        # Each model gets variants in different genes
        selected_genes = np.random.choice(cancer_genes, n_variants_per_model, replace=True)
        
        for i, gene in enumerate(selected_genes):
            # Realistic variant types
            variant_types = ['missense', 'nonsense', 'frameshift', 'splice_site', 'inframe_indel']
            variant_type = np.random.choice(variant_types, p=[0.6, 0.15, 0.1, 0.1, 0.05])
            
            # Generate realistic genomic coordinates
            chromosome = np.random.choice(range(1, 23))
            position = np.random.randint(1000000, 200000000)
            
            variant_data.append({
                'Model': model,
                'Chromosome': f'chr{chromosome}',
                'Position': position,
                'Gene': gene,
                'Variant_Type': variant_type,
                'Impact': 'HIGH' if variant_type in ['nonsense', 'frameshift'] else 'MODERATE'
            })
    
    variant_df = pd.DataFrame(variant_data)
    
    # Create metadata file for volcano plot
    metadata_df = pd.DataFrame({
        'Model': all_models,
        'Treatment_Arm': ['control'] * n_control + ['treatment'] * n_treatment,
        'Batch': [1] * (n_control // 2) + [2] * (n_control - n_control // 2) + 
                [1] * (n_treatment // 2) + [2] * (n_treatment - n_treatment // 2)
    })
    
    # Save all data files
    os.makedirs('data', exist_ok=True)
    
    expression_df.to_csv('data/expression_tpm_effective.csv', index=False)
    volume_df.to_csv('data/tumor_volumes_effective.csv', index=False)
    variant_df.to_csv('data/variants_effective.csv', index=False)
    metadata_df.to_csv('data/metadata_effective.csv', index=False)
    
    print(f"\n✓ Generated effective PDX study data:")
    print(f"  - Expression: {expression_df.shape[0]:,} genes × {expression_df.shape[1]-1} samples")
    print(f"  - Tumor volumes: {len(volume_df)} measurements")
    print(f"  - Variants: {len(variant_df)} variants")
    print(f"  - Metadata: {len(metadata_df)} samples")
    print(f"\nFiles saved:")
    print(f"  - data/expression_tpm_effective.csv")
    print(f"  - data/tumor_volumes_effective.csv") 
    print(f"  - data/variants_effective.csv")
    print(f"  - data/metadata_effective.csv")
    
    return expression_df, volume_df, variant_df, metadata_df

if __name__ == "__main__":
    generate_effective_pdx_data()