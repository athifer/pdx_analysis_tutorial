"""
Generate realistic PDX study data with moderate drug effects
More representative of typical clinical trial outcomes
"""

import pandas as pd
import numpy as np
import os

def generate_realistic_pdx_data():
    """
    Generate data for a realistic PDX study scenario:
    - n=15 per group (well-powered study)
    - Moderate effect sizes (1.2-2 fold changes, not 2-4x)
    - Realistic technical noise
    - Mixed responder patterns
    """
    
    np.random.seed(789)  # Different seed for realistic results
    
    # Study parameters
    n_control = 15
    n_treatment = 15
    n_genes = 20000
    
    print("Generating Realistic PDX Study Data with Moderate Drug Effects...")
    print(f"Study design: {n_control} control + {n_treatment} treatment PDX models")
    print(f"Genes analyzed: {n_genes:,}")
    print("Target: 2-3% FDR-significant genes (realistic for moderate effects)")
    
    # More conservative differential expression parameters
    de_fraction = 0.025  # 2.5% of genes are DE (more realistic for moderate drug)
    
    # Create gene categories with realistic DE rates
    gene_categories = []
    for i in range(n_genes):
        if i < 2000:  # First 2000 are oncogenes/tumor suppressors
            if np.random.random() < 0.08:  # 8% DE rate for cancer genes (reduced from 12%)
                gene_categories.append('cancer_gene')
            else:
                gene_categories.append('background')
        elif i < 5000:  # Next 3000 are immune/pathway genes
            if np.random.random() < 0.04:  # 4% DE rate for pathway genes
                gene_categories.append('pathway_gene')
            else:
                gene_categories.append('background')
        else:  # Remaining are background
            if np.random.random() < 0.015:  # 1.5% DE rate for background
                gene_categories.append('background_de')
            else:
                gene_categories.append('background')
    
    # Generate expression data with realistic distributions
    base_expression = 8.0  # Log2 TPM baseline
    
    # Control samples (realistic technical variance)
    control_noise = 0.5  # Typical technical noise in PDX studies
    control_data = np.random.normal(base_expression, control_noise, (n_genes, n_control))
    
    # Treatment samples (with biological variance)
    treatment_data = np.random.normal(base_expression, control_noise, (n_genes, n_treatment))
    
    # Add realistic differential expression with moderate effect sizes
    de_gene_count = 0
    upregulated_genes = []
    downregulated_genes = []
    
    for i, category in enumerate(gene_categories):
        if category != 'background':
            # Moderate, realistic effect sizes
            if category == 'cancer_gene':
                # Cancer genes: small to moderate effects (1.2-2.5 fold)
                effect_size = np.random.uniform(0.6, 1.3)  # log2 fold change
            elif category == 'pathway_gene':
                # Pathway genes: small effects (1.1-1.8 fold)
                effect_size = np.random.uniform(0.4, 0.9)
            else:
                # Background DE: very small effects (1.1-1.4 fold)
                effect_size = np.random.uniform(0.3, 0.7)
            
            # 55% upregulated, 45% downregulated (more balanced)
            if np.random.random() < 0.55:
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
    expression_df = expression_df.reset_index()
    
    # Generate realistic tumor volume data with moderate treatment effect
    volume_data = []
    
    for i, model in enumerate(all_models):
        arm = 'control' if model.startswith('PDX_C') else 'treatment'
        
        # Initial volume (similar between arms)
        initial_volume = np.random.normal(150, 35)
        initial_volume = max(60, initial_volume)  # Minimum 60 mm3
        
        # Growth parameters - more realistic
        if arm == 'control':
            # Control: steady growth with variation
            growth_rate = np.random.normal(0.14, 0.04)  # Control growth
        else:
            # Treatment: moderate growth inhibition, NOT shrinkage
            # Some models respond well, others poorly (mixed response)
            if np.random.random() < 0.4:  # 40% good responders
                growth_rate = np.random.normal(0.05, 0.03)  # Slow growth
            elif np.random.random() < 0.7:  # 30% moderate responders
                growth_rate = np.random.normal(0.08, 0.03)  # Moderate growth
            else:  # 30% poor responders
                growth_rate = np.random.normal(0.12, 0.03)  # Similar to control
        
        # Generate time series (0, 7, 14, 21, 28 days)
        for day in [0, 7, 14, 21, 28]:
            if day == 0:
                volume = initial_volume
            else:
                # Exponential growth with realistic noise
                volume = initial_volume * np.exp(growth_rate * day / 7)
                volume *= np.random.normal(1.0, 0.12)  # Add measurement noise
                volume = max(20, volume)  # Minimum measurable volume
            
            volume_data.append({
                'Model': model,
                'Day': day,
                'Volume_mm3': round(volume, 1),
                'Arm': arm
            })
    
    volume_df = pd.DataFrame(volume_data)
    
    # Generate variant data (same as before - this is realistic)
    n_variants_per_model = 25
    variant_data = []
    
    cancer_genes = ['TP53', 'KRAS', 'PIK3CA', 'PTEN', 'EGFR', 'MYC', 'RB1', 'BRCA1', 'BRCA2', 'APC',
                   'BRAF', 'ERBB2', 'CDKN2A', 'VHL', 'MLH1', 'MSH2', 'ATM', 'CHEK2', 'PALB2', 'CDH1']
    
    for model in all_models:
        selected_genes = np.random.choice(cancer_genes, n_variants_per_model, replace=True)
        
        for i, gene in enumerate(selected_genes):
            variant_types = ['missense', 'nonsense', 'frameshift', 'splice_site', 'inframe_indel']
            variant_type = np.random.choice(variant_types, p=[0.6, 0.15, 0.1, 0.1, 0.05])
            
            chromosome = np.random.choice(range(1, 23))
            position = np.random.randint(1000000, 200000000)
            
            variant_data.append({
                'Model': model,
                'Chromosome': f'chr{chromosome}',
                'Position': position,
                'Gene': gene,
                'Ref': np.random.choice(['A', 'T', 'G', 'C']),
                'Alt': np.random.choice(['A', 'T', 'G', 'C']),
                'Variant_Type': variant_type,
                'Impact': 'HIGH' if variant_type in ['nonsense', 'frameshift'] else 'MODERATE'
            })
    
    variant_df = pd.DataFrame(variant_data)
    
    # Create sample metadata
    metadata = []
    for model in all_models:
        arm = 'control' if model.startswith('PDX_C') else 'treatment'
        metadata.append({
            'Sample': model,
            'Arm': arm,
            'Cancer_Type': np.random.choice(['NSCLC', 'BRCA', 'CRC', 'PDAC'], p=[0.4, 0.25, 0.2, 0.15]),
            'Patient_Age': np.random.randint(45, 80),
            'Stage': np.random.choice(['II', 'III', 'IV'], p=[0.2, 0.3, 0.5])
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Save files with "realistic" suffix
    expression_df.to_csv('data/expression_tpm_realistic.csv', index=False)
    volume_df.to_csv('data/tumor_volumes_realistic.csv', index=False)
    variant_df.to_csv('data/variants_realistic.csv', index=False)
    metadata_df.to_csv('data/metadata_realistic.csv', index=False)
    
    print("\n✓ Generated realistic PDX study data with moderate drug effects:")
    print(f"  - Expression: {len(gene_names):,} genes × {len(all_models)} samples")
    print(f"  - Tumor volumes: {len(volume_df)} measurements")
    print(f"  - Variants: {len(variant_df)} variants")
    print(f"  - Metadata: {len(all_models)} samples")
    print("\nFiles saved:")
    print("  - data/expression_tpm_realistic.csv")
    print("  - data/tumor_volumes_realistic.csv")
    print("  - data/variants_realistic.csv")
    print("  - data/metadata_realistic.csv")
    print("\nExpected results:")
    print("  - Modest treatment effect in tumor volumes")
    print("  - ~500-600 FDR-significant genes (2.5% discovery rate)")
    print("  - Mixed responder patterns (realistic for clinical trials)")

if __name__ == "__main__":
    generate_realistic_pdx_data()