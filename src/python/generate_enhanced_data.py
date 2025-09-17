"""
Enhanced data generation script with more realistic mock data
Includes biological correlations, batch effects, and metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedPDXDataGenerator:
    """Generate realistic mock PDX data with biological correlations"""
    
    def __init__(self, n_models=20, n_genes=1000, n_timepoints=10):
        self.n_models = n_models
        self.n_genes = n_genes
        self.n_timepoints = n_timepoints
        self.timepoints = np.arange(0, n_timepoints * 3, 3)  # Every 3 days
        
    def generate_model_metadata(self):
        """Generate metadata for PDX models"""
        
        # Cancer types and their characteristics
        cancer_types = ['NSCLC', 'BRCA', 'CRC', 'PDAC']
        cancer_probs = [0.3, 0.25, 0.25, 0.2]
        
        # Patient demographics
        demographics = {
            'Age': np.random.normal(65, 12, self.n_models).astype(int),
            'Sex': np.random.choice(['M', 'F'], self.n_models, p=[0.6, 0.4]),
            'Cancer_Type': np.random.choice(cancer_types, self.n_models, p=cancer_probs),
            'Stage': np.random.choice(['II', 'III', 'IV'], self.n_models, p=[0.2, 0.3, 0.5]),
            'Prior_Treatment': np.random.choice(['Naive', 'Pretreated'], self.n_models, p=[0.3, 0.7])
        }
        
        # Assign treatment arms (ensure exactly 10+10)
        treatment_arms = ['control'] * 10 + ['treatment'] * 10
        
        # Shuffle to avoid systematic bias
        np.random.shuffle(treatment_arms)
        demographics['Treatment_Arm'] = treatment_arms[:self.n_models]
        
        # Model characteristics
        model_metadata = pd.DataFrame(demographics)
        model_metadata['Model'] = [f'PDX{i+1}' for i in range(self.n_models)]
        model_metadata['Passage'] = np.random.randint(2, 6, self.n_models)  # P2-P5
        model_metadata['Engraftment_Rate'] = np.random.beta(8, 2, self.n_models)  # High success rate
        
        # Add batch information (sequencing/experimental batches)
        model_metadata['RNA_Batch'] = np.random.randint(1, 4, self.n_models)
        model_metadata['Experiment_Date'] = pd.date_range('2024-01-01', periods=self.n_models, freq='W')
        
        return model_metadata
    
    def generate_tumor_volume_data(self, metadata):
        """Generate realistic tumor volume data with biological variation"""
        
        tumor_data = []
        
        for _, model_info in metadata.iterrows():
            model = model_info['Model']
            arm = model_info['Treatment_Arm']
            cancer_type = model_info['Cancer_Type']
            
            # Cancer type-specific growth parameters
            growth_params = {
                'NSCLC': {'base_growth': 0.08, 'variance': 0.02},
                'BRCA': {'base_growth': 0.06, 'variance': 0.015},
                'CRC': {'base_growth': 0.10, 'variance': 0.025},
                'PDAC': {'base_growth': 0.12, 'variance': 0.03}
            }
            
            base_growth = growth_params[cancer_type]['base_growth']
            variance = growth_params[cancer_type]['variance']
            
            # Treatment effect (varies by cancer type)
            treatment_effects = {
                'NSCLC': 0.6,  # 40% inhibition
                'BRCA': 0.7,   # 30% inhibition  
                'CRC': 0.5,    # 50% inhibition
                'PDAC': 0.8    # 20% inhibition (resistant)
            }
            
            if arm == 'treatment':
                growth_rate = base_growth * treatment_effects[cancer_type]
            else:
                growth_rate = base_growth
                
            # Add individual model variation
            growth_rate += np.random.normal(0, variance)
            growth_rate = max(0.01, growth_rate)  # Ensure positive growth
            
            # Initial tumor volume (varies by cancer type and individual)
            base_volumes = {'NSCLC': 120, 'BRCA': 100, 'CRC': 140, 'PDAC': 160}
            initial_volume = base_volumes[cancer_type] + np.random.normal(0, 20)
            initial_volume = max(50, initial_volume)  # Minimum volume
            
            # Generate volume time series
            for day in self.timepoints:
                # Exponential growth with noise
                expected_volume = initial_volume * np.exp(growth_rate * day)
                
                # Add measurement noise (proportional to volume)
                noise_level = 0.05  # 5% measurement error
                actual_volume = expected_volume * (1 + np.random.normal(0, noise_level))
                actual_volume = max(10, actual_volume)  # Minimum detectable volume
                
                # Add occasional missing measurements (dropout)
                if np.random.random() < 0.02:  # 2% missing rate
                    continue
                    
                tumor_data.append({
                    'Model': model,
                    'Arm': arm,
                    'Day': day,
                    'Volume_mm3': actual_volume,
                    'Cancer_Type': cancer_type,
                    'Measurement_Date': model_info['Experiment_Date'] + pd.Timedelta(days=day)
                })
        
        return pd.DataFrame(tumor_data)
    
    def generate_expression_data(self, metadata):
        """Generate realistic gene expression data with biological structure"""
        
        models = metadata['Model'].tolist()
        genes = [f'GENE{i+1}' for i in range(self.n_genes)]
        
        # Create gene categories with different expression patterns
        n_de_genes = 200  # More differentially expressed genes
        oncogenes = genes[:50]  # First 50 genes are oncogenes
        tumor_suppressors = genes[50:100]  # Next 50 are tumor suppressors
        immune_genes = genes[100:150]  # Immune response genes
        drug_targets = genes[150:200]  # Drug target pathway genes
        housekeeping_genes = genes[200:250]  # Stable housekeeping genes
        random_genes = genes[250:]  # Remaining genes
        
        expression_matrix = np.zeros((self.n_genes, len(models)))
        
        for i, model in enumerate(models):
            model_info = metadata[metadata['Model'] == model].iloc[0]
            cancer_type = model_info['Cancer_Type']
            treatment = model_info['Treatment_Arm']
            rna_batch = model_info['RNA_Batch']
            
            # Base expression levels (tissue-specific)
            cancer_base = {
                'NSCLC': np.random.lognormal(2.0, 1.5, self.n_genes),
                'BRCA': np.random.lognormal(2.2, 1.4, self.n_genes),
                'CRC': np.random.lognormal(1.8, 1.6, self.n_genes),
                'PDAC': np.random.lognormal(1.9, 1.7, self.n_genes)
            }
            
            base_expr = cancer_base[cancer_type]
            
            # Gene category effects
            gene_effects = np.ones(self.n_genes)
            
            # Oncogenes: higher in aggressive cancers
            oncogene_indices = [genes.index(g) for g in oncogenes]
            if cancer_type in ['PDAC', 'CRC']:
                gene_effects[oncogene_indices] *= 1.5
            
            # Tumor suppressors: lower in aggressive cancers
            ts_indices = [genes.index(g) for g in tumor_suppressors]
            if cancer_type in ['PDAC', 'CRC']:
                gene_effects[ts_indices] *= 0.7
            
            # Treatment effects on specific gene sets
            if treatment == 'treatment':
                # Strong treatment effects for better differential expression
                # Treatment downregulates oncogenes significantly
                gene_effects[oncogene_indices] *= np.random.normal(0.4, 0.1, len(oncogene_indices))  # Stronger effect
                
                # Treatment upregulates tumor suppressors
                gene_effects[ts_indices] *= np.random.normal(1.8, 0.2, len(ts_indices))  # Stronger effect
                
                # Drug target pathway strongly affected
                drug_target_indices = [genes.index(g) for g in drug_targets]
                gene_effects[drug_target_indices] *= np.random.normal(0.3, 0.15, len(drug_target_indices))
                
                # Immune response varies by cancer type
                immune_indices = [genes.index(g) for g in immune_genes]
                if cancer_type == 'NSCLC':  # More immunogenic
                    gene_effects[immune_indices] *= np.random.normal(2.0, 0.3, len(immune_indices))
                elif cancer_type == 'BRCA':
                    gene_effects[immune_indices] *= np.random.normal(1.5, 0.2, len(immune_indices))
                else:
                    gene_effects[immune_indices] *= np.random.normal(1.2, 0.15, len(immune_indices))
            
            # Housekeeping genes remain stable
            hk_indices = [genes.index(g) for g in housekeeping_genes]
            gene_effects[hk_indices] = np.random.normal(1.0, 0.05, len(hk_indices))
            
            # Batch effects (technical variation)
            batch_effects = {1: 1.0, 2: 1.1, 3: 0.95}
            batch_multiplier = batch_effects[rna_batch]
            
            # Final expression values
            expression_values = base_expr * gene_effects * batch_multiplier
            
            # Add noise
            noise = np.random.lognormal(0, 0.3, self.n_genes)
            expression_values *= noise
            
            # Ensure reasonable TPM range (0.1 - 10000)
            expression_values = np.clip(expression_values, 0.1, 10000)
            
            expression_matrix[:, i] = expression_values
        
        # Create DataFrame
        expression_df = pd.DataFrame(expression_matrix, index=genes, columns=models)
        
        return expression_df
    
    def generate_variant_data(self, metadata):
        """Generate realistic variant data"""
        
        variants = []
        
        # Common cancer genes and their mutation rates
        cancer_genes = {
            'TP53': 0.6, 'KRAS': 0.3, 'PIK3CA': 0.25, 'EGFR': 0.2,
            'BRAF': 0.15, 'APC': 0.4, 'BRCA1': 0.1, 'BRCA2': 0.08,
            'PTEN': 0.2, 'MYC': 0.15, 'RB1': 0.12, 'ATM': 0.18
        }
        
        # Cancer type-specific mutation patterns
        cancer_signatures = {
            'NSCLC': {'TP53': 0.8, 'KRAS': 0.4, 'EGFR': 0.3, 'BRAF': 0.05},
            'BRCA': {'TP53': 0.7, 'PIK3CA': 0.4, 'BRCA1': 0.3, 'BRCA2': 0.25},
            'CRC': {'APC': 0.8, 'TP53': 0.6, 'KRAS': 0.5, 'PIK3CA': 0.3},
            'PDAC': {'KRAS': 0.9, 'TP53': 0.7, 'CDKN2A': 0.6, 'SMAD4': 0.4}
        }
        
        for _, model_info in metadata.iterrows():
            model = model_info['Model']
            cancer_type = model_info['Cancer_Type']
            
            # Get cancer-specific mutation rates
            mutation_probs = cancer_signatures.get(cancer_type, cancer_genes)
            
            for gene, prob in mutation_probs.items():
                if np.random.random() < prob:
                    # Generate realistic variant
                    chromosome = f"chr{np.random.randint(1, 23)}"
                    position = np.random.randint(1000000, 200000000)
                    ref = np.random.choice(['A', 'T', 'G', 'C'])
                    alt = np.random.choice(['A', 'T', 'G', 'C'])
                    while alt == ref:
                        alt = np.random.choice(['A', 'T', 'G', 'C'])
                    
                    # VAF depends on tumor purity and clonality
                    # Most mutations are clonal (VAF ~0.5 * purity)
                    tumor_purity = np.random.beta(6, 2)  # High purity
                    if np.random.random() < 0.8:  # 80% clonal
                        vaf = tumor_purity * 0.5 * np.random.normal(1, 0.1)
                    else:  # 20% subclonal
                        vaf = tumor_purity * np.random.uniform(0.1, 0.4)
                    
                    vaf = np.clip(vaf, 0.01, 0.95)
                    
                    variants.append({
                        'Model': model,
                        'Gene': gene,
                        'Chr': chromosome,
                        'Pos': position,
                        'Ref': ref,
                        'Alt': alt,
                        'VAF': vaf,
                        'Cancer_Type': cancer_type,
                        'Tumor_Purity': tumor_purity
                    })
            
            # Add some random passenger mutations
            n_passengers = np.random.poisson(5)  # Average 5 passenger mutations
            for _ in range(n_passengers):
                passenger_gene = f"GENE{np.random.randint(1, 1000)}"
                
                variants.append({
                    'Model': model,
                    'Gene': passenger_gene,
                    'Chr': f"chr{np.random.randint(1, 23)}",
                    'Pos': np.random.randint(1000000, 200000000),
                    'Ref': np.random.choice(['A', 'T', 'G', 'C']),
                    'Alt': np.random.choice(['A', 'T', 'G', 'C']),
                    'VAF': np.random.uniform(0.05, 0.4),
                    'Cancer_Type': cancer_type,
                    'Tumor_Purity': np.random.beta(6, 2)
                })
        
        return pd.DataFrame(variants)
    
    def generate_all_data(self):
        """Generate all datasets and save to files"""
        
        print("Generating enhanced PDX datasets...")
        
        # Generate metadata
        metadata = self.generate_model_metadata()
        print(f"Generated metadata for {len(metadata)} models")
        
        # Generate tumor volume data
        tumor_data = self.generate_tumor_volume_data(metadata)
        print(f"Generated {len(tumor_data)} tumor volume measurements")
        
        # Generate expression data
        expression_data = self.generate_expression_data(metadata)
        print(f"Generated expression data: {expression_data.shape[0]} genes × {expression_data.shape[1]} samples")
        
        # Generate variant data
        variant_data = self.generate_variant_data(metadata)
        print(f"Generated {len(variant_data)} variants")
        
        return metadata, tumor_data, expression_data, variant_data
    
    def save_data(self, metadata, tumor_data, expression_data, variant_data, output_dir='data'):
        """Save all datasets to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main datasets
        tumor_data[['Model', 'Arm', 'Day', 'Volume_mm3']].to_csv(
            output_path / 'tumor_volumes_mock.csv', index=False)
        
        expression_data.to_csv(output_path / 'expression_tpm_mock.csv')
        
        variant_data[['Model', 'Gene', 'Chr', 'Pos', 'Ref', 'Alt', 'VAF']].to_csv(
            output_path / 'variants_mock.csv', index=False)
        
        # Save enhanced metadata
        metadata.to_csv(output_path / 'model_metadata.csv', index=False)
        tumor_data.to_csv(output_path / 'tumor_volumes_enhanced.csv', index=False)
        variant_data.to_csv(output_path / 'variants_enhanced.csv', index=False)
        
        print(f"All datasets saved to {output_path}")
    
    def create_data_summary_plots(self, metadata, tumor_data, expression_data, variant_data):
        """Create summary plots of the generated data"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sample distribution by cancer type and treatment
        ax = axes[0, 0]
        metadata_summary = metadata.groupby(['Cancer_Type', 'Treatment_Arm']).size().unstack(fill_value=0)
        metadata_summary.plot(kind='bar', ax=ax, color=['lightblue', 'orange'])
        ax.set_title('Sample Distribution by Cancer Type')
        ax.set_ylabel('Number of Models')
        ax.legend(title='Treatment')
        
        # 2. Tumor growth curves by treatment
        ax = axes[0, 1]
        for arm in tumor_data['Arm'].unique():
            subset = tumor_data[tumor_data['Arm'] == arm]
            mean_volumes = subset.groupby('Day')['Volume_mm3'].mean()
            sem_volumes = subset.groupby('Day')['Volume_mm3'].sem()
            
            ax.plot(mean_volumes.index, mean_volumes.values, 'o-', 
                   label=arm, linewidth=2, markersize=6)
            ax.fill_between(mean_volumes.index, 
                           mean_volumes - sem_volumes, 
                           mean_volumes + sem_volumes, 
                           alpha=0.3)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.set_title('Mean Tumor Growth Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Expression distribution
        ax = axes[0, 2]
        expr_flat = expression_data.values.flatten()
        ax.hist(np.log10(expr_flat + 0.1), bins=50, alpha=0.7, color='green')
        ax.set_xlabel('Log10(TPM + 0.1)')
        ax.set_ylabel('Frequency')
        ax.set_title('Expression Level Distribution')
        
        # 4. Mutation burden by cancer type
        ax = axes[1, 0]
        mut_burden = variant_data.groupby(['Model', 'Cancer_Type']).size().reset_index(name='N_mutations')
        mut_summary = mut_burden.groupby('Cancer_Type')['N_mutations'].mean().sort_values(ascending=False)
        mut_summary.plot(kind='bar', ax=ax, color='red')
        ax.set_title('Mean Mutation Burden by Cancer Type')
        ax.set_ylabel('Number of Mutations')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. VAF distribution
        ax = axes[1, 1]
        ax.hist(variant_data['VAF'], bins=30, alpha=0.7, color='purple')
        ax.set_xlabel('Variant Allele Frequency')
        ax.set_ylabel('Frequency')
        ax.set_title('VAF Distribution')
        
        # 6. Expression correlation heatmap (subset)
        ax = axes[1, 2]
        expr_subset = expression_data.iloc[:20, :]  # First 20 genes
        corr_matrix = expr_subset.T.corr()
        sns.heatmap(corr_matrix, ax=ax, cmap='RdBu_r', center=0, square=True)
        ax.set_title('Gene Expression Correlations\n(First 20 genes)')
        
        plt.tight_layout()
        return fig

def main():
    """Generate enhanced PDX datasets"""
    
    # Create generator with 20 models (10+10 design) and 1000 genes
    generator = EnhancedPDXDataGenerator(n_models=20, n_genes=1000, n_timepoints=10)
    
    # Generate all data
    metadata, tumor_data, expression_data, variant_data = generator.generate_all_data()
    
    # Save data
    generator.save_data(metadata, tumor_data, expression_data, variant_data)
    
    # Create summary plots
    fig = generator.create_data_summary_plots(metadata, tumor_data, expression_data, variant_data)
    fig.savefig('data/data_summary_plots.png', dpi=300, bbox_inches='tight')
    print("Summary plots saved to data/data_summary_plots.png")
    
    print("\n=== DATA GENERATION COMPLETE ===")
    print(f"Generated datasets with enhanced realism:")
    print(f"  • {len(metadata)} PDX models with clinical metadata")
    print(f"  • {len(tumor_data)} tumor volume measurements")  
    print(f"  • {expression_data.shape[0]} genes × {expression_data.shape[1]} samples")
    print(f"  • {len(variant_data)} genomic variants")

if __name__ == "__main__":
    main()