#!/usr/bin/env python3
"""
PDX Variant Analysis Script

This script analyzes genomic variants in PDX models and correlates them with 
treatment response data. It demonstrates typical workflows for variant 
annotation, filtering, and biomarker discovery.

Author: PDX Analysis Tutorial
Date: 2025
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from python.plotting import create_publication_plot, save_plot
    from python.preprocessing import load_and_validate_data
except ImportError:
    print("Warning: Could not import custom modules. Some functions may not be available.")
    
    def create_publication_plot():
        """Fallback plotting function"""
        return plt.figure(figsize=(10, 6))
    
    def save_plot(fig, filename, output_dir):
        """Fallback save function"""
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        
    def load_and_validate_data(filepath, required_columns):
        """Fallback data loading function"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        return pd.read_csv(filepath)


class VariantAnalyzer:
    """Class for analyzing genomic variants in PDX models"""
    
    def __init__(self, data_dir="data", output_dir="results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.variants_df = None
        self.tumor_volumes_df = None
        self.expression_df = None
        
    def load_data(self):
        """Load variant, tumor volume, and expression data"""
        print("Loading variant analysis data...")
        
        # Load variants data
        variants_file = self.data_dir / "variants_mock.csv"
        if not variants_file.exists():
            # Try enhanced data
            variants_file = self.data_dir / "variants_enhanced.csv"
            
        if variants_file.exists():
            self.variants_df = pd.read_csv(variants_file)
            print(f"Loaded {len(self.variants_df)} variant records")
        else:
            raise FileNotFoundError("No variant data file found")
            
        # Load tumor volume data for response correlation
        tumor_file = self.data_dir / "tumor_volumes_mock.csv"
        if tumor_file.exists():
            self.tumor_volumes_df = pd.read_csv(tumor_file)
            print(f"Loaded tumor volume data for {len(self.tumor_volumes_df['Model'].unique())} models")
            
        # Load expression data for pathway analysis
        expr_file = self.data_dir / "expression_tpm_mock.csv"
        if expr_file.exists():
            self.expression_df = pd.read_csv(expr_file)
            print(f"Loaded expression data for {len(self.expression_df.columns)-1} genes")
            
    def analyze_variant_distribution(self):
        """Analyze the distribution of variants across models and genes"""
        print("\n=== VARIANT DISTRIBUTION ANALYSIS ===")
        
        if self.variants_df is None:
            print("No variant data available")
            return
            
        # Basic statistics
        print(f"Total variants: {len(self.variants_df)}")
        print(f"Unique models: {self.variants_df['Model'].nunique()}")
        print(f"Unique genes: {self.variants_df['Gene'].nunique()}")
        
        # Variant type distribution
        if 'Type' in self.variants_df.columns:
            variant_types = self.variants_df['Type'].value_counts()
            print(f"\nVariant types:")
            for vtype, count in variant_types.items():
                print(f"  {vtype}: {count}")
                
        # Most frequently mutated genes
        gene_counts = self.variants_df['Gene'].value_counts().head(10)
        print(f"\nTop 10 most frequently mutated genes:")
        for gene, count in gene_counts.items():
            print(f"  {gene}: {count} models")
            
        return {
            'total_variants': len(self.variants_df),
            'unique_models': self.variants_df['Model'].nunique(),
            'unique_genes': self.variants_df['Gene'].nunique(),
            'top_genes': gene_counts
        }
    
    def correlate_variants_with_response(self):
        """Correlate variant status with treatment response"""
        print("\n=== VARIANT-RESPONSE CORRELATION ===")
        
        if self.variants_df is None or self.tumor_volumes_df is None:
            print("Insufficient data for response correlation")
            return pd.DataFrame()
            
        # For this mock data structure, models are in different arms
        # We'll compare treatment vs control models directly
        
        # Separate models by arm
        control_models = set(self.tumor_volumes_df[self.tumor_volumes_df['Arm'] == 'control']['Model'].unique())
        treatment_models = set(self.tumor_volumes_df[self.tumor_volumes_df['Arm'] == 'treatment']['Model'].unique())
        
        print(f"Control models: {len(control_models)}")
        print(f"Treatment models: {len(treatment_models)}")
        
        # Calculate final volumes for each model
        model_responses = {}
        
        for model in self.tumor_volumes_df['Model'].unique():
            model_data = self.tumor_volumes_df[self.tumor_volumes_df['Model'] == model]
            final_volume = model_data['Volume_mm3'].iloc[-1]
            arm = model_data['Arm'].iloc[0]
            
            model_responses[model] = {
                'FinalVolume': final_volume,
                'Arm': arm,
                'IsResponder': arm == 'treatment' and final_volume < 300  # Simple threshold
            }
        
        # Convert to DataFrame
        response_df = pd.DataFrame.from_dict(model_responses, orient='index')
        response_df.index.name = 'Model'
        response_df = response_df.reset_index()
        
        print(f"Response classification based on final volume threshold:")
        print(f"Total responders: {response_df['IsResponder'].sum()}")
        print(f"Total models: {len(response_df)}")
        
        # Analyze variant-response associations
        variant_response_results = []
        
        for gene in self.variants_df['Gene'].unique():
            gene_models = set(self.variants_df[self.variants_df['Gene'] == gene]['Model'].unique())
            
            # Calculate response rates
            mutated_models = response_df[response_df['Model'].isin(gene_models)]
            wild_type_models = response_df[~response_df['Model'].isin(gene_models)]
            
            if len(mutated_models) > 0 and len(wild_type_models) > 0:
                mut_response_rate = mutated_models['IsResponder'].mean()
                wt_response_rate = wild_type_models['IsResponder'].mean()
                
                variant_response_results.append({
                    'Gene': gene,
                    'MutatedModels': len(mutated_models),
                    'WildTypeModels': len(wild_type_models),
                    'MutatedResponseRate': mut_response_rate,
                    'WildTypeResponseRate': wt_response_rate,
                    'ResponseDifference': mut_response_rate - wt_response_rate
                })
        
        variant_response_df = pd.DataFrame(variant_response_results)
        if len(variant_response_df) > 0:
            variant_response_df = variant_response_df.sort_values('ResponseDifference', key=abs, ascending=False)
            
            print(f"Analyzed {len(variant_response_df)} genes for response association")
            print("\nTop genes with strongest response associations:")
            
            for _, row in variant_response_df.head(5).iterrows():
                print(f"  {row['Gene']}: {row['ResponseDifference']:.3f} difference "
                      f"({row['MutatedModels']} mut vs {row['WildTypeModels']} WT)")
        else:
            print("No variant-response associations found")
            
        return variant_response_df
    
    def create_variant_plots(self):
        """Create visualization plots for variant analysis"""
        print("\n=== CREATING VARIANT VISUALIZATIONS ===")
        
        if self.variants_df is None:
            print("No variant data available for plotting")
            return
        
        # 1. Variant frequency plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top mutated genes
        gene_counts = self.variants_df['Gene'].value_counts().head(15)
        gene_counts.plot(kind='barh', ax=axes[0,0])
        axes[0,0].set_title('Top 15 Most Frequently Mutated Genes')
        axes[0,0].set_xlabel('Number of Models')
        
        # Variant type distribution (if available)
        if 'Type' in self.variants_df.columns:
            type_counts = self.variants_df['Type'].value_counts()
            axes[0,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[0,1].set_title('Variant Type Distribution')
        else:
            axes[0,1].text(0.5, 0.5, 'Variant type\ndata not available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Variant Type Distribution')
        
        # Variants per model
        model_counts = self.variants_df['Model'].value_counts()
        axes[1,0].hist(model_counts.values, bins=10, edgecolor='black', alpha=0.7)
        axes[1,0].set_title('Distribution of Variants per Model')
        axes[1,0].set_xlabel('Number of Variants')
        axes[1,0].set_ylabel('Number of Models')
        
        # Sample variant positions (if chromosome data available)
        if 'Chromosome' in self.variants_df.columns and 'Position' in self.variants_df.columns:
            # Simple chromosome plot
            chr_counts = self.variants_df['Chromosome'].value_counts().sort_index()
            chr_counts.plot(kind='bar', ax=axes[1,1])
            axes[1,1].set_title('Variants by Chromosome')
            axes[1,1].set_xlabel('Chromosome')
            axes[1,1].set_ylabel('Number of Variants')
            axes[1,1].tick_params(axis='x', rotation=45)
        else:
            axes[1,1].text(0.5, 0.5, 'Chromosome position\ndata not available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Genomic Distribution')
        
        plt.tight_layout()
        
        # Save plot
        output_file = self.output_dir / "variant_analysis_summary.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Variant analysis plots saved to: {output_file}")
        
        return str(output_file)
    
    def run_complete_analysis(self):
        """Run the complete variant analysis workflow"""
        print("Starting PDX Variant Analysis")
        print("=" * 50)
        
        try:
            # Load data
            self.load_data()
            
            # Run analyses
            distribution_results = self.analyze_variant_distribution()
            response_results = self.correlate_variants_with_response()
            plot_file = self.create_variant_plots()
            
            # Summary
            print("\n" + "=" * 50)
            print("VARIANT ANALYSIS COMPLETE")
            print("=" * 50)
            print(f"Analysis results saved to: {self.output_dir}")
            if plot_file:
                print(f"Visualizations saved to: {plot_file}")
            
            return {
                'distribution': distribution_results,
                'response_correlation': response_results,
                'plots': plot_file
            }
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise


def main():
    """Main function to run variant analysis"""
    # Change to project root if needed
    if not os.path.exists('data'):
        if os.path.exists('../../data'):
            os.chdir('../..')
        else:
            print("Error: Cannot find data directory")
            sys.exit(1)
    
    # Create analyzer and run analysis
    analyzer = VariantAnalyzer()
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()