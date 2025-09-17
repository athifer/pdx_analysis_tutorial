"""
Simulate different PDX study scenarios to demonstrate
typical differential expression results
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction for multiple testing
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    q_values = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        if i == n-1:
            q_values[sorted_indices[i]] = sorted_pvals[i]
        else:
            q_val = min(sorted_pvals[i] * n / (i + 1), 
                       q_values[sorted_indices[i+1]])
            q_values[sorted_indices[i]] = q_val
    
    q_values = np.minimum(q_values, 1.0)
    significant = q_values < alpha
    
    return q_values, significant

def simulate_pdx_scenario(n_control, n_treatment, effect_size="moderate", noise_level="typical"):
    """
    Simulate PDX study with different parameters
    
    Parameters:
    - n_control, n_treatment: Sample sizes
    - effect_size: "small", "moderate", "large", "very_large"
    - noise_level: "low", "typical", "high"
    """
    
    np.random.seed(42)  # For reproducibility
    n_genes = 20000
    
    # Define effect sizes (log2 fold changes)
    effect_params = {
        "small": {"mean_fc": 0.8, "max_fc": 2.0, "de_fraction": 0.05},
        "moderate": {"mean_fc": 1.5, "max_fc": 3.0, "de_fraction": 0.08},
        "large": {"mean_fc": 2.0, "max_fc": 4.0, "de_fraction": 0.12},
        "very_large": {"mean_fc": 3.0, "max_fc": 6.0, "de_fraction": 0.15}
    }
    
    # Define noise levels (standard deviations)
    noise_params = {
        "low": 0.3,
        "typical": 0.5,
        "high": 0.8
    }
    
    params = effect_params[effect_size]
    noise_sd = noise_params[noise_level]
    
    # Determine which genes are DE
    n_de_genes = int(n_genes * params["de_fraction"])
    de_indices = np.random.choice(n_genes, n_de_genes, replace=False)
    
    # Generate expression data
    control_data = np.random.normal(8, noise_sd, (n_genes, n_control))
    treatment_data = np.random.normal(8, noise_sd, (n_genes, n_treatment))
    
    # Add differential expression to selected genes
    for i, gene_idx in enumerate(de_indices):
        # Random fold change within range
        if i < n_de_genes * 0.6:  # 60% upregulated
            fold_change = np.random.uniform(params["mean_fc"], params["max_fc"])
        else:  # 40% downregulated
            fold_change = -np.random.uniform(params["mean_fc"], params["max_fc"])
        
        treatment_data[gene_idx, :] += fold_change
    
    # Calculate statistics
    p_values = []
    fold_changes = []
    
    for gene_idx in range(n_genes):
        try:
            _, p_val = ttest_ind(treatment_data[gene_idx, :], control_data[gene_idx, :])
            p_val = max(p_val, 1e-50)
        except:
            p_val = 1.0
        
        control_mean = np.mean(control_data[gene_idx, :])
        treatment_mean = np.mean(treatment_data[gene_idx, :])
        
        if control_mean > 0:
            fc = treatment_mean - control_mean  # Log2 fold change (already in log space)
        else:
            fc = 0
        
        p_values.append(p_val)
        fold_changes.append(fc)
    
    # Apply FDR correction
    q_values, fdr_significant = benjamini_hochberg_correction(p_values)
    
    # Count results
    raw_sig = sum(p < 0.05 for p in p_values)
    fdr_sig = sum(fdr_significant)
    
    return {
        'n_control': n_control,
        'n_treatment': n_treatment,
        'effect_size': effect_size,
        'noise_level': noise_level,
        'true_de_genes': n_de_genes,
        'raw_significant': raw_sig,
        'fdr_significant': fdr_sig,
        'raw_percentage': raw_sig / n_genes * 100,
        'fdr_percentage': fdr_sig / n_genes * 100,
        'power': fdr_sig / n_de_genes * 100 if n_de_genes > 0 else 0
    }

def run_pdx_study_comparison():
    """
    Compare different PDX study scenarios
    """
    scenarios = [
        # Typical small studies
        (5, 5, "small", "typical"),
        (8, 8, "moderate", "typical"),
        (10, 10, "moderate", "typical"),
        
        # Medium-sized studies
        (15, 15, "moderate", "typical"),
        (20, 20, "moderate", "typical"),
        
        # Large studies (rare)
        (30, 30, "moderate", "typical"),
        (50, 50, "moderate", "typical"),
        
        # Effective studies (optimized)
        (10, 10, "large", "low"),
        (15, 15, "large", "typical"),
        (25, 25, "large", "typical"),
        
        # Very effective studies
        (15, 15, "very_large", "low"),
        (20, 20, "very_large", "typical"),
    ]
    
    results = []
    
    print("PDX Study Scenarios Comparison")
    print("=" * 80)
    print(f"{'Study Type':<20} {'n':<8} {'Effect':<12} {'Noise':<10} {'Raw':<8} {'FDR':<8} {'Power':<8}")
    print("-" * 80)
    
    for n_ctrl, n_treat, effect, noise in scenarios:
        result = simulate_pdx_scenario(n_ctrl, n_treat, effect, noise)
        results.append(result)
        
        # Determine study type
        total_n = n_ctrl + n_treat
        if total_n <= 16:
            study_type = "Small Study"
        elif total_n <= 40:
            study_type = "Medium Study"
        elif effect == "large" or effect == "very_large":
            study_type = "Effective Study"
        else:
            study_type = "Large Study"
        
        print(f"{study_type:<20} {total_n:<8} {effect:<12} {noise:<10} "
              f"{result['raw_significant']:<8} {result['fdr_significant']:<8} "
              f"{result['power']:.1f}%")
    
    return results

if __name__ == "__main__":
    results = run_pdx_study_comparison()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("- Raw: Number of genes with p < 0.05")
    print("- FDR: Number of genes with FDR < 0.05 (multiple testing corrected)")
    print("- Power: Percentage of true DE genes successfully detected")
    print("- Our current study (10+10, moderate effect): Typical small study scenario")
    print("- Effective studies need: larger effects, more samples, or lower noise")