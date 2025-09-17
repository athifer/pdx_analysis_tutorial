"""
Advanced PDX Analysis Workflows
Comprehensive visualization and analysis functions for PDX studies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction for multiple testing
    Returns adjusted p-values (q-values) and significance boolean array
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    # Calculate adjusted p-values (q-values)
    q_values = np.zeros(n)
    
    for i in range(n-1, -1, -1):  # Start from largest p-value
        if i == n-1:
            q_values[sorted_indices[i]] = sorted_pvals[i]
        else:
            # BH correction: q = min(p * n / (i+1), q_next)
            q_val = min(sorted_pvals[i] * n / (i + 1), 
                       q_values[sorted_indices[i+1]])
            q_values[sorted_indices[i]] = q_val
    
    # Ensure q-values don't exceed 1
    q_values = np.minimum(q_values, 1.0)
    
    # Determine significance
    significant = q_values < alpha
    
    return q_values, significant

class PDXWorkflows:
    """Advanced PDX analysis workflows and visualizations"""
    
    def __init__(self, data_dir='data/', results_dir='results/'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.tumor_data = None
        self.expression_data = None
        self.variant_data = None
        
    def load_data(self):
        """Load all PDX data files"""
        try:
            self.tumor_data = pd.read_csv(f'{self.data_dir}/tumor_volumes_mock.csv')
            self.expression_data = pd.read_csv(f'{self.data_dir}/expression_tpm_mock.csv')
            self.variant_data = pd.read_csv(f'{self.data_dir}/variants_mock.csv')
            print(f"✓ Loaded data: {len(self.tumor_data)} tumor measurements, "
                  f"{self.expression_data.shape[1]-1} genes, {len(self.variant_data)} variants")
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        return True
    
    def growth_curves_analysis(self, save_plots=True):
        """
        Generate comprehensive growth curve analysis
        - Per-mouse growth curves
        - Aggregated treatment arm curves with statistics
        """
        if self.tumor_data is None:
            print("Error: Tumor data not loaded")
            return
            
        print("\n=== GROWTH CURVE ANALYSIS ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PDX Tumor Growth Analysis', fontsize=16, fontweight='bold')
        
        # 1. Individual mouse trajectories by treatment
        ax1 = axes[0, 0]
        for arm in self.tumor_data['Arm'].unique():
            arm_data = self.tumor_data[self.tumor_data['Arm'] == arm]
            for model in arm_data['Model'].unique():
                model_data = arm_data[arm_data['Model'] == model]
                alpha = 0.6 if arm == 'control' else 0.8
                color = 'red' if arm == 'control' else 'blue'
                ax1.plot(model_data['Day'], model_data['Volume_mm3'], 
                        color=color, alpha=alpha, linewidth=1)
        
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Tumor Volume (mm³)')
        ax1.set_title('Individual Mouse Growth Trajectories')
        ax1.legend(['Control', 'Treatment'])
        ax1.grid(True, alpha=0.3)
        
        # 2. Mean growth curves with error bars
        ax2 = axes[0, 1]
        summary_stats = []
        
        for day in sorted(self.tumor_data['Day'].unique()):
            day_data = self.tumor_data[self.tumor_data['Day'] == day]
            for arm in ['control', 'treatment']:
                arm_day_data = day_data[day_data['Arm'] == arm]['Volume_mm3']
                if len(arm_day_data) > 0:
                    summary_stats.append({
                        'Day': day,
                        'Arm': arm,
                        'Mean': arm_day_data.mean(),
                        'SEM': arm_day_data.sem(),
                        'N': len(arm_day_data)
                    })
        
        summary_df = pd.DataFrame(summary_stats)
        
        for arm in ['control', 'treatment']:
            arm_summary = summary_df[summary_df['Arm'] == arm]
            color = 'red' if arm == 'control' else 'blue'
            ax2.errorbar(arm_summary['Day'], arm_summary['Mean'], 
                        yerr=arm_summary['SEM'], 
                        color=color, linewidth=2, capsize=5, 
                        label=f'{arm.title()} (n={arm_summary["N"].iloc[0]})')
        
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Mean Tumor Volume (mm³)')
        ax2.set_title('Mean Growth Curves ± SEM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Growth rate analysis
        ax3 = axes[1, 0]
        growth_rates = []
        
        for model in self.tumor_data['Model'].unique():
            model_data = self.tumor_data[self.tumor_data['Model'] == model].sort_values('Day')
            if len(model_data) >= 3:
                # Calculate exponential growth rate
                log_volumes = np.log(model_data['Volume_mm3'].values + 1)
                slope, _, r_value, _, _ = stats.linregress(model_data['Day'].values, log_volumes)
                
                growth_rates.append({
                    'Model': model,
                    'Arm': model_data['Arm'].iloc[0],
                    'GrowthRate': slope,
                    'R_squared': r_value**2
                })
        
        growth_df = pd.DataFrame(growth_rates)
        
        # Box plot of growth rates
        sns.boxplot(data=growth_df, x='Arm', y='GrowthRate', ax=ax3)
        sns.swarmplot(data=growth_df, x='Arm', y='GrowthRate', ax=ax3, 
                     color='black', alpha=0.6, size=6)
        
        # Statistical test
        control_rates = growth_df[growth_df['Arm'] == 'control']['GrowthRate']
        treatment_rates = growth_df[growth_df['Arm'] == 'treatment']['GrowthRate']
        
        if len(control_rates) > 0 and len(treatment_rates) > 0:
            stat, p_value = mannwhitneyu(control_rates, treatment_rates, alternative='two-sided')
            ax3.set_title(f'Growth Rates by Treatment\n(Mann-Whitney U p={p_value:.3f})')
        else:
            ax3.set_title('Growth Rates by Treatment')
            
        ax3.set_ylabel('Growth Rate (log scale)')
        
        # 4. Final volume comparison
        ax4 = axes[1, 1]
        final_volumes = []
        
        for model in self.tumor_data['Model'].unique():
            model_data = self.tumor_data[self.tumor_data['Model'] == model]
            final_vol = model_data.loc[model_data['Day'].idxmax(), 'Volume_mm3']
            final_volumes.append({
                'Model': model,
                'Arm': model_data['Arm'].iloc[0],
                'FinalVolume': final_vol
            })
        
        final_df = pd.DataFrame(final_volumes)
        
        sns.boxplot(data=final_df, x='Arm', y='FinalVolume', ax=ax4)
        sns.swarmplot(data=final_df, x='Arm', y='FinalVolume', ax=ax4,
                     color='black', alpha=0.6, size=6)
        
        # Statistical test for final volumes
        control_final = final_df[final_df['Arm'] == 'control']['FinalVolume']
        treatment_final = final_df[final_df['Arm'] == 'treatment']['FinalVolume']
        
        if len(control_final) > 0 and len(treatment_final) > 0:
            stat, p_value = mannwhitneyu(control_final, treatment_final, alternative='two-sided')
            ax4.set_title(f'Final Tumor Volumes\n(Mann-Whitney U p={p_value:.3f})')
        else:
            ax4.set_title('Final Tumor Volumes')
            
        ax4.set_ylabel('Final Volume (mm³)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.results_dir}/growth_curves_comprehensive.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Growth curve analysis saved to {self.results_dir}/growth_curves_comprehensive.png")
        
        plt.close()
        
        # Return summary statistics
        return {
            'growth_rates': growth_df,
            'final_volumes': final_df,
            'summary_stats': summary_df
        }
    
    def waterfall_plot(self, save_plots=True):
        """
        Create waterfall plot showing drug response across PDX models
        """
        if self.tumor_data is None:
            print("Error: Tumor data not loaded")
            return
            
        print("\n=== WATERFALL PLOT ANALYSIS ===")
        
        # Calculate response metrics for each model
        response_data = []
        
        for model in self.tumor_data['Model'].unique():
            model_data = self.tumor_data[self.tumor_data['Model'] == model].sort_values('Day')
            
            if len(model_data) >= 2:
                initial_volume = model_data['Volume_mm3'].iloc[0]
                final_volume = model_data['Volume_mm3'].iloc[-1]
                
                # Calculate percent change
                percent_change = ((final_volume - initial_volume) / initial_volume) * 100
                
                # Classify response
                if percent_change <= -30:
                    response = 'Responder'
                elif percent_change <= 20:
                    response = 'Stable'
                else:
                    response = 'Progressor'
                
                response_data.append({
                    'Model': model,
                    'Arm': model_data['Arm'].iloc[0],
                    'PercentChange': percent_change,
                    'Response': response,
                    'InitialVolume': initial_volume,
                    'FinalVolume': final_volume
                })
        
        response_df = pd.DataFrame(response_data)
        
        # Create waterfall plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Sort by percent change for waterfall effect
        response_df_sorted = response_df.sort_values('PercentChange')
        
        # Color by response category
        colors = []
        for response in response_df_sorted['Response']:
            if response == 'Responder':
                colors.append('green')
            elif response == 'Stable':
                colors.append('gold')
            else:
                colors.append('red')
        
        # Waterfall plot
        bars = ax1.bar(range(len(response_df_sorted)), response_df_sorted['PercentChange'], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add horizontal lines for response thresholds
        ax1.axhline(y=-30, color='green', linestyle='--', alpha=0.7, label='Response threshold (-30%)')
        ax1.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Progression threshold (+20%)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Customize plot
        ax1.set_xlabel('PDX Models (ranked by response)')
        ax1.set_ylabel('Tumor Volume Change (%)')
        ax1.set_title('Waterfall Plot: Drug Response Across PDX Models')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add model labels
        model_labels = [f"{model}\n({arm[0].upper()})" for model, arm in 
                       zip(response_df_sorted['Model'], response_df_sorted['Arm'])]
        ax1.set_xticks(range(len(response_df_sorted)))
        ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=10)
        
        # Response summary pie chart
        response_counts = response_df['Response'].value_counts()
        colors_pie = ['green' if x=='Responder' else 'gold' if x=='Stable' else 'red' 
                     for x in response_counts.index]
        
        wedges, texts, autotexts = ax2.pie(response_counts.values, 
                                          labels=response_counts.index,
                                          colors=colors_pie, autopct='%1.1f%%',
                                          startangle=90)
        ax2.set_title('Response Distribution')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.results_dir}/waterfall_plot.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Waterfall plot saved to {self.results_dir}/waterfall_plot.png")
        
        plt.close()
        
        # Print summary
        print("\nResponse Summary:")
        for response, count in response_counts.items():
            percentage = (count / len(response_df)) * 100
            print(f"  {response}: {count} models ({percentage:.1f}%)")
        
        # Treatment arm comparison
        if len(response_df['Arm'].unique()) > 1:
            print("\nResponse by Treatment Arm:")
            arm_response = pd.crosstab(response_df['Arm'], response_df['Response'], normalize='index') * 100
            print(arm_response.round(1))
        
        return response_df
    
    def survival_analysis(self, save_plots=True):
        """
        Kaplan-Meier survival analysis for treatment outcomes
        """
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            print("Error: lifelines package not installed. Install with: pip install lifelines")
            return
        
        if self.tumor_data is None:
            print("Error: Tumor data not loaded")
            return
            
        print("\n=== SURVIVAL ANALYSIS ===")
        
        # Define survival endpoint (time to progression)
        # Progression defined as tumor volume doubling
        survival_data = []
        
        for model in self.tumor_data['Model'].unique():
            model_data = self.tumor_data[self.tumor_data['Model'] == model].sort_values('Day')
            
            if len(model_data) >= 2:
                initial_volume = model_data['Volume_mm3'].iloc[0]
                progression_threshold = initial_volume * 2  # Doubling
                
                # Find time to progression
                progressed = False
                time_to_progression = model_data['Day'].max()  # Censoring time
                
                for _, row in model_data.iterrows():
                    if row['Volume_mm3'] >= progression_threshold:
                        time_to_progression = row['Day']
                        progressed = True
                        break
                
                survival_data.append({
                    'Model': model,
                    'Arm': model_data['Arm'].iloc[0],
                    'Time': time_to_progression,
                    'Event': 1 if progressed else 0  # 1 = progression, 0 = censored
                })
        
        survival_df = pd.DataFrame(survival_data)
        
        if len(survival_df) == 0:
            print("No survival data available")
            return
        
        # Create Kaplan-Meier plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        kmf = KaplanMeierFitter()
        
        # Plot survival curves by treatment arm
        arms = survival_df['Arm'].unique()
        colors = ['red', 'blue']
        
        for i, arm in enumerate(arms):
            arm_data = survival_df[survival_df['Arm'] == arm]
            kmf.fit(arm_data['Time'], arm_data['Event'], label=f'{arm.title()} (n={len(arm_data)})')
            kmf.plot_survival_function(ax=ax1, color=colors[i], linewidth=2)
        
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Progression-Free Survival')
        ax1.set_title('Kaplan-Meier Survival Curves')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add median survival times
        for i, arm in enumerate(arms):
            arm_data = survival_df[survival_df['Arm'] == arm]
            kmf.fit(arm_data['Time'], arm_data['Event'])
            median_survival = kmf.median_survival_time_
            if not np.isnan(median_survival):
                ax1.axvline(x=median_survival, color=colors[i], linestyle='--', alpha=0.7)
        
        # Log-rank test
        if len(arms) == 2:
            control_data = survival_df[survival_df['Arm'] == arms[0]]
            treatment_data = survival_df[survival_df['Arm'] == arms[1]]
            
            results = logrank_test(control_data['Time'], treatment_data['Time'],
                                 control_data['Event'], treatment_data['Event'])
            
            ax1.text(0.02, 0.02, f'Log-rank test p={results.p_value:.3f}',
                    transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        # Survival summary table
        summary_data = []
        for arm in arms:
            arm_data = survival_df[survival_df['Arm'] == arm]
            kmf.fit(arm_data['Time'], arm_data['Event'])
            
            summary_data.append({
                'Treatment': arm.title(),
                'N': len(arm_data),
                'Events': arm_data['Event'].sum(),
                'Median_Survival': kmf.median_survival_time_,
                'Survival_at_15d': kmf.survival_function_at_times(15).values[0] if 15 <= arm_data['Time'].max() else np.nan
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Plot summary table
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=summary_df.round(2).values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax2.set_title('Survival Summary', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.results_dir}/survival_analysis.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Survival analysis saved to {self.results_dir}/survival_analysis.png")
        
        plt.close()
        
        return survival_df, summary_df
    
    def molecular_heatmaps(self, save_plots=True):
        """
        Create heatmaps showing molecular features vs drug response correlations
        """
        if self.expression_data is None or self.tumor_data is None:
            print("Error: Expression and tumor data not loaded")
            return
            
        print("\n=== MOLECULAR HEATMAPS ===")
        
        # Calculate response metrics for each model
        response_metrics = {}
        for model in self.tumor_data['Model'].unique():
            model_data = self.tumor_data[self.tumor_data['Model'] == model].sort_values('Day')
            
            if len(model_data) >= 2:
                initial = model_data['Volume_mm3'].iloc[0]
                final = model_data['Volume_mm3'].iloc[-1]
                
                response_metrics[model] = {
                    'PercentChange': ((final - initial) / initial) * 100,
                    'FinalVolume': final,
                    'Arm': model_data['Arm'].iloc[0]
                }
        
        # Prepare expression data matrix
        # Handle different possible gene column names
        gene_col = None
        if 'Gene' in self.expression_data.columns:
            gene_col = 'Gene'
        elif 'Unnamed: 0' in self.expression_data.columns:
            gene_col = 'Unnamed: 0'
        else:
            # Use index if no clear gene column
            self.expression_data = self.expression_data.set_index(self.expression_data.columns[0])
            gene_col = None
        
        if gene_col:
            expr_models = [col for col in self.expression_data.columns if col != gene_col]
            common_models = list(set(expr_models) & set(response_metrics.keys()))
            
            if len(common_models) < 3:
                print("Insufficient overlapping models for analysis")
                return
            
            # Create expression matrix
            expr_matrix = self.expression_data.set_index(gene_col)[common_models]
        else:
            expr_models = [col for col in self.expression_data.columns]
            common_models = list(set(expr_models) & set(response_metrics.keys()))
            
            if len(common_models) < 3:
                print("Insufficient overlapping models for analysis")
                return
            
            # Use existing index
            expr_matrix = self.expression_data[common_models]
        
        # Calculate correlations with response
        correlations = []
        p_values = []
        response_values = [response_metrics[model]['PercentChange'] for model in common_models]
        
        for gene in expr_matrix.index:
            gene_expr = expr_matrix.loc[gene].values
            corr, p_val = stats.pearsonr(gene_expr, response_values)
            correlations.append(corr)
            p_values.append(p_val)
        
        # Create correlation dataframe
        corr_df = pd.DataFrame({
            'Gene': expr_matrix.index,
            'Correlation': correlations,
            'P_value': p_values,
            'Significant': [p < 0.05 for p in p_values]
        })
        
        # Select top genes for heatmap
        top_genes = corr_df.nlargest(25, 'Correlation')['Gene'].tolist()
        bottom_genes = corr_df.nsmallest(25, 'Correlation')['Gene'].tolist()
        selected_genes = top_genes + bottom_genes
        
        # Create comprehensive heatmap figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 2, 0.3], width_ratios=[2, 0.5, 0.5])
        
        # Main heatmap: Expression vs Response
        ax_main = fig.add_subplot(gs[1, 0])
        
        # Prepare data for heatmap
        heatmap_data = expr_matrix.loc[selected_genes, common_models].T
        
        # Standardize expression values
        scaler = StandardScaler()
        heatmap_data_scaled = pd.DataFrame(
            scaler.fit_transform(heatmap_data),
            index=heatmap_data.index,
            columns=heatmap_data.columns
        )
        
        # Sort models by response
        model_response = [(model, response_metrics[model]['PercentChange']) 
                         for model in common_models]
        model_response.sort(key=lambda x: x[1])
        sorted_models = [model for model, _ in model_response]
        
        # Create heatmap
        sns.heatmap(heatmap_data_scaled.loc[sorted_models], 
                   cmap='RdBu_r', center=0, 
                   xticklabels=True, yticklabels=True,
                   cbar_kws={'label': 'Standardized Expression'},
                   ax=ax_main)
        
        ax_main.set_title('Gene Expression vs Drug Response\n(Models sorted by response)', 
                         fontweight='bold')
        ax_main.set_xlabel('Genes')
        ax_main.set_ylabel('PDX Models')
        
        # Response bar plot
        ax_response = fig.add_subplot(gs[1, 1])
        response_colors = ['green' if resp < -30 else 'red' if resp > 20 else 'gold' 
                          for _, resp in model_response]
        
        ax_response.barh(range(len(sorted_models)), 
                        [response_metrics[model]['PercentChange'] for model in sorted_models],
                        color=response_colors, alpha=0.7)
        ax_response.set_ylim(-0.5, len(sorted_models)-0.5)
        ax_response.set_xlabel('Response (%)')
        ax_response.set_title('Drug\nResponse')
        ax_response.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax_response.axvline(x=-30, color='green', linestyle='--', alpha=0.7)
        ax_response.axvline(x=20, color='red', linestyle='--', alpha=0.7)
        
        # Treatment arm annotation
        ax_arm = fig.add_subplot(gs[1, 2])
        arm_colors = ['blue' if response_metrics[model]['Arm'] == 'treatment' else 'red' 
                     for model in sorted_models]
        
        for i, color in enumerate(arm_colors):
            ax_arm.barh(i, 1, color=color, alpha=0.7)
        ax_arm.set_ylim(-0.5, len(sorted_models)-0.5)
        ax_arm.set_xlim(0, 1)
        ax_arm.set_title('Treatment\nArm')
        ax_arm.set_xticks([])
        
        # Gene correlation plot at top
        ax_corr = fig.add_subplot(gs[0, 0])
        gene_corrs = [corr_df[corr_df['Gene'] == gene]['Correlation'].values[0] 
                     for gene in selected_genes]
        gene_colors = ['red' if corr > 0 else 'blue' for corr in gene_corrs]
        
        ax_corr.bar(range(len(selected_genes)), gene_corrs, 
                   color=gene_colors, alpha=0.7)
        ax_corr.set_xlim(-0.5, len(selected_genes)-0.5)
        ax_corr.set_ylabel('Correlation\nwith Response')
        ax_corr.set_title('Gene-Response Correlations')
        ax_corr.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax_corr.set_xticks([])
        
        # Legend
        ax_legend = fig.add_subplot(gs[2, :])
        ax_legend.axis('off')
        
        legend_text = ("Expression heatmap shows standardized gene expression (red=high, blue=low).\n"
                      "Models sorted by drug response. Gene correlations show association with response.\n"
                      "Response: Green=Responder (<-30%), Gold=Stable (-30% to +20%), Red=Progressor (>+20%)")
        ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', 
                      fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.results_dir}/molecular_heatmaps.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Molecular heatmaps saved to {self.results_dir}/molecular_heatmaps.png")
        
        plt.close()
        
        # Print top correlations
        print("\nTop Gene-Response Correlations:")
        top_corr = corr_df.nlargest(10, 'Correlation')
        for _, row in top_corr.iterrows():
            sig_mark = "*" if row['P_value'] < 0.05 else ""
            print(f"  {row['Gene']}: r={row['Correlation']:.3f} (p={row['P_value']:.3f}){sig_mark}")
        
        print("\nBottom Gene-Response Correlations:")
        bottom_corr = corr_df.nsmallest(10, 'Correlation')
        for _, row in bottom_corr.iterrows():
            sig_mark = "*" if row['P_value'] < 0.05 else ""
            print(f"  {row['Gene']}: r={row['Correlation']:.3f} (p={row['P_value']:.3f}){sig_mark}")
        
        return corr_df
    
    def circos_plot(self, save_plots=True):
        """
        Create Circos plot for genome-wide CNVs or structural variants
        """
        if self.variant_data is None:
            print("Error: Variant data not loaded")
            return
            
        print("\n=== CIRCOS PLOT ANALYSIS ===")
        
        try:
            from matplotlib.patches import Wedge, Rectangle
            from matplotlib.collections import LineCollection
        except ImportError:
            print("Error: Required matplotlib components not available")
            return
        
        # Simulate genomic coordinates for variants
        np.random.seed(42)
        
        # Define chromosome lengths (simplified human genome)
        chromosomes = {
            'chr1': 249250621, 'chr2': 242193529, 'chr3': 198295559,
            'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
            'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
            'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
            'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
            'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
            'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
            'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
        }
        
        # Assign chromosomal positions to variants
        variant_data_extended = self.variant_data.copy()
        variant_data_extended['Chromosome'] = np.random.choice(
            list(chromosomes.keys()), len(variant_data_extended)
        )
        
        # Create variant types based on Ref/Alt columns
        def determine_variant_type(ref, alt):
            if len(ref) == 1 and len(alt) == 1:
                return 'SNV'  # Single nucleotide variant
            elif len(ref) != len(alt):
                return 'INDEL'  # Insertion or deletion
            else:
                return 'MNV'  # Multi-nucleotide variant
        
        variant_data_extended['Type'] = variant_data_extended.apply(
            lambda row: determine_variant_type(row['Ref'], row['Alt']), axis=1
        )
        
        # Assign positions within chromosomes
        positions = []
        for _, row in variant_data_extended.iterrows():
            chr_len = chromosomes[row['Chromosome']]
            pos = np.random.randint(1, chr_len)
            positions.append(pos)
        variant_data_extended['Position'] = positions
        
        # Create circular plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        # Calculate chromosome positions on circle
        total_genome_size = sum(chromosomes.values())
        chr_positions = {}
        current_angle = 0
        
        for chr_name, chr_len in chromosomes.items():
            chr_fraction = chr_len / total_genome_size
            chr_angle = chr_fraction * 2 * np.pi
            
            chr_positions[chr_name] = {
                'start_angle': current_angle,
                'end_angle': current_angle + chr_angle,
                'length': chr_len
            }
            current_angle += chr_angle
        
        # Draw chromosome ideograms
        for chr_name, chr_info in chr_positions.items():
            # Chromosome background
            theta = np.linspace(chr_info['start_angle'], chr_info['end_angle'], 100)
            r = np.ones_like(theta) * 0.9
            ax.plot(theta, r, linewidth=8, color='lightgray', alpha=0.7)
            
            # Chromosome label
            mid_angle = (chr_info['start_angle'] + chr_info['end_angle']) / 2
            ax.text(mid_angle, 1.05, chr_name.replace('chr', ''), 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Plot variants
        variant_colors = {'SNV': 'red', 'INDEL': 'blue', 'CNV': 'green'}
        
        for variant_type in variant_data_extended['Type'].unique():
            if variant_type not in variant_colors:
                variant_colors[variant_type] = np.random.choice(['orange', 'purple', 'brown'])
            
            type_variants = variant_data_extended[variant_data_extended['Type'] == variant_type]
            
            variant_angles = []
            variant_radii = []
            
            for _, variant in type_variants.iterrows():
                chr_info = chr_positions[variant['Chromosome']]
                
                # Calculate position within chromosome
                pos_fraction = variant['Position'] / chr_info['length']
                variant_angle = (chr_info['start_angle'] + 
                               pos_fraction * (chr_info['end_angle'] - chr_info['start_angle']))
                
                variant_angles.append(variant_angle)
                
                # Vary radius based on impact (simplified)
                if 'HIGH' in str(variant.get('Impact', '')):
                    radius = 0.8
                elif 'MODERATE' in str(variant.get('Impact', '')):
                    radius = 0.7
                else:
                    radius = 0.6
                
                variant_radii.append(radius)
            
            # Plot variants of this type
            ax.scatter(variant_angles, variant_radii, 
                      c=variant_colors[variant_type], 
                      s=30, alpha=0.7, label=variant_type)
        
        # Add connections between related variants (simulated)
        if len(variant_data_extended) > 10:
            # Select random variant pairs for connections
            n_connections = min(5, len(variant_data_extended) // 4)
            
            for _ in range(n_connections):
                var1, var2 = np.random.choice(len(variant_data_extended), 2, replace=False)
                
                # Get angles for both variants
                var1_data = variant_data_extended.iloc[var1]
                var2_data = variant_data_extended.iloc[var2]
                
                chr1_info = chr_positions[var1_data['Chromosome']]
                chr2_info = chr_positions[var2_data['Chromosome']]
                
                pos1_fraction = var1_data['Position'] / chr1_info['length']
                pos2_fraction = var2_data['Position'] / chr2_info['length']
                
                angle1 = (chr1_info['start_angle'] + 
                         pos1_fraction * (chr1_info['end_angle'] - chr1_info['start_angle']))
                angle2 = (chr2_info['start_angle'] + 
                         pos2_fraction * (chr2_info['end_angle'] - chr2_info['start_angle']))
                
                # Draw curved connection
                angles = np.linspace(angle1, angle2, 50)
                radii = 0.4 * np.sin(np.linspace(0, np.pi, 50)) + 0.1
                
                ax.plot(angles, radii, color='gray', alpha=0.3, linewidth=1)
        
        # Customize plot
        ax.set_ylim(0, 1.2)
        ax.set_title('Circos Plot: Genomic Variants Across PDX Models', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        # Add legend
        ax.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        
        # Add summary statistics
        summary_text = f"Total Variants: {len(variant_data_extended)}\n"
        summary_text += f"Models: {variant_data_extended['Model'].nunique()}\n"
        summary_text += f"Genes: {variant_data_extended['Gene'].nunique()}"
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=10, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save_plots:
            plt.savefig(f'{self.results_dir}/circos_plot.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Circos plot saved to {self.results_dir}/circos_plot.png")
        
        plt.close()
        
        return variant_data_extended
    
    def volcano_plot(self, save_plots=True):
        """
        Generate volcano plot for differential gene expression analysis
        Shows fold change vs -log10(p-value) for treatment vs control comparison
        """
        print("\n=== VOLCANO PLOT ANALYSIS ===")
        
        # Prepare expression data
        if self.expression_data is None:
            print("Error: Expression data not loaded")
            return
        
        # Load metadata to get treatment assignments
        metadata_path = f'{self.data_dir}/metadata_effective.csv'
        try:
            metadata = pd.read_csv(metadata_path)
        except:
            # Try alternative metadata files
            try:
                metadata_path = f'{self.data_dir}/metadata_mock.csv'
                metadata = pd.read_csv(metadata_path)
            except:
                print("Warning: Could not load metadata. Using simulated treatment assignments.")
                # Create simulated metadata based on actual model names
                models = [col for col in self.expression_data.columns if col.startswith('PDX')]
                n_models = len(models)
                n_control = n_models // 2
                metadata = pd.DataFrame({
                    'Model': models,
                    'Treatment_Arm': ['control'] * n_control + ['treatment'] * (n_models - n_control)
                })
        
        # Get gene column name
        gene_col = None
        if 'Gene' in self.expression_data.columns:
            gene_col = 'Gene'
        elif 'Unnamed: 0' in self.expression_data.columns:
            gene_col = 'Unnamed: 0'
        else:
            # Use index if no clear gene column
            expr_data = self.expression_data.set_index(self.expression_data.columns[0])
            gene_col = None
        
        if gene_col:
            expr_data = self.expression_data.set_index(gene_col)
        else:
            expr_data = self.expression_data
        
        # Get control and treatment samples
        control_models = metadata[metadata['Treatment_Arm'] == 'control']['Model'].tolist()
        treatment_models = metadata[metadata['Treatment_Arm'] == 'treatment']['Model'].tolist()
        
        # Filter for available models in expression data
        control_models = [m for m in control_models if m in expr_data.columns]
        treatment_models = [m for m in treatment_models if m in expr_data.columns]
        
        print(f"Comparing {len(treatment_models)} treatment vs {len(control_models)} control samples")
        
        if len(control_models) < 3 or len(treatment_models) < 3:
            print("Insufficient samples for differential expression analysis")
            return
        
        # Calculate statistics for each gene
        fold_changes = []
        p_values = []
        gene_names = []
        
        for gene in expr_data.index:
            control_expr = expr_data.loc[gene, control_models].values
            treatment_expr = expr_data.loc[gene, treatment_models].values
            
            # Calculate mean expression
            control_mean = np.mean(control_expr)
            treatment_mean = np.mean(treatment_expr)
            
            # Calculate log2 fold change (data is already in log2 space)
            # For log2-transformed data: log2FC = treatment_mean - control_mean
            log2_fc = treatment_mean - control_mean
            
            # Statistical test (t-test)
            try:
                _, p_val = ttest_ind(treatment_expr, control_expr)
                p_val = max(p_val, 1e-50)  # Avoid log(0)
            except:
                p_val = 1.0
            
            fold_changes.append(log2_fc)
            p_values.append(p_val)
            gene_names.append(gene)
        
        # Create results dataframe
        volcano_data = pd.DataFrame({
            'Gene': gene_names,
            'Log2FoldChange': fold_changes,
            'PValue': p_values,
            'MinusLog10PValue': [-np.log10(p) for p in p_values]
        })
        
        # Apply multiple testing correction (Benjamini-Hochberg)
        q_values, fdr_significant = benjamini_hochberg_correction(volcano_data['PValue'].values, alpha=0.05)
        volcano_data['QValue'] = q_values
        volcano_data['MinusLog10QValue'] = [-np.log10(q) for q in q_values]
        volcano_data['FDR_Significant'] = fdr_significant
        
        # Define significance thresholds
        fc_threshold = 1.0  # |log2FC| > 1
        p_threshold = 0.05  # raw p < 0.05
        fdr_threshold = 0.05  # FDR < 0.05
        
        # Classify genes using FDR-corrected significance
        volcano_data['Significant'] = (
            (np.abs(volcano_data['Log2FoldChange']) > fc_threshold) & 
            volcano_data['FDR_Significant']
        )
        volcano_data['Direction'] = np.where(
            volcano_data['Log2FoldChange'] > fc_threshold, 'Upregulated',
            np.where(volcano_data['Log2FoldChange'] < -fc_threshold, 'Downregulated', 'Not Significant')
        )
        # Use FDR significance instead of raw p-value
        volcano_data.loc[~volcano_data['FDR_Significant'], 'Direction'] = 'Not Significant'
        
        # Create volcano plot - Mobile-friendly dimensions
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color scheme
        colors = {
            'Upregulated': '#d62728',     # Red
            'Downregulated': '#2ca02c',   # Green  
            'Not Significant': '#7f7f7f'  # Gray
        }
        
        # Plot points by category - use FDR-corrected q-values for y-axis
        for direction in ['Not Significant', 'Upregulated', 'Downregulated']:
            data_subset = volcano_data[volcano_data['Direction'] == direction]
            ax.scatter(data_subset['Log2FoldChange'], data_subset['MinusLog10QValue'],
                      c=colors[direction], alpha=0.6, s=30, label=direction)
        
        # Add significance thresholds
        ax.axhline(-np.log10(fdr_threshold), color='black', linestyle='--', alpha=0.5, 
                  label=f'FDR = {fdr_threshold}')
        ax.axvline(fc_threshold, color='black', linestyle='--', alpha=0.5)
        ax.axvline(-fc_threshold, color='black', linestyle='--', alpha=0.5)
        
        # Remove gene labels for cleaner appearance
        # top_genes = volcano_data.nlargest(10, 'MinusLog10QValue')
        # for _, gene_data in top_genes.iterrows():
        #     if gene_data['Significant']:
        #         ax.annotate(gene_data['Gene'], 
        #                    xy=(gene_data['Log2FoldChange'], gene_data['MinusLog10QValue']),
        #                    xytext=(5, 5), textcoords='offset points',
        #                    fontsize=10, alpha=0.8)
        
        # Formatting - Optimized for mobile viewing
        ax.set_xlabel('Log₂ Fold Change (Treatment/Control)', fontsize=16, fontweight='bold')
        ax.set_ylabel('-Log₁₀ P-Value (FDR-corrected)', fontsize=16, fontweight='bold')
        # Remove title for cleaner mobile appearance
        # ax.set_title('Volcano Plot: Differential Gene Expression\nTreatment vs Control (FDR-corrected)', 
        #              fontsize=14, fontweight='bold', pad=20)
        
        # Increase tick label font sizes for mobile
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Legend with larger font
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=14)
        
        # Statistics summary
        n_up = len(volcano_data[volcano_data['Direction'] == 'Upregulated'])
        n_down = len(volcano_data[volcano_data['Direction'] == 'Downregulated']) 
        n_total = len(volcano_data)
        n_raw_sig = len(volcano_data[volcano_data['PValue'] < p_threshold])
        n_fdr_sig = len(volcano_data[volcano_data['FDR_Significant']])
        
        stats_text = f"Total genes: {n_total}\n"
        stats_text += f"Upregulated: {n_up} ({n_up/n_total*100:.1f}%)\n"
        stats_text += f"Downregulated: {n_down} ({n_down/n_total*100:.1f}%)\n"
        stats_text += f"Raw p<0.05: {n_raw_sig}\n"
        stats_text += f"FDR<0.05: {n_fdr_sig}\n"
        stats_text += f"Thresholds: |log₂FC| > {fc_threshold}, FDR < {fdr_threshold}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=12, va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{self.results_dir}/volcano_plot.png', 
                       dpi=300, bbox_inches='tight')
            print(f"✓ Volcano plot saved to {self.results_dir}/volcano_plot.png")
        
        # Close plot to prevent blocking
        plt.close()
        
        # Print summary
        print(f"\nDifferential Expression Summary (FDR-corrected):")
        print(f"  Upregulated genes: {n_up}")
        print(f"  Downregulated genes: {n_down}") 
        print(f"  Not significant: {n_total - n_up - n_down}")
        print(f"  Total genes analyzed: {n_total}")
        print(f"  Raw p<0.05: {n_raw_sig} ({n_raw_sig/n_total*100:.1f}%)")
        print(f"  FDR<0.05: {n_fdr_sig} ({n_fdr_sig/n_total*100:.1f}%)")
        print(f"  Multiple testing correction: Benjamini-Hochberg (FDR)")
        
        return volcano_data
    
    def run_all_workflows(self):
        """
        Run all advanced workflows in sequence
        """
        print("="*60)
        print("RUNNING ALL PDX ADVANCED WORKFLOWS")
        print("="*60)
        
        if not self.load_data():
            print("Failed to load data. Exiting.")
            return
        
        results = {}
        
        # Run all analyses
        print("\n1. Growth Curve Analysis...")
        results['growth'] = self.growth_curves_analysis()
        
        print("\n2. Waterfall Plot Analysis...")
        results['waterfall'] = self.waterfall_plot()
        
        print("\n3. Survival Analysis...")
        results['survival'] = self.survival_analysis()
        
        print("\n4. Molecular Heatmaps...")
        results['molecular'] = self.molecular_heatmaps()
        
        print("\n5. Volcano Plot...")
        results['volcano'] = self.volcano_plot()
        
        print("\n6. Circos Plot...")
        results['circos'] = self.circos_plot()
        
        print("\n" + "="*60)
        print("ALL WORKFLOWS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {self.results_dir}")
        
        return results


if __name__ == "__main__":
    # Example usage
    workflows = PDXWorkflows()
    results = workflows.run_all_workflows()