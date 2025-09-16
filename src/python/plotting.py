"""
Comprehensive plotting functions for PDX analysis
Publication-ready figures with interactive capabilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")

class PDXPlotter:
    """Class for creating publication-ready PDX analysis plots"""
    
    def __init__(self, style='publication', figsize=(10, 6), dpi=300):
        """Initialize plotter with style settings"""
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Color palettes
        self.colors = {
            'treatment': '#E74C3C',  # Red
            'control': '#3498DB',    # Blue
            'significant': '#E74C3C',
            'non_significant': '#BDC3C7',
            'upregulated': '#E74C3C',
            'downregulated': '#3498DB',
            'neutral': '#95A5A6'
        }
        
        # Set default figure parameters
        if style == 'publication':
            plt.rcParams.update({
                'figure.figsize': figsize,
                'figure.dpi': dpi,
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 11,
                'font.family': 'DejaVu Sans',
                'axes.linewidth': 1.2,
                'xtick.major.width': 1.2,
                'ytick.major.width': 1.2,
                'grid.linewidth': 0.8,
                'lines.linewidth': 2
            })
    
    def plot_tumor_growth_curves(self, tumor_data, save_path=None, individual_curves=True):
        """Create comprehensive tumor growth curve plots"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Mean growth curves with error bars
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_mean_growth_curves(tumor_data, ax1)
        
        # 2. Individual growth curves (spaghetti plot)
        ax2 = fig.add_subplot(gs[0, 1])
        if individual_curves:
            self._plot_individual_curves(tumor_data, ax2)
        
        # 3. Growth rate distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_growth_rate_distribution(tumor_data, ax3)
        
        # 4. Volume at specific timepoints
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_volume_boxplots(tumor_data, ax4)
        
        # 5. Percent change from baseline
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_percent_change(tumor_data, ax5)
        
        # 6. Waterfall plot (final volume change)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_waterfall(tumor_data, ax6)
        
        plt.suptitle('Comprehensive Tumor Growth Analysis', fontsize=16, y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def _plot_mean_growth_curves(self, tumor_data, ax):
        """Plot mean growth curves with confidence intervals"""
        
        for arm in tumor_data['Arm'].unique():
            subset = tumor_data[tumor_data['Arm'] == arm]
            
            # Calculate mean and SEM for each timepoint
            stats = subset.groupby('Day')['Volume_mm3'].agg(['mean', 'sem', 'count']).reset_index()
            
            # Plot mean curve
            ax.plot(stats['Day'], stats['mean'], 'o-', 
                   color=self.colors[arm], label=f'{arm.title()} (n={stats["count"].iloc[0]})',
                   linewidth=2, markersize=6)
            
            # Add confidence interval
            ci = 1.96 * stats['sem']  # 95% CI
            ax.fill_between(stats['Day'], 
                           stats['mean'] - ci, 
                           stats['mean'] + ci,
                           color=self.colors[arm], alpha=0.2)
        
        ax.set_xlabel('Days After Treatment Start')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.set_title('Mean Growth Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_individual_curves(self, tumor_data, ax):
        """Plot individual growth curves (spaghetti plot)"""
        
        for arm in tumor_data['Arm'].unique():
            subset = tumor_data[tumor_data['Arm'] == arm]
            
            for model in subset['Model'].unique():
                model_data = subset[subset['Model'] == model].sort_values('Day')
                ax.plot(model_data['Day'], model_data['Volume_mm3'], 
                       color=self.colors[arm], alpha=0.6, linewidth=1)
        
        # Add legend with custom patches
        legend_elements = [Patch(facecolor=self.colors['control'], alpha=0.6, label='Control'),
                          Patch(facecolor=self.colors['treatment'], alpha=0.6, label='Treatment')]
        ax.legend(handles=legend_elements)
        
        ax.set_xlabel('Days After Treatment Start')
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.set_title('Individual Growth Curves')
        ax.grid(True, alpha=0.3)
    
    def _plot_growth_rate_distribution(self, tumor_data, ax):
        """Plot distribution of growth rates"""
        
        # Calculate growth rates for each model
        growth_rates = []
        
        for model, group in tumor_data.groupby('Model'):
            if len(group) >= 3:  # Need at least 3 points
                group = group.sort_values('Day')
                log_volumes = np.log(group['Volume_mm3'])
                days = group['Day']
                
                # Linear regression to get growth rate
                coeffs = np.polyfit(days, log_volumes, 1)
                growth_rate = coeffs[0]
                
                growth_rates.append({
                    'Model': model,
                    'Arm': group['Arm'].iloc[0],
                    'GrowthRate': growth_rate
                })
        
        growth_df = pd.DataFrame(growth_rates)
        
        # Create violin plot
        arms = growth_df['Arm'].unique()
        positions = range(len(arms))
        
        for i, arm in enumerate(arms):
            subset = growth_df[growth_df['Arm'] == arm]['GrowthRate']
            parts = ax.violinplot([subset], positions=[i], widths=0.6)
            
            # Color the violin plots
            for pc in parts['bodies']:
                pc.set_facecolor(self.colors[arm])
                pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([arm.title() for arm in arms])
        ax.set_ylabel('Growth Rate (log volume/day)')
        ax.set_title('Growth Rate Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_volume_boxplots(self, tumor_data, ax):
        """Plot volume distributions at key timepoints"""
        
        # Select key timepoints
        key_timepoints = [0, 14, 28]  # Baseline, mid-point, endpoint
        available_timepoints = sorted(tumor_data['Day'].unique())
        
        # Find closest available timepoints
        plot_timepoints = []
        for target in key_timepoints:
            closest = min(available_timepoints, key=lambda x: abs(x - target))
            if closest not in plot_timepoints:
                plot_timepoints.append(closest)
        
        # Prepare data for boxplot
        plot_data = []
        plot_labels = []
        
        for day in plot_timepoints:
            for arm in tumor_data['Arm'].unique():
                subset = tumor_data[(tumor_data['Day'] == day) & (tumor_data['Arm'] == arm)]
                if len(subset) > 0:
                    plot_data.append(subset['Volume_mm3'])
                    plot_labels.append(f'{arm.title()}\nDay {day}')
        
        # Create boxplot
        box_parts = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
        
        # Color boxes
        colors = []
        for label in plot_labels:
            if 'Control' in label:
                colors.append(self.colors['control'])
            else:
                colors.append(self.colors['treatment'])
        
        for patch, color in zip(box_parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Tumor Volume (mm³)')
        ax.set_title('Volume Distribution at Key Timepoints')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_percent_change(self, tumor_data, ax):
        """Plot percent change from baseline"""
        
        # Calculate percent change for each model
        pct_change_data = []
        
        for model, group in tumor_data.groupby('Model'):
            group = group.sort_values('Day')
            baseline = group[group['Day'] == group['Day'].min()]['Volume_mm3'].iloc[0]
            
            for _, row in group.iterrows():
                pct_change = ((row['Volume_mm3'] - baseline) / baseline) * 100
                pct_change_data.append({
                    'Model': model,
                    'Arm': row['Arm'],
                    'Day': row['Day'],
                    'PctChange': pct_change
                })
        
        pct_df = pd.DataFrame(pct_change_data)
        
        # Plot mean percent change
        for arm in pct_df['Arm'].unique():
            subset = pct_df[pct_df['Arm'] == arm]
            mean_pct = subset.groupby('Day')['PctChange'].mean()
            sem_pct = subset.groupby('Day')['PctChange'].sem()
            
            ax.plot(mean_pct.index, mean_pct.values, 'o-',
                   color=self.colors[arm], label=arm.title(), linewidth=2)
            
            # Add error bars
            ax.fill_between(mean_pct.index,
                           mean_pct - 1.96 * sem_pct,
                           mean_pct + 1.96 * sem_pct,
                           color=self.colors[arm], alpha=0.2)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Days After Treatment Start')
        ax.set_ylabel('Percent Change from Baseline (%)')
        ax.set_title('Tumor Growth: Percent Change')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_waterfall(self, tumor_data, ax):
        """Create waterfall plot of final volume changes"""
        
        # Calculate final volume change for each model
        final_changes = []
        
        for model, group in tumor_data.groupby('Model'):
            group = group.sort_values('Day')
            initial = group['Volume_mm3'].iloc[0]
            final = group['Volume_mm3'].iloc[-1]
            pct_change = ((final - initial) / initial) * 100
            
            final_changes.append({
                'Model': model,
                'Arm': group['Arm'].iloc[0],
                'PctChange': pct_change
            })
        
        change_df = pd.DataFrame(final_changes)
        
        # Sort by percent change
        change_df = change_df.sort_values('PctChange', ascending=False)
        
        # Create waterfall plot
        x_pos = range(len(change_df))
        colors = [self.colors[arm] for arm in change_df['Arm']]
        
        bars = ax.bar(x_pos, change_df['PctChange'], color=colors, alpha=0.7)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Customize
        ax.set_xlabel('PDX Models')
        ax.set_ylabel('Final Volume Change (%)')
        ax.set_title('Waterfall Plot: Final Volume Changes')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(change_df['Model'], rotation=45)
        
        # Add legend
        legend_elements = [Patch(facecolor=self.colors['control'], label='Control'),
                          Patch(facecolor=self.colors['treatment'], label='Treatment')]
        ax.legend(handles=legend_elements)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_expression_analysis(self, expression_data, deg_results, metadata, save_path=None):
        """Create comprehensive expression analysis plots"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 4, hspace=0.3, wspace=0.3)
        
        # 1. Volcano plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_volcano(deg_results, ax1)
        
        # 2. MA plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_ma(deg_results, ax2)
        
        # 3. Expression heatmap
        ax3 = fig.add_subplot(gs[0, 2:])
        self._plot_expression_heatmap(expression_data, deg_results, metadata, ax3)
        
        # 4. PCA plot
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_pca(expression_data, metadata, ax4)
        
        # 5. Expression distribution
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_expression_distribution(expression_data, ax5)
        
        # 6. Top genes barplot
        ax6 = fig.add_subplot(gs[1, 2:])
        self._plot_top_genes(deg_results, ax6)
        
        plt.suptitle('Comprehensive Gene Expression Analysis', fontsize=16, y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig
    
    def _plot_volcano(self, deg_results, ax):
        """Create volcano plot"""
        
        # Prepare data
        plot_data = deg_results.dropna(subset=['Log2FoldChange', 'P_adjusted'])
        plot_data = plot_data[np.isfinite(plot_data['Log2FoldChange'])]
        
        # Calculate -log10(p-value)
        plot_data['NegLog10P'] = -np.log10(plot_data['P_adjusted'] + 1e-10)
        
        # Create scatter plot
        nonsig = plot_data[~plot_data['Significant']]
        sig = plot_data[plot_data['Significant']]
        
        ax.scatter(nonsig['Log2FoldChange'], nonsig['NegLog10P'], 
                  c=self.colors['non_significant'], alpha=0.5, s=20, label='Non-significant')
        ax.scatter(sig['Log2FoldChange'], sig['NegLog10P'],
                  c=self.colors['significant'], alpha=0.8, s=30, label='Significant')
        
        # Add threshold lines
        ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=-1, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Log₂ Fold Change')
        ax.set_ylabel('-Log₁₀(Adjusted P-value)')
        ax.set_title('Volcano Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_ma(self, deg_results, ax):
        """Create MA plot (log fold change vs mean expression)"""
        
        # Calculate mean expression (approximation)
        plot_data = deg_results.dropna(subset=['Log2FoldChange', 'Mean_Control', 'Mean_Treatment'])
        plot_data['MeanExpr'] = (plot_data['Mean_Control'] + plot_data['Mean_Treatment']) / 2
        plot_data['LogMeanExpr'] = np.log2(plot_data['MeanExpr'] + 1)
        
        # Create scatter plot
        nonsig = plot_data[~plot_data['Significant']]
        sig = plot_data[plot_data['Significant']]
        
        ax.scatter(nonsig['LogMeanExpr'], nonsig['Log2FoldChange'],
                  c=self.colors['non_significant'], alpha=0.5, s=20)
        ax.scatter(sig['LogMeanExpr'], sig['Log2FoldChange'],
                  c=self.colors['significant'], alpha=0.8, s=30)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.7)
        
        ax.set_xlabel('Log₂ Mean Expression')
        ax.set_ylabel('Log₂ Fold Change')
        ax.set_title('MA Plot')
        ax.grid(True, alpha=0.3)
    
    def _plot_expression_heatmap(self, expression_data, deg_results, metadata, ax, top_n=30):
        """Create expression heatmap of top DE genes"""
        
        # Get top significant genes
        sig_genes = deg_results[deg_results['Significant']].copy()
        if len(sig_genes) == 0:
            ax.text(0.5, 0.5, 'No significant genes found', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        sig_genes['AbsLog2FC'] = np.abs(sig_genes['Log2FoldChange'])
        top_genes = sig_genes.nlargest(min(top_n, len(sig_genes)), 'AbsLog2FC')
        
        # Prepare data
        heatmap_data = expression_data.loc[top_genes['Gene'], :]
        
        # Z-score normalization
        heatmap_data_z = (heatmap_data.T - heatmap_data.T.mean()) / heatmap_data.T.std()
        heatmap_data_z = heatmap_data_z.T
        
        # Create heatmap
        sns.heatmap(heatmap_data_z, cmap='RdBu_r', center=0,
                   xticklabels=True, yticklabels=True,
                   cbar_kws={'label': 'Z-score'}, ax=ax)
        
        ax.set_title(f'Top {len(top_genes)} DE Genes')
    
    def _plot_pca(self, expression_data, metadata, ax):
        """Create PCA plot"""
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        data_for_pca = expression_data.T
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_for_pca)
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data_scaled)
        
        # Plot
        for arm in metadata['Arm'].unique():
            mask = metadata['Arm'] == arm
            ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                      c=self.colors[arm], label=arm.title(), s=100, alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_expression_distribution(self, expression_data, ax):
        """Plot expression level distribution"""
        
        expr_flat = expression_data.values.flatten()
        ax.hist(np.log2(expr_flat + 1), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Log₂(TPM + 1)')
        ax.set_ylabel('Frequency')
        ax.set_title('Expression Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_top_genes(self, deg_results, ax, top_n=20):
        """Plot top differentially expressed genes"""
        
        sig_genes = deg_results[deg_results['Significant']].copy()
        if len(sig_genes) == 0:
            ax.text(0.5, 0.5, 'No significant genes found',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get top upregulated and downregulated
        sig_genes = sig_genes.sort_values('Log2FoldChange', ascending=False)
        top_up = sig_genes.head(top_n//2)
        top_down = sig_genes.tail(top_n//2)
        
        plot_genes = pd.concat([top_up, top_down])
        
        # Create barplot
        y_pos = range(len(plot_genes))
        colors = [self.colors['upregulated'] if fc > 0 else self.colors['downregulated'] 
                 for fc in plot_genes['Log2FoldChange']]
        
        bars = ax.barh(y_pos, plot_genes['Log2FoldChange'], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_genes['Gene'])
        ax.set_xlabel('Log₂ Fold Change')
        ax.set_title(f'Top {len(plot_genes)} DE Genes')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')
    
    def create_multi_panel_summary(self, tumor_data, expression_data, deg_results, 
                                 metadata, save_path=None):
        """Create comprehensive multi-panel summary figure"""
        
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Tumor growth analysis
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_mean_growth_curves(tumor_data, ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_waterfall(tumor_data, ax2)
        
        # Row 2: Expression analysis
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_volcano(deg_results, ax3)
        
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_pca(expression_data, metadata, ax4)
        
        # Row 3: Detailed heatmap and top genes
        ax5 = fig.add_subplot(gs[1:, :2])
        self._plot_expression_heatmap(expression_data, deg_results, metadata, ax5, top_n=40)
        
        ax6 = fig.add_subplot(gs[1:, 2:])
        self._plot_top_genes(deg_results, ax6, top_n=30)
        
        plt.suptitle('PDX Analysis Summary', fontsize=20, y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            
        return fig

def create_example_plots():
    """Create example plots with mock data"""
    
    # Create plotter
    plotter = PDXPlotter()
    
    # Generate some mock data for demonstration
    np.random.seed(42)
    
    # Mock tumor data
    models = ['PDX1', 'PDX2', 'PDX3', 'PDX4']
    arms = ['control', 'control', 'treatment', 'treatment']
    days = range(0, 29, 4)
    
    tumor_data = []
    for model, arm in zip(models, arms):
        growth_rate = 0.08 if arm == 'control' else 0.04
        initial_vol = np.random.normal(120, 20)
        
        for day in days:
            volume = initial_vol * np.exp(growth_rate * day) * (1 + np.random.normal(0, 0.1))
            tumor_data.append({
                'Model': model,
                'Arm': arm,
                'Day': day,
                'Volume_mm3': volume
            })
    
    tumor_df = pd.DataFrame(tumor_data)
    
    # Create tumor growth plots
    fig1 = plotter.plot_tumor_growth_curves(tumor_df, save_path='results/comprehensive_growth_plots.png')
    plt.show()
    
    print("Example plots created successfully!")
    print("Check results/comprehensive_growth_plots.png for output")

if __name__ == "__main__":
    create_example_plots()