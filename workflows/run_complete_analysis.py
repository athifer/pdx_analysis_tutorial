"""
Complete PDX analysis workflow in Python
Orchestrates data processing, analysis, and reporting
"""

import sys
import os
from pathlib import Path
import logging
import argparse
from typing import Optional, Dict, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src" / "python"))

from preprocessing import PDXDataProcessor
from plotting import PDXPlotter
from reporting import PDXReportGenerator

def setup_logging(log_file: str = "logs/workflow.log"):
    """Setup logging for the workflow"""
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_complete_pdx_analysis(config_file: str = "config/config.ini",
                             output_dir: str = "results",
                             generate_data: bool = False) -> Dict[str, Any]:
    """
    Run complete PDX analysis workflow
    
    Args:
        config_file: Path to configuration file
        output_dir: Output directory for results
        generate_data: Whether to generate new mock data
        
    Returns:
        Dictionary with analysis results
    """
    
    logger = setup_logging()
    logger.info("Starting complete PDX analysis workflow")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        results = {}
        
        # Step 1: Generate data (if requested)
        if generate_data:
            logger.info("Step 1: Generating enhanced mock data")
            from generate_enhanced_data import EnhancedPDXDataGenerator
            
            generator = EnhancedPDXDataGenerator(n_models=8, n_genes=500, n_timepoints=8)
            metadata, tumor_data, expression_data, variant_data = generator.generate_all_data()
            generator.save_data(metadata, tumor_data, expression_data, variant_data)
            
            results['data_generation'] = {
                'metadata_shape': metadata.shape,
                'tumor_data_shape': tumor_data.shape,
                'expression_shape': expression_data.shape,
                'variant_data_shape': variant_data.shape
            }
            logger.info("Data generation completed")
        
        # Step 2: Data preprocessing and analysis
        logger.info("Step 2: Data preprocessing and analysis")
        processor = PDXDataProcessor(config_file)
        
        # Load and process tumor data
        tumor_data = processor.load_tumor_volumes("data/tumor_volumes_mock.csv")
        tumor_data = processor.detect_outliers(tumor_data)
        tumor_data = processor.baseline_normalize(tumor_data)
        growth_metrics = processor.calculate_growth_metrics(tumor_data)
        
        # Load expression data
        expression_data = processor.load_expression_data("data/expression_tpm_mock.csv")
        
        results['preprocessing'] = {
            'tumor_data_shape': tumor_data.shape,
            'expression_shape': expression_data.shape,
            'growth_metrics_shape': growth_metrics.shape,
            'n_outliers': tumor_data['outlier'].sum()
        }
        logger.info("Data preprocessing completed")
        
        # Step 3: Statistical analysis (simplified DEG analysis)
        logger.info("Step 3: Statistical analysis")
        
        # Create sample metadata
        sample_metadata = tumor_data[tumor_data['Day'] == 0][['Model', 'Arm']].copy()
        sample_metadata = sample_metadata.rename(columns={'Model': 'Sample'})
        sample_metadata = sample_metadata.set_index('Sample')
        
        # Filter to common samples
        common_samples = list(set(expression_data.columns) & set(sample_metadata.index))
        expression_filtered = expression_data[common_samples]
        metadata_filtered = sample_metadata.loc[common_samples]
        
        # Simple differential expression analysis
        import numpy as np
        import pandas as pd
        from scipy import stats
        from scipy.stats import false_discovery_control
        
        deg_results = []
        control_samples = metadata_filtered[metadata_filtered['Arm'] == 'control'].index
        treatment_samples = metadata_filtered[metadata_filtered['Arm'] == 'treatment'].index
        
        expression_log = np.log2(expression_filtered + 1)
        
        for gene in expression_log.index:
            if len(control_samples) > 0 and len(treatment_samples) > 0:
                control_expr = expression_log.loc[gene, control_samples]
                treatment_expr = expression_log.loc[gene, treatment_samples]
                
                # Calculate fold change
                mean_control = control_expr.mean()
                mean_treatment = treatment_expr.mean()
                fold_change = mean_treatment / mean_control if mean_control > 0 else np.inf
                log2_fc = np.log2(fold_change) if fold_change > 0 and np.isfinite(fold_change) else np.nan
                
                # T-test
                try:
                    t_stat, p_value = stats.ttest_ind(treatment_expr, control_expr)
                except:
                    t_stat, p_value = np.nan, 1.0
                
                deg_results.append({
                    'Gene': gene,
                    'Mean_Control': mean_control,
                    'Mean_Treatment': mean_treatment,
                    'Log2FoldChange': log2_fc,
                    'P_value': p_value
                })
        
        deg_df = pd.DataFrame(deg_results)
        if len(deg_df) > 0:
            deg_df['P_adjusted'] = false_discovery_control(deg_df['P_value'].fillna(1.0))
            deg_df['Significant'] = (deg_df['P_adjusted'] < 0.05) & (np.abs(deg_df['Log2FoldChange']) > 1)
        
        results['statistical_analysis'] = {
            'n_genes_tested': len(deg_df),
            'n_significant': deg_df['Significant'].sum() if len(deg_df) > 0 else 0
        }
        logger.info("Statistical analysis completed")
        
        # Step 4: Generate visualizations
        logger.info("Step 4: Generating visualizations")
        plotter = PDXPlotter()
        
        # Create growth curve plots
        growth_fig = plotter.plot_tumor_growth_curves(
            tumor_data, 
            save_path=f"{output_dir}/comprehensive_growth_analysis.png"
        )
        
        # Create expression plots (if we have results)
        if len(deg_df) > 0 and deg_df['Significant'].sum() > 0:
            expr_fig = plotter.plot_expression_analysis(
                expression_log,
                deg_df,
                metadata_filtered,
                save_path=f"{output_dir}/comprehensive_expression_analysis.png"
            )
        
        # Save data files
        tumor_data.to_csv(f"{output_dir}/processed_tumor_data.csv", index=False)
        growth_metrics.to_csv(f"{output_dir}/growth_metrics.csv", index=False)
        if len(deg_df) > 0:
            deg_df.to_csv(f"{output_dir}/differential_expression_results.csv", index=False)
        
        results['visualization'] = {
            'plots_generated': ['growth_analysis', 'expression_analysis'],
            'files_saved': ['processed_tumor_data.csv', 'growth_metrics.csv', 'differential_expression_results.csv']
        }
        logger.info("Visualization completed")
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating comprehensive report")
        reporter = PDXReportGenerator("Complete PDX Analysis Report")
        
        report_path = reporter.generate_comprehensive_report(
            tumor_data=tumor_data,
            expression_data=expression_log,
            deg_results=deg_df,
            growth_metrics=growth_metrics,
            correlation_results=None,  # Could add correlation analysis here
            metadata=metadata_filtered,
            output_path=f"{output_dir}/pdx_analysis_report.html"
        )
        
        results['reporting'] = {
            'report_path': report_path
        }
        logger.info("Report generation completed")
        
        # Step 6: Summary
        logger.info("Workflow completed successfully")
        results['summary'] = {
            'workflow_status': 'completed',
            'output_directory': output_dir,
            'main_results': {
                'n_models': tumor_data['Model'].nunique(),
                'n_timepoints': tumor_data['Day'].nunique(),
                'n_genes': expression_data.shape[0],
                'n_significant_genes': deg_df['Significant'].sum() if len(deg_df) > 0 else 0,
                'tgi_percent': ((growth_metrics[growth_metrics['Arm'] == 'control']['GrowthRate'].mean() - 
                               growth_metrics[growth_metrics['Arm'] == 'treatment']['GrowthRate'].mean()) /
                               growth_metrics[growth_metrics['Arm'] == 'control']['GrowthRate'].mean() * 100)
                               if len(growth_metrics) > 0 else None
            }
        }
        
        # Print summary
        summary = results['summary']['main_results']
        print(f"""
=== PDX ANALYSIS WORKFLOW COMPLETED ===
📊 Study Overview:
   • {summary['n_models']} PDX models analyzed
   • {summary['n_timepoints']} timepoints collected
   • {summary['n_genes']} genes analyzed
   
🎯 Key Results:
   • {summary['n_significant_genes']} significantly differentially expressed genes
   • {summary['tgi_percent']:.1f}% tumor growth inhibition (TGI)
   
📁 Output Files:
   • Results saved to: {output_dir}/
   • Comprehensive report: {report_path}
   • Visualizations: {output_dir}/*.png
        """)
        
        return results
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        raise

def main():
    """Main function for command-line execution"""
    
    parser = argparse.ArgumentParser(description="Run complete PDX analysis workflow")
    parser.add_argument("--config", default="config/config.ini", 
                       help="Path to configuration file")
    parser.add_argument("--output", default="results", 
                       help="Output directory")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate new mock data")
    
    args = parser.parse_args()
    
    # Run workflow
    results = run_complete_pdx_analysis(
        config_file=args.config,
        output_dir=args.output,
        generate_data=args.generate_data
    )
    
    print("Workflow completed successfully!")
    return results

if __name__ == "__main__":
    main()