"""
Automated reporting and results interpretation for PDX analysis
Generates comprehensive HTML reports with statistical summaries and clinical relevance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from typing import Dict, List, Tuple, Optional
import json

class PDXReportGenerator:
    """Generate comprehensive analysis reports for PDX studies"""
    
    def __init__(self, study_title: str = "PDX Analysis Report"):
        self.study_title = study_title
        self.report_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.clinical_thresholds = {
            'tgi_excellent': 80,      # >80% TGI = excellent response
            'tgi_good': 50,           # 50-80% TGI = good response  
            'tgi_moderate': 20,       # 20-50% TGI = moderate response
            'significant_fc': 2.0,    # 2-fold change threshold
            'strong_correlation': 0.7, # Strong correlation threshold
        }
    
    def generate_comprehensive_report(self, 
                                    tumor_data: pd.DataFrame,
                                    expression_data: pd.DataFrame,
                                    deg_results: pd.DataFrame,
                                    growth_metrics: pd.DataFrame,
                                    correlation_results: pd.DataFrame = None,
                                    metadata: pd.DataFrame = None,
                                    output_path: str = "results/pdx_analysis_report.html") -> str:
        """Generate comprehensive HTML report"""
        
        # Calculate key statistics
        stats = self._calculate_key_statistics(tumor_data, expression_data, deg_results, 
                                             growth_metrics, correlation_results)
        
        # Generate interpretations
        interpretations = self._generate_interpretations(stats, deg_results, correlation_results)
        
        # Create HTML report
        html_content = self._create_html_report(stats, interpretations, metadata)
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comprehensive report saved to: {output_file}")
        return str(output_file)
    
    def _calculate_key_statistics(self, tumor_data, expression_data, deg_results, 
                                growth_metrics, correlation_results) -> Dict:
        """Calculate key statistical summaries"""
        
        stats = {
            'study_overview': {},
            'tumor_analysis': {},
            'expression_analysis': {},
            'integration_analysis': {}
        }
        
        # Study overview
        stats['study_overview'] = {
            'n_models': len(tumor_data['Model'].unique()),
            'n_control': len(tumor_data[tumor_data['Arm'] == 'control']['Model'].unique()),
            'n_treatment': len(tumor_data[tumor_data['Arm'] == 'treatment']['Model'].unique()),
            'n_timepoints': len(tumor_data['Day'].unique()),
            'study_duration': tumor_data['Day'].max(),
            'n_measurements': len(tumor_data)
        }
        
        # Tumor growth analysis
        if len(growth_metrics) > 0:
            control_metrics = growth_metrics[growth_metrics['Arm'] == 'control']
            treatment_metrics = growth_metrics[growth_metrics['Arm'] == 'treatment']
            
            # Calculate TGI
            if len(control_metrics) > 0 and len(treatment_metrics) > 0:
                control_growth = control_metrics['GrowthRate'].mean()
                treatment_growth = treatment_metrics['GrowthRate'].mean()
                tgi = ((control_growth - treatment_growth) / control_growth) * 100
            else:
                tgi = np.nan
            
            stats['tumor_analysis'] = {
                'control_growth_rate': control_metrics['GrowthRate'].mean() if len(control_metrics) > 0 else np.nan,
                'treatment_growth_rate': treatment_metrics['GrowthRate'].mean() if len(treatment_metrics) > 0 else np.nan,
                'tgi_percent': tgi,
                'control_doubling_time': control_metrics['DoublingTime'].mean() if len(control_metrics) > 0 else np.nan,
                'treatment_doubling_time': treatment_metrics['DoublingTime'].mean() if len(treatment_metrics) > 0 else np.nan,
                'control_fold_change': control_metrics['FoldChange'].mean() if len(control_metrics) > 0 else np.nan,
                'treatment_fold_change': treatment_metrics['FoldChange'].mean() if len(treatment_metrics) > 0 else np.nan
            }
        
        # Expression analysis
        stats['expression_analysis'] = {
            'n_genes_analyzed': len(deg_results),
            'n_genes_significant': deg_results['Significant'].sum() if 'Significant' in deg_results.columns else 0,
            'n_genes_upregulated': ((deg_results.get('Significant', False)) & (deg_results.get('Log2FoldChange', 0) > 0)).sum(),
            'n_genes_downregulated': ((deg_results.get('Significant', False)) & (deg_results.get('Log2FoldChange', 0) < 0)).sum(),
            'max_log2fc_up': deg_results['Log2FoldChange'].max() if 'Log2FoldChange' in deg_results.columns else np.nan,
            'max_log2fc_down': deg_results['Log2FoldChange'].min() if 'Log2FoldChange' in deg_results.columns else np.nan,
            'median_expression': expression_data.values.flatten().median(),
            'expression_range': (expression_data.values.min(), expression_data.values.max())
        }
        
        # Integration analysis
        if correlation_results is not None and len(correlation_results) > 0:
            stats['integration_analysis'] = {
                'n_correlations_tested': len(correlation_results),
                'n_significant_correlations': correlation_results.get('Significant_Spearman', pd.Series()).sum(),
                'max_correlation': correlation_results['Spearman_r'].abs().max() if 'Spearman_r' in correlation_results.columns else np.nan,
                'mean_correlation': correlation_results['Spearman_r'].abs().mean() if 'Spearman_r' in correlation_results.columns else np.nan
            }
        
        return stats
    
    def _generate_interpretations(self, stats, deg_results, correlation_results) -> Dict:
        """Generate clinical and biological interpretations"""
        
        interpretations = {
            'efficacy_assessment': '',
            'expression_summary': '',
            'biomarker_potential': '',
            'clinical_relevance': '',
            'limitations': [],
            'recommendations': []
        }
        
        # Efficacy assessment
        tgi = stats['tumor_analysis'].get('tgi_percent', np.nan)
        if not np.isnan(tgi):
            if tgi >= self.clinical_thresholds['tgi_excellent']:
                interpretations['efficacy_assessment'] = f"Excellent anti-tumor activity observed with {tgi:.1f}% tumor growth inhibition (TGI). This level of efficacy suggests strong therapeutic potential and warrants further investigation in clinical trials."
            elif tgi >= self.clinical_thresholds['tgi_good']:
                interpretations['efficacy_assessment'] = f"Good anti-tumor activity with {tgi:.1f}% TGI. This represents clinically meaningful efficacy that could translate to patient benefit."
            elif tgi >= self.clinical_thresholds['tgi_moderate']:
                interpretations['efficacy_assessment'] = f"Moderate anti-tumor activity with {tgi:.1f}% TGI. While statistically significant, the clinical relevance may be limited."
            else:
                interpretations['efficacy_assessment'] = f"Limited anti-tumor activity with {tgi:.1f}% TGI. The treatment shows minimal efficacy in these PDX models."
        else:
            interpretations['efficacy_assessment'] = "Unable to calculate TGI due to insufficient data."
        
        # Expression summary
        n_sig = stats['expression_analysis']['n_genes_significant']
        n_total = stats['expression_analysis']['n_genes_analyzed']
        n_up = stats['expression_analysis']['n_genes_upregulated']
        n_down = stats['expression_analysis']['n_genes_downregulated']
        
        if n_sig > 0:
            pct_sig = (n_sig / n_total) * 100
            interpretations['expression_summary'] = f"Significant transcriptional response detected with {n_sig} genes ({pct_sig:.1f}%) showing differential expression. {n_up} genes were upregulated and {n_down} genes were downregulated in treated samples."
        else:
            interpretations['expression_summary'] = "No significant differential gene expression detected, suggesting minimal transcriptional response to treatment."
        
        # Biomarker potential
        if correlation_results is not None and len(correlation_results) > 0:
            n_corr = stats['integration_analysis']['n_significant_correlations']
            max_corr = stats['integration_analysis']['max_correlation']
            
            if n_corr > 0:
                interpretations['biomarker_potential'] = f"Identified {n_corr} genes with significant expression-growth correlations (max correlation: {max_corr:.3f}). These genes represent potential predictive biomarkers for treatment response."
            else:
                interpretations['biomarker_potential'] = "No significant correlations between gene expression and tumor growth identified."
        else:
            interpretations['biomarker_potential'] = "Correlation analysis not performed or insufficient data available."
        
        # Clinical relevance
        control_dt = stats['tumor_analysis'].get('control_doubling_time', np.nan)
        treatment_dt = stats['tumor_analysis'].get('treatment_doubling_time', np.nan)
        
        clinical_notes = []
        
        if not np.isnan(control_dt):
            if control_dt < 7:
                clinical_notes.append("Rapidly growing tumors (doubling time < 7 days) may represent aggressive disease.")
            elif control_dt > 21:
                clinical_notes.append("Slowly growing tumors (doubling time > 21 days) may represent indolent disease.")
        
        if not np.isnan(treatment_dt) and not np.isnan(control_dt):
            dt_ratio = treatment_dt / control_dt
            if dt_ratio > 2:
                clinical_notes.append("Treatment significantly slowed tumor growth (>2-fold increase in doubling time).")
        
        interpretations['clinical_relevance'] = " ".join(clinical_notes) if clinical_notes else "Standard tumor growth kinetics observed."
        
        # Limitations
        limitations = []
        
        if stats['study_overview']['n_models'] < 6:
            limitations.append("Small sample size may limit statistical power and generalizability.")
        
        if stats['study_overview']['study_duration'] < 21:
            limitations.append("Short study duration may not capture delayed treatment effects.")
        
        if n_sig < 10:
            limitations.append("Limited differential gene expression may indicate insufficient treatment exposure or resistance.")
        
        interpretations['limitations'] = limitations
        
        # Recommendations
        recommendations = []
        
        if tgi >= 50:
            recommendations.append("Consider advancing to clinical trials based on strong preclinical efficacy.")
        
        if n_sig > 100:
            recommendations.append("Perform pathway enrichment analysis to understand biological mechanisms.")
        
        if correlation_results is not None and stats['integration_analysis']['n_significant_correlations'] > 5:
            recommendations.append("Validate identified biomarkers in independent patient cohorts.")
        
        recommendations.append("Consider dose-response studies to optimize therapeutic window.")
        recommendations.append("Evaluate combination therapies to enhance efficacy.")
        
        interpretations['recommendations'] = recommendations
        
        return interpretations
    
    def _create_html_report(self, stats, interpretations, metadata) -> str:
        """Create comprehensive HTML report"""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.study_title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .section h3 {{
            color: #34495e;
            margin-bottom: 15px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .interpretation {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }}
        .efficacy {{
            border-left-color: #27ae60;
        }}
        .warning {{
            border-left-color: #f39c12;
            background-color: #fef9e7;
        }}
        .limitation {{
            border-left-color: #e74c3c;
            background-color: #fdedec;
        }}
        .recommendation {{
            border-left-color: #9b59b6;
            background-color: #f4ecf7;
        }}
        .table-container {{
            overflow-x: auto;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.study_title}</h1>
            <p>Generated on {self.report_date}</p>
        </div>
        
        {self._generate_overview_section(stats)}
        {self._generate_efficacy_section(stats, interpretations)}
        {self._generate_expression_section(stats, interpretations)}
        {self._generate_biomarker_section(stats, interpretations)}
        {self._generate_conclusions_section(interpretations)}
        
        <div class="footer">
            <p>This report was automatically generated by the PDX Analysis Pipeline</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_overview_section(self, stats) -> str:
        """Generate study overview section"""
        
        overview = stats['study_overview']
        
        return f"""
        <div class="section">
            <h2>📊 Study Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{overview['n_models']}</div>
                    <div class="stat-label">PDX Models</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overview['study_duration']}</div>
                    <div class="stat-label">Study Duration (days)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overview['n_measurements']}</div>
                    <div class="stat-label">Total Measurements</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{overview['n_control']} / {overview['n_treatment']}</div>
                    <div class="stat-label">Control / Treatment</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_efficacy_section(self, stats, interpretations) -> str:
        """Generate efficacy assessment section"""
        
        tumor = stats['tumor_analysis']
        
        return f"""
        <div class="section">
            <h2>🎯 Treatment Efficacy</h2>
            
            <div class="interpretation efficacy">
                <h3>Primary Efficacy Assessment</h3>
                <p>{interpretations['efficacy_assessment']}</p>
            </div>
            
            <div class="table-container">
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Control Group</th>
                        <th>Treatment Group</th>
                        <th>Difference</th>
                    </tr>
                    <tr>
                        <td>Growth Rate (log volume/day)</td>
                        <td>{tumor.get('control_growth_rate', 'N/A'):.4f}</td>
                        <td>{tumor.get('treatment_growth_rate', 'N/A'):.4f}</td>
                        <td>{(tumor.get('control_growth_rate', 0) - tumor.get('treatment_growth_rate', 0)):.4f}</td>
                    </tr>
                    <tr>
                        <td>Doubling Time (days)</td>
                        <td>{tumor.get('control_doubling_time', 'N/A'):.1f}</td>
                        <td>{tumor.get('treatment_doubling_time', 'N/A'):.1f}</td>
                        <td>{(tumor.get('treatment_doubling_time', 0) - tumor.get('control_doubling_time', 0)):.1f}</td>
                    </tr>
                    <tr>
                        <td>Final Volume Fold Change</td>
                        <td>{tumor.get('control_fold_change', 'N/A'):.2f}</td>
                        <td>{tumor.get('treatment_fold_change', 'N/A'):.2f}</td>
                        <td>{(tumor.get('control_fold_change', 0) - tumor.get('treatment_fold_change', 0)):.2f}</td>
                    </tr>
                    <tr style="background-color: #e8f5e8;">
                        <td><strong>Tumor Growth Inhibition (TGI)</strong></td>
                        <td colspan="2" style="text-align: center;"><strong>{tumor.get('tgi_percent', 'N/A'):.1f}%</strong></td>
                        <td>-</td>
                    </tr>
                </table>
            </div>
            
            <div class="interpretation">
                <h3>Clinical Interpretation</h3>
                <p>{interpretations['clinical_relevance']}</p>
            </div>
        </div>
        """
    
    def _generate_expression_section(self, stats, interpretations) -> str:
        """Generate gene expression analysis section"""
        
        expr = stats['expression_analysis']
        
        return f"""
        <div class="section">
            <h2>🧬 Gene Expression Analysis</h2>
            
            <div class="interpretation">
                <h3>Transcriptional Response</h3>
                <p>{interpretations['expression_summary']}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{expr['n_genes_analyzed']}</div>
                    <div class="stat-label">Genes Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{expr['n_genes_significant']}</div>
                    <div class="stat-label">Significant Genes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{expr['n_genes_upregulated']}</div>
                    <div class="stat-label">Upregulated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{expr['n_genes_downregulated']}</div>
                    <div class="stat-label">Downregulated</div>
                </div>
            </div>
            
            <div class="table-container">
                <table>
                    <tr>
                        <th>Expression Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Maximum Log2 Fold Change (Up)</td>
                        <td>{expr.get('max_log2fc_up', 'N/A'):.2f}</td>
                        <td>{'Strong upregulation' if expr.get('max_log2fc_up', 0) > 2 else 'Moderate upregulation'}</td>
                    </tr>
                    <tr>
                        <td>Maximum Log2 Fold Change (Down)</td>
                        <td>{expr.get('max_log2fc_down', 'N/A'):.2f}</td>
                        <td>{'Strong downregulation' if expr.get('max_log2fc_down', 0) < -2 else 'Moderate downregulation'}</td>
                    </tr>
                    <tr>
                        <td>Median Expression Level (TPM)</td>
                        <td>{expr.get('median_expression', 'N/A'):.2f}</td>
                        <td>Overall expression level</td>
                    </tr>
                </table>
            </div>
        </div>
        """
    
    def _generate_biomarker_section(self, stats, interpretations) -> str:
        """Generate biomarker discovery section"""
        
        integration = stats.get('integration_analysis', {})
        
        return f"""
        <div class="section">
            <h2>🔬 Biomarker Discovery</h2>
            
            <div class="interpretation">
                <h3>Predictive Biomarker Potential</h3>
                <p>{interpretations['biomarker_potential']}</p>
            </div>
            
            {self._generate_integration_stats(integration)}
        </div>
        """
    
    def _generate_integration_stats(self, integration) -> str:
        """Generate integration analysis statistics"""
        
        if not integration:
            return "<p>No integration analysis performed.</p>"
        
        return f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{integration.get('n_correlations_tested', 0)}</div>
                <div class="stat-label">Correlations Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{integration.get('n_significant_correlations', 0)}</div>
                <div class="stat-label">Significant Correlations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{integration.get('max_correlation', 0):.3f}</div>
                <div class="stat-label">Maximum Correlation</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{integration.get('mean_correlation', 0):.3f}</div>
                <div class="stat-label">Mean Correlation</div>
            </div>
        </div>
        """
    
    def _generate_conclusions_section(self, interpretations) -> str:
        """Generate conclusions and recommendations section"""
        
        limitations_html = "".join([f"<li>{limitation}</li>" for limitation in interpretations['limitations']])
        recommendations_html = "".join([f"<li>{rec}</li>" for rec in interpretations['recommendations']])
        
        return f"""
        <div class="section">
            <h2>📝 Conclusions and Recommendations</h2>
            
            <div class="interpretation limitation">
                <h3>Study Limitations</h3>
                <ul>
                    {limitations_html}
                </ul>
            </div>
            
            <div class="interpretation recommendation">
                <h3>Recommendations for Future Studies</h3>
                <ul>
                    {recommendations_html}
                </ul>
            </div>
        </div>
        """

def generate_sample_report():
    """Generate a sample report with mock data"""
    
    # Create mock data
    np.random.seed(42)
    
    # Mock tumor data
    tumor_data = pd.DataFrame({
        'Model': ['PDX1', 'PDX2', 'PDX3', 'PDX4'] * 8,
        'Arm': ['control', 'control', 'treatment', 'treatment'] * 8,
        'Day': list(range(0, 32, 4)) * 4,
        'Volume_mm3': np.random.exponential(200, 32)
    })
    
    # Mock expression data
    expression_data = pd.DataFrame(np.random.lognormal(2, 1, (100, 4)),
                                 columns=['PDX1', 'PDX2', 'PDX3', 'PDX4'])
    
    # Mock DE results
    deg_results = pd.DataFrame({
        'Gene': [f'GENE{i}' for i in range(100)],
        'Log2FoldChange': np.random.normal(0, 1.5, 100),
        'P_adjusted': np.random.uniform(0, 1, 100),
        'Significant': np.random.choice([True, False], 100, p=[0.2, 0.8])
    })
    
    # Mock growth metrics
    growth_metrics = pd.DataFrame({
        'Model': ['PDX1', 'PDX2', 'PDX3', 'PDX4'],
        'Arm': ['control', 'control', 'treatment', 'treatment'],
        'GrowthRate': [0.08, 0.09, 0.04, 0.05],
        'DoublingTime': [8.7, 7.7, 17.3, 13.9],
        'FoldChange': [3.2, 3.8, 1.8, 2.1]
    })
    
    # Generate report
    reporter = PDXReportGenerator("Sample PDX Analysis Report")
    report_path = reporter.generate_comprehensive_report(
        tumor_data, expression_data, deg_results, growth_metrics
    )
    
    print(f"Sample report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_sample_report()