"""
Enhanced Python preprocessing module with error handling and logging
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import configparser
from typing import Optional, Tuple, Dict, Any

# Setup logging
def setup_logging(log_file: str = "logs/preprocessing.log"):
    """Setup logging configuration"""
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

logger = setup_logging()

class PDXDataProcessor:
    """Class for processing PDX data with robust error handling"""
    
    def __init__(self, config_file: str = "config/config.ini"):
        """Initialize processor with configuration"""
        self.config = self._load_config(config_file)
        logger.info("PDX Data Processor initialized")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if Path(config_file).exists():
                config = configparser.ConfigParser()
                config.read(config_file)
                logger.info(f"Configuration loaded from: {config_file}")
                return config
            else:
                logger.warning(f"Configuration file not found: {config_file}, using defaults")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'analysis_parameters': {
                'min_timepoints': '4',
                'outlier_threshold': '3.0',
                'min_expression_tpm': '0.1'
            }
        }
    
    def load_tumor_volumes(self, path: str) -> pd.DataFrame:
        """
        Load and validate tumor volume data
        
        Args:
            path: Path to tumor volume CSV file
            
        Returns:
            Validated DataFrame with tumor volume data
        """
        logger.info(f"Loading tumor volume data from: {path}")
        
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            
            df = pd.read_csv(path)
            logger.info(f"Successfully loaded {len(df)} rows of data")
            
            # Validate required columns
            required_cols = ['Model', 'Arm', 'Day', 'Volume_mm3']
            missing_cols = set(required_cols) - set(df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Data type validation and conversion
            df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
            df['Volume_mm3'] = pd.to_numeric(df['Volume_mm3'], errors='coerce')
            df['Model'] = df['Model'].astype('category')
            df['Arm'] = df['Arm'].astype('category')
            
            # Remove rows with missing data
            initial_rows = len(df)
            df = df.dropna(subset=['Volume_mm3', 'Day'])
            df = df[df['Volume_mm3'] > 0]  # Remove non-positive volumes
            removed_rows = initial_rows - len(df)
            
            if removed_rows > 0:
                logger.warning(f"Removed {removed_rows} rows with missing or invalid data")
            
            logger.info("Data validation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error loading tumor volume data: {e}")
            raise
    
    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in tumor volume data using z-score method
        
        Args:
            df: DataFrame with tumor volume data
            threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outlier flag added
        """
        logger.info(f"Detecting outliers with threshold: {threshold}")
        
        df = df.copy()
        df['outlier'] = False
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].copy()
            
            if len(model_data) > 3:  # Need sufficient data points
                log_volumes = np.log(model_data['Volume_mm3'] + 1)
                z_scores = np.abs((log_volumes - log_volumes.mean()) / log_volumes.std())
                outlier_mask = z_scores > threshold
                
                if outlier_mask.sum() > 0:
                    df.loc[df['Model'] == model, 'outlier'] = outlier_mask
                    logger.warning(f"Found {outlier_mask.sum()} outliers in model {model}")
        
        n_outliers = df['outlier'].sum()
        logger.info(f"Total outliers detected: {n_outliers}")
        
        return df
    
    def baseline_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percent change vs baseline (day 0) for each model
        
        Args:
            df: DataFrame with tumor volume data
            
        Returns:
            DataFrame with percent change added
        """
        logger.info("Computing baseline normalization")
        
        try:
            result_dfs = []
            
            for model, group in df.groupby('Model'):
                group = group.copy()
                
                # Get baseline volume (day 0)
                baseline_data = group[group['Day'] == 0]
                
                if len(baseline_data) == 0:
                    logger.warning(f"No baseline data (Day 0) found for model {model}")
                    baseline_volume = group['Volume_mm3'].iloc[0]  # Use first measurement
                else:
                    baseline_volume = baseline_data['Volume_mm3'].iloc[0]
                
                # Calculate percent change
                group['PctChange'] = ((group['Volume_mm3'] / baseline_volume) - 1) * 100
                group['BaselineVolume'] = baseline_volume
                
                result_dfs.append(group)
            
            result = pd.concat(result_dfs, ignore_index=True)
            logger.info("Baseline normalization completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in baseline normalization: {e}")
            raise
    
    def calculate_growth_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth metrics for each model
        
        Args:
            df: DataFrame with tumor volume data
            
        Returns:
            DataFrame with growth metrics by model
        """
        logger.info("Calculating growth metrics")
        
        try:
            metrics = []
            
            for model, group in df.groupby('Model'):
                if len(group) < 2:
                    logger.warning(f"Insufficient data for growth calculation in model {model}")
                    continue
                
                # Sort by day
                group = group.sort_values('Day')
                
                # Calculate doubling time using exponential fit
                try:
                    log_volumes = np.log(group['Volume_mm3'])
                    days = group['Day']
                    
                    # Linear regression on log-transformed data
                    coeffs = np.polyfit(days, log_volumes, 1)
                    growth_rate = coeffs[0]  # slope in log space
                    doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf
                    
                except Exception as e:
                    logger.warning(f"Could not calculate growth rate for model {model}: {e}")
                    growth_rate = np.nan
                    doubling_time = np.nan
                
                # Calculate other metrics
                initial_volume = group['Volume_mm3'].iloc[0]
                final_volume = group['Volume_mm3'].iloc[-1]
                fold_change = final_volume / initial_volume
                
                metrics.append({
                    'Model': model,
                    'Arm': group['Arm'].iloc[0],
                    'InitialVolume': initial_volume,
                    'FinalVolume': final_volume,
                    'FoldChange': fold_change,
                    'GrowthRate': growth_rate,
                    'DoublingTime': doubling_time,
                    'NumTimepoints': len(group)
                })
            
            metrics_df = pd.DataFrame(metrics)
            logger.info(f"Growth metrics calculated for {len(metrics_df)} models")
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error calculating growth metrics: {e}")
            raise
    
    def load_expression_data(self, path: str) -> pd.DataFrame:
        """
        Load and validate gene expression data
        
        Args:
            path: Path to expression CSV file
            
        Returns:
            Validated DataFrame with expression data
        """
        logger.info(f"Loading expression data from: {path}")
        
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Expression file not found: {path}")
            
            df = pd.read_csv(path, index_col=0)
            logger.info(f"Loaded expression data: {df.shape[0]} genes, {df.shape[1]} samples")
            
            # Validate numeric data
            if not df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]:
                logger.warning("Non-numeric values detected in expression data")
            
            # Filter low-expressed genes
            min_tpm = float(self.config.get('analysis_parameters', {}).get('min_expression_tpm', 0.1))
            expressed_genes = (df >= min_tpm).any(axis=1)
            df_filtered = df[expressed_genes]
            
            removed_genes = df.shape[0] - df_filtered.shape[0]
            if removed_genes > 0:
                logger.info(f"Filtered out {removed_genes} low-expressed genes (< {min_tpm} TPM)")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error loading expression data: {e}")
            raise

def main():
    """Main function for testing the preprocessing module"""
    try:
        processor = PDXDataProcessor()
        
        # Test tumor volume processing
        tumor_data = processor.load_tumor_volumes("data/tumor_volumes_mock.csv")
        tumor_data = processor.detect_outliers(tumor_data)
        tumor_data = processor.baseline_normalize(tumor_data)
        
        # Calculate growth metrics
        growth_metrics = processor.calculate_growth_metrics(tumor_data)
        print("\nGrowth Metrics Summary:")
        print(growth_metrics.groupby('Arm')[['FoldChange', 'GrowthRate', 'DoublingTime']].mean())
        
        # Test expression data loading
        expression_data = processor.load_expression_data("data/expression_tpm_mock.csv")
        print(f"\nExpression data shape: {expression_data.shape}")
        
        logger.info("All preprocessing tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()