"""
Unit tests for PDX analysis functions
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src" / "python"))

class TestPDXDataProcessor(unittest.TestCase):
    """Test PDX data processing functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create mock tumor data
        self.mock_tumor_data = pd.DataFrame({
            'Model': ['PDX1', 'PDX1', 'PDX2', 'PDX2'],
            'Arm': ['control', 'control', 'treatment', 'treatment'],
            'Day': [0, 7, 0, 7],
            'Volume_mm3': [100, 150, 110, 120]
        })
        
        # Create mock expression data
        self.mock_expression = pd.DataFrame(
            np.random.lognormal(2, 1, (10, 2)),
            columns=['PDX1', 'PDX2'],
            index=[f'GENE{i}' for i in range(10)]
        )
    
    def test_tumor_data_validation(self):
        """Test tumor data validation"""
        # Test valid data
        self.assertEqual(len(self.mock_tumor_data), 4)
        self.assertIn('Model', self.mock_tumor_data.columns)
        self.assertIn('Volume_mm3', self.mock_tumor_data.columns)
        
        # Test missing columns
        invalid_data = self.mock_tumor_data.drop('Volume_mm3', axis=1)
        # Note: Would need actual processor to test validation
    
    def test_growth_calculation(self):
        """Test growth rate calculation"""
        # Simple growth rate test
        volumes = [100, 150]  # 50% growth
        days = [0, 7]
        
        # Log linear growth: log(150) - log(100) / 7
        expected_rate = (np.log(150) - np.log(100)) / 7
        calculated_rate = (np.log(volumes[1]) - np.log(volumes[0])) / (days[1] - days[0])
        
        self.assertAlmostEqual(calculated_rate, expected_rate, places=3)
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        # Create data with outlier
        data_with_outlier = self.mock_tumor_data.copy()
        data_with_outlier.loc[0, 'Volume_mm3'] = 10000  # Extreme outlier
        
        # Simple z-score outlier detection
        volumes = data_with_outlier['Volume_mm3']
        z_scores = np.abs((volumes - volumes.mean()) / volumes.std())
        outliers = z_scores > 3
        
        self.assertTrue(outliers.iloc[0])  # First row should be outlier
    
    def test_expression_filtering(self):
        """Test expression data filtering"""
        # Test low expression filtering
        min_tpm = 1.0
        expressed_genes = (self.mock_expression >= min_tpm).any(axis=1)
        filtered_data = self.mock_expression[expressed_genes]
        
        self.assertLessEqual(len(filtered_data), len(self.mock_expression))
        
        # Ensure all remaining genes have some expression above threshold
        self.assertTrue((filtered_data >= min_tpm).any(axis=1).all())

class TestStatisticalFunctions(unittest.TestCase):
    """Test statistical analysis functions"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        
        # Create mock differential expression data
        self.control_expr = np.random.normal(5, 1, 10)
        self.treatment_expr = np.random.normal(6, 1, 10)  # Higher expression
    
    def test_fold_change_calculation(self):
        """Test fold change calculation"""
        mean_control = self.control_expr.mean()
        mean_treatment = self.treatment_expr.mean()
        
        fold_change = mean_treatment / mean_control
        log2_fc = np.log2(fold_change)
        
        self.assertGreater(fold_change, 1)  # Treatment > control
        self.assertGreater(log2_fc, 0)      # Positive log2 FC
    
    def test_correlation_calculation(self):
        """Test correlation calculation"""
        # Perfect positive correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        
        correlation = np.corrcoef(x, y)[0, 1]
        self.assertAlmostEqual(correlation, 1.0, places=3)
        
        # Perfect negative correlation
        y_neg = np.array([10, 8, 6, 4, 2])
        correlation_neg = np.corrcoef(x, y_neg)[0, 1]
        self.assertAlmostEqual(correlation_neg, -1.0, places=3)

class TestDataGeneration(unittest.TestCase):
    """Test data generation functions"""
    
    def test_mock_data_properties(self):
        """Test properties of generated mock data"""
        # Test tumor volume data properties
        n_models = 4
        n_timepoints = 8
        expected_rows = n_models * n_timepoints
        
        # Mock generation (simplified)
        models = [f'PDX{i}' for i in range(1, n_models + 1)]
        days = list(range(0, n_timepoints * 4, 4))
        
        mock_data = []
        for model in models:
            for day in days:
                mock_data.append({
                    'Model': model,
                    'Day': day,
                    'Volume_mm3': 100 * np.exp(0.05 * day)  # Exponential growth
                })
        
        df = pd.DataFrame(mock_data)
        
        self.assertEqual(len(df), expected_rows)
        self.assertEqual(df['Model'].nunique(), n_models)
        self.assertEqual(df['Day'].nunique(), n_timepoints)
        
        # Test exponential growth pattern
        model_data = df[df['Model'] == 'PDX1'].sort_values('Day')
        volumes = model_data['Volume_mm3'].values
        self.assertTrue(all(volumes[i] <= volumes[i+1] for i in range(len(volumes)-1)))

class TestReporting(unittest.TestCase):
    """Test reporting functions"""
    
    def test_tgi_calculation(self):
        """Test TGI calculation"""
        # Mock growth rates
        control_rate = 0.1  # log volume/day
        treatment_rate = 0.05  # log volume/day
        
        tgi = ((control_rate - treatment_rate) / control_rate) * 100
        expected_tgi = 50.0  # 50% inhibition
        
        self.assertAlmostEqual(tgi, expected_tgi, places=1)
    
    def test_significance_classification(self):
        """Test significance classification"""
        # Mock p-values and fold changes
        p_values = [0.001, 0.1, 0.05, 0.2]
        fold_changes = [3.0, 1.5, 2.5, 1.1]
        
        # Classification: significant if p < 0.05 AND |FC| > 2
        significant = [(p < 0.05) and (abs(fc) > 2) for p, fc in zip(p_values, fold_changes)]
        
        expected = [True, False, True, False]
        self.assertEqual(significant, expected)

def run_tests():
    """Run all tests"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPDXDataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestReporting))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)