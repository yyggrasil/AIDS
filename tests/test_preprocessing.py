import unittest
import pandas as pd
import numpy as np
from src.preprocessing import drop_irrelevant_features, get_preprocessor

class TestPreprocessing(unittest.TestCase):
    def test_drop_irrelevant_features(self):
        df = pd.DataFrame({
            'Flow ID': ['1', '2'],
            'Src IP': ['192.168.1.1', '192.168.1.2'],
            'Dst IP': ['10.0.0.1', '10.0.0.2'],
            'Timestamp': ['2023-01-01', '2023-01-02'],
            'Bwd PSH Flags': [0, 0],
            'Valid_Feature': [10.5, 20.3]
        })
        
        clean_df = drop_irrelevant_features(df)
        self.assertNotIn('Flow ID', clean_df.columns)
        self.assertNotIn('Src IP', clean_df.columns)
        self.assertNotIn('Dst IP', clean_df.columns)
        self.assertNotIn('Timestamp', clean_df.columns)
        self.assertNotIn('Bwd PSH Flags', clean_df.columns)
        self.assertIn('Valid_Feature', clean_df.columns)

    def test_get_preprocessor(self):
        df = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 100.0],
            'num2': [10.0, np.nan, 30.0, 40.0],
            'cat1': ['TCP', 'UDP', 'TCP', 'ICMP']
        })
        
        preprocessor = get_preprocessor(
            numeric_features=['num1', 'num2'],
            categorical_features=['cat1'],
            scale_type='robust'
        )
        
        transformed = preprocessor.fit_transform(df)
        self.assertIsNotNone(transformed)
        self.assertEqual(transformed.shape[0], 4)

if __name__ == '__main__':
    unittest.main()
