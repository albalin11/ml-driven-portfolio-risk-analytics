"""Small saved-output contract retained for the complete pipeline check."""
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class SavedResultTests(unittest.TestCase):
    def test_part2_contract(self):
        data = pd.read_csv(ROOT/'data/processed/volatility_modeling_dataset.csv')
        expected = ['Date', 'return_1d', 'return_5d', 'return_20d',
                    'volatility_5d', 'volatility_20d', 'volatility_60d',
                    'drawdown', 'average_correlation_20d',
                    'future_5d_realised_volatility']
        self.assertEqual(data.shape, (4136, 10))
        self.assertEqual(list(data.columns), expected)
        self.assertEqual((data.Date.min(), data.Date.max()), ('2010-03-31', '2026-09-09'))
        self.assertFalse(data.duplicated('Date').any())

if __name__ == '__main__':
    unittest.main()
