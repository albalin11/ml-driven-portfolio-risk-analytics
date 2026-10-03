"""Independent validation for Part 3 model selection and saved forecasts."""
from pathlib import Path
import hashlib
import json
import unittest

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs/tables'
FEATURES = [
    'return_1d', 'return_5d', 'return_20d',
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'drawdown', 'average_correlation_20d',
]
TARGET = 'future_5d_realised_volatility'


class VolatilityForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(
            ROOT / 'data/processed/volatility_modeling_dataset.csv',
            index_col='Date', parse_dates=True, float_precision='round_trip',
        )
        cls.return_dates = pd.DatetimeIndex(pd.read_csv(
            ROOT / 'data/processed/daily_returns.csv', usecols=['Date'], parse_dates=['Date']
        )['Date'])
        positions = cls.return_dates.get_indexer(cls.data.index)
        cls.target_start = pd.Series(cls.return_dates[positions + 1], index=cls.data.index)
        cls.target_end = pd.Series(cls.return_dates[positions + 5], index=cls.data.index)
        cls.splits = pd.read_csv(TABLES / 'time_splits.csv').set_index('split')
        cls.candidates = pd.read_csv(TABLES / 'validation_candidates.csv')
        cls.validation_predictions = pd.read_csv(
            TABLES / 'validation_predictions.csv', index_col='Date', parse_dates=True
        )
        cls.selection = json.loads((TABLES / 'model_selection.json').read_text())
        cls.test_predictions = pd.read_csv(
            TABLES / 'test_predictions.csv', index_col='Date', parse_dates=True
        )
        cls.test_metrics = pd.read_csv(TABLES / 'test_metrics.csv')

    def test_frozen_part2_input_contract(self):
        self.assertEqual(self.data.shape, (4136, 9))
        self.assertEqual(list(self.data.columns), FEATURES + [TARGET])
        self.assertEqual((self.data.index[0], self.data.index[-1]),
                         (pd.Timestamp('2010-03-31'), pd.Timestamp('2026-09-09')))
        self.assertTrue(self.data.index.is_unique)
        self.assertTrue(self.data.index.is_monotonic_increasing)
        self.assertFalse(self.data.isna().any().any())
        self.assertTrue(np.isfinite(self.data.to_numpy()).all())
        report = json.loads((TABLES / 'part2_quality_report.json').read_text())
        path = ROOT / 'data/processed/volatility_modeling_dataset.csv'
        self.assertEqual(report['dataset_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_split_boundaries_and_five_day_purge(self):
        train = self.data.loc[:pd.Timestamp('2017-12-31')]
        validation = self.data.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2021-12-31')]
        test = self.data.loc[pd.Timestamp('2022-01-01'):]
        train = train.loc[self.target_end.loc[train.index] < validation.index[0]]
        validation = validation.loc[self.target_end.loc[validation.index] < test.index[0]]

        self.assertEqual(len(train), self.splits.loc['train', 'rows'])
        self.assertEqual(len(validation), self.splits.loc['validation', 'rows'])
        self.assertEqual(len(test), self.splits.loc['test', 'rows'])
        self.assertEqual(self.splits.loc['train', 'purged_before_next_split'], 5)
        self.assertEqual(self.splits.loc['validation', 'purged_before_next_split'], 5)
        self.assertLess(self.target_end.loc[train.index].max(), validation.index.min())
        self.assertLess(self.target_end.loc[validation.index].max(), test.index.min())
        self.assertTrue(train.index.is_monotonic_increasing)
        self.assertTrue(validation.index.is_monotonic_increasing)
        self.assertTrue(test.index.is_monotonic_increasing)

    def test_validation_metrics_and_mae_first_selection(self):
        actual = self.validation_predictions['actual_volatility'].to_numpy()
        for _, row in self.candidates.iterrows():
            prediction = self.validation_predictions[row['candidate']].to_numpy()
            errors = prediction - actual
            self.assertAlmostEqual(row['MAE'], np.mean(np.abs(errors)), delta=1e-10)
            self.assertAlmostEqual(row['RMSE'], np.sqrt(np.mean(errors ** 2)), delta=1e-10)

        ml = self.candidates[self.candidates['model'].isin(
            ['Linear Regression', 'Random Forest', 'XGBoost']
        )].sort_values(['MAE', 'RMSE', 'candidate'])
        self.assertEqual(ml.iloc[0]['candidate'], self.selection['selected_candidate'])
        self.assertEqual(ml.iloc[0]['model'], self.selection['selected_ml_model'])
        self.assertEqual(self.selection['primary_metric'], 'validation MAE')
        self.assertEqual(self.selection['secondary_metric'], 'validation RMSE')
        self.assertFalse(self.selection['test_used_for_selection'])

    def test_baselines_and_train_only_scaling(self):
        validation_index = self.validation_predictions.index
        np.testing.assert_allclose(
            self.validation_predictions['Historical Volatility'],
            self.data.loc[validation_index, 'volatility_20d'],
        )

        expected_ewma = np.sqrt(
            252 * self.data['return_1d'].pow(2).ewm(alpha=0.06, adjust=False).mean()
        )
        np.testing.assert_allclose(
            self.validation_predictions['EWMA'], expected_ewma.loc[validation_index]
        )
        np.testing.assert_allclose(
            self.test_predictions['EWMA'], expected_ewma.loc[self.test_predictions.index]
        )

        validation_first = pd.Timestamp(self.splits.loc['validation', 'first_date'])
        train = self.data.loc[self.data.index < validation_first]
        train = train.loc[self.target_end.loc[train.index] < validation_first]
        np.testing.assert_allclose(
            self.selection['linear_validation_scaler_mean'], train[FEATURES].mean().to_numpy()
        )

        if self.selection['selected_ml_model'] == 'Linear Regression':
            test_first = self.test_predictions.index[0]
            refit = self.data.loc[(self.data.index < test_first) & (self.target_end < test_first)]
            model = joblib.load(ROOT / 'outputs/models/selected_volatility_model.joblib')
            np.testing.assert_allclose(
                model.named_steps['standardscaler'].mean_, refit[FEATURES].mean().to_numpy()
            )

    def test_test_metrics_recompute_from_saved_predictions(self):
        self.assertEqual(
            set(self.test_metrics['model']),
            {'Historical Volatility', 'EWMA', self.selection['selected_ml_model']},
        )
        actual = self.test_predictions['actual_volatility'].to_numpy()
        for _, row in self.test_metrics.iterrows():
            errors = self.test_predictions[row['model']].to_numpy() - actual
            self.assertAlmostEqual(row['MAE'], np.mean(np.abs(errors)), delta=1e-10)
            self.assertAlmostEqual(row['RMSE'], np.sqrt(np.mean(errors ** 2)), delta=1e-10)
        self.assertTrue(self.test_predictions.index.is_monotonic_increasing)
        self.assertTrue((self.test_predictions['target_start'] > self.test_predictions.index).all())
        self.assertTrue((self.test_predictions['target_end'] > self.test_predictions['target_start']).all())


if __name__ == '__main__':
    unittest.main()
