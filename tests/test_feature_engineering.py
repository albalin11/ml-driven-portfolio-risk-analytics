"""Independent checks for the equal-weight Part 2 dataset."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from feature_engineering import (
    ASSETS, FEATURE_COLUMNS, TARGET_COLUMN,
    build_portfolio_features, build_future_volatility_target,
)


class FeatureEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = pd.read_csv(
            ROOT / 'data/processed/daily_returns.csv',
            index_col='Date', parse_dates=True, float_precision='round_trip',
        )[ASSETS]
        cls.saved = pd.read_csv(
            ROOT / 'data/processed/volatility_modeling_dataset.csv',
            index_col='Date', parse_dates=True, float_precision='round_trip',
        )

    def test_equal_weight_features_match_independent_arithmetic(self):
        features = build_portfolio_features(self.returns)
        portfolio = self.returns.to_numpy().mean(axis=1)
        wealth = np.cumprod(1 + portfolio)
        for date in pd.to_datetime(['2010-03-31', '2020-03-16', '2026-09-09']):
            position = self.returns.index.get_loc(date)
            row = features.loc[date]
            expected = {
                'return_1d': portfolio[position],
                'return_5d': np.prod(1 + portfolio[position - 4:position + 1]) - 1,
                'return_20d': np.prod(1 + portfolio[position - 19:position + 1]) - 1,
                'volatility_5d': np.std(portfolio[position - 4:position + 1], ddof=1) * np.sqrt(252),
                'volatility_20d': np.std(portfolio[position - 19:position + 1], ddof=1) * np.sqrt(252),
                'volatility_60d': np.std(portfolio[position - 59:position + 1], ddof=1) * np.sqrt(252),
                'drawdown': wealth[position] / max(1.0, wealth[:position + 1].max()) - 1,
            }
            correlations = np.corrcoef(
                self.returns.iloc[position - 19:position + 1], rowvar=False
            )
            expected['average_correlation_20d'] = correlations[np.triu_indices(7, 1)].mean()
            for name, value in expected.items():
                with self.subTest(date=date, feature=name):
                    self.assertAlmostEqual(row[name], value, places=12)

    def test_target_uses_only_next_five_trading_returns(self):
        portfolio = self.returns.mean(axis=1)
        target = build_future_volatility_target(portfolio)
        self.assertTrue(target.iloc[-5:].isna().all())
        self.assertTrue(target.iloc[:-5].notna().all())
        for position in range(len(self.returns) - 5):
            expected = np.std(portfolio.iloc[position + 1:position + 6], ddof=1) * np.sqrt(252)
            self.assertAlmostEqual(target.iloc[position], expected, places=12)

    def test_features_are_prefix_invariant(self):
        full = build_portfolio_features(self.returns)
        for cutoff in pd.to_datetime(['2010-03-31', '2020-03-16', '2026-09-09']):
            prefix = build_portfolio_features(self.returns.loc[:cutoff])
            pd.testing.assert_series_equal(prefix.loc[cutoff], full.loc[cutoff])

    def test_saved_dataset_matches_direct_formulas(self):
        self.assertEqual(list(self.saved.columns), FEATURE_COLUMNS + [TARGET_COLUMN])
        self.assertTrue(self.saved.index.equals(self.returns.index[59:-5]))
        self.assertFalse(self.saved.isna().any().any())
        self.assertTrue(np.isfinite(self.saved.to_numpy()).all())

        portfolio = self.returns.to_numpy().mean(axis=1)
        wealth = np.cumprod(1 + portfolio)
        for date in pd.to_datetime(['2010-03-31', '2020-03-16', '2026-09-09']):
            position = self.returns.index.get_loc(date)
            row = self.saved.loc[date]
            self.assertAlmostEqual(row['return_1d'], portfolio[position], places=12)
            self.assertAlmostEqual(
                row['return_5d'], np.prod(1 + portfolio[position - 4:position + 1]) - 1, places=12
            )
            self.assertAlmostEqual(
                row['return_20d'], np.prod(1 + portfolio[position - 19:position + 1]) - 1, places=12
            )
            for window in [5, 20, 60]:
                expected = np.std(portfolio[position - window + 1:position + 1], ddof=1) * np.sqrt(252)
                self.assertAlmostEqual(row[f'volatility_{window}d'], expected, places=12)
            self.assertAlmostEqual(
                row['drawdown'], wealth[position] / max(1.0, wealth[:position + 1].max()) - 1,
                places=12,
            )
            matrix = np.corrcoef(self.returns.iloc[position - 19:position + 1], rowvar=False)
            self.assertAlmostEqual(
                row['average_correlation_20d'], matrix[np.triu_indices(7, 1)].mean(), places=12
            )
            target = np.std(portfolio[position + 1:position + 6], ddof=1) * np.sqrt(252)
            self.assertAlmostEqual(row[TARGET_COLUMN], target, places=12)

    def test_saved_outputs_and_quality_report_are_consistent(self):
        report_path = ROOT / 'outputs/tables/part2_quality_report.json'
        summary_path = ROOT / 'outputs/tables/feature_summary.csv'
        dataset_path = ROOT / 'data/processed/volatility_modeling_dataset.csv'
        report = json.loads(report_path.read_text())
        summary = pd.read_csv(summary_path)

        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['feature_list'], FEATURE_COLUMNS)
        self.assertEqual(report['target_name'], TARGET_COLUMN)
        self.assertEqual(report['final_modeling_rows'], len(self.saved))
        self.assertEqual(report['final_modeling_columns'], 10)
        self.assertEqual(report['missing_values_final'], 0)
        self.assertEqual(report['non_finite_feature_values'], 0)
        self.assertEqual(report['non_finite_target_values'], 0)
        self.assertEqual(report['target_alignment_validation']['windows_checked'], len(self.returns) - 5)
        self.assertEqual(summary['feature'].tolist(), FEATURE_COLUMNS)
        self.assertTrue(summary['missing_count'].eq(0).all())
        self.assertEqual(report['dataset_sha256'], hashlib.sha256(dataset_path.read_bytes()).hexdigest())
        self.assertEqual(report['feature_summary_sha256'], hashlib.sha256(summary_path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
