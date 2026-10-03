"""Independent validation for Part 4 VaR calculations and backtests."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from var_backtesting import kupiec_test

TABLES = ROOT / 'outputs' / 'tables'
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'DBC']
MODELS = ['Historical Volatility', 'EWMA', 'Linear Regression']
Z_SCORES = {0.95: 1.645, 0.99: 2.326}
EXPECTED_RATES = {0.95: 0.05, 0.99: 0.01}


class VaRBacktestingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.returns = pd.read_csv(
            ROOT / 'data/processed/daily_returns.csv',
            index_col='Date', parse_dates=True, float_precision='round_trip',
        )[ASSETS]
        cls.predictions = pd.read_csv(
            TABLES / 'test_predictions.csv', index_col='Date',
            parse_dates=['Date', 'target_start', 'target_end'],
            float_precision='round_trip',
        )
        cls.risk = pd.read_csv(
            TABLES / 'risk_forecasts.csv',
            parse_dates=['Date', 'target_start', 'target_end'],
            float_precision='round_trip',
        )
        cls.nonoverlap = pd.read_csv(
            TABLES / 'nonoverlapping_risk_forecasts.csv',
            parse_dates=['Date', 'target_start', 'target_end'],
            float_precision='round_trip',
        )
        cls.summary = pd.read_csv(TABLES / 'kupiec_backtests.csv')

    def test_frozen_part3_forecasts_are_used_without_retraining(self):
        report = json.loads((TABLES / 'part3_quality_report.json').read_text())
        path = TABLES / 'test_predictions.csv'
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            report['test_predictions_sha256'],
        )
        self.assertEqual(len(self.predictions), 1175)
        self.assertEqual(
            list(self.predictions.columns),
            ['actual_volatility', 'target_start', 'target_end', *MODELS],
        )
        self.assertTrue(self.predictions.index.is_unique)
        self.assertTrue(self.predictions.index.is_monotonic_increasing)

        for model in MODELS:
            saved = self.risk.loc[
                (self.risk.model == model) & (self.risk.confidence == 0.95)
            ].set_index('Date')
            pd.testing.assert_series_equal(
                saved['annualized_volatility'], self.predictions[model],
                check_names=False,
            )

    def test_future_returns_and_target_dates(self):
        portfolio = self.returns.mean(axis=1)
        positions = self.returns.index.get_indexer(self.predictions.index)
        self.assertTrue((positions >= 0).all())
        self.assertTrue((np.diff(positions) == 1).all())
        np.testing.assert_array_equal(
            self.predictions['target_start'].to_numpy(),
            self.returns.index[positions + 1].to_numpy(),
        )
        np.testing.assert_array_equal(
            self.predictions['target_end'].to_numpy(),
            self.returns.index[positions + 5].to_numpy(),
        )

        for date in pd.to_datetime(['2022-01-03', '2024-05-06', '2026-09-09']):
            position = self.returns.index.get_loc(date)
            window = portfolio.iloc[position + 1:position + 6]
            expected = np.prod(1 + window.to_numpy()) - 1
            saved = self.risk.loc[self.risk.Date == date, 'actual_return_5d']
            np.testing.assert_allclose(saved, expected, atol=1e-14)
            self.assertEqual(window.index[0], self.predictions.loc[date, 'target_start'])
            self.assertEqual(window.index[-1], self.predictions.loc[date, 'target_end'])

    def test_var_units_constants_signs_and_violations(self):
        self.assertEqual(
            list(self.risk.columns),
            [
                'Date', 'model', 'confidence', 'target_start', 'target_end',
                'annualized_volatility', 'sigma_daily', 'sigma_5d',
                'actual_return_5d', 'z_score', 'VaR_loss',
                'return_threshold', 'violation',
            ],
        )
        np.testing.assert_allclose(
            self.risk['sigma_daily'],
            self.risk['annualized_volatility'] / np.sqrt(252),
        )
        np.testing.assert_allclose(
            self.risk['sigma_5d'],
            self.risk['sigma_daily'] * np.sqrt(5),
        )
        expected_z = self.risk['confidence'].map(Z_SCORES)
        np.testing.assert_array_equal(self.risk['z_score'], expected_z)
        np.testing.assert_allclose(
            self.risk['VaR_loss'], expected_z * self.risk['sigma_5d']
        )
        self.assertTrue((self.risk['VaR_loss'] > 0).all())
        np.testing.assert_allclose(
            self.risk['return_threshold'], -self.risk['VaR_loss']
        )
        np.testing.assert_array_equal(
            self.risk['violation'],
            self.risk['actual_return_5d'] < -self.risk['VaR_loss'],
        )

    def test_kupiec_known_counts_and_boundaries(self):
        for count in [0, 1, 5, 20, 100]:
            flags = np.arange(100) < count
            statistic, p_value = kupiec_test(flags, 0.05)
            expected = 2 * (
                binom.logpmf(count, 100, count / 100)
                - binom.logpmf(count, 100, 0.05)
            )
            self.assertAlmostEqual(statistic, expected, places=10)
            self.assertAlmostEqual(p_value, chi2.sf(expected, 1), places=10)
        self.assertAlmostEqual(kupiec_test(np.arange(100) < 5, 0.05)[1], 1.0)
        with self.assertRaises(ValueError):
            kupiec_test([], 0.05)
        with self.assertRaises(ValueError):
            kupiec_test([False, True], 0)

    def test_nonoverlap_and_saved_summary(self):
        expected_dates = self.predictions.index[::5]
        self.assertEqual(len(expected_dates), 235)
        for model in MODELS:
            for confidence in Z_SCORES:
                group = self.nonoverlap.loc[
                    (self.nonoverlap.model == model)
                    & (self.nonoverlap.confidence == confidence)
                ].sort_values('Date')
                pd.testing.assert_index_equal(
                    pd.DatetimeIndex(group['Date']), expected_dates,
                    check_names=False,
                )
                self.assertTrue(np.all(
                    group.target_start.iloc[1:].to_numpy()
                    > group.target_end.iloc[:-1].to_numpy()
                ))

                saved = self.summary.loc[
                    (self.summary.model == model)
                    & (self.summary.confidence == confidence)
                ].iloc[0]
                count = int(group.violation.sum())
                statistic, p_value = kupiec_test(
                    group.violation.to_numpy(), EXPECTED_RATES[confidence]
                )
                self.assertEqual(saved.observations, len(group))
                self.assertEqual(saved.violations, count)
                self.assertAlmostEqual(saved.violation_rate, count / len(group))
                self.assertEqual(
                    saved.expected_violation_rate, EXPECTED_RATES[confidence]
                )
                self.assertAlmostEqual(saved.kupiec_statistic, statistic, places=12)
                self.assertAlmostEqual(saved.kupiec_p_value, p_value, places=12)


if __name__ == '__main__':
    unittest.main()
