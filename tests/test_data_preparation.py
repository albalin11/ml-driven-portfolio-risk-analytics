"""Part 1 checks against the retained raw snapshot and deliberately damaged copies."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from contextlib import chdir, redirect_stdout
import hashlib
import io
import json
import runpy
import shutil
import subprocess
import sys
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'DBC']

sys.path.insert(0, str(ROOT/'src'))
from data_preparation_checks import check_trading_dates as check_dates, check_adjusted_prices, check_ohlcv


class DataPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT/'data/raw/yahoo_ohlcv_20100101_20260916.csv',
                              header=[0, 1], index_col=0, parse_dates=True, float_precision='round_trip')

    def test_snapshot_covers_expected_sessions_and_boundaries(self):
        report = check_dates(self.raw.index, '2010-01-01', '2026-09-16')
        self.assertEqual(report['expected_sessions'], 4201)
        self.assertEqual(report['expected_first'], '2010-01-04')
        self.assertEqual(report['expected_last'], '2026-09-16')

    def test_missing_common_trading_day_in_temporary_csv_is_detected(self):
        removed = pd.Timestamp('2020-03-16')
        self.assertTrue(self.raw.loc[removed, 'Adj Close'].notna().all())
        with TemporaryDirectory() as directory:
            copy = Path(directory)/'missing_session.csv'
            self.raw.drop(index=removed).to_csv(copy)
            damaged = pd.read_csv(copy, header=[0, 1], index_col=0, parse_dates=True)
            # The old cell-based missing check still passes, but the calendar must fail.
            self.assertFalse(damaged['Adj Close'].isna().any().any())
            with self.assertRaisesRegex(ValueError, 'Missing trading sessions.*2020-03-16'):
                check_dates(damaged.index, '2010-01-01', '2026-09-16', ASSETS)

    def test_missing_start_or_cutoff_is_detected(self):
        for date in [self.raw.index[0], self.raw.index[-1]]:
            with self.subTest(date=date), self.assertRaisesRegex(ValueError, str(date.date())):
                check_dates(self.raw.index.drop(date), '2010-01-01', '2026-09-16')

    def test_holidays_special_closures_and_half_days(self):
        check_dates(self.raw.index, '2010-01-01', '2026-09-16')
        for date in ['2010-01-01', '2024-07-04', '2026-09-07',
                     '2012-10-29', '2012-10-30', '2018-12-05', '2025-01-09']:
            self.assertNotIn(pd.Timestamp(date), self.raw.index)
        # Christmas Eve 2018 was an early close, not a full-day closure.
        with self.assertRaisesRegex(ValueError, '2018-12-24'):
            check_dates(self.raw.index.drop(pd.Timestamp('2018-12-24')), '2010-01-01', '2026-09-16')
        with self.assertRaisesRegex(ValueError, 'unexpected dates.*2026-09-12'):
            check_dates(self.raw.index.union(pd.DatetimeIndex(['2026-09-12'])), '2010-01-01', '2026-09-16')

    def test_duplicate_unsorted_and_invalid_date_labels_fail(self):
        cases = [
            (self.raw.index.append(self.raw.index[:1]), 'Duplicate dates.*2010-01-04'),
            (self.raw.index[::-1], 'chronological'),
            (self.raw.index.tz_localize('UTC'), 'timezone-naive'),
            (self.raw.index + pd.Timedelta(hours=1), 'intraday'),
            (self.raw.index.insert(0, pd.NaT), 'NaT'),
        ]
        for dates, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                check_dates(dates, '2010-01-01', '2026-09-16')

    def test_missing_adjusted_price_or_asset_is_detected(self):
        prices = self.raw['Adj Close'][ASSETS].copy()
        prices.loc[pd.Timestamp('2020-03-16'), 'SPY'] = np.nan
        with self.assertRaisesRegex(ValueError, 'SPY: missing Adj Close.*2020-03-16.*value=nan'):
            check_adjusted_prices(prices, ASSETS)
        with self.assertRaisesRegex(ValueError, 'Expected adjusted-price columns'):
            check_adjusted_prices(prices.drop(columns='QQQ'), ASSETS)

    def test_invalid_adjusted_prices_are_detected(self):
        for value in [0, -1, np.inf, -np.inf, 'invalid']:
            prices = self.raw['Adj Close'][ASSETS].copy()
            if isinstance(value, str):
                prices['SPY'] = prices['SPY'].astype(object)
            prices.loc[prices.index[0], 'SPY'] = value
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                check_adjusted_prices(prices, ASSETS)
            message = str(caught.exception)
            self.assertIn('SPY:', message)
            self.assertIn('date=2010-01-04', message)
            self.assertIn(f'value={value!r}', message)

    def test_saved_prices_and_returns_match_raw_adj_close(self):
        raw_prices = self.raw['Adj Close'][ASSETS]
        check_adjusted_prices(raw_prices, ASSETS)
        prices = pd.read_csv(ROOT/'data/processed/clean_adjusted_close_prices.csv',
                             index_col=0, parse_dates=True, float_precision='round_trip')
        returns = pd.read_csv(ROOT/'data/processed/daily_returns.csv',
                              index_col=0, parse_dates=True, float_precision='round_trip')
        self.assertEqual(list(prices.columns), ASSETS)
        self.assertEqual(list(returns.columns), ASSETS)
        self.assertTrue(prices.index.equals(raw_prices.index))
        self.assertTrue(returns.index.equals(raw_prices.index[1:]))
        np.testing.assert_array_equal(prices.to_numpy(), raw_prices.to_numpy())
        expected = raw_prices.to_numpy()[1:] / raw_prices.to_numpy()[:-1] - 1
        np.testing.assert_allclose(returns.to_numpy(), expected, rtol=0, atol=0)
        self.assertTrue(np.isfinite(returns.to_numpy()).all())
        self.assertTrue((returns.to_numpy() < 0).any())
        self.assertTrue((returns.to_numpy() == 0).any())

    def test_existing_ohlcv_checks_keep_adj_close_separate(self):
        check_ohlcv(self.raw, ASSETS)
        self.assertTrue((self.raw['Adj Close']['HYG'] < self.raw['Low']['HYG']).any())
        copied = self.raw.copy()
        copied.loc[copied.index[0], ('Volume', 'SPY')] = 0
        check_ohlcv(copied, ASSETS)
        check_ohlcv(self.raw[['Adj Close']], ASSETS)  # No extra fields are required.
        for field, value in [('Open', 0), ('High', 0.01), ('Low', 10000),
                             ('Close', np.inf), ('Volume', -1), ('Volume', np.nan)]:
            copied = self.raw.copy()
            copied.loc[copied.index[0], (field, 'SPY')] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'SPY:'):
                check_ohlcv(copied, ASSETS)

    def test_invalid_download_stops_before_saving_snapshot_or_outputs(self):
        missing_price = self.raw.copy()
        missing_price.loc[pd.Timestamp('2020-03-16'), ('Adj Close', 'SPY')] = np.nan
        cases = [
            (self.raw.drop(pd.Timestamp('2020-03-16')), 'Missing trading sessions'),
            (self.raw.drop(columns=[('Adj Close', 'SPY')]), 'Missing Adj Close field.*SPY'),
            (missing_price, 'missing Adj Close'),
        ]
        for raw, message in cases:
            with self.subTest(message=message), TemporaryDirectory() as directory:
                project = Path(directory)
                (project/'notebooks').mkdir()
                (project/'src').mkdir()
                (project/'notebooks/01_data_preparation.ipynb').touch()
                with chdir(project), redirect_stdout(io.StringIO()), \
                     patch('yfinance.download', return_value=raw), patch('yfinance.set_tz_cache_location'):
                    with self.assertRaisesRegex(ValueError, message):
                        runpy.run_path(str(ROOT/'scripts/part_01_data_preparation.py'))
                self.assertEqual(list((project/'data/raw').iterdir()), [])
                self.assertEqual(list((project/'data/processed').iterdir()), [])
                self.assertEqual(list((project/'outputs/tables').iterdir()), [])

    def test_quality_report_matches_saved_data(self):
        report = json.loads((ROOT/'outputs/tables/part1_quality_report.json').read_text())
        self.assertEqual(report['assets'], ASSETS)
        self.assertEqual(report['requested_start'], '2010-01-01')
        self.assertEqual(report['cutoff'], '2026-09-16')
        self.assertEqual(report['cross_asset_missing_prices'], 0)
        self.assertEqual(report['calendar_audit']['missing_sessions'], 0)
        self.assertEqual(report['calendar_audit']['unexpected_dates'], 0)
        for name, section in [('clean_adjusted_close_prices.csv', 'cleaned_prices'), ('daily_returns.csv', 'daily_returns')]:
            path = ROOT/'data/processed'/name
            frame = pd.read_csv(path, index_col=0, parse_dates=True, float_precision='round_trip')
            summary = report[section]
            self.assertEqual(summary['rows'], len(frame))
            self.assertEqual(summary['columns'], 7)
            self.assertEqual(summary['start_date'], str(frame.index[0].date()))
            self.assertEqual(summary['end_date'], str(frame.index[-1].date()))
            for key in ['missing_values', 'duplicate_dates', 'infinite_values', 'invalid_values']:
                self.assertEqual(summary[key], 0)
            self.assertTrue(summary['chronological'])
            self.assertEqual(report['processed_sha256'][name], hashlib.sha256(path.read_bytes()).hexdigest())
            if section == 'daily_returns':
                self.assertEqual(summary['zero_returns'], int(frame.eq(0).sum().sum()))
                self.assertEqual(summary['negative_returns'], int(frame.lt(0).sum().sum()))
            else:
                self.assertEqual(summary['non_positive_values'], 0)
        metadata = json.loads((ROOT/'data/raw/yahoo_ohlcv_20100101_20260916_metadata.json').read_text())
        self.assertEqual(report['raw_snapshot_sha256'], metadata['sha256'])
        self.assertEqual(report['source'], metadata['source'])

    def test_runner_rejects_stale_notebook_before_execution(self):
        with TemporaryDirectory() as directory:
            project = Path(directory)
            for folder in ['src', 'scripts', 'notebooks']:
                (project/folder).mkdir()
            for name in ['run_pipeline.py', 'sync_notebooks.py']:
                shutil.copy2(ROOT/'src'/name, project/'src'/name)
            (project/'scripts/part_01_data_preparation.py').write_text('# %%\nprint("current")\n')
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('print("stale")')]),
                           project/'notebooks/01_data_preparation.ipynb')
            result = subprocess.run([sys.executable, 'src/run_pipeline.py', '--part', '1', '--mode', 'scripts'],
                                    cwd=project, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Part 1 notebook differs from its script', result.stderr)
            self.assertFalse((project/'outputs').exists())


if __name__ == '__main__':
    unittest.main()
