"""Part 1 checks against the retained raw snapshot and deliberately damaged copies."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import shutil
import subprocess
import sys
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'DBC']

sys.path.insert(0, str(ROOT/'src'))
from data_preparation_checks import check_trading_dates as check_dates, check_raw_bars as check_bars


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
                check_dates(damaged.index, '2010-01-01', '2026-09-16')

    def test_missing_start_or_cutoff_is_detected(self):
        for date in [self.raw.index[0], self.raw.index[-1]]:
            with self.subTest(date=date), self.assertRaisesRegex(ValueError, str(date.date())):
                check_dates(self.raw.index.drop(date), '2010-01-01', '2026-09-16')

    def test_holidays_special_closures_and_half_days(self):
        report = check_dates(self.raw.index, '2010-01-01', '2026-09-16')
        self.assertEqual(report['special_closures'], ['2012-10-29', '2012-10-30', '2018-12-05', '2025-01-09'])
        for date in ['2010-01-01', '2024-07-04', '2026-09-07']:
            self.assertNotIn(pd.Timestamp(date), self.raw.index)
        # Christmas Eve 2018 was an early close, not a full-day closure.
        with self.assertRaisesRegex(ValueError, '2018-12-24'):
            check_dates(self.raw.index.drop(pd.Timestamp('2018-12-24')), '2010-01-01', '2026-09-16')
        with self.assertRaisesRegex(ValueError, 'unexpected dates.*2026-09-12'):
            check_dates(self.raw.index.union(pd.DatetimeIndex(['2026-09-12'])), '2010-01-01', '2026-09-16')

    def test_duplicate_and_unsorted_dates_fail(self):
        for dates in [self.raw.index.append(self.raw.index[:1]), self.raw.index[::-1]]:
            with self.assertRaisesRegex(ValueError, 'unique and chronological'):
                check_dates(dates, '2010-01-01', '2026-09-16')

    def test_real_bars_pass_without_comparing_adj_close_to_ohlc_range(self):
        summary = check_bars(self.raw, ASSETS)
        self.assertEqual(len(summary), 7)
        self.assertTrue((self.raw['Adj Close']['HYG'] < self.raw['Low']['HYG']).any())
        self.assertTrue(all(item['invalid_ohlc_rows'] == 0 for item in summary.values()))

    def test_invalid_ohlcv_and_adjusted_prices_fail(self):
        for field, value in [('High', 0.01), ('Adj Close', 0), ('Volume', -1), ('Open', np.nan), ('Close', np.inf)]:
            damaged = self.raw.copy()
            damaged.loc[damaged.index[0], (field, 'SPY')] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_bars(damaged, ASSETS)

    def test_zero_volume_is_reported_without_price_imputation(self):
        copied = self.raw.copy()
        copied.loc[copied.index[0], ('Volume', 'SPY')] = 0
        report = check_bars(copied, ASSETS)
        self.assertEqual(report['SPY']['zero_volume_rows'], 1)
        pd.testing.assert_frame_equal(copied['Adj Close'], self.raw['Adj Close'])

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
