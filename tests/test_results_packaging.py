"""Checks that the final package agrees with the frozen Parts 1–4 outputs."""
from pathlib import Path
import hashlib
import json
import re
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs' / 'tables'
FIGURES = ROOT / 'outputs' / 'figures'


class ResultsPackagingTests(unittest.TestCase):
    def test_model_comparison_matches_saved_part3_metrics(self):
        validation = pd.read_csv(TABLES / 'validation_metrics.csv')
        test = pd.read_csv(TABLES / 'test_metrics.csv')
        expected = validation[['model', 'MAE', 'RMSE']].merge(
            test,
            on='model',
            how='left',
            suffixes=('_validation', '_test'),
        )
        expected['test_status'] = expected['MAE_test'].notna().map({
            True: 'evaluated',
            False: 'not evaluated: not selected on validation',
        })
        actual = pd.read_csv(TABLES / 'model_comparison.csv')
        assert_frame_equal(actual, expected, check_exact=False, rtol=1e-14, atol=1e-14)

    def test_summary_records_current_upstream_hashes_and_values(self):
        summary = json.loads((TABLES / 'results_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['status'], 'PASS')
        self.assertEqual(summary['selected_ml_model'], 'Linear Regression')

        test = pd.DataFrame(summary['test_metrics']).set_index('model')
        self.assertAlmostEqual(test.loc['Linear Regression', 'MAE'], 0.03550219872099334)
        self.assertAlmostEqual(test.loc['Linear Regression', 'RMSE'], 0.052967046471455)

        backtests = pd.DataFrame(summary['nonoverlapping_var_backtests'])
        violations = backtests.pivot(index='model', columns='confidence', values='violations')
        self.assertEqual(int(violations.loc['Historical Volatility', 0.95]), 10)
        self.assertEqual(int(violations.loc['EWMA', 0.95]), 8)
        self.assertEqual(int(violations.loc['Linear Regression', 0.95]), 15)
        self.assertTrue((backtests['observations'] == 235).all())

        for name, saved_digest in summary['upstream_sha256'].items():
            digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(digest, saved_digest, name)

    def test_readme_is_current_and_has_no_old_risk_measure(self):
        readme = (ROOT / 'README.md').read_text(encoding='utf-8')
        first_screen = '\n'.join(readme.splitlines()[:20])
        for model in ['Historical Volatility', 'EWMA', 'Linear Regression',
                      'Random Forest', 'XGBoost']:
            self.assertIn(model, first_screen)
        self.assertNotRegex(readme, re.compile(r'Expected Shortfall|\bES\b', re.IGNORECASE))
        self.assertIn('0.035502', readme)
        self.assertIn('0.052967', readme)
        self.assertIn('| Historical Volatility | 95.00% | 235 | 10 |', readme)
        self.assertIn('| EWMA | 95.00% | 235 | 8 |', readme)
        self.assertIn('| Linear Regression | 95.00% | 235 | 15 |', readme)
        self.assertIn('2010-01-04 to 2026-09-16', readme)
        self.assertIn('2010-03-31 to 2026-09-09', readme)

    def test_findings_and_core_figures_are_current(self):
        findings = (TABLES / 'findings.md').read_text(encoding='utf-8')
        self.assertIn('MAE was 0.035502', findings)
        self.assertIn('RMSE was 0.052967', findings)
        self.assertIn('Linear Regression at 95%: 15/235', findings)
        self.assertNotIn('13/235', findings)

        for name in ['model_errors.png', 'test_volatility_forecasts.png', 'var_violations.png']:
            path = FIGURES / name
            self.assertTrue(path.is_file(), name)
            self.assertGreater(path.stat().st_size, 10_000, name)


if __name__ == '__main__':
    unittest.main()
