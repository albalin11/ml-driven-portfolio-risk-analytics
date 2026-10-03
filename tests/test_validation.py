"""Independent arithmetic checks using synthetic examples and saved pipeline outputs."""
from pathlib import Path
import ast
import json
import unittest
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, binom
from scipy.special import xlogy
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs/tables'

def load_function(script, name):
    # Import only a pure function, without retraining models during unit tests.
    tree = ast.parse((ROOT / 'scripts' / script).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(np=np, pd=pd, chi2=chi2, xlogy=xlogy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), script, 'exec'), namespace)
    return namespace[name]

class FormulaTests(unittest.TestCase):
    def test_kupiec_known_counts_and_extremes(self):
        kupiec = load_function('part_04_var_backtesting.py', 'kupiec_test')
        for count in [0, 1, 5, 20, 100]:
            flags = np.arange(100) < count
            statistic, p_value = kupiec(flags, 0.05)
            expected = 2*(binom.logpmf(count, 100, count/100)-binom.logpmf(count, 100, 0.05))
            self.assertAlmostEqual(statistic, expected, places=10)
            self.assertAlmostEqual(p_value, chi2.sf(expected, 1), places=10)
        self.assertAlmostEqual(kupiec(np.arange(100)<5, 0.05)[1], 1.0)

    def test_kupiec_rejects_empty_sample(self):
        kupiec = load_function('part_04_var_backtesting.py', 'kupiec_test')
        with self.assertRaises(AssertionError):
            kupiec([], 0.05)

    def test_portfolio_features_independent_arithmetic(self):
        build = load_function('part_03_volatility_forecasting.py', 'portfolio_features')
        dates = pd.bdate_range('2000-01-03', periods=181)
        rng = np.random.default_rng(23)
        values = rng.normal(0, 0.01, size=(180, 7))
        frame = pd.DataFrame(values, index=dates[1:], columns=['SPY','QQQ','IWM','TLT','HYG','GLD','DBC'])
        result = build(frame, dates)
        daily = np.sum(values / 7, axis=1)
        wealth = np.r_[1, np.cumprod(1+daily)]
        for position in [120, 150, 180]:
            row = result.loc[dates[position]]
            for feature, lag in [('return_1d',1),('return_5d',5),('return_20d',20),('momentum_60d',60),('momentum_120d',120)]:
                self.assertAlmostEqual(row[feature], np.prod(1+daily[position-lag:position])-1, places=12)
            for window in [5,20,60]:
                self.assertAlmostEqual(row[f'volatility_{window}d'], np.std(daily[position-window:position],ddof=1)*np.sqrt(252),places=12)
            self.assertAlmostEqual(row.drawdown, wealth[position]/np.max(wealth[:position+1])-1, places=12)
            self.assertAlmostEqual(row.correlation_to_spy_20d, np.corrcoef(daily[position-20:position],values[position-20:position,0])[0,1],places=12)

    def test_future_perturbation_cannot_change_past_features(self):
        build = load_function('part_03_volatility_forecasting.py', 'portfolio_features')
        dates = pd.bdate_range('2000-01-03', periods=181)
        frame = pd.DataFrame(np.random.default_rng(7).normal(0,0.01,(180,7)),index=dates[1:],columns=['SPY','QQQ','IWM','TLT','HYG','GLD','DBC'])
        before = build(frame,dates)
        frame.loc[dates[151]:] *= 5
        after = build(frame,dates)
        pd.testing.assert_frame_equal(before.loc[:dates[150]],after.loc[:dates[150]])

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

    def test_split_target_windows(self):
        splits = pd.read_csv(TABLES/'time_splits.csv').set_index('split')
        self.assertLess(splits.loc['train','last_target_end'], splits.loc['validation','first_date'])
        self.assertLess(splits.loc['validation','last_target_end'], splits.loc['test','first_date'])

    def test_test_scores_recomputed(self):
        predictions = pd.read_csv(TABLES/'test_predictions.csv')
        metrics = pd.read_csv(TABLES/'test_metrics.csv')
        for _, row in metrics.iterrows():
            errors = predictions[row['model']].to_numpy()-predictions.actual_volatility.to_numpy()
            self.assertAlmostEqual(row.MAE,np.mean(np.abs(errors)),places=12)
            self.assertAlmostEqual(row.RMSE,np.sqrt(np.mean(errors**2)),places=12)
        selection = json.loads((TABLES/'model_selection.json').read_text())
        candidates = pd.read_csv(TABLES/'validation_candidates.csv').sort_values(['RMSE','MAE','candidate'])
        best_ml = candidates.loc[candidates.model.isin(['Linear Regression','Random Forest','XGBoost'])].iloc[0]
        self.assertEqual(best_ml.candidate,selection['selected_candidate'])
        self.assertEqual(set(metrics.model), {'Historical Volatility','EWMA',selection['selected_ml_model']})

    def test_saved_risk_units_signs_and_es_integration(self):
        risk = pd.read_csv(TABLES/'risk_forecasts.csv')
        np.testing.assert_allclose(risk.sigma_5d,risk.annualized_volatility/np.sqrt(252/5),atol=1e-14)
        np.testing.assert_array_equal(risk.violation,-risk.actual_return_5d > risk.VaR_loss)
        np.testing.assert_allclose(risk.return_threshold,-risk.VaR_loss,atol=1e-14)
        for confidence in [0.95,0.99]:
            rows=risk.loc[risk.confidence==confidence]
            integral=quad(lambda z:-z*norm.pdf(z),-np.inf,norm.ppf(1-confidence))[0]
            np.testing.assert_allclose(rows.ES_loss,rows.sigma_5d*integral/(1-confidence),atol=1e-12)

    def test_nonoverlap_and_backtest_counts(self):
        risk=pd.read_csv(TABLES/'nonoverlapping_risk_forecasts.csv',parse_dates=['Date','target_start','target_end'])
        summary=pd.read_csv(TABLES/'kupiec_backtests.csv')
        for _, row in summary.iterrows():
            group=risk.loc[(risk.model==row['model']) & (risk.confidence==row.confidence)].sort_values('Date')
            self.assertEqual(len(group),row.observations)
            self.assertEqual(group.violation.sum(),row.violations)
            self.assertTrue(np.all(group.target_start.iloc[1:].to_numpy()>group.target_end.iloc[:-1].to_numpy()))

if __name__ == '__main__':
    unittest.main()
