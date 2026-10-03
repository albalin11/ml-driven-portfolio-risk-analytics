"""Write the public README from the saved aggregate project results."""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'outputs' / 'tables'


def markdown_table(frame):
    """Return a small DataFrame as a Markdown table without another dependency."""
    lines = [
        '| ' + ' | '.join(frame.columns) + ' |',
        '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    return '\n'.join(lines)


part1 = json.loads((TABLES / 'part1_quality_report.json').read_text(encoding='utf-8'))
part2 = json.loads((TABLES / 'part2_quality_report.json').read_text(encoding='utf-8'))
selection = json.loads((TABLES / 'model_selection.json').read_text(encoding='utf-8'))
summary = json.loads((TABLES / 'results_summary.json').read_text(encoding='utf-8'))

validation = pd.read_csv(TABLES / 'validation_metrics.csv').rename(
    columns={'MAE': 'Validation MAE', 'RMSE': 'Validation RMSE'}
)
test = pd.read_csv(TABLES / 'test_metrics.csv').rename(
    columns={'MAE': 'Test MAE', 'RMSE': 'Test RMSE'}
)
metrics = validation.merge(test, on='model', how='left')[
    ['model', 'Validation MAE', 'Validation RMSE', 'Test MAE', 'Test RMSE']
]
for column in metrics.columns[1:]:
    metrics[column] = metrics[column].map(
        lambda value: f'{value:.6f}' if pd.notna(value) else 'Not evaluated'
    )
metrics.columns = ['Model', 'Validation MAE', 'Validation RMSE', 'Test MAE', 'Test RMSE']

splits = pd.read_csv(TABLES / 'time_splits.csv')
splits['Dates'] = splits['first_date'] + ' to ' + splits['last_date']
splits = splits[['split', 'Dates', 'rows']]
splits.columns = ['Split', 'Dates', 'Rows']

backtests = pd.read_csv(TABLES / 'kupiec_backtests.csv')[
    ['model', 'confidence', 'observations', 'violations', 'violation_rate',
     'expected_violation_rate', 'kupiec_statistic', 'kupiec_p_value']
].copy()
for column in ['confidence', 'violation_rate', 'expected_violation_rate']:
    backtests[column] = backtests[column].map(lambda value: f'{value:.2%}')
for column in ['kupiec_statistic', 'kupiec_p_value']:
    backtests[column] = backtests[column].map(lambda value: f'{value:.4f}')
backtests.columns = [
    'Model', 'Confidence', 'Windows', 'Violations', 'Rate',
    'Expected', 'Kupiec LR', 'p-value',
]

selected_model = selection['selected_ml_model']
relative_rmse = summary['relative_test_rmse_to_best_baseline_percent']
test_source = pd.read_csv(TABLES / 'test_metrics.csv')
selected_test = test_source.set_index('model').loc[selected_model]
model_order = [
    'Historical Volatility', 'EWMA', 'Linear Regression', 'Random Forest', 'XGBoost'
]
assert set(validation['model']) == set(model_order)
model_text = ', '.join(model_order[:-1]) + ' and ' + model_order[-1]
best_baseline = (
    test_source[test_source['model'].isin(['Historical Volatility', 'EWMA'])]
    .sort_values('RMSE')
    .iloc[0]['model']
)
feature_text = ', '.join(f"`{name}`" for name in part2['feature_list'])

text = f'''# ML-Driven Portfolio Risk Analytics

## Research question

Can simple machine-learning models improve forecasts of an equal-weight ETF portfolio's
next-five-trading-day realised volatility? Do those forecasts produce well-calibrated
five-day Value at Risk (VaR) limits on a held-out test period?

**Models:** {model_text}.

**Main result:** Linear Regression was selected on validation and recorded test MAE
**{selected_test['MAE']:.6f}** and RMSE **{selected_test['RMSE']:.6f}**, the lowest errors
among the three final test models.

![Held-out volatility forecasts](outputs/figures/test_volatility_forecasts.png)

## Data and portfolio

The project uses Yahoo Finance adjusted close prices for seven ETFs on the XNYS calendar.
The fixed sample contains **{part1['cleaned_prices']['rows']:,} price dates** from
**{part1['cleaned_prices']['start_date']} to {part1['cleaned_prices']['end_date']}** and
**{part1['daily_returns']['rows']:,} daily return dates**. Part 1 found no missing expected
sessions or cross-asset price gaps. Raw and observation-level data remain local because
market-data redistribution rights are separate from the yfinance software license; see
[data provenance](data/README.md).

| ETF | Main exposure |
| --- | --- |
| SPY | US large-cap equities |
| QQQ | Nasdaq-100 equities |
| IWM | US small-cap equities |
| TLT | Long-duration US Treasury bonds |
| HYG | High-yield corporate bonds |
| GLD | Gold |
| DBC | Broad commodities |

The portfolio keeps a constant **1/7 weight** in each ETF. Its daily return is the mean of
the seven ETF returns, so portfolio volatility includes cross-asset co-movement.

## Features and target

Part 2 creates eight backward-looking features: {feature_text}. The target at date `t` is
the annualised sample standard deviation of portfolio returns from `t+1` through `t+5`.
The final modeling data contain **{part2['final_modeling_rows']:,} rows** from
**{part2['final_start_date']} to {part2['final_end_date']}**, with no missing or non-finite
values. No future information enters the features. Independent formulas, full target
alignment and prefix-invariance checks passed.

## Forecasting models and time split

The two baselines are trailing 20-day Historical Volatility and EWMA with lambda 0.94.
The ML models are Linear Regression, Random Forest and XGBoost. Validation MAE is the
primary selection metric and RMSE is secondary. The chronological split is never shuffled;
five boundary observations are purged before validation and test so target windows do not
cross split boundaries. Scaling and model fitting use only eligible earlier observations.

{markdown_table(splits)}

## Forecast results

Errors are annualised volatility decimals. Random Forest and XGBoost were not evaluated on
test because the test set was not used for model selection.

{markdown_table(metrics)}

**{selected_model}** was selected on validation. Its test RMSE was
**{abs(relative_rmse):.2f}% {'lower' if relative_rmse < 0 else 'higher'}** than
{best_baseline}, the lower-RMSE baseline. This point comparison does not establish
statistical superiority or trading profitability.

![Model forecast errors](outputs/figures/model_errors.png)

## Five-day VaR backtesting

Part 4 uses the saved annualised test volatility forecasts and zero expected return:

```text
sigma_5d = sigma_annual * sqrt(5 / 252)
VaR_loss(95%) = 1.645 * sigma_5d
VaR_loss(99%) = 2.326 * sigma_5d
violation = actual future 5-day compounded return < -VaR_loss
```

Actual outcomes compound the five portfolio returns after each forecast date. Formal
Kupiec unconditional coverage tests use every fifth forecast from the first test date,
giving **235 non-overlapping windows** per model and confidence level.

{markdown_table(backtests)}

None of the six Kupiec tests rejects the expected violation rate at 5%. This does not prove
correct calibration: only 2.35 violations are expected at 99%, and the test checks frequency
rather than independence. Linear Regression has the lowest test forecast error but the
highest observed VaR violation rate, so point accuracy and tail calibration differ.

![Five-day VaR violations](outputs/figures/var_violations.png)

## Findings

- Linear Regression had the lowest validation MAE and the lowest test MAE and RMSE.
- Both tree models ranked behind Linear Regression on the fixed validation period.
- All six non-overlapping Kupiec p-values exceeded 0.05, with limited power at 99%.
- Lower volatility forecast error did not imply fewer VaR violations.

## Limitations

- Five daily returns make the realised-volatility target noisy.
- One chronological split and one test period cannot establish a stable model ranking.
- Normal VaR, zero expected return and square-root-of-time scaling are restrictive.
- The Kupiec test checks unconditional frequency; 235 windows give little 99% tail evidence.
- Daily rebalancing ignores costs and liquidity, and seven ETFs limit generalisation.

## Project structure

```text
scripts/       # Five complete analysis stages
notebooks/     # Matching notebooks with saved execution output
src/           # Small execution and validation helpers
tests/         # Formula, leakage, alignment and saved-result checks
data/          # Provenance note; raw and processed observations stay local
outputs/       # Public aggregate tables and figures; detailed rows stay local
```

## Reproduce the analysis

The project was tested with Python 3.14. In a virtual environment, run from the project root:

```bash
python -m pip install -r requirements.txt
python src/run_pipeline.py
python src/update_readme.py
python -m unittest discover -s tests -v
python src/verify_pipeline.py
```

The first run needs network access if no local raw snapshot exists. A fresh Yahoo download
may include provider revisions, so exact published numbers require the recorded local data
vintage.

The five scripts are the authoritative implementation. `src/run_pipeline.py` runs all five
scripts and all five notebooks, preserves notebook output, and checks that both paths save
byte-identical CSV files. Aggregate results are under `outputs/tables`; core figures are
under `outputs/figures`.

## References

- [yfinance documentation and data-use note](https://github.com/ranaroussi/yfinance#readme)
- [RiskMetrics Technical Document (1996)](https://www.msci.com/www/research-report/1996-riskmetrics-technical/018482266)
- [Kupiec proportion-of-failures test](https://www.mathworks.com/help/risk/risk.validation.proportionoffailurestest.html)
'''

(ROOT / 'README.md').write_text(text, encoding='utf-8')
print('README updated from the saved aggregate results.')
