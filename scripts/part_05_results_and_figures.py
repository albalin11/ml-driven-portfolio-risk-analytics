# %% [markdown]
# # Part 5: Results and Discussion
#
# This notebook brings together the data audit, volatility forecasts and tail-risk checks.
# All displayed numbers and charts are loaded from or calculated from the executed pipeline.
# Volatility forecast errors and VaR calibration answer different questions: an accurate
# typical forecast can still miss unusually large losses.

# %%
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 160)

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts').is_dir() and (p / 'data').is_dir())
TABLE_DIR = PROJECT_ROOT / 'outputs' / 'tables'
FIGURE_DIR = PROJECT_ROOT / 'outputs' / 'figures'
returns = pd.read_csv(PROJECT_ROOT / 'data/processed/daily_returns.csv', index_col='Date', parse_dates=True)
returns['Equal-weight portfolio'] = returns.mean(axis=1)
validation = pd.read_csv(TABLE_DIR / 'validation_metrics.csv')
test = pd.read_csv(TABLE_DIR / 'test_metrics.csv')
backtests = pd.read_csv(TABLE_DIR / 'kupiec_backtests.csv')
selection = json.loads((TABLE_DIR / 'model_selection.json').read_text())
predictions = pd.read_csv(TABLE_DIR / 'test_predictions.csv', index_col='Date', parse_dates=True)

# %% [markdown]
# ## Data and portfolio overview
#
# SPY, QQQ and IWM represent US equities; TLT holds long-duration US Treasuries;
# HYG represents high-yield corporate bonds; GLD tracks gold and DBC broad commodities.
# These are different exposures, but they are not independent. Equal weights do not mean
# equal risk contributions. The table reports daily means and annualized sample volatility
# for the full descriptive sample; these full-sample statistics never enter model fitting.

# %%
overview = pd.DataFrame({'observations': returns.count(), 'mean_daily_return': returns.mean(),
                         'annualized_volatility': returns.std(ddof=1)*np.sqrt(252),
                         'zero_returns': returns.eq(0).sum(), 'negative_returns': returns.lt(0).sum()})
overview['start_date'] = str(returns.index.min().date())
overview['end_date'] = str(returns.index.max().date())
overview.to_csv(TABLE_DIR / 'asset_portfolio_overview.csv', index_label='asset')
display(overview)
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.bar(overview.index, overview.annualized_volatility*100, color=['#235789']*7+['#c1292e'])
ax.set_ylabel('Annualized sample volatility (%)')
ax.set_title('Daily returns: full-sample volatility of seven ETFs and the equal-weight portfolio')
ax.tick_params(axis='x', rotation=20)
fig.savefig(FIGURE_DIR / 'asset_portfolio_volatility.png', dpi=160)
if plt.get_backend().lower() != 'agg':
    plt.show()
plt.close(fig)

# %% [markdown]
# ## Validation and test errors
#
# The validation comparison includes all five model families. Each tree model's reported
# row is its best validation specification. The test table deliberately leaves unselected
# ML models unevaluated. Units below are annualized volatility decimals, not percentages
# of forecast accuracy. No claim of statistical superiority is made from point errors alone.

# %%
comparison = validation[['model', 'MAE', 'RMSE']].merge(test, on='model', how='left', suffixes=('_validation', '_test'))
comparison['test_status'] = np.where(comparison.MAE_test.notna(), 'evaluated', 'not evaluated: not selected on validation')
comparison.to_csv(TABLE_DIR / 'model_comparison.csv', index=False)
display(comparison)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
for ax, metric in zip(axes, ['MAE', 'RMSE']):
    ax.barh(comparison.model, comparison[f'{metric}_validation']*100, color='#235789', label='Validation')
    ax.scatter(comparison[f'{metric}_test']*100, comparison.model, color='#c1292e', label='Test (selected ML and baselines)', zorder=3)
    ax.set_xlabel(f'{metric} (annualized volatility percentage points)')
    ax.set_title(f'Five-day volatility forecast {metric}')
    ax.legend(fontsize=7)
fig.savefig(FIGURE_DIR / 'model_errors.png', dpi=160)
if plt.get_backend().lower() != 'agg':
    plt.show()
plt.close(fig)
display(Image(filename=str(FIGURE_DIR / 'test_volatility_forecasts.png')))

# %% [markdown]
# ## Five-day risk calibration
#
# The following tests use non-overlapping future five-session return windows. The p-values
# refer to unconditional violation frequency, not independence, ES calibration or profit.
# Multiple models/confidence levels are inspected without a multiple-testing adjustment;
# treat these as exploratory diagnostics. Few expected 99% violations make inference imprecise.

# %%
display(backtests)
display(Image(filename=str(FIGURE_DIR / 'var_violations.png')))
selected_name = selection['selected_ml_model']
selected_test = test.set_index('model').loc[selected_name]
best_baseline = test.loc[test.model.isin(['Historical Volatility', 'EWMA'])].sort_values('RMSE').iloc[0]
relative_rmse = (selected_test.RMSE / best_baseline.RMSE - 1)*100
direction = 'higher' if relative_rmse >= 0 else 'lower'
findings = [
    f"The validation-selected ML model was {selected_name}; the overall validation winner was {selection['overall_validation_winner']}.",
    f"On the held-out test period, {selected_name} had MAE {selected_test.MAE:.6f} and RMSE {selected_test.RMSE:.6f} in annualized volatility units.",
    f"Its test RMSE was {abs(relative_rmse):.2f}% {direction} than the lower-error baseline ({best_baseline['model']}). This is a descriptive comparison, not a significance test.",
]
for _, row in backtests.iterrows():
    findings.append(f"{row['model']} at {row.confidence:.0%}: {int(row.violations)}/{int(row.observations)} violations ({row.violation_rate:.2%}); Kupiec p={row.kupiec_p_value:.4f}.")
findings_text = '\n\n'.join(findings)
(TABLE_DIR / 'findings.md').write_text(findings_text, encoding='utf-8')
display(Markdown(findings_text))

# %% [markdown]
# ## What this study can and cannot show
#
# Five observations give a noisy volatility target. The test period covers one historical
# sequence, and model rankings may change with another period. The small validation search
# is transparent but not an exhaustive model comparison. No trading or economic-value claim
# follows from lower RMSE. Non-overlapping windows remove shared returns, not all dependence.
#
# Normal VaR can underestimate heavy-tailed losses. The square-root-of-time conversion
# assumes no conditional serial correlation and treats compound returns approximately.
# ES is a model-implied estimate here, without a dedicated ES backtest. Zero drift ignores
# the conditional mean. Daily equal-weight rebalancing ignores transaction costs and liquidity.
# Adjusted Yahoo prices are a historical vintage, not a point-in-time archive; public reruns
# may change if the provider revises prices. Seven chosen ETFs also limit generalization.
#
# Useful next steps would be a prespecified expanding-window evaluation with purging,
# Student-t or filtered-historical tail models, conditional coverage and dedicated ES tests,
# and explicit rebalancing costs. These are future work, not results claimed in this project.

# %%
summary = dict(selected_ml_model=selected_name, test_rmse=float(selected_test.RMSE),
               test_mae=float(selected_test.MAE), best_baseline=best_baseline['model'],
               relative_rmse_percent=float(relative_rmse), main_backtest_windows=int(backtests.observations.iloc[0]),
               figures=['asset_portfolio_volatility.png', 'model_errors.png', 'test_volatility_forecasts.png', 'var_violations.png'])
(TABLE_DIR / 'results_summary.json').write_text(json.dumps(summary, indent=2))
for figure in summary['figures']:
    assert (FIGURE_DIR / figure).is_file()
print('Part 5 completed: tables, figures and discussion use the executed results.')
