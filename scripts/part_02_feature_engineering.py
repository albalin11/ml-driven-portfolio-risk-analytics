# %% [markdown]
# # Part 2: Feature Engineering
#
# Part 2 turns the seven ETF return series from Part 1 into one constant-weight portfolio.
# Each ETF has weight 1/7, so the portfolio return is their daily mean. Eight historical
# features use information available on or before date t. The target uses t+1 through t+5.

# %%
from pathlib import Path
from IPython.display import display
import hashlib
import json
import sys
import numpy as np
import pandas as pd


def find_project_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / 'notebooks' / '02_feature_engineering.ipynb').is_file() and (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError('Start Jupyter inside the project root or one of its subdirectories.')


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from feature_engineering import (
    ASSETS, FEATURE_COLUMNS, TARGET_COLUMN, TRADING_DAYS_PER_YEAR,
    build_portfolio_features, build_future_volatility_target,
)

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
REPORT_DIR = PROJECT_ROOT / 'outputs' / 'tables'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

returns_path = PROCESSED_DIR / 'daily_returns.csv'
part1_report = json.loads((REPORT_DIR / 'part1_quality_report.json').read_text())
expected_hash = part1_report['processed_sha256']['daily_returns.csv']
assert hashlib.sha256(returns_path.read_bytes()).hexdigest() == expected_hash

returns = pd.read_csv(
    returns_path, parse_dates=['Date'], index_col='Date', float_precision='round_trip'
)
if list(returns.columns) != ASSETS:
    raise ValueError(f'Expected Part 1 return columns in this order: {ASSETS}')
if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
    raise ValueError('Part 1 return dates must be unique and chronological.')
if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
    raise ValueError('Part 1 returns contain missing or non-finite values.')

print('Loaded verified Part 1 daily returns:', returns.shape)
print('Input date range:', returns.index[0].date(), 'to', returns.index[-1].date())

# %% [markdown]
# ## 1. Build the equal-weight portfolio and eight features
#
# `return_5d` and `return_20d` compound the last 5 or 20 portfolio returns, including t.
# Volatility is the trailing sample standard deviation (`ddof=1`) times `sqrt(252)`.
# Drawdown compares current compounded wealth with its running historical peak.
# Average correlation is the mean of the 21 distinct ETF pairs over the last 20 days.

# %%
features = build_portfolio_features(returns)
portfolio_returns = features['return_1d'].copy()
target = build_future_volatility_target(portfolio_returns)

all_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
modeling_before_drop = features.copy()
modeling_before_drop[TARGET_COLUMN] = target
missing_before = modeling_before_drop.isna().sum()
incomplete_rows = modeling_before_drop[all_columns].isna().any(axis=1)
modeling_dataset = modeling_before_drop.dropna(subset=all_columns).copy()
modeling_dataset.index.name = 'Date'

print('Equal-weight portfolio return rows:', len(portfolio_returns))
print('Natural missing values before complete-window removal:')
display(missing_before.to_frame('missing_values'))
print('Complete modeling rows:', len(modeling_dataset))

# %% [markdown]
# ## 2. Future five-day realised volatility target
#
# On date t, the target is the sample standard deviation of portfolio returns at
# t+1, t+2, t+3, t+4 and t+5, annualized by `sqrt(252)`. The return on t is excluded.
# Overlapping target windows are retained because non-overlapping backtests belong to Part 4.

# %%
target_dates = returns.index[:-5]
expected_all_targets = np.array([
    np.std(portfolio_returns.iloc[position + 1:position + 6], ddof=1)
    * np.sqrt(TRADING_DAYS_PER_YEAR)
    for position in range(len(returns) - 5)
])
np.testing.assert_allclose(
    target.loc[target_dates].to_numpy(), expected_all_targets, rtol=1e-12, atol=1e-14
)
assert target.loc[target_dates].notna().all()
assert target.iloc[-5:].isna().all()
assert target.index[-6] == returns.index[-6]

print('First target date:', target_dates[0].date())
print('Last complete target date:', target_dates[-1].date())
print('Unavailable target tail:', target.index[-5:].strftime('%Y-%m-%d').tolist())

# %% [markdown]
# ## 3. Independent formula and leakage checks
#
# Three fixed dates are checked directly from `daily_returns.csv`, without using the feature
# builder for the expected values. Prefix checks then confirm that adding future observations
# cannot change an earlier feature row.

# %%
formula_check_dates = pd.to_datetime(['2010-03-31', '2020-03-16', '2026-09-09'])
portfolio_values = returns.to_numpy().mean(axis=1)
wealth_values = np.cumprod(1 + portfolio_values)
independent_comparisons = 0

for date in formula_check_dates:
    position = returns.index.get_loc(date)
    row = modeling_dataset.loc[date]
    expected_portfolio_return = returns.loc[date].to_numpy().mean()
    np.testing.assert_allclose(portfolio_returns.loc[date], expected_portfolio_return, atol=1e-14)
    independent_comparisons += 1

    expected_features = {
        'return_1d': expected_portfolio_return,
        'return_5d': np.prod(1 + portfolio_values[position - 4:position + 1]) - 1,
        'return_20d': np.prod(1 + portfolio_values[position - 19:position + 1]) - 1,
        'volatility_5d': np.std(portfolio_values[position - 4:position + 1], ddof=1) * np.sqrt(252),
        'volatility_20d': np.std(portfolio_values[position - 19:position + 1], ddof=1) * np.sqrt(252),
        'volatility_60d': np.std(portfolio_values[position - 59:position + 1], ddof=1) * np.sqrt(252),
        'drawdown': wealth_values[position] / max(1.0, wealth_values[:position + 1].max()) - 1,
    }
    correlation_matrix = np.corrcoef(
        returns.iloc[position - 19:position + 1].to_numpy(), rowvar=False
    )
    expected_features['average_correlation_20d'] = correlation_matrix[np.triu_indices(7, 1)].mean()

    for feature_name, expected_value in expected_features.items():
        np.testing.assert_allclose(
            row[feature_name], expected_value, rtol=1e-11, atol=1e-13,
            err_msg=f'{date.date()} {feature_name}',
        )
        independent_comparisons += 1

    expected_target = np.std(portfolio_values[position + 1:position + 6], ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(row[TARGET_COLUMN], expected_target, rtol=1e-11, atol=1e-13)
    independent_comparisons += 1

prefix_check_dates = formula_check_dates
for cutoff in prefix_check_dates:
    prefix_features = build_portfolio_features(returns.loc[:cutoff])
    pd.testing.assert_series_equal(
        prefix_features.loc[cutoff], features.loc[cutoff], rtol=1e-12, atol=1e-14
    )

print(f'Independent formula checks passed: {independent_comparisons} comparisons.')
print(f'Feature prefix-invariance checks passed: {len(prefix_check_dates)} dates.')
print(f'All target windows checked: {len(target_dates)}.')

# %% [markdown]
# ## 4. Validate and save the final dataset
#
# Rows are removed only after all eight features and the target have been calculated.
# The existing filename `volatility_modeling_dataset.csv` is retained for project compatibility.

# %%
expected_index = returns.index[59:-5]
assert modeling_dataset.index.equals(expected_index)
assert list(modeling_dataset.columns) == all_columns
assert modeling_dataset.index.is_unique and modeling_dataset.index.is_monotonic_increasing
assert not modeling_dataset.isna().any().any()
assert np.isfinite(modeling_dataset.to_numpy()).all()
assert (modeling_dataset[['volatility_5d', 'volatility_20d', 'volatility_60d', TARGET_COLUMN]] >= 0).all().all()
assert (modeling_dataset['drawdown'] <= 1e-14).all()
assert modeling_dataset['average_correlation_20d'].between(-1, 1).all()

modeling_path = PROCESSED_DIR / 'volatility_modeling_dataset.csv'
modeling_dataset.reset_index().to_csv(modeling_path, index=False)

feature_summary = modeling_dataset[FEATURE_COLUMNS].describe().T[
    ['count', 'mean', 'std', 'min', 'max']
]
feature_summary.insert(0, 'feature', feature_summary.index)
feature_summary['missing_count'] = modeling_dataset[FEATURE_COLUMNS].isna().sum().to_numpy()
feature_summary.to_csv(REPORT_DIR / 'feature_summary.csv', index=False)

report = {
    'status': 'PASS',
    'assets': ASSETS,
    'portfolio_weights': {asset: 1 / len(ASSETS) for asset in ASSETS},
    'input_file': 'data/processed/daily_returns.csv',
    'input_rows': len(returns),
    'input_start_date': str(returns.index[0].date()),
    'input_end_date': str(returns.index[-1].date()),
    'portfolio_return_rows': len(portfolio_returns),
    'final_modeling_rows': len(modeling_dataset),
    'final_modeling_columns': len(modeling_dataset.columns) + 1,
    'final_start_date': str(modeling_dataset.index[0].date()),
    'final_end_date': str(modeling_dataset.index[-1].date()),
    'feature_list': FEATURE_COLUMNS,
    'target_name': TARGET_COLUMN,
    'rows_removed_for_natural_nan': int(incomplete_rows.sum()),
    'feature_warmup_rows': 59,
    'future_target_tail_rows': 5,
    'missing_before_complete_row_filter': {name: int(value) for name, value in missing_before.items()},
    'missing_values_final': int(modeling_dataset.isna().sum().sum()),
    'duplicate_dates_final': int(modeling_dataset.index.duplicated().sum()),
    'non_finite_feature_values': int((~np.isfinite(modeling_dataset[FEATURE_COLUMNS].to_numpy())).sum()),
    'non_finite_target_values': int((~np.isfinite(modeling_dataset[TARGET_COLUMN].to_numpy())).sum()),
    'independent_formula_validation': {
        'status': 'PASS',
        'dates': formula_check_dates.strftime('%Y-%m-%d').tolist(),
        'comparisons': independent_comparisons,
    },
    'target_alignment_validation': {
        'status': 'PASS',
        'windows_checked': len(target_dates),
        'first_target_date': str(target_dates[0].date()),
        'last_complete_target_date': str(target_dates[-1].date()),
        'unavailable_tail_dates': target.index[-5:].strftime('%Y-%m-%d').tolist(),
        'window_definition': 't+1 through t+5',
    },
    'prefix_invariance_validation': {
        'status': 'PASS',
        'dates': prefix_check_dates.strftime('%Y-%m-%d').tolist(),
    },
    'input_sha256': expected_hash,
    'dataset_sha256': hashlib.sha256(modeling_path.read_bytes()).hexdigest(),
    'feature_summary_sha256': hashlib.sha256((REPORT_DIR / 'feature_summary.csv').read_bytes()).hexdigest(),
}
(REPORT_DIR / 'part2_quality_report.json').write_text(json.dumps(report, indent=2))

print('Final modeling dataset:', modeling_dataset.shape)
print('Date range:', modeling_dataset.index[0].date(), 'to', modeling_dataset.index[-1].date())
display(feature_summary)
print('Saved volatility_modeling_dataset.csv, feature_summary.csv and part2_quality_report.json.')
