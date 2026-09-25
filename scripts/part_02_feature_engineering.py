# %% [markdown]
# # Part 2: Feature Engineering
#
# This notebook builds a clean, chronological modeling dataset for future volatility forecasting. All features use information available on the current date or earlier. The target uses the next five trading days and is kept separate from the feature calculations.

# %%
from pathlib import Path
from IPython.display import display
import hashlib
import json
import numpy as np
import pandas as pd

def find_project_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / 'notebooks' / '01_data_preparation.ipynb').is_file() and (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError('Start Jupyter inside the project root or one of its subdirectories.')

PROJECT_ROOT = find_project_root()
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
REPORT_DIR = PROJECT_ROOT / 'outputs' / 'tables'
for directory in (PROCESSED_DIR, RAW_DIR, REPORT_DIR):
    assert directory.resolve().is_relative_to(PROJECT_ROOT)
    directory.mkdir(parents=True, exist_ok=True)
print('Project root verified.')
ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'DBC']
TRADING_DAYS_PER_YEAR = 252
VOLATILITY_WINDOWS = [5, 20, 60]

prices = pd.read_csv(
    PROCESSED_DIR / 'clean_adjusted_close_prices.csv',
    parse_dates=['Date'],
    index_col='Date'
)
returns = pd.read_csv(
    PROCESSED_DIR / 'daily_returns.csv',
    parse_dates=['Date'],
    index_col='Date'
)

prices = prices[ASSETS]
returns = returns[ASSETS]
print(f'Loaded prices: {prices.shape}')
print(f'Loaded returns: {returns.shape}')
part1_report = json.loads((REPORT_DIR / 'part1_quality_report.json').read_text())
for name, digest in part1_report['processed_sha256'].items():
    assert hashlib.sha256((PROCESSED_DIR / name).read_bytes()).hexdigest() == digest
for frame in (prices, returns):
    assert list(frame.columns) == ASSETS
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert np.isfinite(frame.to_numpy()).all()
assert (prices > 0).all().all()
assert returns.index.equals(prices.index[1:])
np.testing.assert_allclose(returns, prices.pct_change(fill_method=None).iloc[1:], atol=1e-14)

# %% [markdown]
# ## 1. Return, Volatility, and Drawdown Features
#
# Return features use `pct_change` over trailing windows. Historical volatility is the rolling sample standard deviation of daily returns, annualized by `sqrt(252)`. Drawdown is the current price relative to the running historical peak for each asset.

# %%
def build_features(prices, returns):
    feature_frames = []

    for asset in ASSETS:
        asset_prices = prices[asset]
        asset_returns = returns[asset]
        asset_features = pd.DataFrame(index=prices.index)
        asset_features['Date'] = asset_features.index
        asset_features['asset'] = asset

        asset_features['return_1d'] = asset_returns
        asset_features['return_5d'] = asset_prices.pct_change(5, fill_method=None)
        asset_features['return_20d'] = asset_prices.pct_change(20, fill_method=None)

        for window in VOLATILITY_WINDOWS:
            asset_features[f'volatility_{window}d'] = (
                asset_returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
            )

        running_peak = asset_prices.cummax()
        asset_features['drawdown'] = asset_prices / running_peak - 1

        spy_returns = returns['SPY']
        if asset == 'SPY':
            asset_features['correlation_to_spy_20d'] = 1.0
        else:
            asset_features['correlation_to_spy_20d'] = asset_returns.rolling(20).corr(spy_returns)

        asset_features['momentum_60d'] = asset_prices.pct_change(60, fill_method=None)
        asset_features['momentum_120d'] = asset_prices.pct_change(120, fill_method=None)
        feature_frames.append(asset_features.reset_index(drop=True))

    features = pd.concat(feature_frames, ignore_index=True)
    return features

features = build_features(prices, returns)
print("Feature rows before target construction:", len(features))

# %% [markdown]
# ## 2. Future 5-Day Realised Volatility Target
#
# For date `t`, the target is the sample standard deviation of returns at `t+1` through `t+5`, annualized by `sqrt(252)`. The explicit negative shifts ensure that the current date is excluded and the target is forward-looking.

# %%
target_frames = []
for asset in ASSETS:
    future_returns = pd.concat(
        [returns[asset].shift(-step) for step in range(1, 6)],
        axis=1
    )
    complete_future_window = future_returns.notna().sum(axis=1).eq(5)
    future_realised_volatility = (
        future_returns.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).where(complete_future_window)
    target_frames.append(pd.DataFrame({
        'Date': returns.index,
        'asset': asset,
        'future_5d_realised_volatility': future_realised_volatility.to_numpy()
    }))

targets = pd.concat(target_frames, ignore_index=True)
modeling_dataset = features.merge(targets, on=['Date', 'asset'], how='left', validate='one_to_one')
modeling_dataset = modeling_dataset.sort_values(['Date', 'asset']).reset_index(drop=True)

feature_columns = [
    'return_1d', 'return_5d', 'return_20d',
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'drawdown', 'correlation_to_spy_20d',
    'momentum_60d', 'momentum_120d'
]
target_column = 'future_5d_realised_volatility'
assert not np.isinf(modeling_dataset[feature_columns + [target_column]].to_numpy()).any(), 'Investigate infinite values.'
missing_before = modeling_dataset[feature_columns + [target_column]].isna().sum()
feature_invalid = modeling_dataset[feature_columns].isna().any(axis=1)
target_invalid = modeling_dataset[target_column].isna()
rows_before = len(modeling_dataset)
print('NaNs before warm-up and target-tail removal:')
display(missing_before.to_frame('missing_values'))
print('Feature warm-up rows:', int(feature_invalid.sum()))
print('Target unavailable rows (including initial price date):', int(target_invalid.sum()))
modeling_dataset = modeling_dataset.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

print(f'Modeling dataset shape: {modeling_dataset.shape}')
print('Feature columns:', list(modeling_dataset.columns))

# %% [markdown]
# ## 3. Data Quality and Leakage Checks
#
# The checks below verify chronology, uniqueness, missing values, infinite values, feature coverage, and the expected target alignment. No random split or model training is performed in Part 2.
#
# Features are available **after the close of date t**. The target excludes t and uses t+1 through t+5. Removing future data is tested at three cutoffs to check feature causality; every retained target is independently checked with NumPy. A later train/validation split must be chronological and purge training labels whose five-day outcome window overlaps validation. No fitted scaler, imputer, or random split is introduced here.
#
# Adjusted prices are a downloaded historical snapshot, not a point-in-time vintage database. These tests establish calculation causality, not historical provider revision immunity. Momentum retains the original 60/120-day definitions; SPY correlation to itself is 1.0.

# %%
expected_columns = ['Date', 'asset'] + feature_columns + [target_column]
assert list(modeling_dataset.columns) == expected_columns
assert modeling_dataset['Date'].is_monotonic_increasing
assert not modeling_dataset.duplicated().any()
assert not modeling_dataset.duplicated(['Date', 'asset']).any()
assert not modeling_dataset[feature_columns + [target_column]].isna().any().any()
assert not np.isinf(modeling_dataset[feature_columns + [target_column]].to_numpy()).any()
assert set(modeling_dataset['asset']) == set(ASSETS)
assert modeling_dataset['Date'].max() == returns.index[-6]

for asset in ASSETS:
    asset_rows = modeling_dataset[modeling_dataset['asset'].eq(asset)].set_index('Date')
    assert np.allclose(asset_rows['return_1d'], returns.loc[asset_rows.index, asset])
    assert np.allclose(asset_rows['return_5d'], prices[asset].pct_change(5, fill_method=None).loc[asset_rows.index])
    assert np.allclose(asset_rows['drawdown'], (prices[asset] / prices[asset].cummax() - 1).loc[asset_rows.index])

first_modeling_date = modeling_dataset['Date'].min()
last_modeling_date = modeling_dataset['Date'].max()
print('Feature names:', feature_columns)
print(f'Rows: {len(modeling_dataset):,}')
print(f'Columns: {len(modeling_dataset.columns)}')
print(f'Date range: {first_modeling_date.date()} to {last_modeling_date.date()}')
print('Observations by asset:')
print(modeling_dataset.groupby('asset').size().to_string())
print('Missing values:')
print(modeling_dataset.isna().sum().to_string())
print('Duplicate rows:', modeling_dataset.duplicated().sum())
print('Duplicate Date + asset combinations:', modeling_dataset.duplicated(['Date', 'asset']).sum())
print('Infinite values:', np.isinf(modeling_dataset[feature_columns + [target_column]].to_numpy()).sum())
print('Chronological order:', modeling_dataset['Date'].is_monotonic_increasing)
print('Feature formula checks passed.')
print('Basic descriptive statistics:')
display(modeling_dataset[feature_columns + [target_column]].describe().T)
print('Per-asset coverage verified above.')
# The only row exclusions are the 120-date warm-up and final five dates.
expected_dates = prices.index[120:-5]
assert len(modeling_dataset) == len(expected_dates) * len(ASSETS)
for asset in ASSETS:
    rows = modeling_dataset.loc[modeling_dataset.asset.eq(asset)].set_index('Date')
    assert rows.index.equals(expected_dates)
    # Independent NumPy target verification for every retained date.
    locations = returns.index.get_indexer(rows.index)
    values = returns[asset].to_numpy()
    expected = np.array([np.std(values[i+1:i+6], ddof=1) * np.sqrt(252) for i in locations])
    np.testing.assert_allclose(rows[target_column], expected, rtol=1e-10, atol=1e-12)

# Independent formula checks on fixed dates for all assets, including SPY itself.
# Use direct price ratios and NumPy statistics, never build_features().
formula_check_dates = pd.to_datetime(['2010-06-25', '2020-03-16', '2026-09-09'])
for asset in ASSETS:
    asset_rows = modeling_dataset.loc[modeling_dataset.asset.eq(asset)].set_index('Date')
    for date in formula_check_dates:
        row = asset_rows.loc[date]
        price_position = prices.index.get_loc(date)
        return_position = returns.index.get_loc(date)
        for feature, lag in [('return_20d', 20), ('momentum_60d', 60), ('momentum_120d', 120)]:
            expected = prices[asset].iloc[price_position] / prices[asset].iloc[price_position-lag] - 1
            np.testing.assert_allclose(row[feature], expected, rtol=1e-10, atol=1e-12,
                                       err_msg=f'{asset} {date.date()} {feature}')
        for window in [5, 20, 60]:
            trailing_returns = returns[asset].iloc[return_position-window+1:return_position+1].to_numpy()
            assert len(trailing_returns) == window
            expected = np.std(trailing_returns, ddof=1) * np.sqrt(252)
            np.testing.assert_allclose(row[f'volatility_{window}d'], expected, rtol=1e-10, atol=1e-12,
                                       err_msg=f'{asset} {date.date()} volatility_{window}d')
        asset_window = returns[asset].iloc[return_position-19:return_position+1].to_numpy()
        spy_window = returns['SPY'].iloc[return_position-19:return_position+1].to_numpy()
        assert len(asset_window) == len(spy_window) == 20
        expected_correlation = np.corrcoef(asset_window, spy_window)[0, 1]
        np.testing.assert_allclose(row['correlation_to_spy_20d'], expected_correlation,
                                   rtol=1e-10, atol=1e-12,
                                   err_msg=f'{asset} {date.date()} correlation_to_spy_20d')
print('Independent feature formula checks passed: 7 features x 7 assets x 3 fixed dates = 147 checks.')

# Causality regression: removing all later data cannot change existing features.
for cutoff in [expected_dates[0], expected_dates[len(expected_dates)//2], expected_dates[-1]]:
    prefix = build_features(prices.loc[:cutoff], returns.loc[:cutoff])
    full = features.loc[features.Date.le(cutoff)]
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), full.reset_index(drop=True), rtol=1e-10, atol=1e-12)
print('All-row target alignment and feature prefix-invariance checks passed.')

# %%
sample_asset = 'SPY'
sample_date = modeling_dataset.loc[modeling_dataset['asset'].eq(sample_asset), 'Date'].iloc[0]
sample_target = modeling_dataset.loc[
    (modeling_dataset['asset'] == sample_asset) & (modeling_dataset['Date'] == sample_date),
    target_column
].iloc[0]
sample_future_returns = returns.loc[returns.index > sample_date, sample_asset].iloc[:5]
expected_target = sample_future_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
assert np.isclose(sample_target, expected_target)
assert sample_date < sample_future_returns.index.min()
assert sample_future_returns.index.max() > sample_date
print('Target alignment check passed.')
print('Feature date:', sample_date.date())
print('Future return dates used by the target:', sample_future_returns.index.strftime('%Y-%m-%d').tolist())
print(f'Calculated target: {sample_target:.6f}')
print(f'Independent check: {expected_target:.6f}')

# %% [markdown]
# ## 4. Save the Modeling Dataset
#
# The final dataset is sorted chronologically and saved as one CSV for Part 3.

# %%
modeling_dataset_path = PROCESSED_DIR / 'volatility_modeling_dataset.csv'
modeling_dataset.to_csv(modeling_dataset_path, index=False)
print(f'Saved modeling dataset: {modeling_dataset_path.relative_to(PROJECT_ROOT)}')
print(f'File size: {modeling_dataset_path.stat().st_size:,} bytes')
report = dict(rows=len(modeling_dataset), columns=len(modeling_dataset.columns),
              start_date=str(first_modeling_date.date()), end_date=str(last_modeling_date.date()),
              rows_before=rows_before, rows_removed=rows_before-len(modeling_dataset),
              feature_warmup_rows=int(feature_invalid.sum()), target_unavailable_rows=int(target_invalid.sum()),
              missing_before={k:int(v) for k,v in missing_before.items()}, missing_after=int(modeling_dataset.isna().sum().sum()),
              infinite_values=0, duplicate_date_asset=0, all_targets_verified=True,
              prefix_invariance_checks=3, input_sha256=part1_report['processed_sha256'],
              dataset_sha256=hashlib.sha256(modeling_dataset_path.read_bytes()).hexdigest())
(REPORT_DIR / 'part2_quality_report.json').write_text(json.dumps(report, indent=2))
print('Part 2 completed successfully using verified Part 1 inputs.')
