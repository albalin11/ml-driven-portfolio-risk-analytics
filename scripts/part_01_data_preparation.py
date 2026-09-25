# %% [markdown]
# # Part 1: Data Preparation
#
# Download and retain a fixed Yahoo Finance snapshot, audit adjusted prices, select common observed dates without price imputation, and calculate simple daily returns.
#
# The cutoff is **2026-09-16**, matching the previous dataset for comparison. The download end is exclusive. Existing snapshots are reused and hash-checked: Yahoo may revise historical adjusted prices, so a fixed cutoff alone is insufficient for exact reproduction. Raw OHLCV fields, cleaned adjusted prices, and returns are separate files.

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

import platform
from datetime import datetime, timezone
import yfinance as yf

TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'DBC']
START_DATE = '2010-01-01'
DATA_CUTOFF = '2026-09-16'
END_DATE_EXCLUSIVE = '2026-09-17'
SNAPSHOT_PATH = RAW_DIR / 'yahoo_ohlcv_20100101_20260916.csv'
METADATA_PATH = RAW_DIR / 'yahoo_ohlcv_20100101_20260916_metadata.json'
# Keep the provider cache inside the project as well.
yf.set_tz_cache_location(str(RAW_DIR / 'yfinance_cache'))

# %% [markdown]
# ## 1. Download or Load the Immutable Raw Snapshot

# %%
if SNAPSHOT_PATH.exists():
    metadata = json.loads(METADATA_PATH.read_text())
    assert hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest() == metadata['sha256']
    assert metadata['tickers'] == TICKERS and metadata['end_exclusive'] == END_DATE_EXCLUSIVE
    raw_data = pd.read_csv(SNAPSHOT_PATH, header=[0, 1], index_col=0, parse_dates=True, float_precision='round_trip')
    print('Loaded verified raw snapshot:', SNAPSHOT_PATH.relative_to(PROJECT_ROOT))
else:
    raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE_EXCLUSIVE,
                           auto_adjust=False, keepna=True, progress=False, threads=False)
    assert not raw_data.empty, 'Download failed; do not substitute old processed data.'
    assert 'Adj Close' in raw_data.columns.get_level_values(0)
    assert set(TICKERS).issubset(raw_data['Adj Close'].columns)
    assert raw_data['Adj Close'][TICKERS].notna().any().all(), 'An asset download failed.'
    raw_data.to_csv(SNAPSHOT_PATH, index_label='Date')
    metadata = dict(source='Yahoo Finance via yfinance', downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                    start=START_DATE, cutoff_inclusive=DATA_CUTOFF, end_exclusive=END_DATE_EXCLUSIVE,
                    tickers=TICKERS, auto_adjust=False, keepna=True,
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, yfinance=yf.__version__),
                    sha256=hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest())
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

prices = raw_data['Adj Close'].reindex(columns=TICKERS).copy()
prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
prices.index.name = 'Date'
assert prices.index.notna().all()
assert prices.index.min() >= pd.Timestamp(START_DATE)
assert prices.index.max() <= pd.Timestamp(DATA_CUTOFF)
print('Raw adjusted-price shape:', prices.shape)

# %% [markdown]
# ## 2. Audit Missing Values and Select Common Observed Dates
#
# No `ffill`, backward fill, interpolation, or zero-return insertion is used. Missing observations are classified as outside asset coverage or inside coverage; an internal gap alone cannot prove a holiday, trading halt, or provider error. Weekends and market holidays absent for every asset are not missing price cells.
#
# Duplicate dates and invalid prices stop execution rather than being silently discarded. Incomplete rows are reported and removed. An internal removed session would make a common-date return span multiple market sessions; this notebook stops in that case pending investigation rather than labelling it a one-day return.

# %%
def audit_frame(frame, *, is_returns=False):
    values = frame.to_numpy(dtype=float)
    audit = dict(rows=len(frame), columns=frame.shape[1], start_date=str(frame.index.min().date()),
                end_date=str(frame.index.max().date()), missing_values=int(frame.isna().sum().sum()),
                duplicate_dates=int(frame.index.duplicated().sum()), infinite_values=int(np.isinf(values).sum()),
                chronological=bool(frame.index.is_monotonic_increasing))
    if is_returns:
        # Negative and zero returns are valid observations, not data errors.
        audit.update(zero_returns=int((values == 0).sum()), negative_returns=int((values < 0).sum()))
    else:
        audit['non_positive_values'] = int((values <= 0).sum())
    return audit

before = audit_frame(prices)
asset_audit = pd.DataFrame({
    'start_date': prices.apply(lambda s: s.first_valid_index()),
    'end_date': prices.apply(lambda s: s.last_valid_index()),
    'observed_prices': prices.count(), 'missing_before': prices.isna().sum()
})
missing_records = []
for asset in TICKERS:
    first, last = prices[asset].first_valid_index(), prices[asset].last_valid_index()
    for date in prices.index[prices[asset].isna()]:
        reason = 'outside_asset_coverage' if date < first or date > last else 'internal_gap_cause_unconfirmed'
        missing_records.append(dict(Date=date, asset=asset, reason=reason))
missing_audit = pd.DataFrame(missing_records, columns=['Date', 'asset', 'reason'])
missing_audit.to_csv(REPORT_DIR / 'part1_missing_observations.csv', index=False)
assert before['duplicate_dates'] == 0, 'Duplicate dates require investigation.'
assert before['non_positive_values'] == 0 and before['infinite_values'] == 0
prices = prices.sort_index()
clean_prices = prices.dropna(how='any').copy()
assert not clean_prices.empty
removed_dates = prices.index.difference(clean_prices.index)
internal_removed = removed_dates[(removed_dates >= clean_prices.index.min()) & (removed_dates <= clean_prices.index.max())]
assert len(internal_removed) == 0, 'Investigate missing internal sessions before calculating daily returns.'
asset_audit['missing_after'] = clean_prices.isna().sum()
asset_audit.to_csv(REPORT_DIR / 'part1_asset_audit.csv', index_label='asset')
display(asset_audit)
display(missing_audit)
print('Before cleaning:', before)
print('After cleaning:', audit_frame(clean_prices))
print('Removed dates:', removed_dates.strftime('%Y-%m-%d').tolist())
print('Missing-price diagnosis:', 'No missing adjusted-price cells in this snapshot.' if not missing_records else 'See missing observation report; causes are not assumed.')

# %% [markdown]
# ## 3. Calculate and Validate Daily Returns
#
# For each asset, `return[t] = price[t] / price[t-1] - 1`. Only the first row lacks a prior observed price. A genuine unchanged price may produce a zero return; no zeros are manufactured by filling missing prices.

# %%
untrimmed_returns = clean_prices.pct_change(fill_method=None)
assert untrimmed_returns.iloc[0].isna().all()
daily_returns = untrimmed_returns.iloc[1:].copy()
assert daily_returns.index.equals(clean_prices.index[1:])
for frame in (clean_prices, daily_returns):
    assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert np.isfinite(frame.to_numpy()).all()
np.testing.assert_allclose(daily_returns, clean_prices.to_numpy()[1:] / clean_prices.to_numpy()[:-1] - 1)
print('Initial return NaNs (expected):', int(untrimmed_returns.isna().sum().sum()))
print('Validated daily returns:', audit_frame(daily_returns, is_returns=True))
print('Observed zero returns by asset:')
display(daily_returns.eq(0).sum().to_frame('zero_returns'))
print('Daily return shape:', daily_returns.shape)

# %% [markdown]
# ## 4. Save Processed Data and the Audit Trail

# %%
clean_prices.to_csv(PROCESSED_DIR / 'clean_adjusted_close_prices.csv', index_label='Date')
daily_returns.to_csv(PROCESSED_DIR / 'daily_returns.csv', index_label='Date')
report = dict(raw_adjusted_prices=before, cleaned_prices=audit_frame(clean_prices),
              daily_returns=audit_frame(daily_returns, is_returns=True), removed_price_rows=len(prices)-len(clean_prices),
              expected_initial_return_nans=len(TICKERS), raw_snapshot_sha256=metadata['sha256'],
              cutoff=DATA_CUTOFF, processed_sha256={name: hashlib.sha256((PROCESSED_DIR / name).read_bytes()).hexdigest()
                  for name in ['clean_adjusted_close_prices.csv', 'daily_returns.csv']})
(REPORT_DIR / 'part1_quality_report.json').write_text(json.dumps(report, indent=2))
print('Part 1 completed successfully. Raw snapshot and audit reports retained.')
