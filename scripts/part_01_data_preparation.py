# %% [markdown]
# # Part 1: Data Preparation
#
# Download seven ETFs, align their trading dates, calculate daily returns and save clean CSVs.
# SPY, QQQ and IWM cover US equities; TLT and HYG cover Treasury and high-yield bonds;
# GLD covers gold and DBC covers commodity futures.
# Returns use Adj Close to account for the provider's split and distribution adjustments.
# The 2010-01-01 start and 2026-09-16 inclusive cutoff are fixed project choices.

# %%
from pathlib import Path
from IPython.display import display
import hashlib
import json
import numpy as np
import pandas as pd
import sys


def find_project_root():
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / 'notebooks' / '01_data_preparation.ipynb').is_file() and (candidate / 'src').is_dir():
            return candidate
    raise RuntimeError('Start Jupyter inside the project root or one of its subdirectories.')

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from data_preparation_checks import check_trading_dates, check_adjusted_prices, check_ohlcv

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
# ## 1. Download or Load Historical Prices
#
# Reuse the saved Yahoo Finance snapshot and verify its checksum. Download only if it is absent.
# Keeping the snapshot preserves this data vintage if the provider later revises historical prices.

# %%
snapshot_exists = SNAPSHOT_PATH.exists()
if snapshot_exists:
    metadata = json.loads(METADATA_PATH.read_text())
    assert hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest() == metadata['sha256'], 'Raw snapshot checksum mismatch.'
    assert metadata['tickers'] == TICKERS and metadata['end_exclusive'] == END_DATE_EXCLUSIVE
    assert metadata['start'] == START_DATE and metadata['cutoff_inclusive'] == DATA_CUTOFF
    raw_data = pd.read_csv(SNAPSHOT_PATH, header=[0, 1], index_col=0, parse_dates=True, float_precision='round_trip')
    print('Loaded verified raw snapshot:', SNAPSHOT_PATH.relative_to(PROJECT_ROOT))
else:
    raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE_EXCLUSIVE,
                           auto_adjust=False, keepna=True, progress=False, threads=False)
    assert not raw_data.empty, 'Download failed; do not substitute old processed data.'

# Both loaded and downloaded data must pass before any new snapshot is saved.
raw_data.index = pd.DatetimeIndex(raw_data.index)
calendar_audit = check_trading_dates(raw_data.index, START_DATE, DATA_CUTOFF, TICKERS)
if not isinstance(raw_data.columns, pd.MultiIndex) or raw_data.columns.nlevels != 2 or not raw_data.columns.is_unique:
    raise ValueError('Expected unique (field, asset) columns in the raw snapshot.')
missing_fields = [asset for asset in TICKERS if ('Adj Close', asset) not in raw_data.columns]
if missing_fields:
    raise ValueError(f'Missing Adj Close field for tickers: {missing_fields}; no Close substitution allowed.')
clean_prices = raw_data['Adj Close'][TICKERS].copy()
clean_prices.index.name = 'Date'
check_adjusted_prices(clean_prices, TICKERS)
check_ohlcv(raw_data, TICKERS)
if not snapshot_exists:
    raw_data.to_csv(SNAPSHOT_PATH, index_label='Date')
    metadata = dict(source='Yahoo Finance via yfinance', downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                    start=START_DATE, cutoff_inclusive=DATA_CUTOFF, end_exclusive=END_DATE_EXCLUSIVE,
                    tickers=TICKERS, auto_adjust=False, keepna=True,
                    versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, yfinance=yf.__version__),
                    sha256=hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest())
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


# %% [markdown]
# ## 2. Check Dates and Missing Prices
#
# The XNYS calendar supplies US equity trading sessions, including early-close days and
# special closures. Each ETF must have a positive, finite Adj Close on every expected date.
# Missing dates or prices stop the process for investigation; nothing is filled or dropped.
# Existing OHLCV fields receive basic consistency checks; Adj Close is not compared with OHLC ranges.
# This snapshot is complete, so the aligned price table needs no further cleaning.

# %%
print('Trading-calendar check:', calendar_audit)
display(clean_prices.head())

def audit_frame(frame, *, is_returns=False):
    values = frame.to_numpy(dtype=float)
    audit = dict(rows=len(frame), columns=frame.shape[1], start_date=str(frame.index.min().date()),
                end_date=str(frame.index.max().date()), missing_values=int(frame.isna().sum().sum()),
                duplicate_dates=int(frame.index.duplicated().sum()), infinite_values=int(np.isinf(values).sum()),
                chronological=bool(frame.index.is_monotonic_increasing),
                invalid_values=int((~np.isfinite(values)).sum()))
    if is_returns:
        # Negative and zero returns are valid observations, not data errors.
        audit.update(zero_returns=int((values == 0).sum()), negative_returns=int((values < 0).sum()))
    else:
        audit['non_positive_values'] = int((values <= 0).sum())
        audit['invalid_values'] = int((~np.isfinite(values) | (values <= 0)).sum())
    return audit

print('Clean prices:', audit_frame(clean_prices))

# %% [markdown]
# ## 3. Calculate and Validate Daily Returns
#
# For each asset, `return[t] = price[t] / price[t-1] - 1`. Only the first row lacks a prior observed price. A genuine unchanged price may produce a zero return; no zeros are manufactured by filling missing prices.

# %%
untrimmed_returns = clean_prices.pct_change(fill_method=None)
assert untrimmed_returns.iloc[0].isna().all()
daily_returns = untrimmed_returns.iloc[1:].copy()
assert len(daily_returns) == len(clean_prices) - 1
assert daily_returns.index.equals(clean_prices.index[1:])
assert np.isfinite(daily_returns.to_numpy()).all(), 'Daily returns contain NaN or infinity.'
np.testing.assert_allclose(daily_returns, clean_prices.to_numpy()[1:] / clean_prices.to_numpy()[:-1] - 1)
print('Initial return NaNs (expected):', int(untrimmed_returns.isna().sum().sum()))
print('Validated daily returns:', audit_frame(daily_returns, is_returns=True))
display(daily_returns.head())

# %% [markdown]
# ## 4. Save Clean Data
#
# Save the aligned prices and returns as separate CSVs. A short quality report records their
# shape, date range and checksums so later stages can verify which data they use.

# %%
clean_prices.to_csv(PROCESSED_DIR / 'clean_adjusted_close_prices.csv', index_label='Date')
daily_returns.to_csv(PROCESSED_DIR / 'daily_returns.csv', index_label='Date')
report = dict(assets=TICKERS, source=metadata['source'], requested_start=START_DATE,
              cross_asset_missing_prices=int(clean_prices.isna().sum().sum()),
              cleaned_prices=audit_frame(clean_prices),
              daily_returns=audit_frame(daily_returns, is_returns=True),
              calendar_audit=calendar_audit, cutoff=DATA_CUTOFF,
              raw_snapshot_sha256=metadata['sha256'],
              processed_sha256={name: hashlib.sha256((PROCESSED_DIR / name).read_bytes()).hexdigest()
                  for name in ['clean_adjusted_close_prices.csv', 'daily_returns.csv']})
(REPORT_DIR / 'part1_quality_report.json').write_text(json.dumps(report, indent=2))
print('Saved clean_adjusted_close_prices.csv, daily_returns.csv and part1_quality_report.json.')
