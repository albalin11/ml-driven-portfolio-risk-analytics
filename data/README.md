# Data provenance and reproduction

The seven ETFs are SPY, QQQ, IWM, TLT, HYG, GLD and DBC. Part 1 downloads daily
Yahoo Finance OHLCV fields with `auto_adjust=False` and uses `Adj Close` for returns.
The start request is 2010-01-01 and the end request is 2026-09-17 (exclusive).
The analysis therefore ends on 2026-09-16. No price imputation is used.

The local raw snapshot and metadata are retained with SHA-256 verification. They
are not committed. The published project does not redistribute raw quotes,
per-asset price/return tables, or a complete derived observation-level dataset.
The public figures and aggregate tables document this research; they are not
a market-data feed or a grant of rights to underlying third-party data.

The [yfinance project](https://github.com/ranaroussi/yfinance#readme) distinguishes
its software license from rights to the downloaded data and points to
[Yahoo's terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html).
Review applicable provider terms for your own use; downloading through an
open-source library does not grant redistribution permission.

Running Part 1 without the local snapshot downloads the data and records a new
vintage. Provider revisions can change prices and results even with a fixed end
date. Exact reproduction of the reported numbers requires the recorded vintage
and dependency versions. Tests compare byte-level hashes only when the raw
snapshot matches the recorded reference; all mathematical checks run regardless.

There is no bundled real-data sample. The unit tests include synthetic returns
for checking formulas without publishing market observations.
