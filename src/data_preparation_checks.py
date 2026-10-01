"""Trading-session and raw-price checks shared by Part 1 and its tests."""
import numpy as np
import pandas as pd
import exchange_calendars as xcals


def check_trading_dates(dates, start, end):
    """Require every expected US equity session, including both sample boundaries."""
    dates = pd.DatetimeIndex(dates)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError('Expected timezone-naive daily session labels without NaT or intraday times.')
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique and chronological.')
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Padding lets the requested boundary itself be a holiday or weekend.
    calendar = xcals.get_calendar('XNYS', start=start-pd.Timedelta(days=7), end=end+pd.Timedelta(days=7))
    expected = calendar.sessions_in_range(start, end).tz_localize(None)
    missing = expected.difference(dates)
    unexpected = dates.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f'Missing trading sessions ({len(missing)}): {missing.strftime("%Y-%m-%d").tolist()}; '
                         f'unexpected dates ({len(unexpected)}): {unexpected.strftime("%Y-%m-%d").tolist()}')
    days = pd.date_range(start, end)
    weekends = days[days.dayofweek >= 5]
    regular = calendar.regular_holidays.holidays(start, end)
    special = pd.DatetimeIndex(calendar.adhoc_holidays)
    special = special[(special >= start) & (special <= end)]
    closed_weekdays = days.difference(weekends).difference(expected)
    unexplained = closed_weekdays.difference(regular).difference(special)
    if len(unexplained):
        raise ValueError(f'Calendar closure classification needs review: {unexplained.tolist()}')
    return dict(calendar='XNYS', package_version=xcals.__version__, expected_sessions=len(expected),
                expected_first=str(expected[0].date()), expected_last=str(expected[-1].date()),
                weekend_dates=len(weekends), regular_holiday_dates=len(closed_weekdays.intersection(regular)),
                special_closures=special.strftime('%Y-%m-%d').tolist(), missing_sessions=0, unexpected_dates=0)


def check_raw_bars(raw, tickers):
    """Check comparable OHLC fields together; Adj Close has a different adjustment basis."""
    fields = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not isinstance(raw.columns, pd.MultiIndex) or not raw.columns.is_unique:
        raise ValueError('Expected unique (field, asset) raw columns.')
    summary = {}
    for asset in tickers:
        missing_columns = [field for field in fields if (field, asset) not in raw.columns]
        if missing_columns:
            raise ValueError(f'{asset}: missing raw fields {missing_columns}')
        bars = raw.xs(asset, axis=1, level=1)[fields]
        if not np.isfinite(bars.to_numpy(dtype=float)).all():
            raise ValueError(f'{asset}: missing or non-finite OHLCV/Adj Close; missing counts {bars.isna().sum().to_dict()}')
        if (bars[fields[:-1]] <= 0).any().any() or (bars.Volume < 0).any():
            raise ValueError(f'{asset}: non-positive price or negative volume.')
        tolerance = bars.Close * 1e-8
        invalid_range = ((bars.Low > bars[['Open', 'Close', 'High']].min(axis=1) + tolerance)
                         | (bars.High < bars[['Open', 'Close', 'Low']].max(axis=1) - tolerance))
        if invalid_range.any():
            raise ValueError(f'{asset}: inconsistent OHLC range on {bars.index[invalid_range].strftime("%Y-%m-%d").tolist()}')
        summary[asset] = dict(missing_values=0, non_finite_values=0, non_positive_prices=0,
                              invalid_ohlc_rows=0, negative_volume_rows=0,
                              zero_volume_rows=int(bars.Volume.eq(0).sum()))
    return summary
