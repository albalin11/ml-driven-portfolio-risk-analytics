"""Feature and target formulas for the equal-weight portfolio in Part 2."""
import numpy as np
import pandas as pd


ASSETS = ['SPY', 'QQQ', 'IWM', 'TLT', 'HYG', 'GLD', 'DBC']
FEATURE_COLUMNS = [
    'return_1d', 'return_5d', 'return_20d',
    'volatility_5d', 'volatility_20d', 'volatility_60d',
    'drawdown', 'average_correlation_20d',
]
TARGET_COLUMN = 'future_5d_realised_volatility'
TRADING_DAYS_PER_YEAR = 252


def build_portfolio_features(asset_returns):
    """Build eight backward-looking features for a constant-weight portfolio."""
    if list(asset_returns.columns) != ASSETS:
        raise ValueError(f'Expected return columns in this order: {ASSETS}')

    portfolio_returns = asset_returns.mean(axis=1)
    features = pd.DataFrame(index=asset_returns.index)
    features['return_1d'] = portfolio_returns

    for window in [5, 20]:
        features[f'return_{window}d'] = (
            (1 + portfolio_returns).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
        )

    for window in [5, 20, 60]:
        features[f'volatility_{window}d'] = (
            portfolio_returns.rolling(window, min_periods=window).std(ddof=1)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    wealth = (1 + portfolio_returns).cumprod()
    running_peak = wealth.cummax().clip(lower=1.0)
    features['drawdown'] = wealth / running_peak - 1

    pairwise_correlations = []
    for first_position, first_asset in enumerate(ASSETS):
        for second_asset in ASSETS[first_position + 1:]:
            correlation = asset_returns[first_asset].rolling(20, min_periods=20).corr(
                asset_returns[second_asset]
            )
            pairwise_correlations.append(correlation)
    features['average_correlation_20d'] = pd.concat(
        pairwise_correlations, axis=1
    ).mean(axis=1, skipna=False)

    return features[FEATURE_COLUMNS]


def build_future_volatility_target(portfolio_returns):
    """Use returns t+1 through t+5 for the annualized realised-volatility target."""
    future_returns = pd.concat(
        [portfolio_returns.shift(-step) for step in range(1, 6)], axis=1
    )
    complete_window = future_returns.notna().all(axis=1)
    target = future_returns.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return target.where(complete_window).rename(TARGET_COLUMN)
