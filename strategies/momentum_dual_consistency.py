import pandas as pd
import numpy as np

# Updated Parameter Space:
# Reduced lookback periods to capture trends faster and ensure trades occur 
# even in shorter datasets (e.g., 2-3 years).
param_space = {
    'long_lookback': {'type': 'int', 'low': 40, 'high': 150},   # Was 120-260
    'short_lookback': {'type': 'int', 'low': 10, 'high': 40},   # Was 40-120
    'consistency_window': {'type': 'int', 'low': 20, 'high': 60}, # Window in Days (approx 1-3 months)
    'consistency_threshold': {'type': 'float', 'low': 0.5, 'high': 0.8}
}

compatible_timeframes = ['1 day', '4 hours']

def _calculate_consistency_daily(series, window_days):
    """
    Calculates consistency based on daily positive returns over a rolling window.
    (More granular than monthly for shorter timeframes).
    """
    daily_returns = series.pct_change()
    is_positive = (daily_returns > 0).astype(int)
    # Percentage of days in the window that were positive
    consistency = is_positive.rolling(window=window_days).mean()
    return consistency

def strategy_function(
    df: pd.DataFrame,
    long_lookback: int = 100,
    short_lookback: int = 20,
    consistency_window: int = 40,
    consistency_threshold: float = 0.55
) -> pd.DataFrame:
    """
    Dual Momentum with Trend Consistency.
    
    Logic:
    1. Long Term Momentum must be positive.
    2. Short Term Momentum must be positive.
    3. The quality of the trend (Consistency) must be smooth, not just one big jump.
    """
    strategy_df = df.copy()
    
    # 1. Momentum Calculation (Rate of Change)
    # Using 'close' vs 'close' N days ago
    strategy_df['long_momentum'] = strategy_df['close'].pct_change(periods=long_lookback)
    strategy_df['short_momentum'] = strategy_df['close'].pct_change(periods=short_lookback)
    
    # 2. Consistency Calculation
    strategy_df['consistency'] = _calculate_consistency_daily(strategy_df['close'], consistency_window)
    
    # 3. Entry Logic
    go_long = (
        (strategy_df['long_momentum'] > 0) &
        (strategy_df['short_momentum'] > 0) &
        (strategy_df['consistency'] > consistency_threshold)
    )
    
    # 4. Exit Logic (Simple Regime Filter)
    # If Long Momentum turns negative, we exit immediately (Risk Off)
    go_flat = (strategy_df['long_momentum'] < 0)

    # 5. Position Sizing
    # 1 for Long, 0 for Flat. (This strategy is Long Only for safety, can be adapted)
    raw_signal = np.where(go_long, 1, np.where(go_flat, 0, np.nan))
    
    strategy_df['position'] = pd.Series(raw_signal, index=strategy_df.index).ffill().fillna(0).astype(int)

    return strategy_df