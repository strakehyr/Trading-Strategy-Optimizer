import pandas as pd
import numpy as np


# Each strategy file must define these two variables:
# 1. param_space: A dictionary defining the optimizable parameters.
# 2. strategy_function: The function that implements the strategy logic.

# This strategy is timeframe-agnostic and can run on any timeframe.
compatible_timeframes = ['1 hour', '4 hours', '1 day']

param_space = {
    'trend_ema_period': {'type': 'int', 'low': 100, 'high': 250},
    'rsi_period': {'type': 'int', 'low': 7, 'high': 21},
    'rsi_bullish_level': {'type': 'int', 'low': 55, 'high': 70},
    'rsi_bearish_level': {'type': 'int', 'low': 30, 'high': 45}
}
def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def strategy_function(
    df: pd.DataFrame,
    trend_ema_period: int = 200,
    rsi_period: int = 14,
    rsi_bullish_level: int = 60,
    rsi_bearish_level: int = 40
) -> pd.DataFrame:
    """
    Implements a Trend-Filtered RSI strategy, capturing the logic of indicators
    like the "Trading Oracle Platinum Plus".

    This is a "stop-and-reverse" strategy that aims to be in the market at all times,
    either long or short, based on confirmed trend and momentum.

    - Long Entry: Price must be above the long-term EMA, AND the RSI must cross above a bullish level.
    - Short Entry: Price must be below the long-term EMA, AND the RSI must cross below a bearish level.
    """
    strategy_df = df.copy()

    # 1. Calculate Indicators using pandas_ta for convenience and accuracy
    # The trend baseline
    strategy_df['ema_trend'] = strategy_df['close'].ewm(span=trend_ema_period, adjust=False).mean()
    strategy_df['rsi'] = _rsi(strategy_df['close'], rsi_period)

    # 2. Define Trend Conditions
    is_bullish_trend = strategy_df['close'] > strategy_df['ema_trend']
    is_bearish_trend = strategy_df['close'] < strategy_df['ema_trend']

    # 3. Define Momentum Crossover Signals
    crossed_above_bull_level = (strategy_df['rsi'].shift(1) < rsi_bullish_level) & (strategy_df['rsi'] > rsi_bullish_level)
    crossed_below_bear_level = (strategy_df['rsi'].shift(1) > rsi_bearish_level) & (strategy_df['rsi'] < rsi_bearish_level)

    # 4. Generate Entry Signals based on confirmed trend and momentum
    long_entry_signal = is_bullish_trend & crossed_above_bull_level
    short_entry_signal = is_bearish_trend & crossed_below_bear_level

    # 5. Generate Position State
    # Create raw signals (1 for buy, -1 for sell) ONLY on the signal bar.
    raw_signal = np.where(long_entry_signal, 1, np.nan)
    raw_signal = np.where(short_entry_signal, -1, raw_signal)

    # Propagate the last valid signal forward to hold the position (stop-and-reverse logic)
    strategy_df['position'] = pd.Series(raw_signal, index=strategy_df.index).ffill().fillna(0).astype(int)
    
    # Return the dataframe with indicator columns so the plotter can visualize them
    return strategy_df