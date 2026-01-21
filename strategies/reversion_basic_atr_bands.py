import pandas as pd
import numpy as np

# Each strategy file must define these two variables:
# 1. param_space: A dictionary defining the optimizable parameters.
# 2. strategy_function: The function that implements the strategy logic.

compatible_timeframes = ['1 hour', '4 hours', '1 day']

param_space = {
    'trend_length': {'type': 'int', 'low': 2, 'high': 50},
    'atr_period': {'type': 'int', 'low': 25, 'high': 200},
    'atr_multiplier': {'type': 'float', 'low': 0.5, 'high': 4.0}
}

def strategy_function(df: pd.DataFrame, trend_length: int = 20, atr_period: int = 100, atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    MEAN-REVERSION (Stop-and-Reverse): Sells on upside breakout, buys on downside breakout.
    This is the "basic" version that only flips its position when the price crosses the opposite band.
    """
    strategy_df = df.copy()

    # 1. ATR Bands Calculation
    high_low = strategy_df['high'] - strategy_df['low']
    high_prev_close = np.abs(strategy_df['high'] - strategy_df['close'].shift(1))
    low_prev_close = np.abs(strategy_df['low'] - strategy_df['close'].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    raw_atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
    atr_smoothed = raw_atr.rolling(window=atr_period).mean()
    atr_value = atr_smoothed * atr_multiplier
    
    strategy_df['sma_high'] = strategy_df['high'].rolling(window=trend_length).mean() + atr_value
    strategy_df['sma_low'] = strategy_df['low'].rolling(window=trend_length).mean() - atr_value

    # 2. Explicit Crossover Signal Generation
    crossed_up = (strategy_df['close'].shift(1) < strategy_df['sma_high'].shift(1)) & (strategy_df['close'] > strategy_df['sma_high'])
    crossed_down = (strategy_df['close'].shift(1) > strategy_df['sma_low'].shift(1)) & (strategy_df['close'] < strategy_df['sma_low'])

    # 3. Generate INVERTED raw signals for mean-reversion
    raw_signal = np.where(crossed_up, -1, np.nan)    # SELL on upside breakout
    raw_signal = np.where(crossed_down, 1, raw_signal)   # BUY on downside breakout

    # 4. Propagate the last valid signal forward to hold the position (stop-and-reverse)
    strategy_df['position'] = pd.Series(raw_signal, index=strategy_df.index).ffill().fillna(0).astype(int)
    
    return strategy_df