import pandas as pd
import numpy as np

# Compatible with all timeframes
compatible_timeframes = ['1 hour', '4 hours', '1 day']

# Parameters optimized for identifying structural breakouts
# 1% (0.01) is too large for hourly breakouts, causing missed trades.
# 0.0 to 0.005 (0.5%) is much more realistic for ensuring execution.
param_space = {
    'lookback': {'type': 'int', 'low': 10, 'high': 100},
    'breakout_margin': {'type': 'float', 'low': 0.0, 'high': 0.005}, 
}

def strategy_function(
    df: pd.DataFrame,
    lookback: int = 20,
    breakout_margin: float = 0.001
) -> pd.DataFrame:
    """
    Revised Market Structure Strategy (Break of Structure).
    
    Instead of waiting for complex completed patterns (which often repaints or lags too much),
    this strategy identifies the 'Structure High' and 'Structure Low' over the lookback period.
    
    - HH (Bullish BOS): Price closes above the highest high of the last N bars.
    - LL (Bearish BOS): Price closes below the lowest low of the last N bars.
    
    This ensures trades are generated whenever the market breaks its recent structure range.
    """
    strategy_df = df.copy()
    
    # 1. Identify Structure Highs and Lows (shifted to avoid lookahead bias)
    # We look at the max/min of the *previous* 'lookback' bars to define the range.
    structure_high = strategy_df['high'].rolling(window=lookback).max().shift(1)
    structure_low = strategy_df['low'].rolling(window=lookback).min().shift(1)
    
    # 2. Define Break of Structure (BOS) Criteria
    # Long: Close breaks above Structure High + Margin
    long_condition = strategy_df['close'] > (structure_high * (1 + breakout_margin))
    
    # Short: Close breaks below Structure Low - Margin
    short_condition = strategy_df['close'] < (structure_low * (1 - breakout_margin))
    
    # 3. Generate Signals
    # 1 = Long Signal, -1 = Short Signal, 0 = No new signal (hold)
    raw_signal = np.where(long_condition, 1, np.where(short_condition, -1, np.nan))
    
    # 4. Create Position State (Stop-and-Reverse)
    # Forward fill the last signal to hold position until structure breaks the other way
    strategy_df['position'] = pd.Series(raw_signal, index=strategy_df.index).ffill().fillna(0).astype(int)
    
    # Optional: Visualization columns for the plotter
    strategy_df['sma_high'] = structure_high # Reusing names plotter understands for bands
    strategy_df['sma_low'] = structure_low
    
    return strategy_df