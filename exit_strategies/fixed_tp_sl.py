# exit_strategies/fixed_tp_sl.py

import pandas as pd
from typing import Dict, Any

# Drastically reduced parameter ranges to be learnable and realistic for hourly/daily data.
# Range is now 0.1% to 5.0% for TP and 0.1% to 3.0% for SL.
param_space = {
    'take_profit_pct': {'type': 'float', 'low': 0.001, 'high': 0.05},
    'stop_loss_pct': {'type': 'float', 'low': 0.001, 'high': 0.03},
}

# This strategy uses its own logic, not the native entry signal's exit.
uses_native_exit = False

def exit_logic_function(df: pd.DataFrame, entry_price: float, position_direction: int, params: Dict) -> pd.Series:
    """
    Vectorized calculation of fixed percentage take profit and stop loss.

    Returns:
        pd.Series: A boolean Series indicating an exit signal on a given bar.
    """
    # Defaults in case optimization passes None
    take_profit_pct = params.get('take_profit_pct', 0.01)
    stop_loss_pct = params.get('stop_loss_pct', 0.01)

    if position_direction == 1:  # Long position
        tp_price = entry_price * (1 + take_profit_pct)
        sl_price = entry_price * (1 - stop_loss_pct)
        # Exit if the high hits the take profit OR the low hits the stop loss
        exit_signal = (df['high'] >= tp_price) | (df['low'] <= sl_price)
    elif position_direction == -1:  # Short position
        tp_price = entry_price * (1 - take_profit_pct)
        sl_price = entry_price * (1 + stop_loss_pct)
        # Exit if the low hits the take profit OR the high hits the stop loss
        exit_signal = (df['low'] <= tp_price) | (df['high'] >= sl_price)
    else:
        # No position, so no exit signal
        return pd.Series(False, index=df.index)

    return exit_signal