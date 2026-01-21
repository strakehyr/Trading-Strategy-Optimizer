# exit_strategies/none.py

import pandas as pd
from typing import Dict, Any, Callable

# This strategy has no parameters.
param_space = {}

# This is a special variable. If True, the backtester will not use this exit logic
# and will rely solely on the entry strategy's position changes.
# This is useful for "exit-on-revert" or pure "stop-and-reverse" strategies.
uses_native_exit = True

def exit_logic_function(df: pd.DataFrame, entry_price: float, position_direction: int, params: Dict) -> pd.Series:
    """
    A null exit strategy. It always returns False, signifying no exit signal.
    The backtester will rely on the entry signal's position column changing to 0 or flipping.
    """
    return pd.Series(False, index=df.index)

