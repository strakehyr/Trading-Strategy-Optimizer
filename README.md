Modular Algorithmic Trading & Optimization Framework

Overview

This repository contains a strategy-agnostic framework designed for the rigorous development, backtesting, and optimization of algorithmic trading strategies. The system utilizes the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm via Optuna to navigate complex parameter landscapes. It prioritizes statistical robustness by enforcing strict In-Sample (IS) and Out-of-Sample (OOS) separation, degradation analysis, and regime-specific performance metrics.

1. Installation

Clone the repository and install the required dependencies.
code
Bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
System Requirements:
Python 3.8+
Operating System: Linux, macOS, or Windows
Dependencies:
pandas
numpy
optuna
plotly
scipy

2. Data Integration

The framework supports two methods for ingesting market data: a custom API fetcher or local CSV files.


Option A: Custom Data Fetcher (API/Database)

To connect to an external source (Yahoo Finance, Alpaca, Binance, SQL, etc.), you must implement a data_fetcher.py module in the root directory. This acts as the primary interface.
Required Interface:
The data_fetcher.py file must contain the following function signatures:


import pandas as pd
def get_market_data(symbol: str, timeframe: str, min_data_days: int) -> pd.DataFrame:
    """
    Retrieves historical market data.

    Args:
        symbol (str): Ticker symbol (e.g., "BTCUSD", "AAPL").
        timeframe (str): Resolution (e.g., "1 hour", "1 day").
        min_data_days (int): Minimum required history in days.

    Returns:
        pd.DataFrame: A DataFrame containing OHLCV data.
                      Index must be DatetimeIndex.
                      Columns must include: 'open', 'high', 'low', 'close', 'volume'.
    """
    # Implementation logic here (e.g., read_csv or API call)
    pass

def initialize_data_service():
    """
    Performs any necessary setup (e.g., API authentication).
    Returns True if successful, False otherwise.
    """
    pass

def shutdown_data_service():
    """
    Performs cleanup tasks (e.g., closing connections).
    """
    pass



Option B: Offline Mode (Local CSV)

If initialize_data_service() fails or returns False, the system defaults to Offline Mode. It will look for CSV files in a data/ directory within the project root.
File Naming Convention:
Files must be named using the format: {SYMBOL}_{TIMEFRAME}.csv. Note that spaces in timeframes should be removed in the filename.
Example: QQQ_1hour.csv
Example: BTCUSD_1day.csv
CSV Format Requirements:
The CSV must contain standard OHLCV headers. The date column must be parseable by pandas.
code
CSV
datetime,open,high,low,close,volume
2020-01-01 09:30:00,100.50,101.20,100.00,100.80,5000
2020-01-01 10:30:00,100.80,102.00,100.50,101.50,6200
Note: Ensure your CSV data covers the requested --days lookback period, otherwise the backtest for that symbol will be skipped.

3. Adding Entry Strategies

Strategies are defined as standalone Python modules within the strategies/ directory. The framework dynamically loads any .py file found in this folder.
Strategy Module Structure:
Each strategy file must define:
compatible_timeframes: A list of supported timeframes.
param_space: A dictionary defining the hyperparameter search space for Optuna.
strategy_function: The core logic that accepts a DataFrame and parameters, returning the DataFrame with a position column.
Example: strategies/moving_average_crossover.py
code
Python
import pandas as pd
import numpy as np

compatible_timeframes = ['1 hour', '4 hours', '1 day']

param_space = {
    'short_window': {'type': 'int', 'low': 5, 'high': 20},
    'long_window': {'type': 'int', 'low': 21, 'high': 50}
}

def strategy_function(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    strategy_df = df.copy()
    
    # Calculate indicators
    strategy_df['short_mavg'] = strategy_df['close'].rolling(window=short_window).mean()
    strategy_df['long_mavg'] = strategy_df['close'].rolling(window=long_window).mean()
    
    # Generate signals
    # 1 = Long, -1 = Short, 0 = Neutral/Flat
    condition_long = strategy_df['short_mavg'] > strategy_df['long_mavg']
    condition_short = strategy_df['short_mavg'] < strategy_df['long_mavg']
    
    raw_signal = np.where(condition_long, 1, np.where(condition_short, -1, 0))
    
    # Apply signal to position column
    strategy_df['position'] = raw_signal
    
    return strategy_df


4. Adding Exit Strategies

Exit logic is handled separately in the exit_strategies/ directory.
Exit Module Structure:
param_space: Hyperparameters for the exit logic (e.g., TP/SL percentages).
uses_native_exit: Boolean flag.
True: The backtester relies solely on the entry strategy's position column changes to close trades (e.g., Stop and Reverse).
False: The backtester runs exit_logic_function at every step to check for exit conditions.
exit_logic_function: Logic to determine if an active trade should be closed.
Example: exit_strategies/fixed_tp_sl.py
code
Python
import pandas as pd

param_space = {
    'tp_pct': {'type': 'float', 'low': 0.01, 'high': 0.05},
    'sl_pct': {'type': 'float', 'low': 0.01, 'high': 0.03}
}

uses_native_exit = False

def exit_logic_function(df: pd.DataFrame, entry_price: float, position_direction: int, params: dict) -> pd.Series:
    tp = params['tp_pct']
    sl = params['sl_pct']
    
    if position_direction == 1:
        # Long Exit: High hits TP or Low hits SL
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)
        return (df['high'] >= tp_price) | (df['low'] <= sl_price)
        
    elif position_direction == -1:
        # Short Exit: Low hits TP or High hits SL
        tp_price = entry_price * (1 - tp)
        sl_price = entry_price * (1 + sl)
        return (df['low'] <= tp_price) | (df['high'] >= sl_price)
        
    return pd.Series(False, index=df.index)


5. Usage

Run the workflow using main.py. The system accepts command-line arguments to control the scope of the backtest.
Command Line Arguments:
Argument	Description	Default
--symbols	List of ticker symbols to process.	QQQ
--strategies	Specific entry strategy files to run (or 'all').	all
--exit_strategies	Specific exit strategy files to run (or 'all').	all
--timeframes	Timeframes to test.	1 hour, 4 hours, 1 day
--days	Total lookback period in days (IS + OOS).	1825
--is_years	Number of years allocated to In-Sample training.	3
--trials	Number of Optuna trials per combination.	100
--objective	Metric to optimize (Sharpe Ratio, Calmar Ratio, Total Return).	Calmar Ratio
Example Execution:
code
Bash
python main.py --symbols BTCUSD ETHUSD --strategies momentum_dual_consistency --exit_strategies fixed_tp_sl --timeframes "1 hour" "4 hours" --trials 200 --objective "Sharpe Ratio"


6. Output

Results are saved in the results/run_<timestamp>/ directory. Each symbol-strategy-timeframe combination produces a subdirectory containing:
HTML Reports: Visualization of backtests, robustness, and optimization landscapes.
CSV Data: Raw data for trials and robustness analysis.
Logs: Execution details and error tracking.