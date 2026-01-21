import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Callable, Optional
import logging

logger = logging.getLogger(__name__)

def _calculate_benchmark_metrics(df: pd.DataFrame, initial_capital: float) -> Tuple[Dict[str, Any], Optional[pd.Series]]:
    """
    Calculates benchmark metrics on the exact same DataFrame slice the strategy is run on.
    """
    benchmark_metrics = {}
    benchmark_df = pd.DataFrame(index=df.index)
    benchmark_df['close'] = df['close']
    valid_closes = benchmark_df['close'].dropna()

    if len(valid_closes) < 2:
        return {
            'Benchmark': "N/A", 'Benchmark Return Pct': 0,
            'Benchmark Sharpe Ratio': 0, 'Benchmark Calmar Ratio': 0, 'Benchmark Max Drawdown Pct': 0
        }, pd.Series(initial_capital, index=df.index, name='benchmark_value')

    first_price, last_price = valid_closes.iloc[0], valid_closes.iloc[-1]
    
    if first_price <= 0:
        return { 'Benchmark': "N/A", 'Benchmark Return Pct': 0, 'Benchmark Sharpe Ratio': 0, 'Benchmark Calmar Ratio': 0, 'Benchmark Max Drawdown Pct': 0 }, pd.Series(initial_capital, index=df.index, name='benchmark_value')

    benchmark_name = "Buy & Hold"
    benchmark_df['value'] = (initial_capital / first_price) * benchmark_df['close']
    benchmark_return_pct = (last_price / first_price - 1) * 100

    benchmark_df['value'] = benchmark_df['value'].ffill().bfill()
    benchmark_df['value'] = benchmark_df['value'].fillna(initial_capital)

    bench_pct_change = benchmark_df['value'].pct_change().dropna()
    bench_sharpe = (bench_pct_change.mean() / bench_pct_change.std()) * np.sqrt(252) if bench_pct_change.std() != 0 else 0
    bench_expanding_max = benchmark_df['value'].expanding().max()
    bench_drawdown = (benchmark_df['value'] / bench_expanding_max - 1).min() * 100
    
    raw_bench_calmar = benchmark_return_pct / abs(bench_drawdown) if bench_drawdown != 0 else 0
    bench_calmar = min(raw_bench_calmar, 20.0)

    benchmark_metrics = {
        'Benchmark': benchmark_name,
        'Benchmark Return Pct': benchmark_return_pct,
        'Benchmark Sharpe Ratio': bench_sharpe,
        'Benchmark Calmar Ratio': bench_calmar,
        'Benchmark Max Drawdown Pct': bench_drawdown
    }
    
    return benchmark_metrics, benchmark_df['value']


def calculate_pnl_and_metrics(
    df: pd.DataFrame,
    initial_capital: float,
    commission: float,
    exit_logic_func: Optional[Callable] = None,
    exit_params: Optional[Dict] = None,
    uses_native_exit: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    
    if 'position' not in df.columns:
        raise ValueError("Backtest DataFrame must have a 'position' column.")

    df = df.copy()
    df['position_shifted'] = df['position'].shift(1).fillna(0)
    df['long_entry_marker'], df['short_entry_marker'], df['exit_marker'] = np.nan, np.nan, np.nan
    df['tp_level'], df['sl_level'] = np.nan, np.nan

    cash = initial_capital
    shares, entry_price, position_direction = 0.0, 0.0, 0
    active_trade = False
    current_tp, current_sl = np.nan, np.nan
    cash_list, shares_list, total_value_list, trade_pnl_list = [], [], [], []

    for i, row in df.iterrows():
        if active_trade:
            exit_signal, exit_price = False, row['close']
            if exit_logic_func and not uses_native_exit and exit_params:
                if position_direction == 1:
                    if row['low'] <= current_sl: exit_signal, exit_price = True, current_sl
                    elif row['high'] >= current_tp: exit_signal, exit_price = True, current_tp
                elif position_direction == -1:
                    if row['high'] >= current_sl: exit_signal, exit_price = True, current_sl
                    elif row['low'] <= current_tp: exit_signal, exit_price = True, current_tp
            
            if not exit_signal and (row['position'] == 0 or row['position'] != position_direction):
                exit_signal, exit_price = True, row['close']

            if exit_signal:
                if position_direction == 1: 
                    pnl = shares * (exit_price - entry_price)
                    cash += shares * exit_price - commission
                else: 
                    pnl = shares * (entry_price - exit_price)
                    cash -= (shares * exit_price) + commission 

                trade_pnl_list.append(pnl)
                df.loc[i, 'exit_marker'] = exit_price
                shares, entry_price, position_direction, active_trade = 0.0, 0.0, 0, False
                current_tp, current_sl = np.nan, np.nan
            else:
                df.loc[i, 'tp_level'], df.loc[i, 'sl_level'] = current_tp, current_sl

        if not active_trade and row['position'] != 0 and row['position'] != row['position_shifted']:
            if cash > commission and row['close'] > 0:
                shares_to_trade = (cash * 0.95) / row['close']
                position_direction = int(row['position'])
                
                if position_direction == 1: 
                    cash -= shares_to_trade * row['close'] + commission
                    df.loc[i, 'long_entry_marker'] = row['close']
                else: 
                    cash += shares_to_trade * row['close'] - commission 
                    df.loc[i, 'short_entry_marker'] = row['close']

                shares, entry_price, active_trade = shares_to_trade, row['close'], True
                
                if exit_logic_func and not uses_native_exit and exit_params:
                    sl_pct = exit_params.get('stop_loss_pct', 0)
                    tp_pct = exit_params.get('take_profit_pct', float('inf'))
                    current_sl = entry_price * (1 - sl_pct * position_direction)
                    current_tp = entry_price * (1 + tp_pct * position_direction)
                    df.loc[i, 'tp_level'], df.loc[i, 'sl_level'] = current_sl, current_tp
            else:
                logger.warning(f"Skipping trade at {i} due to insufficient capital or zero price.")

        current_value = cash
        if active_trade:
            if position_direction == 1:
                current_value += shares * row['close']
            else: 
                current_value = cash - (shares * row['close']) 

        cash_list.append(cash); shares_list.append(shares); total_value_list.append(current_value)

    df['cash'], df['shares'], df['total_value'] = cash_list, shares_list, total_value_list

    trades = pd.Series([pnl for pnl in trade_pnl_list if pnl != 0]).dropna()
    total_return_pct = (df['total_value'].iloc[-1] / initial_capital - 1) * 100 if initial_capital > 0 else 0
    expanding_max = df['total_value'].expanding().max()
    drawdown = (df['total_value'] / expanding_max - 1)
    max_drawdown_pct = drawdown.min() * 100
    returns_pct = df['total_value'].pct_change().dropna()
    sharpe_ratio = (returns_pct.mean() / returns_pct.std()) * np.sqrt(252) if returns_pct.std() != 0 else 0
    
    raw_calmar = total_return_pct / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
    calmar_ratio = min(raw_calmar, 20.0) 

    total_trades = len(trades)
    win_rate = (trades > 0).sum() / total_trades * 100 if total_trades > 0 else 0
    
    metrics = {
        'Total Return Pct': total_return_pct, 'Max Drawdown Pct': max_drawdown_pct,
        'Sharpe Ratio': sharpe_ratio, 'Calmar Ratio': calmar_ratio, 'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Max Consecutive Wins': ((trades > 0).astype(int).groupby((trades <= 0).cumsum()).cumsum()).max() if total_trades > 0 else 0,
        'Max Consecutive Losses': ((trades <= 0).astype(int).groupby((trades > 0).cumsum()).cumsum()).max() if total_trades > 0 else 0,
    }
    return df, metrics


def run_backtest(
    df_full_history: pd.DataFrame,
    strategy_func: Callable,
    strategy_params: Dict[str, Any],
    exit_logic_func: Optional[Callable] = None,
    uses_native_exit: bool = False,
    initial_capital: float = 100000.0,
    commission: float = 1.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    if df_full_history.empty:
        return pd.DataFrame(), {}
    
    df_strategy_input = df_full_history.copy()
    sig_params = {k.replace('sig_', ''): v for k, v in strategy_params.items() if k.startswith('sig_')}
    strategy_df_full = strategy_func(df_strategy_input, **sig_params)

    if strategy_df_full.empty:
        logger.warning("Strategy function returned an empty DataFrame. Skipping backtest.")
        return pd.DataFrame(), {}

    benchmark_metrics, benchmark_value_series = _calculate_benchmark_metrics(strategy_df_full, initial_capital)
    
    ex_params = {k.replace('exit_', ''): v for k, v in strategy_params.items() if k.startswith('exit_')}
    result_df, strategy_metrics = calculate_pnl_and_metrics(
        strategy_df_full, initial_capital, commission,
        exit_logic_func, ex_params, uses_native_exit
    )
    
    strategy_metrics.update(benchmark_metrics)
    strategy_metrics['Return vs Benchmark Pct'] = strategy_metrics.get('Total Return Pct', 0) - benchmark_metrics.get('Benchmark Return Pct', 0)
    
    if benchmark_value_series is not None:
        result_df['benchmark_value'] = benchmark_value_series
    else:
        result_df['benchmark_value'] = initial_capital
        
    return result_df, strategy_metrics