import argparse
import logging
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import importlib.util

from optimizer import find_robust_parameters
from plotter import (create_strategy_plot, create_strategy_distribution_boxplot,
                    create_optimization_history_plot, create_parallel_coordinates_plot,
                    create_is_oos_comparison_plot, create_robustness_scatter_plot,
                    create_top_performers_plot, create_regime_performance_matrix)
from data_fetcher import get_market_data, initialize_data_service, shutdown_data_service
from backtester import run_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_strategies_from_folder(folder: str, required_attribs: List[str]) -> Dict[str, Any]:
    """Dynamically discovers and loads strategies from a specified folder."""
    strategies = {}
    if not os.path.isdir(folder):
        logging.warning(f"Directory not found: {folder}")
        return strategies
    
    for filename in os.listdir(folder):
        if filename.endswith('.py') and not filename.startswith('__'):
            strategy_name = filename[:-3]
            module_path = os.path.join(folder, filename)
            spec = importlib.util.spec_from_file_location(strategy_name, module_path)
            if spec and spec.loader:
                strategy_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(strategy_module)
                if all(hasattr(strategy_module, attr) for attr in required_attribs):
                    strategies[strategy_name] = {
                        'function': getattr(strategy_module, required_attribs[0]),
                        'params': getattr(strategy_module, 'param_space', {}),
                        'timeframes': getattr(strategy_module, 'compatible_timeframes', []),
                        'uses_native_exit': getattr(strategy_module, 'uses_native_exit', False)
                    }
                    logging.info(f"Successfully loaded '{strategy_name}' from '{folder}'")
                else:
                    logging.warning(f"Could not load {strategy_name}: missing one of {required_attribs}.")
    return strategies

def run_workflow(symbols: List[str], strategies_to_run: List[str], exit_strategies_to_run: List[str], timeframes: List[str], total_days: int, in_sample_years: int, trials: int, objective_metric: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Full workflow started. Results will be saved in: {results_dir}")
    
    available_strategies = load_strategies_from_folder('strategies', ['strategy_function', 'param_space', 'compatible_timeframes'])
    available_exit_strategies = load_strategies_from_folder('exit_strategies', ['exit_logic_function', 'param_space', 'uses_native_exit'])
    
    if not available_strategies:
        logging.error("No entry strategies found in 'strategies' folder. Aborting."); return
    if not available_exit_strategies:
        logging.error("No exit strategies found in 'exit_strategies' folder. Aborting."); return
        
    if not initialize_data_service():
        logging.warning("Data Service initialization failed. Attempting to run in OFFLINE mode using local CSV data only.")
    
    all_trials_data = []
    top_performers_data = [] 
    
    try:
        for symbol in symbols:
            strategies_to_process = available_strategies.keys() if 'all' in strategies_to_run else strategies_to_run
            exit_strategies_to_process = available_exit_strategies.keys() if 'all' in exit_strategies_to_run else exit_strategies_to_run
            
            for strategy_name in strategies_to_process:
                if strategy_name not in available_strategies:
                    logging.warning(f"Requested entry strategy '{strategy_name}' not found. Skipping."); continue
                
                for exit_strategy_name in exit_strategies_to_process:
                    if exit_strategy_name not in available_exit_strategies:
                        logging.warning(f"Requested exit strategy '{exit_strategy_name}' not found. Skipping."); continue
                    
                    strategy_info = available_strategies[strategy_name]
                    exit_strategy_info = available_exit_strategies[exit_strategy_name]
                    
                    for timeframe in timeframes:
                        if timeframe not in strategy_info['timeframes']:
                            logging.info(f"Skipping: {strategy_name} not compatible with {timeframe}")
                            continue
                        
                        combination_label = f"{symbol}_{strategy_name}_{exit_strategy_name}_{timeframe.replace(' ', '')}"
                        logging.info(f"--- Running combination: {combination_label} ---")
                        
                        df_full_history = get_market_data(symbol, timeframe, min_data_days=total_days)
                        
                        if df_full_history.empty:
                            logging.warning(f"No data found for {symbol} {timeframe}. Skipping.")
                            continue
                            
                        if len(df_full_history) < total_days * 0.6: 
                             logging.warning(f"Insufficient data length for {symbol} {timeframe}. Skipping.")
                             continue
                        
                        split_date = df_full_history.index.min() + pd.DateOffset(years=in_sample_years)
                        df_in_sample = df_full_history.loc[df_full_history.index < split_date]
                        df_out_of_sample = df_full_history.loc[df_full_history.index >= split_date]
                        
                        if df_in_sample.empty or df_out_of_sample.empty: continue
                        
                        robust_params, study, analysis_df = find_robust_parameters(
                            df_in_sample, df_out_of_sample,
                            strategy_info['function'], strategy_info['params'],
                            exit_strategy_info['function'], exit_strategy_info['params'],
                            exit_strategy_info['uses_native_exit'],
                            trials, objective_metric
                        )
                        
                        combo_plot_dir = os.path.join(results_dir, combination_label)
                        os.makedirs(combo_plot_dir, exist_ok=True)

                        if study and study.trials:
                            study_df = study.trials_dataframe()
                            study_df['combination'] = combination_label
                            cols_to_keep = [c for c in study_df.columns if c.startswith('user_attrs_') or c.startswith('params_') or c == 'value']
                            if cols_to_keep:
                                trials_clean = study_df[cols_to_keep].sort_values(by='value', ascending=False)
                                trials_clean.to_csv(os.path.join(combo_plot_dir, "optimization_all_trials.csv"))
                            
                            all_trials_data.append(study_df)
                        
                        try:
                            if not analysis_df.empty:
                                analysis_df.to_csv(os.path.join(combo_plot_dir, "robustness_analysis.csv"))
                                create_robustness_scatter_plot(analysis_df, objective_metric, os.path.join(combo_plot_dir, "robustness_scatter_plot.html"))
                        except Exception as e:
                            logging.error(f"Error creating robustness plots for {combination_label}: {e}")
                        
                        if robust_params:
                            try:
                                is_df, is_metrics = run_backtest(df_in_sample, strategy_info['function'], robust_params, exit_strategy_info['function'], uses_native_exit=exit_strategy_info['uses_native_exit'])
                                oos_df, oos_metrics = run_backtest(df_out_of_sample, strategy_info['function'], robust_params, exit_strategy_info['function'], uses_native_exit=exit_strategy_info['uses_native_exit'])
                                
                                if is_metrics and oos_metrics:
                                    create_is_oos_comparison_plot(is_metrics, oos_metrics, os.path.join(combo_plot_dir, "is_vs_oos_comparison.html"))
                                    
                                    top_performers_data.append({
                                        'combination': combination_label,
                                        'is_metrics': is_metrics,
                                        'oos_metrics': oos_metrics,
                                        'is_curves': is_df[['open', 'high', 'low', 'close', 'total_value', 'benchmark_value']].copy() if not is_df.empty else pd.DataFrame(),
                                        'oos_curves': oos_df[['open', 'high', 'low', 'close', 'total_value', 'benchmark_value']].copy() if not oos_df.empty else pd.DataFrame()
                                    })
                                
                                if not is_df.empty: 
                                    create_strategy_plot(is_df, {**is_metrics, 'label': f"{combination_label} (In-Sample)"}, timeframe, os.path.join(combo_plot_dir, "backtest_IN_SAMPLE.html"))
                                if not oos_df.empty: 
                                    create_strategy_plot(oos_df, {**oos_metrics, 'label': f"{combination_label} (Out-of-Sample)"}, timeframe, os.path.join(combo_plot_dir, "backtest_OUT_OF_SAMPLE.html"))
                            except Exception as e:
                                logging.error(f"Error running backtests for {combination_label}: {e}")
                        
                        if study and study.trials:
                            try:
                                create_optimization_history_plot(study, objective_metric, os.path.join(combo_plot_dir, "optimization_history.html"))
                                create_parallel_coordinates_plot(study, os.path.join(combo_plot_dir, "parallel_coordinates.html"))
                            except Exception as e:
                                logging.error(f"Error creating optimization plots for {combination_label}: {e}")
    
    finally:
        shutdown_data_service()
    
    if all_trials_data:
        create_strategy_distribution_boxplot(pd.concat(all_trials_data, ignore_index=True), os.path.join(results_dir, "strategy_distribution_results.html"))
    
    if top_performers_data:
        create_top_performers_plot(top_performers_data, os.path.join(results_dir, "IS_OOS_strategies_performance.html"))
        create_regime_performance_matrix(top_performers_data, os.path.join(results_dir, "regime_performance_matrix.html"))
    
    logging.info(f"Workflow complete! All results saved in {results_dir}")

def main():
    parser = argparse.ArgumentParser(description="Strategy-Agnostic Backtesting & Optimization Framework")
    parser.add_argument("--symbols", nargs='+', default=["QQQ"], help="List of stock symbols")
    parser.add_argument("--strategies", nargs='+', default=['all'], help="Entry strategies to run, or 'all'")
    parser.add_argument("--exit_strategies", nargs='+', default=['all'], help="Exit strategies to run, or 'all'")
    parser.add_argument("--timeframes", nargs='+', default=['1 hour', '4 hours', '1 day'], help="List of timeframes")
    parser.add_argument("--days", type=int, default=1825, help="Total lookback period in days (IS + OOS)")
    parser.add_argument("--is_years", type=int, default=3, help="Years for In-Sample optimization period")
    parser.add_argument("--trials", type=int, default=100, help="Optimization trials per combination")
    parser.add_argument("--objective", type=str, default="Calmar Ratio", choices=["Sharpe Ratio", "Calmar Ratio", "Total Return Pct"], help="Metric to optimize for.")
    
    args, _ = parser.parse_known_args()
    run_workflow(args.symbols, args.strategies, args.exit_strategies, args.timeframes, args.days, args.is_years, args.trials, args.objective)

if __name__ == "__main__":
    main()