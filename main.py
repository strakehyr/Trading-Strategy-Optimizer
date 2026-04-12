import argparse
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import importlib.util

from optimizer import find_robust_parameters
from plotter import (create_strategy_plot, create_strategy_distribution_boxplot,
                    create_optimization_history_plot, create_parallel_coordinates_plot,
                    create_is_oos_comparison_plot, create_robustness_scatter_plot,
                    create_top_performers_plot, create_regime_performance_matrix,
                    create_return_calendar_heatmap, create_joint_performance_summary)
from data_fetcher import get_market_data, initialize_data_service, shutdown_data_service
from backtester import run_backtest
import regime_analyzer

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

def _save_combo_reports(
    combo_plot_dir: str,
    combination_label: str,
    strategy_info: Dict,
    exit_strategy_info: Dict,
    timeframe: str,
    robust_params: Optional[Dict],
    df_in_sample: pd.DataFrame,
    df_out_of_sample: pd.DataFrame,
    study,
    analysis_df: pd.DataFrame,
    objective_metric: str,
    all_trials_data: list,
    top_performers_data: list,
    symbol: str = '',
):
    """Helper: saves all per-combination CSV and HTML reports."""
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
                entry = {
                    'combination': combination_label,
                    'is_metrics': is_metrics,
                    'oos_metrics': oos_metrics,
                    'is_curves': is_df[['open', 'high', 'low', 'close', 'total_value', 'benchmark_value']].copy() if not is_df.empty else pd.DataFrame(),
                    'oos_curves': oos_df[['open', 'high', 'low', 'close', 'total_value', 'benchmark_value']].copy() if not oos_df.empty else pd.DataFrame(),
                }
                if symbol:
                    entry['symbol'] = symbol
                top_performers_data.append(entry)

            if not is_df.empty:
                create_strategy_plot(is_df, {**is_metrics, 'label': f"{combination_label} (In-Sample)"}, timeframe, os.path.join(combo_plot_dir, "backtest_IN_SAMPLE.html"))
                is_df[['open', 'high', 'low', 'close', 'total_value', 'benchmark_value']].to_csv(os.path.join(combo_plot_dir, "equity_curve_IS.csv"))
            if not oos_df.empty:
                create_strategy_plot(oos_df, {**oos_metrics, 'label': f"{combination_label} (Out-of-Sample)"}, timeframe, os.path.join(combo_plot_dir, "backtest_OUT_OF_SAMPLE.html"))
                oos_df[['open', 'high', 'low', 'close', 'total_value', 'benchmark_value']].to_csv(os.path.join(combo_plot_dir, "equity_curve_OOS.csv"))
            if not is_df.empty or not oos_df.empty:
                create_return_calendar_heatmap(
                    is_df if not is_df.empty else pd.DataFrame(),
                    oos_df if not oos_df.empty else pd.DataFrame(),
                    f"{combination_label} - Daily Returns Calendar",
                    os.path.join(combo_plot_dir, "calendar_returns.html"),
                )
        except Exception as e:
            logging.error(f"Error running backtests for {combination_label}: {e}")

    if study and study.trials:
        try:
            create_optimization_history_plot(study, objective_metric, os.path.join(combo_plot_dir, "optimization_history.html"))
            create_parallel_coordinates_plot(study, os.path.join(combo_plot_dir, "parallel_coordinates.html"))
        except Exception as e:
            logging.error(f"Error creating optimisation plots for {combination_label}: {e}")


_MAX_DRAWDOWN_KEYS = {'Max Drawdown Pct', 'Benchmark Max Drawdown Pct'}

def _avg_metrics(metrics_list: List[Dict]) -> Dict:
    """
    Aggregate numeric metrics across a list of metric dicts.
    Drawdown metrics use the worst-case (min) across symbols;
    all other numeric metrics are averaged.
    """
    if not metrics_list:
        return {}
    all_keys = set().union(*[m.keys() for m in metrics_list])
    combined = {}
    for key in all_keys:
        vals = [m[key] for m in metrics_list if isinstance(m.get(key), (int, float))]
        if not vals:
            combined[key] = metrics_list[0].get(key)
        elif key in _MAX_DRAWDOWN_KEYS:
            combined[key] = float(np.min(vals))   # worst drawdown (most negative)
        else:
            combined[key] = float(np.mean(vals))
    return combined


def _combine_symbol_curves(curves_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines per-symbol equity curves into a single averaged curve.

    Each curve is normalized to its starting value (→ relative returns),
    averaged across symbols, then scaled back to 100 000 initial capital.
    The first non-empty curve's OHLCV columns are used as the price reference
    so regime indicators (SMA200 etc.) have something to work with.
    """
    non_empty = [df for df in curves_list if not df.empty and 'total_value' in df.columns]
    if not non_empty:
        return pd.DataFrame()

    price_ref = None
    tv_series: Dict[int, pd.Series] = {}
    bv_series: Dict[int, pd.Series] = {}

    for i, df in enumerate(non_empty):
        start_val = df['total_value'].iloc[0]
        if start_val == 0:
            continue
        tv_series[i] = df['total_value'] / start_val
        if 'benchmark_value' in df.columns and df['benchmark_value'].iloc[0] != 0:
            bv_series[i] = df['benchmark_value'] / df['benchmark_value'].iloc[0]
        if price_ref is None:
            ohlcv_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
            if ohlcv_cols:
                price_ref = df[ohlcv_cols].copy()

    if not tv_series:
        return pd.DataFrame()

    tv_mean = pd.concat(tv_series, axis=1).mean(axis=1)

    combined = pd.DataFrame(index=tv_mean.index)
    combined['total_value'] = tv_mean * 100_000.0

    if bv_series:
        bv_mean = pd.concat(bv_series, axis=1).mean(axis=1)
        combined['benchmark_value'] = bv_mean * 100_000.0

    if price_ref is not None:
        for col in price_ref.columns:
            combined[col] = price_ref[col].reindex(combined.index, method='ffill')

    return combined


def run_workflow(
    symbols: List[str],
    strategies_to_run: List[str],
    exit_strategies_to_run: List[str],
    timeframes: List[str],
    total_days: int,
    in_sample_years: int,
    trials: int,
    objective_metric: str,
    aggregation: str = 'average',
    resume_dir: Optional[str] = None,
    mode: str = 'joint',
):
    if resume_dir:
        results_dir = resume_dir
        logging.info(f"Resuming run in: {results_dir}")
    else:
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
        logging.warning("Data Service initialization failed. Attempting OFFLINE mode.")

    all_trials_data = []
    top_performers_data = []
    # With a single symbol, joint mode makes no sense — always use per_symbol.
    effective_mode = 'per_symbol' if len(symbols) == 1 else mode
    vix_df = None

    try:
        strategies_to_process = list(available_strategies.keys()) if 'all' in strategies_to_run else strategies_to_run
        exit_strategies_to_process = list(available_exit_strategies.keys()) if 'all' in exit_strategies_to_run else exit_strategies_to_run

        if effective_mode == 'joint':
            # ------------------------------------------------------------------
            # JOINT MULTI-SYMBOL MODE
            # One optimisation per strategy/exit/timeframe across all symbols.
            # Summary charts show ONE entry per combination (no symbol in label).
            # Per-symbol drill-down reports are saved in sub-folders.
            # ------------------------------------------------------------------
            logging.info(f"Running in JOINT mode across {len(symbols)} symbols: {symbols}")

            for strategy_name in strategies_to_process:
                if strategy_name not in available_strategies:
                    logging.warning(f"Entry strategy '{strategy_name}' not found. Skipping."); continue

                for exit_strategy_name in exit_strategies_to_process:
                    if exit_strategy_name not in available_exit_strategies:
                        logging.warning(f"Exit strategy '{exit_strategy_name}' not found. Skipping."); continue

                    strategy_info = available_strategies[strategy_name]
                    exit_strategy_info = available_exit_strategies[exit_strategy_name]

                    for timeframe in timeframes:
                        if timeframe not in strategy_info['timeframes']:
                            logging.info(f"Skipping: {strategy_name} not compatible with {timeframe}"); continue

                        tf_tag = timeframe.replace(' ', '')
                        combo_label = f"{strategy_name}_{exit_strategy_name}_{tf_tag}"
                        joint_label = f"JOINT_{combo_label}"

                        # Skip if already completed (resume mode)
                        joint_dir_check = os.path.join(results_dir, joint_label)
                        if os.path.isdir(joint_dir_check) and os.path.exists(
                            os.path.join(joint_dir_check, "equity_curve_IS_combined.csv")
                        ):
                            logging.info(f"--- Skipping (already done): {joint_label} ---")
                            continue

                        logging.info(f"--- Joint optimisation: {joint_label} ---")

                        # Collect (df_is, df_oos) for every symbol that has sufficient data
                        symbol_datasets = []
                        symbol_data_map: Dict[str, tuple] = {}
                        for sym in symbols:
                            df_full = get_market_data(sym, timeframe, min_data_days=total_days)
                            if df_full.empty or len(df_full) < total_days * 0.6:
                                logging.warning(f"Insufficient data for {sym} {timeframe}. Excluding from joint optimisation.")
                                continue
                            split_date = df_full.index.min() + pd.DateOffset(years=in_sample_years)
                            df_is  = df_full.loc[df_full.index < split_date]
                            df_oos = df_full.loc[df_full.index >= split_date]
                            if df_is.empty or df_oos.empty:
                                continue
                            symbol_datasets.append((df_is, df_oos))
                            symbol_data_map[sym] = (df_is, df_oos)

                        if not symbol_datasets:
                            logging.warning(f"No valid symbol data for {joint_label}. Skipping."); continue

                        robust_params, study, analysis_df = find_robust_parameters(
                            symbol_datasets,
                            strategy_info['function'], strategy_info['params'],
                            exit_strategy_info['function'], exit_strategy_info['params'],
                            exit_strategy_info['uses_native_exit'],
                            trials, objective_metric,
                            aggregation=aggregation,
                        )

                        joint_dir = os.path.join(results_dir, joint_label)
                        os.makedirs(joint_dir, exist_ok=True)

                        # Save joint-level CSV artefacts
                        if study and study.trials:
                            study_df = study.trials_dataframe()
                            study_df['combination'] = joint_label
                            cols_to_keep = [c for c in study_df.columns if c.startswith('user_attrs_') or c.startswith('params_') or c == 'value']
                            if cols_to_keep:
                                study_df[cols_to_keep].sort_values(by='value', ascending=False).to_csv(
                                    os.path.join(joint_dir, "optimization_all_trials.csv"))
                            all_trials_data.append(study_df)

                        try:
                            if not analysis_df.empty:
                                analysis_df.to_csv(os.path.join(joint_dir, "robustness_analysis.csv"))
                                create_robustness_scatter_plot(analysis_df, objective_metric, os.path.join(joint_dir, "robustness_scatter_plot.html"))
                        except Exception as e:
                            logging.error(f"Error saving robustness artefacts for {joint_label}: {e}")

                        if study and study.trials:
                            try:
                                create_optimization_history_plot(study, objective_metric, os.path.join(joint_dir, "optimization_history.html"))
                                create_parallel_coordinates_plot(study, os.path.join(joint_dir, "parallel_coordinates.html"))
                            except Exception as e:
                                logging.error(f"Error creating optimisation plots for {joint_label}: {e}")

                        # Per-symbol drill-down reports using the jointly-optimised parameters
                        joint_symbol_entries = []
                        for sym, (df_is, df_oos) in symbol_data_map.items():
                            sym_combo_label = f"{sym}_{combo_label}"
                            sym_dir = os.path.join(joint_dir, sym)
                            sym_top_performers: list = []
                            _save_combo_reports(
                                sym_dir, sym_combo_label,
                                strategy_info, exit_strategy_info,
                                timeframe, robust_params,
                                df_is, df_oos,
                                None, pd.DataFrame(),  # study/analysis already saved at joint level
                                objective_metric,
                                [],          # don't double-add to all_trials_data
                                sym_top_performers,
                                symbol=sym,
                            )
                            for e in sym_top_performers:
                                e['symbol'] = sym
                                joint_symbol_entries.append(e)

                        # Per-symbol summary chart (drill-down within the joint folder)
                        if joint_symbol_entries:
                            try:
                                create_joint_performance_summary(
                                    joint_symbol_entries,
                                    os.path.join(joint_dir, "joint_performance_summary.html"),
                                )
                            except Exception as e:
                                logging.error(f"Error creating joint performance summary: {e}")

                        # ----------------------------------------------------------
                        # ONE combined entry for top-level summary charts.
                        # Equity curves are normalized and averaged across symbols;
                        # metrics are averaged. This is what the regime attribution,
                        # top-performers plot, and regime matrix will show.
                        # ----------------------------------------------------------
                        if joint_symbol_entries:
                            is_metrics_list  = [e['is_metrics']  for e in joint_symbol_entries if e.get('is_metrics')]
                            oos_metrics_list = [e['oos_metrics'] for e in joint_symbol_entries if e.get('oos_metrics')]
                            is_curves_list   = [e['is_curves']  for e in joint_symbol_entries if isinstance(e.get('is_curves'),  pd.DataFrame) and not e['is_curves'].empty]
                            oos_curves_list  = [e['oos_curves'] for e in joint_symbol_entries if isinstance(e.get('oos_curves'), pd.DataFrame) and not e['oos_curves'].empty]

                            combined_is  = _combine_symbol_curves(is_curves_list)
                            combined_oos = _combine_symbol_curves(oos_curves_list)

                            # Persist combined curves to disk so analyze_from_disk can also use them
                            try:
                                if not combined_is.empty:
                                    combined_is.to_csv(os.path.join(joint_dir, "equity_curve_IS_combined.csv"))
                                if not combined_oos.empty:
                                    combined_oos.to_csv(os.path.join(joint_dir, "equity_curve_OOS_combined.csv"))
                            except Exception as e:
                                logging.error(f"Error saving combined equity curves for {joint_label}: {e}")

                            top_performers_data.append({
                                'combination': combo_label,   # no symbol, no JOINT_ prefix
                                'is_metrics':  _avg_metrics(is_metrics_list)  if is_metrics_list  else {},
                                'oos_metrics': _avg_metrics(oos_metrics_list) if oos_metrics_list else {},
                                'is_curves':   combined_is,
                                'oos_curves':  combined_oos,
                                'symbols':     list(symbol_data_map.keys()),
                            })

        else:
            # ------------------------------------------------------------------
            # PER-SYMBOL MODE
            # Each symbol is optimised independently; results are reported per
            # symbol.  Works for both single-symbol and multi-symbol runs.
            # ------------------------------------------------------------------
            for symbol in symbols:
                logging.info(f"Running in PER-SYMBOL mode for: {symbol}")
                for strategy_name in strategies_to_process:
                    if strategy_name not in available_strategies:
                        logging.warning(f"Entry strategy '{strategy_name}' not found. Skipping."); continue

                    for exit_strategy_name in exit_strategies_to_process:
                        if exit_strategy_name not in available_exit_strategies:
                            logging.warning(f"Exit strategy '{exit_strategy_name}' not found. Skipping."); continue

                        strategy_info = available_strategies[strategy_name]
                        exit_strategy_info = available_exit_strategies[exit_strategy_name]

                        for timeframe in timeframes:
                            if timeframe not in strategy_info['timeframes']:
                                logging.info(f"Skipping: {strategy_name} not compatible with {timeframe}"); continue

                            combination_label = f"{symbol}_{strategy_name}_{exit_strategy_name}_{timeframe.replace(' ', '')}"
                            logging.info(f"--- Running combination: {combination_label} ---")

                            df_full_history = get_market_data(symbol, timeframe, min_data_days=total_days)
                            if df_full_history.empty:
                                logging.warning(f"No data for {symbol} {timeframe}. Skipping."); continue
                            if len(df_full_history) < total_days * 0.6:
                                logging.warning(f"Insufficient data for {symbol} {timeframe}. Skipping."); continue

                            split_date = df_full_history.index.min() + pd.DateOffset(years=in_sample_years)
                            df_in_sample = df_full_history.loc[df_full_history.index < split_date]
                            df_out_of_sample = df_full_history.loc[df_full_history.index >= split_date]
                            if df_in_sample.empty or df_out_of_sample.empty: continue

                            robust_params, study, analysis_df = find_robust_parameters(
                                [(df_in_sample, df_out_of_sample)],
                                strategy_info['function'], strategy_info['params'],
                                exit_strategy_info['function'], exit_strategy_info['params'],
                                exit_strategy_info['uses_native_exit'],
                                trials, objective_metric,
                                aggregation=aggregation,
                            )

                            combo_plot_dir = os.path.join(results_dir, combination_label)
                            _save_combo_reports(
                                combo_plot_dir, combination_label,
                                strategy_info, exit_strategy_info,
                                timeframe, robust_params,
                                df_in_sample, df_out_of_sample,
                                study, analysis_df,
                                objective_metric,
                                all_trials_data, top_performers_data,
                                symbol=symbol,
                            )

        # Fetch VIX while IBKR connection is still open
        try:
            vix_df = get_market_data('VIX', '1 day', min_data_days=1825, contract_type='IND')
        except Exception as e:
            logging.warning(f"Could not fetch VIX data: {e}")

    finally:
        shutdown_data_service()

    if top_performers_data:
        try:
            regime_analyzer.analyze(top_performers_data, results_dir, vix_df=vix_df)
        except Exception as e:
            logging.error(f"Regime analysis failed: {e}")

    if effective_mode == 'joint' and top_performers_data:
        try:
            import regime_router
            regime_router.analyze(results_dir, symbols)
        except Exception as e:
            logging.error(f"Regime router analysis failed: {e}")

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
    parser.add_argument("--objective", type=str, default="Calmar Ratio", choices=["Sharpe Ratio", "Calmar Ratio", "Total Return Pct", "Return vs Benchmark Pct"], help="Metric to optimize for.")
    parser.add_argument("--aggregation", type=str, default="average", choices=["average", "worst_case"], help="Multi-symbol objective aggregation: 'average' (mean across symbols) or 'worst_case' (min across symbols).")
    parser.add_argument("--resume_dir", type=str, default=None, help="Path to an existing run directory to resume (skips already-completed combinations).")
    parser.add_argument("--mode", type=str, default="joint", choices=["joint", "per_symbol"],
                        help="'joint': one optimisation across all symbols, combined reporting (default for multi-symbol). "
                             "'per_symbol': optimise each symbol independently, separate results per symbol.")

    args, _ = parser.parse_known_args()
    run_workflow(args.symbols, args.strategies, args.exit_strategies, args.timeframes, args.days, args.is_years, args.trials, args.objective, args.aggregation, args.resume_dir, args.mode)

if __name__ == "__main__":
    main()