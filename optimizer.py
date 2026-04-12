import optuna
import logging
import numpy as np
from typing import Dict, Any, Callable, Tuple, Optional, List
import pandas as pd
from backtester import run_backtest

logger = logging.getLogger(__name__)


def _aggregate(values: List[float], mode: str) -> float:
    """Aggregate per-symbol metric values into a single scalar."""
    valid = [v for v in values if v is not None and v > -99998.0]
    if not valid:
        return -99999.0
    if mode == 'worst_case':
        return float(np.min(valid))
    return float(np.mean(valid))  # default: 'average'


def create_objective(
    symbol_datasets_is: List[pd.DataFrame],
    strategy_func: Callable,
    strategy_param_grid: Dict[str, Any],
    exit_logic_func: Optional[Callable],
    exit_param_grid: Optional[Dict[str, Any]],
    uses_native_exit: bool,
    initial_capital: float,
    commission: float,
    objective_metric: str,
    aggregation: str = 'average',
) -> Callable:
    """
    Creates the Optuna objective function.

    Accepts a list of IS DataFrames (one per symbol). The objective value
    is the aggregated metric across all symbols (mean or worst-case).
    A trial is pruned if ANY symbol yields fewer than 5 trades.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {}
        for name, space in strategy_param_grid.items():
            param_name = f"sig_{name}"
            if space['type'] == 'int': params[param_name] = trial.suggest_int(param_name, space['low'], space['high'])
            elif space['type'] == 'float': params[param_name] = trial.suggest_float(param_name, space['low'], space['high'])
            elif space['type'] == 'categorical': params[param_name] = trial.suggest_categorical(param_name, space['choices'])

        if exit_param_grid:
            for name, space in exit_param_grid.items():
                param_name = f"exit_{name}"
                if space['type'] == 'int': params[param_name] = trial.suggest_int(param_name, space['low'], space['high'])
                elif space['type'] == 'float': params[param_name] = trial.suggest_float(param_name, space['low'], space['high'])
                elif space['type'] == 'categorical': params[param_name] = trial.suggest_categorical(param_name, space['choices'])

        try:
            per_symbol_values: List[float] = []
            combined_metrics: Dict[str, List] = {}

            for df_is in symbol_datasets_is:
                _, metrics = run_backtest(
                    df_full_history=df_is,
                    strategy_func=strategy_func,
                    strategy_params=params,
                    exit_logic_func=exit_logic_func,
                    uses_native_exit=uses_native_exit,
                    initial_capital=initial_capital,
                    commission=commission,
                )
                if not metrics or metrics.get('Total Trades', 0) < 5:
                    return -99999.0
                per_symbol_values.append(metrics.get(objective_metric, -99999.0))
                for key, val in metrics.items():
                    combined_metrics.setdefault(key, []).append(val)

            joint_value = _aggregate(per_symbol_values, aggregation)
            if joint_value <= -99998.0:
                return -99999.0

            # Store aggregated metrics as trial attributes for downstream analysis
            for key, vals in combined_metrics.items():
                numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                if numeric_vals:
                    trial.set_user_attr(key, float(np.mean(numeric_vals)))

            return joint_value

        except Exception as e:
            logger.error(f"Error during trial with params {params}: {e}")
            raise optuna.exceptions.TrialPruned()

    return objective


def find_robust_parameters(
    symbol_datasets: List[Tuple[pd.DataFrame, pd.DataFrame]],
    strategy_func: Callable,
    strategy_param_grid: Dict[str, Any],
    exit_logic_func: Optional[Callable],
    exit_param_grid: Optional[Dict[str, Any]],
    uses_native_exit: bool,
    n_trials: int,
    objective_metric: str,
    aggregation: str = 'average',
    top_n: int = 10,
) -> Tuple[Optional[Dict], Optional[optuna.Study], pd.DataFrame]:
    """
    Finds robust parameters, optionally across multiple symbols jointly.

    Args:
        symbol_datasets: list of (df_in_sample, df_out_of_sample) tuples,
                         one per symbol. For a single symbol, pass a
                         one-element list.
        aggregation:     'average' (default) or 'worst_case' — how to combine
                         per-symbol objective values during joint optimisation.
    """
    n_symbols = len(symbol_datasets)
    mode_str = f"joint ({n_symbols} symbols, {aggregation})" if n_symbols > 1 else "single-symbol"
    logger.info(f"Starting IS optimisation [{mode_str}] for '{objective_metric}'...")

    is_datasets  = [ds[0] for ds in symbol_datasets]
    oos_datasets = [ds[1] for ds in symbol_datasets]

    objective_func = create_objective(
        is_datasets, strategy_func, strategy_param_grid,
        exit_logic_func, exit_param_grid, uses_native_exit,
        100000.0, 1.0, objective_metric, aggregation,
    )
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_func, n_trials=n_trials, n_jobs=-1, catch=(Exception,))

    logger.info("--- IS Optimisation Finished. Starting Robustness Analysis ---")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logger.warning("No successful trials completed. Cannot perform robustness check.")
        return None, study, pd.DataFrame()

    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_n]

    analysis_results = []
    for rank, trial in enumerate(top_trials):
        params = trial.params
        is_metric_value = trial.value
        logger.info(f"Testing OOS performance for IS Rank #{rank + 1} (Value: {is_metric_value:.4f})")

        oos_values: List[float] = []
        for df_oos in oos_datasets:
            _, oos_metrics = run_backtest(
                df_full_history=df_oos,
                strategy_func=strategy_func,
                strategy_params=params,
                exit_logic_func=exit_logic_func,
                uses_native_exit=uses_native_exit,
            )
            oos_values.append(oos_metrics.get(objective_metric, -99999.0))

        oos_metric_value = _aggregate(oos_values, aggregation)
        degradation = is_metric_value - oos_metric_value
        analysis_results.append({
            "rank": rank + 1,
            "params": params,
            "is_metric": is_metric_value,
            "oos_metric": oos_metric_value,
            "degradation": degradation,
        })

    if not analysis_results:
        logger.warning("No results from robustness analysis.")
        return None, study, pd.DataFrame()

    analysis_df = pd.DataFrame(analysis_results)
    sane_candidates = analysis_df[analysis_df['oos_metric'] > -20.0].copy()

    if sane_candidates.empty:
        logger.warning("All candidates had severe OOS degradation. Picking best available.")
        sane_candidates = analysis_df.copy()

    sane_candidates.sort_values(by='degradation', ascending=True, inplace=True)
    most_robust = sane_candidates.iloc[0]

    logger.info("--- Robustness Analysis Complete ---")
    logger.info(f"Selected Most Robust Parameters (from IS Rank #{most_robust['rank']}):")
    logger.info(f" - In-Sample  {objective_metric}: {most_robust['is_metric']:.4f}")
    logger.info(f" - Out-of-Sample {objective_metric}: {most_robust['oos_metric']:.4f}")
    logger.info(f" - Degradation: {most_robust['degradation']:.4f}")
    logger.info(f" - Params: {most_robust['params']}")

    return most_robust['params'], study, analysis_df