import optuna
import logging
from typing import Dict, Any, Callable, Tuple, Optional
import pandas as pd
from backtester import run_backtest

logger = logging.getLogger(__name__)

def create_objective(
    df: pd.DataFrame,
    strategy_func: Callable,
    strategy_param_grid: Dict[str, Any],
    exit_logic_func: Optional[Callable],
    exit_param_grid: Optional[Dict[str, Any]],
    uses_native_exit: bool,
    initial_capital: float,
    commission: float,
    objective_metric: str
) -> Callable:
    """Creates the objective function for Optuna."""
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
            _, metrics = run_backtest(
                df_full_history=df,
                strategy_func=strategy_func,
                strategy_params=params,
                exit_logic_func=exit_logic_func,
                uses_native_exit=uses_native_exit,
                initial_capital=initial_capital,
                commission=commission
            )
            # Minimum trade threshold of 5
            if not metrics or metrics.get('Total Trades', 0) < 5: return -99999.0
            
            for key, value in metrics.items(): trial.set_user_attr(key, value)
            return metrics.get(objective_metric, -99999.0)
        except Exception as e:
            logger.error(f"Error during trial with params {params}: {e}")
            raise optuna.exceptions.TrialPruned()

    return objective

def find_robust_parameters(
    df_in_sample: pd.DataFrame,
    df_out_of_sample: pd.DataFrame,
    strategy_func: Callable,
    strategy_param_grid: Dict[str, Any],
    exit_logic_func: Optional[Callable],
    exit_param_grid: Optional[Dict[str, Any]],
    uses_native_exit: bool,
    n_trials: int,
    objective_metric: str,
    top_n: int = 10
) -> Tuple[Optional[Dict], Optional[optuna.Study], pd.DataFrame]:
    """
    Finds robust parameters.
    """
    logger.info(f"Starting IS optimization for '{objective_metric}'...")
    objective_func = create_objective(
        df_in_sample, strategy_func, strategy_param_grid,
        exit_logic_func, exit_param_grid, uses_native_exit,
        100000.0, 1.0, objective_metric
    )
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_func, n_trials=n_trials, n_jobs=-1, catch=(Exception,))

    logger.info("--- IS Optimization Finished. Starting Robustness Analysis ---")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        logger.warning("No successful trials completed. Cannot perform robustness check.")
        return None, study, pd.DataFrame()

    top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_n]
    if not top_trials:
        logger.warning("No valid trials found. Cannot perform robustness check.")
        return None, study, pd.DataFrame()

    analysis_results = []
    for rank, trial in enumerate(top_trials):
        params = trial.params
        is_metric_value = trial.value
        logger.info(f"Testing OOS performance for IS Rank #{rank+1} (Value: {is_metric_value:.4f})")

        _, oos_metrics = run_backtest(
            df_full_history=df_out_of_sample,
            strategy_func=strategy_func,
            strategy_params=params,
            exit_logic_func=exit_logic_func,
            uses_native_exit=uses_native_exit
        )
        oos_metric_value = oos_metrics.get(objective_metric, -99999.0)
        degradation = is_metric_value - oos_metric_value
        analysis_results.append({
            "rank": rank + 1, "params": params, "is_metric": is_metric_value,
            "oos_metric": oos_metric_value, "degradation": degradation
        })

    if not analysis_results:
        logger.warning("No results from robustness analysis.")
        return None, study, pd.DataFrame()

    analysis_df = pd.DataFrame(analysis_results)
    sane_candidates = analysis_df[analysis_df['oos_metric'] > -20.0].copy()
    
    if sane_candidates.empty:
        logger.warning("All candidates had severe OOS drawdowns. Picking best available based on degradation.")
        sane_candidates = analysis_df.copy()
    
    sane_candidates.sort_values(by='degradation', ascending=True, inplace=True)
    most_robust = sane_candidates.iloc[0]

    logger.info("--- Robustness Analysis Complete ---")
    logger.info(f"Selected Most Robust Parameters (from IS Rank #{most_robust['rank']}):")
    logger.info(f" - In-Sample {objective_metric}: {most_robust['is_metric']:.4f}")
    logger.info(f" - Out-of-Sample {objective_metric}: {most_robust['oos_metric']:.4f}")
    logger.info(f" - Degradation: {most_robust['degradation']:.4f}")
    logger.info(f" - Params: {most_robust['params']}")

    return most_robust['params'], study, analysis_df