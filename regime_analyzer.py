"""
regime_analyzer.py

Post-processing regime analysis module.

Usage modes:
  1. In-memory (called from run_workflow after all backtests):
       from regime_analyzer import analyze
       analyze(top_performers_data, results_dir, vix_df=vix_df)

  2. Standalone CLI (post-hoc analysis on existing run directory):
       python regime_analyzer.py --run_dir results/run_20260329_142500 --top_n 10

For each top-N strategy permutation (ranked by OOS Calmar), computes:
  - Conditional performance (Calmar, Sharpe, Win Rate) across regime buckets
  - Regime indicators: 21d Realized Volatility × SMA200 position (composite),
    RSI(14) zone, and VIX Percentile if VIX data is available
  - Kruskal-Wallis significance test (are regime differences statistically real?)
  - An "activation guide": top conditions that predict good performance per strategy
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI using EWM (equivalent to Wilder's method)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_regime_features(
    df_full: pd.DataFrame,
    vix_df: Optional[pd.DataFrame],
    is_end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Adds regime indicator columns to a full OHLCV price history DataFrame.

    All regime indicators (SMA200, RV21, RSI14) are computed on a DAILY
    resampled close price, then forward-filled back to the original bar
    frequency.  This makes the regime labels timeframe-agnostic:
      - rolling(200) always means 200 calendar trading days, not 200 bars
      - sqrt(252) annualisation is always applied to true daily returns
      - a 4-hour strategy and a 1-hour strategy see identical regime labels

    Quantile thresholds for realized volatility are derived from the IS
    period only to prevent lookahead bias in the OOS period.

    Args:
        df_full:    Full OHLCV history — IS + OOS concatenated (any timeframe).
        vix_df:     Optional daily VIX DataFrame (columns: open/high/low/close).
        is_end_date: The IS/OOS split date. Thresholds are computed on IS only.

    Returns:
        df_full with added columns:
          rv_21, rv_regime, sma200, sma_pos, rsi14, rsi_regime,
          vix_percentile, vix_regime, composite_regime
    """
    df = df_full.copy()

    # ------------------------------------------------------------------
    # Helper: map a daily-indexed Series back to the original bar index
    # by matching each bar's calendar date to the daily value.
    # Using .date() is timezone-safe (works for both tz-aware & tz-naive).
    # ------------------------------------------------------------------
    def _daily_to_bars(daily_series, fill_value=np.nan):
        daily_dict = {ts.date(): v for ts, v in daily_series.items()}
        return pd.Series(
            [daily_dict.get(ts.date(), fill_value) for ts in df.index],
            index=df.index,
        )

    # ------------------------------------------------------------------
    # Resample close to daily (last bar of each calendar day)
    # ------------------------------------------------------------------
    daily_close = df['close'].resample('1D').last().dropna()

    # ------------------------------------------------------------------
    # 1. 21-day Realized Volatility (annualized) — computed on daily rets
    # ------------------------------------------------------------------
    rv_daily = daily_close.pct_change().rolling(21).std() * np.sqrt(252)

    # Thresholds from IS period only (no lookahead)
    rv_is = rv_daily[daily_close.index < is_end_date].dropna()
    rv_low_thresh  = float(rv_is.quantile(0.33)) if len(rv_is) >= 21 else 0.12
    rv_high_thresh = float(rv_is.quantile(0.67)) if len(rv_is) >= 21 else 0.25

    def _rv_bucket(rv):
        if pd.isna(rv):            return 'unknown'
        if rv < rv_low_thresh:     return 'low_vol'
        if rv < rv_high_thresh:    return 'med_vol'
        return 'high_vol'

    rv_regime_daily = rv_daily.map(_rv_bucket)
    df['rv_21']    = _daily_to_bars(rv_daily, fill_value=np.nan)
    df['rv_regime'] = _daily_to_bars(rv_regime_daily, fill_value='unknown')

    # ------------------------------------------------------------------
    # 2. Price vs true 200-day SMA (always 200 daily bars)
    # ------------------------------------------------------------------
    sma200_daily = daily_close.rolling(200, min_periods=50).mean()
    sma_pos_daily = pd.Series(
        np.where(sma200_daily.isna(), 'unknown',
                 np.where(daily_close > sma200_daily, 'above', 'below')),
        index=daily_close.index,
    )
    df['sma200']  = _daily_to_bars(sma200_daily, fill_value=np.nan)
    df['sma_pos'] = _daily_to_bars(sma_pos_daily, fill_value='unknown')

    # ------------------------------------------------------------------
    # 3. RSI(14) — computed on daily close, 3-day persistent consensus
    # ------------------------------------------------------------------
    rsi_daily     = _compute_rsi(daily_close, 14)
    rsi_bull_avg  = (rsi_daily > 50).astype(float).rolling(3, min_periods=1).mean()
    rsi_regime_daily = pd.Series(
        np.where(rsi_daily.isna(), 'unknown',
                 np.where(rsi_bull_avg >= 0.67, 'rsi_bull', 'rsi_bear')),
        index=daily_close.index,
    )
    df['rsi14']      = _daily_to_bars(rsi_daily, fill_value=np.nan)
    df['rsi_regime'] = _daily_to_bars(rsi_regime_daily, fill_value='unknown')

    # ------------------------------------------------------------------
    # 4. VIX Percentile (252-day rolling rank) — VIX is already daily
    # ------------------------------------------------------------------
    if vix_df is not None and not vix_df.empty and 'close' in vix_df.columns:
        vix_close = vix_df['close'].reindex(df.index, method='ffill')

        def _vix_pct_rank(arr):
            if len(arr) < 10:
                return np.nan
            return float((arr[:-1] < arr[-1]).mean() * 100)

        df['vix_percentile'] = vix_close.rolling(252, min_periods=30).apply(
            _vix_pct_rank, raw=True
        )

        def _vix_bucket(pct):
            if pd.isna(pct): return 'unknown'
            if pct < 20:     return 'vix_very_low'
            if pct < 50:     return 'vix_low'
            if pct < 80:     return 'vix_elevated'
            return 'vix_high'

        df['vix_regime'] = df['vix_percentile'].map(_vix_bucket)
    else:
        df['vix_percentile'] = np.nan
        df['vix_regime'] = 'unavailable'

    # ------------------------------------------------------------------
    # Composite label: rv_regime × sma_pos  (e.g. "low_vol_above")
    # ------------------------------------------------------------------
    df['composite_regime'] = np.where(
        (df['rv_regime'] == 'unknown') | (df['sma_pos'] == 'unknown'),
        'unknown',
        df['rv_regime'].astype(str) + '_' + df['sma_pos'].astype(str)
    )

    return df


# ---------------------------------------------------------------------------
# Conditional performance metrics
# ---------------------------------------------------------------------------

def _compute_conditional_metrics(
    strategy_returns: pd.Series,
    regime_labels: pd.Series,
    min_bars: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    For each unique, non-unknown regime label, computes performance metrics
    from the strategy's daily return series.
    """
    results: Dict[str, Dict[str, Any]] = {}
    valid_regimes = sorted(
        r for r in regime_labels.unique()
        if r not in ('unknown', 'unavailable')
    )

    for regime in valid_regimes:
        mask = regime_labels == regime
        rets = strategy_returns[mask].dropna()
        n = len(rets)

        if n < min_bars:
            results[regime] = {
                'n_bars': n, 'mean_return_pct': np.nan,
                'ann_return_pct': np.nan, 'sharpe': np.nan,
                'calmar': np.nan, 'win_rate_pct': np.nan,
                'insufficient_data': True,
            }
            continue

        mean_ret = rets.mean()
        std_ret  = rets.std()
        sharpe   = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-10 else np.nan

        cum      = (1 + rets).cumprod()
        peak     = cum.cummax()
        dd       = (cum - peak) / peak
        max_dd   = dd.min()
        ann_ret  = (1 + mean_ret) ** 252 - 1
        calmar   = min(ann_ret / abs(max_dd), 20.0) if max_dd < -1e-10 else np.nan

        results[regime] = {
            'n_bars':            n,
            'mean_return_pct':   round(mean_ret * 100, 4),
            'ann_return_pct':    round(ann_ret  * 100, 2),
            'max_drawdown_pct':  round(max_dd   * 100, 2),   # negative, e.g. -15.3
            'sharpe':            round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
            'calmar':            round(calmar, 3) if not np.isnan(calmar) else np.nan,
            'win_rate_pct':      round((rets > 0).mean() * 100, 1),
            'insufficient_data': False,
        }

    return results


# ---------------------------------------------------------------------------
# Kruskal-Wallis significance test
# ---------------------------------------------------------------------------

def _run_kruskal_wallis(
    strategy_returns: pd.Series,
    regime_labels: pd.Series,
    min_bars: int = 10,
) -> Dict[str, Any]:
    """
    Tests whether strategy returns differ significantly across regime groups.

    H0: The distribution of daily returns is identical across all regime buckets.
    A p-value < 0.05 means the regime classification discriminates performance.
    """
    from scipy.stats import kruskal

    valid_regimes = [
        r for r in regime_labels.unique()
        if r not in ('unknown', 'unavailable')
    ]
    groups = [
        strategy_returns[regime_labels == r].dropna().values
        for r in valid_regimes
        if len(strategy_returns[regime_labels == r].dropna()) >= min_bars
    ]

    if len(groups) < 2:
        return {
            'statistic': None, 'p_value': None,
            'significant': None, 'n_groups': len(groups),
        }

    try:
        stat, p = kruskal(*groups)
        return {
            'statistic':  round(float(stat), 4),
            'p_value':    round(float(p), 4),
            'significant': bool(p < 0.05),
            'n_groups':   len(groups),
        }
    except Exception as e:
        logger.debug(f"Kruskal-Wallis failed: {e}")
        return {'statistic': None, 'p_value': None, 'significant': None, 'n_groups': len(groups)}


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze(
    top_performers_data: List[Dict],
    results_dir: str,
    vix_df: Optional[pd.DataFrame] = None,
    top_n: int = 25,
) -> None:
    """
    In-memory entry point — called from run_workflow() after backtests complete.

    Args:
        top_performers_data: list of dicts, each containing:
            'combination', 'is_metrics', 'oos_metrics', 'is_curves', 'oos_curves'
        results_dir: path to the current run's result folder
        vix_df: optional pre-loaded VIX daily DataFrame
        top_n: number of top permutations to include (by OOS Calmar)
    """
    if not top_performers_data:
        logger.warning("Regime analyzer: no data to process.")
        return

    logger.info(f"Regime analysis: processing {len(top_performers_data)} combinations...")

    # Rank by OOS Calmar, keep top_n
    sorted_data = sorted(
        top_performers_data,
        key=lambda x: x['oos_metrics'].get('Calmar Ratio', -99.0),
        reverse=True,
    )[:top_n]

    passport_entries: List[Dict] = []

    for entry in sorted_data:
        combo    = entry['combination']
        is_cv    = entry['is_curves']
        oos_cv   = entry['oos_curves']

        if (is_cv.empty and oos_cv.empty) or 'close' not in pd.concat([is_cv, oos_cv]).columns:
            continue

        # Full history for indicator computation (SMA200 needs ~200 bars of lead)
        df_full = pd.concat([is_cv, oos_cv]).sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')]

        if 'total_value' not in df_full.columns:
            continue

        is_end_date = is_cv.index.max() if not is_cv.empty else df_full.index.min()

        df_reg = compute_regime_features(df_full, vix_df, is_end_date)
        df_reg['strategy_return'] = df_reg['total_value'].pct_change()

        for period_label, period_mask in [
            ('IS',  df_reg.index <= is_end_date),
            ('OOS', df_reg.index  > is_end_date),
        ]:
            df_p = df_reg.loc[period_mask]
            if df_p.empty:
                continue

            cond_metrics = _compute_conditional_metrics(
                df_p['strategy_return'], df_p['composite_regime']
            )
            kw_result = _run_kruskal_wallis(
                df_p['strategy_return'], df_p['composite_regime']
            )
            rsi_metrics = _compute_conditional_metrics(
                df_p['strategy_return'], df_p['rsi_regime']
            )
            kw_rsi = _run_kruskal_wallis(
                df_p['strategy_return'], df_p['rsi_regime']
            )
            vix_metrics: Dict = {}
            if 'vix_regime' in df_p.columns and (df_p['vix_regime'] != 'unavailable').any():
                vix_metrics = _compute_conditional_metrics(
                    df_p['strategy_return'], df_p['vix_regime']
                )
                kw_vix = _run_kruskal_wallis(
                    df_p['strategy_return'], df_p['vix_regime']
                )
            else:
                kw_vix = {'statistic': None, 'p_value': None, 'significant': None, 'n_groups': 0}

            # ------------------------------------------------------------------
            # Coverage: how much time fell in each regime bucket this period.
            # Expressed as: n_bars (raw count), pct (% of period bars), and
            # months (approximate calendar months based on calendar date span).
            # ------------------------------------------------------------------
            total_bars_p   = len(df_p)
            total_cal_days = int((df_p.index[-1] - df_p.index[0]).days) if total_bars_p > 1 else 0

            def _cov(col: str) -> Dict[str, Dict]:
                cov: Dict[str, Dict] = {}
                for bucket in df_p[col].unique():
                    if bucket in ('unknown', 'unavailable'):
                        continue
                    n_b = int((df_p[col] == bucket).sum())
                    pct = round(n_b / total_bars_p * 100, 1) if total_bars_p > 0 else 0.0
                    months = round(total_cal_days * pct / 100.0 / 30.44, 1)
                    cov[bucket] = {'n_bars': n_b, 'pct': pct, 'months': months}
                return cov

            regime_coverage = _cov('composite_regime')
            rsi_coverage    = _cov('rsi_regime')
            vix_coverage    = _cov('vix_regime') if 'vix_regime' in df_p.columns else {}

            passport_entries.append({
                'combination':          combo,
                'period':               period_label,
                'is_metrics':           entry['is_metrics'],
                'oos_metrics':          entry['oos_metrics'],
                'conditional_metrics':  cond_metrics,   # composite rv×sma
                'rsi_metrics':          rsi_metrics,
                'vix_metrics':          vix_metrics,
                'kruskal_wallis':       kw_result,
                'kruskal_wallis_rsi':   kw_rsi,
                'kruskal_wallis_vix':   kw_vix,
                'has_vix':              bool(vix_metrics),
                'regime_coverage':      regime_coverage,
                'rsi_coverage':         rsi_coverage,
                'vix_coverage':         vix_coverage,
            })

    if not passport_entries:
        logger.warning("Regime analyzer: no valid passport entries produced.")
        return

    from plotter import create_regime_strategy_passport
    save_path = os.path.join(results_dir, "market_regime_attribution.html")
    create_regime_strategy_passport(passport_entries, save_path)
    logger.info(f"Market Regime Attribution saved to: {save_path}")


# ---------------------------------------------------------------------------
# Standalone post-hoc analysis (reads equity curve CSVs from disk)
# ---------------------------------------------------------------------------

def analyze_from_disk(run_dir: str, top_n: int = 15) -> None:
    """
    Standalone post-processing mode.

    Expects each combination sub-folder inside run_dir to contain:
      equity_curve_IS.csv, equity_curve_OOS.csv, robustness_analysis.csv

    Also attempts to load VIX from market_data/VIX_1day.csv.
    """
    logger.info(f"Loading existing run from: {run_dir}")
    top_performers_data: List[Dict] = []

    def _try_load_combo(combo_path: str, combo_label: str) -> None:
        """Attempt to load one combination folder and append to top_performers_data."""
        is_csv         = os.path.join(combo_path, "equity_curve_IS.csv")
        oos_csv        = os.path.join(combo_path, "equity_curve_OOS.csv")
        robustness_csv = os.path.join(combo_path, "robustness_analysis.csv")

        if not (os.path.exists(is_csv) and os.path.exists(oos_csv)):
            return

        try:
            is_cv  = pd.read_csv(is_csv,  parse_dates=['date'], index_col='date')
            oos_cv = pd.read_csv(oos_csv, parse_dates=['date'], index_col='date')

            is_metrics, oos_metrics = {}, {}
            if os.path.exists(robustness_csv):
                rob_df = pd.read_csv(robustness_csv)
                if not rob_df.empty:
                    best = rob_df.sort_values('degradation').iloc[0]
                    is_metrics  = {'Calmar Ratio': float(best.get('is_metric',  0))}
                    oos_metrics = {'Calmar Ratio': float(best.get('oos_metric', 0))}

            top_performers_data.append({
                'combination': combo_label,
                'is_metrics':  is_metrics,
                'oos_metrics': oos_metrics,
                'is_curves':   is_cv,
                'oos_curves':  oos_cv,
            })
        except Exception as e:
            logger.warning(f"Could not load {combo_label}: {e}")

    for combo_dir in sorted(os.listdir(run_dir)):
        combo_path = os.path.join(run_dir, combo_dir)
        if not os.path.isdir(combo_path):
            continue

        combined_is  = os.path.join(combo_path, "equity_curve_IS_combined.csv")
        combined_oos = os.path.join(combo_path, "equity_curve_OOS_combined.csv")

        if os.path.exists(combined_is) and os.path.exists(combined_oos):
            # Joint mode: load pre-combined curves saved during the run.
            # Strip the JOINT_ prefix so the label matches the in-memory convention.
            label = combo_dir[len("JOINT_"):] if combo_dir.startswith("JOINT_") else combo_dir
            try:
                is_cv  = pd.read_csv(combined_is,  parse_dates=[0], index_col=0)
                oos_cv = pd.read_csv(combined_oos, parse_dates=[0], index_col=0)
                rob_csv = os.path.join(combo_path, "robustness_analysis.csv")
                is_metrics, oos_metrics = {}, {}
                if os.path.exists(rob_csv):
                    rob_df = pd.read_csv(rob_csv)
                    if not rob_df.empty:
                        best = rob_df.sort_values('degradation').iloc[0]
                        is_metrics  = {'Calmar Ratio': float(best.get('is_metric',  0))}
                        oos_metrics = {'Calmar Ratio': float(best.get('oos_metric', 0))}
                top_performers_data.append({
                    'combination': label,
                    'is_metrics':  is_metrics,
                    'oos_metrics': oos_metrics,
                    'is_curves':   is_cv,
                    'oos_curves':  oos_cv,
                })
            except Exception as e:
                logger.warning(f"Could not load combined curves for {combo_dir}: {e}")

        elif os.path.exists(os.path.join(combo_path, "equity_curve_IS.csv")):
            # Single-symbol / per_symbol layout: equity curves directly in combo_path
            _try_load_combo(combo_path, combo_dir)
        else:
            # Fallback: per-symbol sub-folders inside a joint folder (old format)
            for sym_dir in sorted(os.listdir(combo_path)):
                sym_path = os.path.join(combo_path, sym_dir)
                if os.path.isdir(sym_path) and os.path.exists(
                    os.path.join(sym_path, "equity_curve_IS.csv")
                ):
                    _try_load_combo(sym_path, f"{sym_dir}_{combo_dir}")

    # Try to load VIX data
    vix_df = None
    try:
        from data_fetcher import get_market_data
        vix_df = get_market_data('VIX', '1 day', min_data_days=365, contract_type='IND')
    except Exception as e:
        logger.warning(f"VIX data not available: {e}")

    logger.info(f"Loaded {len(top_performers_data)} combinations from disk.")
    analyze(top_performers_data, run_dir, vix_df=vix_df, top_n=top_n)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(
        description="Post-processing regime analysis for existing run directories."
    )
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to a results/run_XXXXXXXX directory')
    parser.add_argument('--top_n',   type=int, default=25,
                        help='Number of top permutations to analyze (default: 15)')
    args = parser.parse_args()
    analyze_from_disk(args.run_dir, top_n=args.top_n)
