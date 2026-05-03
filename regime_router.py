"""
regime_router.py

Closes the end-to-end loop of the framework:

  1. Scans an existing run directory and computes per-regime conditional OOS
     performance for every daily JOINT strategy combination.
  2. Selects the best strategy per regime using filter-first ranking:
       Hard filters (all must pass):
         - max_dd > -8%  (absolute floor)
         - abs(max_dd) <= ann_return / 2  (drawdown <= half of annual return)
         - n_bars >= 63  (~3 months of daily regime data)
       Score (after filters): ann_return × (1 + 0.10 × years_in_regime)
  3. Saves a human-readable JSON routing table.
  4. Runs a regime-switching OOS backtest per symbol:
       - market regime is re-evaluated daily on a reference equity curve
       - a persistence filter (default 3 days) prevents rapid composite whipsawing
       - the condition-stack selector is evaluated on every OOS bar
       - each contiguous same-selected-strategy segment runs its assigned strategy
         on the full prior history (so all indicator lookbacks are warm),
         then the segment equity curve is scaled and chained to the
         running capital from the previous segment
  5. Generates a self-contained HTML report (Auto Generated Strategy Routing Backtest.html):
       - routing table coloured by regime
       - per-symbol equity curves with regime background bands
       - metrics summary (total return, ann return, Sharpe, max DD, Calmar)

Standalone usage:
    python regime_router.py --run_dir results/run_XXXXXXXX
                            [--symbols QQQ SPY IWM]
                            [--min_n 30]
                            [--persistence_days 3]
                            [--timeframe "1 day"]

Also called automatically from main.py at the end of every joint run.
"""

import argparse
import ast
import importlib.util
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_combo(combo: str) -> Tuple[str, str, str]:
    """Parse 'strategy_exit_timeframe' into (strategy_name, exit_name, timeframe)."""
    tf_map = {'1day': '1 day', '4hours': '4 hours', '1hour': '1 hour'}
    for exit_name in ['fixed_tp_sl', 'no_exit']:
        tag = f'_{exit_name}_'
        if tag in combo:
            idx = combo.index(tag)
            strategy_name = combo[:idx]
            tf_tag = combo[idx + len(tag):]
            return strategy_name, exit_name, tf_map.get(tf_tag, tf_tag)
    return combo, '', ''


def _load_module(folder: str, name: str):
    path = os.path.join(folder, f'{name}.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _score(ann_ret: float, max_dd_pct: float, n: int) -> float:
    """
    Filter-first regime-assignment score.

    Hard filters (returns 0.0 if any fail):
      1. max_dd_pct > -8.0      — absolute drawdown floor
      2. abs(max_dd_pct) <= ann_ret / 2  — drawdown <= half of annual return
      3. n >= 63                — at least ~3 months of daily regime data

    Score = ann_return × (1 + 0.10 × years_in_regime)
    Annual return is the primary ranking signal; the years multiplier gives
    a mild boost to strategies with more regime data, capped implicitly by
    the 3-month minimum filter.
    """
    if ann_ret is None or ann_ret <= 0:
        return 0.0
    if max_dd_pct is None or max_dd_pct < -8.0:
        return 0.0
    if abs(max_dd_pct) > ann_ret / 2:
        return 0.0
    if n < 63:
        return 0.0
    years = n / 252.0
    return float(ann_ret * (1.0 + 0.10 * years))


def _robust_selection_score(is_metric: float, oos_metric: float) -> float:
    if oos_metric is None or not np.isfinite(oos_metric) or oos_metric <= -99998.0:
        return -99999.0
    if is_metric is None or not np.isfinite(is_metric) or is_metric <= -99998.0:
        return -99999.0
    positive_degradation = max(float(is_metric) - float(oos_metric), 0.0)
    return float(oos_metric) - 0.25 * positive_degradation


def _select_robust_row(rob_df: pd.DataFrame) -> pd.Series:
    ranked = rob_df.copy()
    if 'selection_score' not in ranked.columns:
        ranked['selection_score'] = [
            _robust_selection_score(is_v, oos_v)
            for is_v, oos_v in zip(ranked.get('is_metric', []), ranked.get('oos_metric', []))
        ]
    return ranked.sort_values(
        by=['selection_score', 'oos_metric', 'is_metric'],
        ascending=[False, False, False],
    ).iloc[0]


def smooth_regime(series: pd.Series, persistence: int) -> pd.Series:
    """
    Require `persistence` consecutive bars in a new regime before accepting
    the switch.  Prevents rapid whipsawing at regime boundaries.
    """
    out = series.copy().astype(object)
    current = series.iloc[0]
    pending = None
    streak = 0
    for i in range(len(series)):
        val = series.iloc[i]
        if val == current:
            pending, streak = None, 0
        else:
            if val == pending:
                streak += 1
                if streak >= persistence:
                    current, pending, streak = val, None, 0
            else:
                pending, streak = val, 1
        out.iloc[i] = current
    return out


def _get_segments(smoothed: pd.Series) -> List[Tuple]:
    """Convert a smoothed regime series to (start, end, regime) tuples."""
    segs: List[Tuple] = []
    if smoothed.empty:
        return segs
    current = smoothed.iloc[0]
    start = smoothed.index[0]
    for i in range(1, len(smoothed)):
        if smoothed.iloc[i] != current:
            segs.append((start, smoothed.index[i - 1], current))
            current = smoothed.iloc[i]
            start = smoothed.index[i]
    segs.append((start, smoothed.index[-1], current))
    return segs


_ROUTING_TIERS: List[Tuple[str, ...]] = [
    ('composite_regime', 'rsi_regime', 'vix_regime'),
    ('composite_regime', 'rsi_regime'),
    ('composite_regime', 'vix_regime'),
    ('rsi_regime', 'vix_regime'),
    ('composite_regime',),
    ('rsi_regime',),
    ('vix_regime',),
]

_ROUTING_TIER_LABELS = {
    'composite_regime': 'Composite',
    'rsi_regime': 'RSI',
    'vix_regime': 'VIX',
}


def _is_live_bucket(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value) not in ('unknown', 'unavailable', '')


def _tier_id(dims: Tuple[str, ...]) -> str:
    return '+'.join(d.replace('_regime', '') for d in dims)


def _tier_label(dims: Tuple[str, ...]) -> str:
    return ' + '.join(_ROUTING_TIER_LABELS.get(d, d) for d in dims)


def _condition_key_from_values(values: Dict[str, Any], dims: Tuple[str, ...]) -> str:
    return '|'.join(str(values.get(d, 'unavailable')) for d in dims)


def _condition_label_from_values(values: Dict[str, Any], dims: Tuple[str, ...]) -> str:
    return ' + '.join(
        f'{_ROUTING_TIER_LABELS.get(d, d)}={values.get(d, "unavailable")}'
        for d in dims
    )


def _key_has_only_live_buckets(key: str) -> bool:
    return all(_is_live_bucket(part) for part in key.split('|'))


def _select_combo_for_state(
    composite: Any,
    rsi: Any,
    vix: Any,
    routing_table: Dict,
) -> Tuple[Dict, float, List[str], str]:
    """
    Select the best strategy for the current condition stack.

    The router first looks for an exact Composite + RSI + VIX winner. If that
    stack has no filtered candidate, it backs off through less-specific tiers
    before using the global fallback. This avoids broad marginal winners
    swamping the actual state-specific evidence.
    """
    combos_meta = routing_table.get('combos_meta', {})
    default = routing_table.get('default', {})
    condition_routing = routing_table.get('condition_routing', {})

    state = {
        'composite_regime': composite,
        'rsi_regime': rsi,
        'vix_regime': vix,
    }
    conditions = [
        _condition_label_from_values(state, (d,))
        for d in ('composite_regime', 'rsi_regime', 'vix_regime')
        if _is_live_bucket(state.get(d))
    ]

    for dims in _ROUTING_TIERS:
        if not all(_is_live_bucket(state.get(d)) for d in dims):
            continue
        tier = _tier_id(dims)
        key = _condition_key_from_values(state, dims)
        candidate = condition_routing.get(tier, {}).get(key)
        if not candidate:
            continue
        combo = candidate.get('combo')
        if combo in combos_meta:
            return combos_meta[combo], float(candidate.get('score', 0.0)), conditions, _tier_label(dims)

    # Backward compatibility for older routing tables that only have marginal scores.
    combo_scores = routing_table.get('combo_scores', {})
    best_name = default.get('combo')
    best_score = 0.0
    raw_conditions = [str(v) for v in (composite, rsi, vix) if _is_live_bucket(v)]
    for combo, bucket_scores in combo_scores.items():
        score = float(sum(bucket_scores.get(c, 0.0) for c in raw_conditions))
        if score > best_score:
            best_score = score
            best_name = combo

    if best_name and best_name in combos_meta:
        return combos_meta[best_name], best_score, conditions, 'Marginal score fallback'
    if best_name == default.get('combo') and default:
        return default, best_score, conditions, 'Global fallback'
    return {}, best_score, conditions, 'Unrouted'


def _add_condition_candidates(
    condition_scores: Dict[str, Dict[str, List[Dict]]],
    df_oos: pd.DataFrame,
    combo: str,
    info: Dict,
) -> None:
    """Score a strategy on exact and partial condition stacks."""
    from regime_analyzer import _compute_conditional_metrics

    for dims in _ROUTING_TIERS:
        if not all(d in df_oos.columns for d in dims):
            continue
        labels = df_oos[list(dims)].astype(str).agg('|'.join, axis=1)
        cond = _compute_conditional_metrics(df_oos['_ret'], labels)
        tier = _tier_id(dims)
        for key, m in cond.items():
            if m.get('insufficient_data') or not _key_has_only_live_buckets(key):
                continue
            n = m.get('n_bars', 0)
            ann_ret = m.get('ann_return_pct') or 0.0
            max_dd = m.get('max_drawdown_pct') or 0.0
            s = _score(ann_ret, max_dd, n)
            if s <= 0:
                continue
            condition_scores.setdefault(tier, {}).setdefault(key, []).append({
                'combo': combo,
                'strategy': info['strategy'],
                'exit': info['exit'],
                'timeframe': info['timeframe'],
                'params': info['params'],
                'oos_calmar_all': info['oos_calmar'],
                'score': s,
                'ann_return_pct': round(ann_ret, 2),
                'max_dd_pct': round(max_dd, 2),
                'n_bars': n,
            })


def _decision_change_reason(prev_row: pd.Series, cur_row: pd.Series) -> str:
    changed: List[str] = []
    for col, label in (
        ('composite_regime', 'Composite'),
        ('rsi_regime', 'RSI'),
        ('vix_regime', 'VIX'),
    ):
        if str(prev_row.get(col, '')) != str(cur_row.get(col, '')):
            changed.append(label)
    return ' + '.join(changed) + ' selector change' if changed else 'Routing selector change'


def _build_routing_decisions(
    df_oos: pd.DataFrame,
    regime_series: pd.Series,
    rsi_series: Optional[pd.Series],
    vix_series: Optional[pd.Series],
    routing_table: Dict,
) -> pd.DataFrame:
    """Build the per-bar strategy selector that the routing backtest executes."""
    if df_oos.empty:
        return pd.DataFrame()

    index = df_oos.index
    decisions = pd.DataFrame(index=index)
    decisions['composite_regime'] = regime_series.reindex(index).ffill().bfill()
    decisions['rsi_regime'] = (
        rsi_series.reindex(index).ffill().bfill()
        if rsi_series is not None and not rsi_series.empty
        else 'unavailable'
    )
    decisions['vix_regime'] = (
        vix_series.reindex(index).ffill().bfill()
        if vix_series is not None and not vix_series.empty
        else 'unavailable'
    )

    rows: List[Dict] = []
    for _, row in decisions.iterrows():
        combo_info, score, conditions, tier = _select_combo_for_state(
            row.get('composite_regime'),
            row.get('rsi_regime'),
            row.get('vix_regime'),
            routing_table,
        )
        rows.append({
            'strategy_combo': combo_info.get('combo', combo_info.get('strategy', 'unknown')),
            'routing_score': score,
            'routing_conditions': ' + '.join(conditions),
            'routing_tier': tier,
        })

    selected = pd.DataFrame(rows, index=index)
    decisions = pd.concat([decisions, selected], axis=1)
    decisions = decisions[decisions['strategy_combo'].notna()]
    decisions = decisions[decisions['strategy_combo'] != 'unknown']
    return decisions


def _get_decision_segments(decisions: pd.DataFrame) -> List[Dict]:
    """Convert per-bar strategy decisions into contiguous selected-strategy segments."""
    segments: List[Dict] = []
    if decisions.empty:
        return segments

    current_combo = decisions['strategy_combo'].iloc[0]
    start_pos = 0
    for i in range(1, len(decisions)):
        if decisions['strategy_combo'].iloc[i] != current_combo:
            prev_row = decisions.iloc[i - 1]
            cur_row = decisions.iloc[i]
            segments.append({
                'start': decisions.index[start_pos],
                'end': decisions.index[i - 1],
                'combo': current_combo,
                'start_state': decisions.iloc[start_pos],
                'end_state': prev_row,
                'next_change_reason': _decision_change_reason(prev_row, cur_row),
            })
            start_pos = i
            current_combo = decisions['strategy_combo'].iloc[i]

    segments.append({
        'start': decisions.index[start_pos],
        'end': decisions.index[-1],
        'combo': current_combo,
        'start_state': decisions.iloc[start_pos],
        'end_state': decisions.iloc[-1],
        'next_change_reason': 'End of OOS window',
    })
    return segments


# ---------------------------------------------------------------------------
# Step 1 — Build routing table
# ---------------------------------------------------------------------------

def build_routing_table(
    run_dir: str,
    min_n: int = 30,
    timeframe_filter: str = '1 day',
    vix_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Scans JOINT_* folders, computes per-regime conditional OOS performance
    for every combination matching `timeframe_filter`, and selects the
    highest-scoring strategy per regime.

    Returns:
        {
          'routing'   : {regime_label → best_combo_entry_dict},
          'default'   : entry for the best overall OOS Calmar (fallback),
          'all_scores': {regime_label → [ranked combo entries]},
        }
    """
    from regime_analyzer import compute_regime_features, _compute_conditional_metrics

    combos: Dict[str, Dict] = {}

    for d in sorted(os.listdir(run_dir)):
        if not d.startswith('JOINT_'):
            continue
        combo = d[len('JOINT_'):]
        strategy_name, exit_name, timeframe = _parse_combo(combo)
        if timeframe_filter and timeframe != timeframe_filter:
            continue

        combo_dir = os.path.join(run_dir, d)
        is_f  = os.path.join(combo_dir, 'equity_curve_IS_combined.csv')
        oos_f = os.path.join(combo_dir, 'equity_curve_OOS_combined.csv')
        rob_f = os.path.join(combo_dir, 'robustness_analysis.csv')

        if not (os.path.exists(is_f) and os.path.exists(oos_f)):
            continue

        try:
            is_cv  = pd.read_csv(is_f,  parse_dates=[0], index_col=0)
            oos_cv = pd.read_csv(oos_f, parse_dates=[0], index_col=0)
        except Exception as e:
            logger.warning(f"Could not load equity curves for {combo}: {e}")
            continue

        params, oos_calmar = {}, 0.0
        if os.path.exists(rob_f):
            try:
                rob_df = pd.read_csv(rob_f)
                if not rob_df.empty:
                    best = _select_robust_row(rob_df)
                    raw = best.get('params', '{}')
                    params = ast.literal_eval(raw) if isinstance(raw, str) else {}
                    oos_calmar = float(best.get('oos_metric', 0))
            except Exception as e:
                logger.warning(f"Params parse error for {combo}: {e}")

        combos[combo] = dict(
            strategy=strategy_name, exit=exit_name, timeframe=timeframe,
            is_cv=is_cv, oos_cv=oos_cv, params=params, oos_calmar=oos_calmar,
        )

    if not combos:
        logger.warning("No JOINT combinations found with timeframe '%s'.", timeframe_filter)
        return {}

    # ── Score every combo across marginal and exact condition-stack analyses ────
    # combo_scores keeps the legacy marginal bucket diagnostics. condition_scores
    # is the runtime map: exact Composite+RSI+VIX winners first, then partial tiers.
    all_scores:   Dict[str, List] = {}              # composite only (for logging/fallback)
    combo_scores: Dict[str, Dict[str, float]] = {}  # unified cross-analysis scores
    condition_scores: Dict[str, Dict[str, List[Dict]]] = {}

    for combo, info in combos.items():
        is_cv, oos_cv = info['is_cv'], info['oos_cv']
        df_full = pd.concat([is_cv, oos_cv]).sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        if 'total_value' not in df_full.columns or 'close' not in df_full.columns:
            continue

        is_end = is_cv.index.max()
        try:
            df_reg = compute_regime_features(df_full, vix_df, is_end)
        except Exception as e:
            logger.warning(f"Regime features failed for {combo}: {e}")
            continue

        df_reg['_ret'] = df_reg['total_value'].pct_change()
        df_oos = df_reg.loc[df_reg.index > is_end]

        cond     = _compute_conditional_metrics(df_oos['_ret'], df_oos['composite_regime'])
        cond_rsi = _compute_conditional_metrics(df_oos['_ret'], df_oos['rsi_regime'])
        cond_vix: Dict = {}
        if 'vix_regime' in df_oos.columns and (df_oos['vix_regime'] != 'unavailable').any():
            cond_vix = _compute_conditional_metrics(df_oos['_ret'], df_oos['vix_regime'])

        _add_condition_candidates(condition_scores, df_oos, combo, info)

        _bucket_scores: Dict[str, float] = {}

        # Composite regime scores (also populate all_scores for logging)
        for regime, m in cond.items():
            if m.get('insufficient_data'):
                continue
            n       = m.get('n_bars', 0)
            ann_ret = m.get('ann_return_pct') or 0.0
            max_dd  = m.get('max_drawdown_pct') or 0.0
            s = _score(ann_ret, max_dd, n)
            if s > 0:
                _bucket_scores[regime] = s
                all_scores.setdefault(regime, []).append({
                    'combo':          combo,
                    'strategy':       info['strategy'],
                    'exit':           info['exit'],
                    'timeframe':      info['timeframe'],
                    'params':         info['params'],
                    'oos_calmar_all': info['oos_calmar'],
                    'score':          s,
                    'ann_return_pct': round(ann_ret, 2),
                    'max_dd_pct':     round(max_dd, 2),
                    'n_bars':         n,
                })

        # RSI regime scores
        for regime, m in cond_rsi.items():
            if m.get('insufficient_data'):
                continue
            n       = m.get('n_bars', 0)
            ann_ret = m.get('ann_return_pct') or 0.0
            max_dd  = m.get('max_drawdown_pct') or 0.0
            s = _score(ann_ret, max_dd, n)
            if s > 0:
                _bucket_scores[regime] = s

        # VIX regime scores
        for regime, m in cond_vix.items():
            if m.get('insufficient_data'):
                continue
            n       = m.get('n_bars', 0)
            ann_ret = m.get('ann_return_pct') or 0.0
            max_dd  = m.get('max_drawdown_pct') or 0.0
            s = _score(ann_ret, max_dd, n)
            if s > 0:
                _bucket_scores[regime] = s

        if _bucket_scores:
            combo_scores[combo] = _bucket_scores

    condition_routing: Dict[str, Dict[str, Dict]] = {}
    for tier, key_map in condition_scores.items():
        condition_routing[tier] = {}
        for key, candidates in key_map.items():
            ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
            key_map[key] = ranked
            if ranked and ranked[0]['score'] > 0:
                condition_routing[tier][key] = ranked[0]

    # Per-composite-bucket rankings (for logging and simple fallback)
    routing: Dict[str, Dict] = {}
    for regime, candidates in all_scores.items():
        ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
        all_scores[regime] = ranked
        if ranked[0]['score'] > 0:
            routing[regime] = ranked[0]

    # combos_meta: strategy/exit/params per combo (needed at backtest execution time)
    combos_meta: Dict[str, Dict] = {
        combo: {
            'combo':     combo,
            'strategy':  info['strategy'],
            'exit':      info['exit'],
            'timeframe': info['timeframe'],
            'params':    info['params'],
        }
        for combo, info in combos.items()
    }

    # Default fallback = best overall OOS Calmar across all daily combos
    default_combo, default_info = max(combos.items(), key=lambda x: x[1]['oos_calmar'])
    default = dict(
        combo=default_combo,
        strategy=default_info['strategy'],
        exit=default_info['exit'],
        timeframe=default_info['timeframe'],
        params=default_info['params'],
    )

    return {
        'routing':     routing,      # per-composite-bucket best (for logging/fallback)
        'combo_scores': combo_scores, # unified {combo: {bucket: score}} — marginal diagnostics
        'condition_scores': condition_scores,
        'condition_routing': condition_routing,
        'combos_meta': combos_meta,   # combo → strategy/exit/params for execution
        'default':     default,
        'all_scores':  all_scores,
        'timeframe_filter': timeframe_filter,
    }


# ---------------------------------------------------------------------------
# Step 2 — Regime-switching OOS backtest
# ---------------------------------------------------------------------------

_MODULE_CACHE: Dict[Tuple, Tuple] = {}


def _get_strategy_funcs(combo_info: Dict, strategies_dir: str, exit_strategies_dir: str):
    key = (combo_info.get('strategy'), combo_info.get('exit'))
    if key not in _MODULE_CACHE:
        strat_mod = _load_module(strategies_dir, combo_info['strategy'])
        exit_mod  = _load_module(exit_strategies_dir, combo_info['exit'])
        if strat_mod is None or exit_mod is None:
            _MODULE_CACHE[key] = (None, None, False)
        else:
            _MODULE_CACHE[key] = (
                getattr(strat_mod, 'strategy_function',   None),
                getattr(exit_mod,  'exit_logic_function', None),
                getattr(exit_mod,  'uses_native_exit',    False),
            )
    return _MODULE_CACHE[key]


def run_regime_switching_backtest(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    regime_series: pd.Series,
    routing_table: Dict,
    rsi_series: Optional[pd.Series] = None,
    vix_series: Optional[pd.Series] = None,
    strategies_dir: str = 'strategies',
    exit_strategies_dir: str = 'exit_strategies',
    initial_capital: float = 100_000.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates the Composite + RSI + VIX condition-stack selector at every OOS
    timestamp, then switches only when the selected strategy changes.

    Falls back through less-specific condition tiers before using the global
    default if no current-state score is available.

    The segment equity curves are chained multiplicatively: each segment
    starts at the capital left by the previous one.

    Returns (equity_df, metrics_dict).
    """
    from backtester import run_backtest

    combos_meta  = routing_table.get('combos_meta', {})
    default      = routing_table.get('default', {})

    df_full = pd.concat([df_is, df_oos]).sort_index()
    df_full = df_full[~df_full.index.duplicated(keep='first')]

    decisions = _build_routing_decisions(
        df_oos,
        regime_series,
        rsi_series,
        vix_series,
        routing_table,
    )
    segments = _get_decision_segments(decisions)
    parts: List[pd.DataFrame] = []
    capital = initial_capital

    for seg_meta in segments:
        seg_start = seg_meta['start']
        seg_end = seg_meta['end']
        combo_name = seg_meta['combo']
        combo_info = combos_meta.get(combo_name)
        if combo_info is None and combo_name == default.get('combo'):
            combo_info = default
        if not combo_info or not combo_info.get('strategy'):
            # Unknown / unrouted regime → hold cash
            ohlcv = [c for c in ['open', 'high', 'low', 'close'] if c in df_oos.columns]
            cash_seg = df_oos.loc[seg_start:seg_end, ohlcv].copy()
            cash_seg['total_value']     = capital
            cash_seg['benchmark_value'] = capital
            parts.append(cash_seg)
            continue

        strat_func, exit_func, native_exit = _get_strategy_funcs(
            combo_info, strategies_dir, exit_strategies_dir
        )
        if strat_func is None:
            logger.warning("Could not load '%s'. Holding cash for this segment.", combo_info.get('combo'))
            continue

        # Run strategy on full history up to segment end (keeps lookbacks warm)
        ohlcv_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df_full.columns]
        df_for_bt  = df_full.loc[:seg_end, ohlcv_cols].copy()

        try:
            result_df, _ = run_backtest(
                df_for_bt, strat_func, combo_info['params'],
                exit_func, uses_native_exit=native_exit,
                initial_capital=100_000.0,
            )
        except Exception as e:
            logger.error("Backtest error for %s [%s]: %s", combo_info.get('combo'), combo_name, e)
            continue

        seg = result_df.loc[seg_start:seg_end].copy()
        if seg.empty or seg['total_value'].iloc[0] == 0:
            continue

        # Tag which combo was active so the report can colour signals per algo
        seg['strategy_combo'] = combo_info.get('combo', combo_info.get('strategy', 'unknown'))
        seg['routing_composite'] = seg_meta['start_state'].get('composite_regime', 'unknown')
        seg['routing_rsi'] = seg_meta['start_state'].get('rsi_regime', 'unavailable')
        seg['routing_vix'] = seg_meta['start_state'].get('vix_regime', 'unavailable')
        seg['routing_conditions'] = seg_meta['start_state'].get('routing_conditions', '')
        seg['routing_score'] = seg_meta['start_state'].get('routing_score', 0.0)
        seg['routing_tier'] = seg_meta['start_state'].get('routing_tier', '')

        # If the routed slice begins with an already-open position, the visible
        # entry happened before this regime segment. Flag that carry-in exposure
        # so the report does not look like it started tracking without a trade.
        seg['segment_carry_in'] = np.nan
        seg['segment_carry_in_label'] = np.nan
        if 'shares' in seg.columns and abs(seg['shares'].iloc[0]) > 1e-9:
            entry_cols = [c for c in ('long_entry_marker', 'short_entry_marker') if c in seg.columns]
            has_entry_on_first = any(pd.notna(seg[c].iloc[0]) for c in entry_cols)
            if not has_entry_on_first:
                seg['segment_carry_in'] = seg['segment_carry_in'].astype(object)
                seg['segment_carry_in_label'] = seg['segment_carry_in_label'].astype(object)
                direction = 'long' if float(seg['shares'].iloc[0]) > 0 else 'short'
                seg.loc[seg.index[0], 'segment_carry_in'] = direction
                seg.loc[seg.index[0], 'segment_carry_in_label'] = (
                    'Adopt LONG state at regime handoff'
                    if direction == 'long'
                    else 'Adopt SHORT state at regime handoff'
                )

        # Detect routing-boundary forced exit: if the last bar still carries an open
        # position (shares != 0), that position is silently dropped when the next
        # segment starts.  Mark the last bar so the plotter can flag it distinctly.
        seg['regime_switch_exit'] = np.nan
        seg['regime_switch_exit_reason'] = np.nan
        if 'shares' in seg.columns and abs(seg['shares'].iloc[-1]) > 1e-9:
            seg['regime_switch_exit'] = seg['regime_switch_exit'].astype(object)
            seg['regime_switch_exit_reason'] = seg['regime_switch_exit_reason'].astype(object)
            seg.loc[seg.index[-1], 'regime_switch_exit'] = True
            seg.loc[seg.index[-1], 'regime_switch_exit_reason'] = seg_meta.get(
                'next_change_reason',
                'Routing selector boundary',
            )

        # Chain: scale this segment to start at current running capital
        rel_return = seg['total_value'].iloc[-1] / seg['total_value'].iloc[0]
        seg['total_value'] = seg['total_value'] / seg['total_value'].iloc[0] * capital
        if 'benchmark_value' in seg.columns and seg['benchmark_value'].iloc[0] != 0:
            seg['benchmark_value'] = (
                seg['benchmark_value'] / seg['benchmark_value'].iloc[0] * capital
            )
        # Marker columns (long_entry_marker etc.) hold raw close prices — kept
        # as-is; plotter uses them as boolean flags and plots arrows at total_value.
        capital *= rel_return
        parts.append(seg)

    if not parts:
        return pd.DataFrame(), {}

    equity_df = pd.concat(parts).sort_index()
    equity_df = equity_df[~equity_df.index.duplicated(keep='first')]

    # Summary metrics
    tv   = equity_df['total_value'].dropna()
    rets = tv.pct_change().dropna()
    total_ret = (tv.iloc[-1] / tv.iloc[0] - 1) * 100 if len(tv) > 1 else 0.0
    n_years   = max(len(tv) / 252, 1e-6)
    ann_ret   = ((1 + total_ret / 100) ** (1 / n_years) - 1) * 100
    sharpe    = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 1e-10 else 0.0
    dd        = tv / tv.cummax() - 1
    max_dd    = float(dd.min() * 100)
    calmar    = min(ann_ret / abs(max_dd), 20.0) if max_dd < -1e-10 else 0.0

    metrics = {
        'Total Return Pct': round(total_ret, 2),
        'Ann Return Pct':   round(ann_ret,   2),
        'Sharpe Ratio':     round(sharpe,    3),
        'Max Drawdown Pct': round(max_dd,    2),
        'Calmar Ratio':     round(calmar,    3),
    }
    return equity_df, metrics


# ---------------------------------------------------------------------------
# Step 3 — Orchestrate
# ---------------------------------------------------------------------------

def analyze(
    run_dir: str,
    symbols: List[str],
    min_n: int = 30,
    persistence_days: int = 3,
    timeframe_filter: str = '1 day',
    vix_df: Optional[pd.DataFrame] = None,
) -> None:
    """Main entry point — called from main.py or the CLI."""
    logger.info("=== Regime Router: building routing table ===")
    if vix_df is None:
        try:
            from data_fetcher import get_market_data
            vix_df = get_market_data('VIX', '1 day', min_data_days=365, contract_type='IND')
        except Exception as e:
            logger.warning("VIX data not available for regime router: %s", e)
            vix_df = None

    rt = build_routing_table(run_dir, min_n=min_n, timeframe_filter=timeframe_filter, vix_df=vix_df)

    if not rt:
        logger.error("Routing table empty — aborting regime-switching analysis.")
        return

    # Log routing decisions
    logger.info("Routing decisions (score = ann_return × (1 + 0.10 × years), filters: maxDD>-8%%, |DD|≤ann/2, n≥63):")
    for regime, info in sorted(rt['routing'].items()):
        logger.info(
            "  %-28s -> %-52s  score=%.2f  ann=%.1f%%  maxDD=%.1f%%  n=%d",
            regime, info['combo'], info['score'],
            info['ann_return_pct'], info['max_dd_pct'], info['n_bars'],
        )
    logger.info("  %-28s -> %s  (fallback)", 'DEFAULT', rt['default']['combo'])

    # Persist routing table JSON (strip DataFrames before serialising)
    rt_json = {
        'routing': {k: {kk: vv for kk, vv in v.items() if kk not in ('is_cv', 'oos_cv')}
                    for k, v in rt['routing'].items()},
        'condition_routing': rt.get('condition_routing', {}),
        'combo_scores': rt.get('combo_scores', {}),
        'default': rt['default'],
        'timeframe_filter': rt.get('timeframe_filter'),
    }
    rt_path = os.path.join(run_dir, 'regime_routing_table.json')
    with open(rt_path, 'w') as f:
        json.dump(rt_json, f, indent=2, default=str)
    logger.info("Routing table saved to %s", rt_path)

    # Compute reference regime series from the default combo's combined curve
    from regime_analyzer import compute_regime_features
    ref_combo_dir = os.path.join(run_dir, f"JOINT_{rt['default']['combo']}")
    ref_is_f  = os.path.join(ref_combo_dir, 'equity_curve_IS_combined.csv')
    ref_oos_f = os.path.join(ref_combo_dir, 'equity_curve_OOS_combined.csv')

    ref_regime_series: Optional[pd.Series] = None
    rsi_series_oos:    Optional[pd.Series] = None
    vix_series_oos:    Optional[pd.Series] = None
    if os.path.exists(ref_is_f) and os.path.exists(ref_oos_f):
        ref_is  = pd.read_csv(ref_is_f,  parse_dates=[0], index_col=0)
        ref_oos = pd.read_csv(ref_oos_f, parse_dates=[0], index_col=0)
        df_ref  = pd.concat([ref_is, ref_oos]).sort_index()
        df_ref  = df_ref[~df_ref.index.duplicated(keep='first')]
        if 'close' in df_ref.columns:
            is_end  = ref_is.index.max()
            df_reg  = compute_regime_features(df_ref, vix_df, is_end)
            _oos_reg = df_reg.loc[df_reg.index > is_end]
            raw      = _oos_reg['composite_regime']
            ref_regime_series = smooth_regime(raw, persistence_days)
            # Extract RSI and VIX series for the same OOS window
            rsi_series_oos: Optional[pd.Series] = _oos_reg.get('rsi_regime')
            vix_series_oos: Optional[pd.Series] = None
            if 'vix_regime' in _oos_reg.columns and (_oos_reg['vix_regime'] != 'unavailable').any():
                vix_series_oos = _oos_reg['vix_regime']
            logger.info(
                "Regime series: %d OOS bars, %d unique regimes: %s",
                len(ref_regime_series),
                ref_regime_series.nunique(),
                sorted(ref_regime_series.unique()),
            )

    if ref_regime_series is None:
        logger.error("Could not compute reference regime series. Aborting.")
        return

    # Run regime-switching backtest per symbol
    results: Dict[str, Dict] = {}

    for symbol in symbols:
        logger.info("Regime-switching backtest: %s", symbol)

        # Locate this symbol's per-symbol IS/OOS OHLCV (from any JOINT subfolder)
        df_is_sym = df_oos_sym = None
        for d in sorted(os.listdir(run_dir)):
            if not d.startswith('JOINT_'):
                continue
            sym_path = os.path.join(run_dir, d, symbol)
            is_f  = os.path.join(sym_path, 'equity_curve_IS.csv')
            oos_f = os.path.join(sym_path, 'equity_curve_OOS.csv')
            if os.path.exists(is_f) and os.path.exists(oos_f):
                df_is_sym  = pd.read_csv(is_f,  parse_dates=[0], index_col=0)
                df_oos_sym = pd.read_csv(oos_f, parse_dates=[0], index_col=0)
                break

        if df_is_sym is None or df_oos_sym is None:
            logger.warning("No per-symbol equity curves found for %s. Skipping.", symbol)
            continue

        equity_df, metrics = run_regime_switching_backtest(
            df_is_sym, df_oos_sym,
            regime_series=ref_regime_series,
            routing_table=rt,
            rsi_series=rsi_series_oos,
            vix_series=vix_series_oos,
        )

        if equity_df.empty:
            logger.warning("Empty regime-switching equity curve for %s.", symbol)
            continue

        results[symbol] = {
            'equity_df':    equity_df,
            'metrics':      metrics,
            'regime_series': ref_regime_series,
            'df_oos':       df_oos_sym,
        }
        logger.info(
            "  %s | total=%.1f%%  ann=%.1f%%  sharpe=%.3f  maxDD=%.1f%%  calmar=%.3f",
            symbol,
            metrics.get('Total Return Pct', 0),
            metrics.get('Ann Return Pct',   0),
            metrics.get('Sharpe Ratio',     0),
            metrics.get('Max Drawdown Pct', 0),
            metrics.get('Calmar Ratio',     0),
        )

    if not results:
        logger.warning("No symbol results produced — skipping report.")
        return

    from plotter import create_auto_generated_strategy_routing_report
    report_path = os.path.join(run_dir, 'Auto Generated Strategy Routing Backtest.html')
    create_auto_generated_strategy_routing_report(
        results, rt, report_path,
        rsi_series=rsi_series_oos,
        vix_series=vix_series_oos,
    )
    logger.info("Auto-generated strategy routing backtest report saved to %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='Regime-switching strategy router & backtester')
    parser.add_argument('--run_dir',          required=True)
    parser.add_argument('--symbols',          nargs='+', default=['QQQ', 'SPY', 'IWM'])
    parser.add_argument('--min_n',            type=int,  default=30,
                        help='Min OOS bars required before a regime can be assigned a strategy')
    parser.add_argument('--persistence_days', type=int,  default=3,
                        help='Consecutive days required before accepting a regime switch')
    parser.add_argument('--timeframe',        type=str,  default='1 day')
    args = parser.parse_args()
    analyze(args.run_dir, args.symbols, args.min_n, args.persistence_days, args.timeframe)
