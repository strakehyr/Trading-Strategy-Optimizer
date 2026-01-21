import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import optuna
import pytz
import plotly.express as px

def _get_strategy_color_map(combinations: list) -> Dict[str, str]:
    """Helper to generate consistent colors based on the strategy name."""
    strategy_colors = {}
    unique_strategies = set()
    for combo in combinations:
        parts = combo.split('_')
        strat_name = parts[1] if len(parts) > 1 else combo
        unique_strategies.add(strat_name)
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
    for i, strat in enumerate(sorted(unique_strategies)):
        strategy_colors[strat] = palette[i % len(palette)]
    return strategy_colors

def _extract_strat_name(combo: str) -> str:
    parts = combo.split('_')
    return parts[1] if len(parts) > 1 else combo

def create_strategy_plot(df: pd.DataFrame, metrics: Dict, timeframe: str, save_path: Optional[str] = None):
    if df.empty: return
    df_plot = df.copy()
    try: df_plot.index = df_plot.index.tz_convert('America/New_York')
    except (TypeError, AttributeError): pass
    
    indicator_sets = {
        'atr_bands': {'columns': ['sma_high', 'sma_low'], 'subplot': 'price', 'name': 'ATR Bands'},
        'momentum': {'columns': ['long_momentum', 'short_momentum'], 'subplot': 'new', 'name': 'Momentum Score'},
        'consistency': {'columns': ['consistency'], 'subplot': 'new', 'name': 'Trend Consistency (%)'},
        'rsi': {'columns': ['rsi'], 'subplot': 'new', 'name': 'RSI'},
        'ema_trend': {'columns': ['ema_trend'], 'subplot': 'price', 'name': 'Trend EMA'}
    }
    
    six_meridian_cols = ['macd_signal', 'kdj_signal', 'rsi_signal', 'lwr_signal', 'bbi_signal', 'mtm_signal']
    has_six_meridian = all(col in df_plot.columns for col in six_meridian_cols)
    
    active_indicators, new_subplots_needed = {}, []
    for key, spec in indicator_sets.items():
        if all(col in df_plot.columns for col in spec['columns']):
            if spec['name'] not in [v['name'] for v in active_indicators.values()]:
                active_indicators[key] = spec
                if spec['subplot'] == 'new': new_subplots_needed.append(spec['name'])
    
    if has_six_meridian:
        new_subplots_needed.append('Six Meridian Indicators')

    base_rows = 3
    total_rows = base_rows + len(new_subplots_needed)
    
    row_heights = [0.5] 
    for name in new_subplots_needed:
        row_heights.append(0.15) 
    row_heights.extend([0.1, 0.25]) 
    
    subplot_titles = ['Price & Signals'] + new_subplots_needed + ['Volume', 'Portfolio Value']
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                        subplot_titles=subplot_titles, row_heights=row_heights)
    
    x_axis = df_plot.index
    
    fig.add_trace(go.Candlestick(x=x_axis, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='OHLC'), row=1, col=1)
    
    for spec in active_indicators.values():
        if spec['subplot'] == 'price':
            for col in spec['columns']:
                fig.add_trace(go.Scatter(x=x_axis, y=df_plot[col], mode='lines', name=col.replace('_', ' ').title()), row=1, col=1)

    if 'tp_level' in df_plot.columns and 'sl_level' in df_plot.columns:
        all_entries = pd.concat([df_plot['long_entry_marker'].dropna().rename('price'), df_plot['short_entry_marker'].dropna().rename('price')]).sort_index()
        all_exits = df_plot['exit_marker'].dropna()
        used_exit_indices = set()
        
        for entry_idx, entry_price in all_entries.items():
            next_exits = all_exits[all_exits.index > entry_idx]
            valid_exits = [idx for idx in next_exits.index if idx not in used_exit_indices]
            
            if valid_exits:
                end_idx = valid_exits[0]
                used_exit_indices.add(end_idx)
            else:
                end_idx = df_plot.index[-1]
            
            trade_period_df = df_plot.loc[entry_idx:end_idx]
            tp_idx = trade_period_df['tp_level'].first_valid_index()
            sl_idx = trade_period_df['sl_level'].first_valid_index()
            
            if tp_idx is not None and sl_idx is not None:
                tp_price = df_plot.loc[tp_idx, 'tp_level']
                sl_price = df_plot.loc[sl_idx, 'sl_level']
                
                fig.add_shape(type="rect", xref="x", yref="y", layer="below",
                            x0=entry_idx, y0=entry_price, x1=end_idx, y1=tp_price,
                            fillcolor="rgba(214, 39, 40, 0.1)", line_width=0, row=1, col=1)
                
                fig.add_shape(type="rect", xref="x", yref="y", layer="below",
                            x0=entry_idx, y0=entry_price, x1=end_idx, y1=sl_price,
                            fillcolor="rgba(44, 160, 44, 0.1)", line_width=0, row=1, col=1)

    if 'tp_level' in df_plot.columns and not df_plot['tp_level'].dropna().empty:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot['tp_level'], mode='lines', name='Take Profit', line=dict(color='lightgreen', dash='dash', width=1), connectgaps=False), row=1, col=1)
    if 'sl_level' in df_plot.columns and not df_plot['sl_level'].dropna().empty:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot['sl_level'], mode='lines', name='Stop Loss', line=dict(color='#FF5252', dash='dash', width=1), connectgaps=False), row=1, col=1)
    
    if 'exit_marker' in df_plot.columns:
        exits = df_plot[df_plot['exit_marker'].notna()]
        fig.add_trace(go.Scatter(x=exits.index, y=exits['exit_marker'], mode='markers', name='Exit', marker=dict(symbol='x', color='orange', size=14, line=dict(width=2))), row=1, col=1)
    if 'long_entry_marker' in df_plot.columns:
        entries = df_plot[df_plot['long_entry_marker'].notna()]
        fig.add_trace(go.Scatter(x=entries.index, y=entries['long_entry_marker'], mode='markers', name='Long Entry', marker=dict(symbol='triangle-up', color='green', size=12, line=dict(width=1, color='black'))), row=1, col=1)
    if 'short_entry_marker' in df_plot.columns:
        entries = df_plot[df_plot['short_entry_marker'].notna()]
        fig.add_trace(go.Scatter(x=entries.index, y=entries['short_entry_marker'], mode='markers', name='Short Entry', marker=dict(symbol='triangle-down', color='red', size=12, line=dict(width=1, color='black'))), row=1, col=1)
    
    current_row = 2
    for name in new_subplots_needed:
        if name == 'Six Meridian Indicators':
            heatmap_z = df_plot[six_meridian_cols].T.values
            colorscale = [[0.0, 'red'], [0.5, 'lightgrey'], [1.0, 'green']]
            fig.add_trace(go.Heatmap(
                z=heatmap_z,
                x=x_axis,
                y=[c.replace('_signal', '').upper() for c in six_meridian_cols],
                colorscale=colorscale,
                showscale=False,
                zmin=-1, zmax=1
            ), row=current_row, col=1)
        else:
            for spec in active_indicators.values():
                if spec['name'] == name:
                    for col in spec['columns']:
                        fig.add_trace(go.Scatter(x=x_axis, y=df_plot[col], mode='lines', name=col.replace('_', ' ').title()), row=current_row, col=1)
        current_row += 1
    
    fig.add_trace(go.Bar(x=x_axis, y=df_plot['volume'], name='Volume', marker_color='lightgrey'), row=total_rows - 1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot['total_value'], mode='lines', name='Strategy Value', line=dict(color='blue', width=2)), row=total_rows, col=1)
    if 'benchmark_value' in df_plot.columns and 'Benchmark' in metrics:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot['benchmark_value'], mode='lines', name=metrics.get('Benchmark', 'Benchmark'), line=dict(dash='dot', color='grey')), row=total_rows, col=1)
    
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if 'hour' in timeframe.lower() or 'min' in timeframe.lower(): rangebreaks.append(dict(pattern='hour', bounds=[16, 9.5]))
    fig.update_xaxes(rangebreaks=rangebreaks)
    
    metrics_text = f"Return: {metrics.get('Total Return Pct', 0):.2f}% | Sharpe: {metrics.get('Sharpe Ratio', 0):.2f}"
    fig.update_layout(title=f"Backtest: {metrics.get('label', 'Strategy')} ({metrics_text})", height=300 * total_rows, xaxis_rangeslider_visible=False)
    if save_path: fig.write_html(save_path)


def _find_longest_regime_period(df: pd.DataFrame) -> Dict[str, tuple]:
    if df.empty: return {}
    monthly = df.resample('ME').last()
    if monthly.empty: monthly = df.resample('M').last()
    if 'benchmark_value' not in monthly.columns: return {}
    
    rets = monthly['benchmark_value'].pct_change().dropna()
    conditions = [(rets > 0.02), (rets < -0.02)]
    choices = ['Bullish', 'Bearish']
    regimes = np.select(conditions, choices, default='Sideways')
    regime_series = pd.Series(regimes, index=rets.index)
    
    longest_periods = {}
    for regime_type in ['Bullish', 'Bearish', 'Sideways']:
        is_type = (regime_series == regime_type).astype(int)
        blocks = is_type.diff().ne(0).cumsum()
        valid_blocks = blocks[is_type == 1]
        
        if valid_blocks.empty:
            longest_periods[regime_type] = None
            continue
            
        counts = valid_blocks.value_counts()
        longest_block_id = counts.idxmax()
        block_dates = valid_blocks[valid_blocks == longest_block_id].index
        start_date = block_dates[0]
        end_date = block_dates[-1]
        
        start_ts = start_date - pd.offsets.MonthBegin(1)
        end_ts = end_date + pd.offsets.MonthEnd(0)
        longest_periods[regime_type] = (start_ts, end_ts)
    return longest_periods

def create_regime_performance_matrix(summary_data: list[Dict], save_path: str):
    if not summary_data: return
    ref_entry = summary_data[0]
    is_curve_ref = ref_entry.get('is_curves')
    oos_curve_ref = ref_entry.get('oos_curves')
    if is_curve_ref is None or is_curve_ref.empty: return
    
    combo_str = ref_entry.get('combination', '').lower()
    is_intraday = 'hour' in combo_str or 'min' in combo_str
    
    rangebreaks = [dict(bounds=["sat", "mon"])] 
    if is_intraday:
        rangebreaks.append(dict(pattern='hour', bounds=[16, 9.5])) 
    
    is_periods = _find_longest_regime_period(is_curve_ref)
    oos_periods = _find_longest_regime_period(oos_curve_ref)
    
    rows_config = []
    regime_meta = {
        'Bearish': {'color': 'rgba(255, 50, 50, 0.1)'},
        'Sideways': {'color': 'rgba(100, 149, 237, 0.1)'},
        'Bullish': {'color': 'rgba(50, 205, 50, 0.1)'}
    }

    for r_type in ['Bearish', 'Sideways', 'Bullish']:
        if is_periods.get(r_type):
            rows_config.append((f"IS {r_type}", 'is_curves', is_periods[r_type], regime_meta[r_type]['color']))
    for r_type in ['Bearish', 'Sideways', 'Bullish']:
        if oos_periods.get(r_type):
            rows_config.append((f"OOS {r_type}", 'oos_curves', oos_periods[r_type], regime_meta[r_type]['color']))
            
    if not rows_config: return

    fig = make_subplots(
        rows=len(rows_config), cols=2,
        column_widths=[0.75, 0.25],
        subplot_titles=[t for row in rows_config for t in [f"{row[0]} Market", "Strategy Returns"]],
        vertical_spacing=0.06
    )
    
    color_map = _get_strategy_color_map([d['combination'] for d in summary_data])
    
    for i, (label, curve_key, (start, end), bg_color) in enumerate(rows_config, 1):
        ref_df = ref_entry[curve_key]
        period_mask = (ref_df.index >= start) & (ref_df.index <= end)
        period_df = ref_df.loc[period_mask].copy()
        
        if period_df.empty: continue
        
        try: period_df.index = period_df.index.tz_convert('America/New_York')
        except (TypeError, AttributeError): pass

        fig.add_trace(go.Candlestick(
            x=period_df.index,
            open=period_df['open'], high=period_df['high'],
            low=period_df['low'], close=period_df['close'],
            name="Benchmark",
            showlegend=False
        ), row=i, col=1)
        
        axis_idx = (i - 1) * 2 + 1
        x_ref = f"x{axis_idx}" if axis_idx > 1 else "x"
        y_ref = f"y{axis_idx} domain" if axis_idx > 1 else "y domain"
        
        fig.add_shape(
            type="rect", xref=x_ref, yref=y_ref, 
            x0=period_df.index[0], x1=period_df.index[-1],
            y0=0, y1=1,
            fillcolor=bg_color, line_width=0, layer="below"
        )
        
        fig.update_xaxes(rangeslider_visible=False, rangebreaks=rangebreaks, row=i, col=1)
        
        bench_start_val = period_df['benchmark_value'].iloc[0]
        bench_end_val = period_df['benchmark_value'].iloc[-1]
        bench_ret = (bench_end_val / bench_start_val - 1) * 100
        
        fig.add_shape(
            type="line", x0=-0.5, x1=len(summary_data)-0.5, 
            y0=bench_ret, y1=bench_ret,
            line=dict(color="black", width=2, dash="dash"),
            row=i, col=2
        )
        
        fig.add_trace(go.Scatter(
            x=[0], y=[bench_ret], mode='text', text=[f"Bench: {bench_ret:.1f}%"],
            textposition="top right", showlegend=False
        ), row=i, col=2)

        x_vals, y_vals, colors, hover_texts = [], [], [], []
        
        for idx, strategy_entry in enumerate(summary_data):
            strat_df = strategy_entry.get(curve_key)
            if strat_df is None or strat_df.empty: continue
            
            strat_period = strat_df.loc[(strat_df.index >= start) & (strat_df.index <= end)]
            if strat_period.empty: continue
            
            start_val = strat_period['total_value'].iloc[0]
            end_val = strat_period['total_value'].iloc[-1]
            strat_ret = (end_val / start_val - 1) * 100
            
            combo_name = strategy_entry['combination']
            
            x_vals.append(idx) 
            y_vals.append(strat_ret)
            colors.append(color_map[_extract_strat_name(combo_name)])
            hover_texts.append(f"{combo_name}<br>Return: {strat_ret:.2f}%")
            
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='markers',
            marker=dict(size=10, color=colors),
            text=hover_texts, hoverinfo='text', showlegend=False
        ), row=i, col=2)
        
        fig.update_xaxes(showticklabels=False, row=i, col=2)
        fig.update_yaxes(title_text="Return %", matches=None, row=i, col=2)

    fig.update_layout(title='Regime-Specific Performance Analysis (Longest Periods)', height=350 * len(rows_config), showlegend=False)
    if save_path: fig.write_html(save_path)


def create_top_performers_plot(summary_data: list[Dict], save_path: str):
    if not summary_data: return
    
    metrics_to_plot = ['Total Return Pct', 'Sharpe Ratio', 'Calmar Ratio', 'Max Drawdown Pct']
    benchmark_key_map = {
        'Total Return Pct': 'Benchmark Return Pct',
        'Sharpe Ratio': 'Benchmark Sharpe Ratio',
        'Calmar Ratio': 'Benchmark Calmar Ratio',
        'Max Drawdown Pct': 'Benchmark Max Drawdown Pct'
    }
    
    fig = make_subplots(
        rows=len(metrics_to_plot), cols=1, shared_xaxes=True,
        subplot_titles=[m.replace(' Pct', ' (%)') for m in metrics_to_plot],
        vertical_spacing=0.1
    )
    combinations = [d['combination'] for d in summary_data]
    color_map = _get_strategy_color_map(combinations)
    bar_colors = [color_map[_extract_strat_name(c)] for c in combinations]

    for i, metric in enumerate(metrics_to_plot, 1):
        is_values = []
        for d in summary_data:
            val = d['is_metrics'].get(metric, 0)
            if metric == 'Calmar Ratio': val = min(val, 20.0)
            is_values.append(val)
            
        oos_values = []
        for d in summary_data:
            val = d['oos_metrics'].get(metric, 0)
            if metric == 'Calmar Ratio': val = min(val, 20.0)
            oos_values.append(val)

        fig.add_trace(go.Bar(name='In-Sample', x=combinations, y=is_values, marker_color=bar_colors, text=[f'{v:.2f}' for v in is_values], legendgroup='is', showlegend=(i==1)), row=i, col=1)
        fig.add_trace(go.Bar(name='Out-of-Sample', x=combinations, y=oos_values, marker_color=bar_colors, marker_pattern_shape="/", text=[f'{v:.2f}' for v in oos_values], legendgroup='oos', showlegend=(i==1)), row=i, col=1)

        bench_key = benchmark_key_map.get(metric)
        if bench_key:
            is_bench = []
            for d in summary_data:
                val = d['is_metrics'].get(bench_key, 0)
                if metric == 'Calmar Ratio': val = min(val, 20.0)
                is_bench.append(val)
            
            oos_bench = []
            for d in summary_data:
                val = d['oos_metrics'].get(bench_key, 0)
                if metric == 'Calmar Ratio': val = min(val, 20.0)
                oos_bench.append(val)

            fig.add_trace(go.Scatter(x=combinations, y=is_bench, mode='lines+markers', name='IS Benchmark', line=dict(color='black', width=2), marker=dict(size=6), legendgroup='benchmark_is', showlegend=(i==1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=combinations, y=oos_bench, mode='lines+markers', name='OOS Benchmark', line=dict(color='black', width=2, dash='dash'), marker=dict(size=6), legendgroup='benchmark_oos', showlegend=(i==1)), row=i, col=1)

    fig.update_layout(height=300 * len(metrics_to_plot), title_text="Top Performers: In-Sample and Out-of-Sample vs Benchmark", barmode='group')
    fig.update_xaxes(tickangle=-45)
    if save_path: fig.write_html(save_path)

def create_strategy_distribution_boxplot(all_studies_data: pd.DataFrame, save_path: str):
    if all_studies_data.empty: return

    all_studies_data = all_studies_data.sort_values(by='combination')
    all_combos = all_studies_data['combination'].unique()
    color_map = _get_strategy_color_map(all_combos)
    all_studies_data['strategy_family'] = all_studies_data['combination'].apply(_extract_strat_name)
    
    metrics_config = [
        {'title': 'Total Return (%)', 'key': 'user_attrs_Total Return Pct', 'bench': 'user_attrs_Benchmark Return Pct'},
        {'title': 'Sharpe Ratio', 'key': 'user_attrs_Sharpe Ratio', 'bench': 'user_attrs_Benchmark Sharpe Ratio'},
        {'title': 'Calmar Ratio', 'key': 'user_attrs_Calmar Ratio', 'bench': 'user_attrs_Benchmark Calmar Ratio'},
        {'title': 'Win Rate (%)', 'key': 'user_attrs_Win Rate', 'bench': 'user_attrs_Benchmark Win Rate'},
        {'title': 'Max Drawdown (%)', 'key': 'user_attrs_Max Drawdown Pct', 'bench': 'user_attrs_Benchmark Max Drawdown Pct'},
        {'title': 'Max Consecutive Wins', 'key': 'user_attrs_Max Consecutive Wins', 'bench': 'user_attrs_Benchmark Max Consecutive Wins'},
        {'title': 'Max Consecutive Losses', 'key': 'user_attrs_Max Consecutive Losses', 'bench': 'user_attrs_Benchmark Max Consecutive Losses'}
    ]
    
    fig = make_subplots(
        rows=len(metrics_config), cols=1,
        subplot_titles=[m['title'] for m in metrics_config],
        vertical_spacing=0.03,
        shared_xaxes=True
    )
    
    unique_combos = sorted(all_combos)
    unique_families = sorted(color_map.keys())
    
    for i, metric_info in enumerate(metrics_config, 1):
        metric_key = metric_info['key']
        bench_key = metric_info['bench']
        if metric_key not in all_studies_data.columns: continue
        
        for family in unique_families:
            family_data = all_studies_data[all_studies_data['strategy_family'] == family]
            if family_data.empty: continue
            
            fig.add_trace(go.Box(
                y=family_data[metric_key], x=family_data['combination'], name=family,
                boxpoints='outliers', marker_color=color_map[family], showlegend=(i == 1), legendgroup=family
            ), row=i, col=1)
        
        if bench_key and bench_key in all_studies_data.columns:
            bench_summary = all_studies_data.groupby('combination')[bench_key].first().reindex(unique_combos)
            fig.add_trace(go.Scatter(
                x=bench_summary.index, y=bench_summary.values, mode='lines+markers', name='Benchmark',
                line=dict(color='black', width=2), marker=dict(size=6, symbol='diamond', color='black'), showlegend=(i == 1)
            ), row=i, col=1)
        
        fig.update_xaxes(tickangle=-45, row=i, col=1)
        fig.update_yaxes(title_text="", row=i, col=1)
    
    fig.update_layout(title_text="Strategy Distribution Boxplot (In-Sample)", showlegend=True, height=600 * len(metrics_config), legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1))
    if save_path: fig.write_html(save_path)

def create_optimization_history_plot(study: optuna.Study, objective_metric: str, save_path: str):
    if not study.trials: return
    trials_df = study.trials_dataframe()
    if 'Sharpe' in objective_metric or 'Calmar' in objective_metric:
        trials_df = trials_df[(trials_df['value'] >= -100) & (trials_df['value'] <= 100)]
    fig = optuna.visualization.plot_optimization_history(study)
    if len(trials_df) < len(study.trials):
        fig.data[0].y = trials_df['value'].values
        fig.data[0].x = trials_df['number'].values
    fig.update_yaxes(title_text=objective_metric)
    if save_path: fig.write_html(save_path)

def create_parallel_coordinates_plot(study: optuna.Study, save_path: str):
    if not study.trials: return
    trials_df = study.trials_dataframe(multi_index=False).dropna()
    if trials_df.empty: return
    if 'value' in trials_df.columns:
        trials_df = trials_df[(trials_df['value'] >= -100) & (trials_df['value'] <= 100)]
    metrics_for_parcoords = {
        'user_attrs_Total Return Pct': 'Return (%)',
        'user_attrs_Return vs Benchmark Pct': 'Return vs. Bench (%)',
        'user_attrs_Win Rate': 'Win Rate (%)',
        'user_attrs_Max Drawdown Pct': 'Max Drawdown (%)',
        'user_attrs_Sharpe Ratio': 'Sharpe Ratio',
        'user_attrs_Max Consecutive Wins': 'Max Consec. Wins',
        'user_attrs_Max Consecutive Losses': 'Max Consec. Losses'
    }
    param_cols = [c for c in trials_df.columns if c.startswith('params_')]
    dims = [dict(label=p.replace('params_', '').replace('_', ' ').title(), values=trials_df[p]) for p in param_cols]
    dims += [dict(label=label, values=trials_df[key]) for key, label in metrics_for_parcoords.items() if key in trials_df.columns]
    fig = go.Figure(data=go.Parcoords(
        line=dict(color=trials_df.get('value', 0), colorscale='Viridis', showscale=True, cmin=trials_df['value'].min(), cmax=trials_df['value'].max(), colorbar=dict(title="Objective<br>Metric<br>Value")),
        dimensions=dims
    ))
    fig.update_layout(title="Parallel Coordinates of Optimization Hyperparameters and Metrics")
    if save_path: fig.write_html(save_path)

def create_is_oos_comparison_plot(is_metrics: Dict, oos_metrics: Dict, save_path: str):
    metrics_to_compare = ['Total Return Pct', 'Sharpe Ratio', 'Calmar Ratio', 'Max Drawdown Pct', 'Total Trades']
    labels = [m.replace(' Pct', ' (%)').replace('_', ' ') for m in metrics_to_compare]
    is_values = [is_metrics.get(m, 0) for m in metrics_to_compare]
    oos_values = [oos_metrics.get(m, 0) for m in metrics_to_compare]
    fig = go.Figure(data=[
        go.Bar(name='In-Sample (Optimized)', x=labels, y=is_values, text=[f'{v:.2f}' for v in is_values], textposition='auto'),
        go.Bar(name='Out-of-Sample (Validation)', x=labels, y=oos_values, text=[f'{v:.2f}' for v in oos_values], textposition='auto')
    ])
    fig.update_layout(barmode='group', title='Performance Stability: In-Sample vs. Out-of-Sample', yaxis_title="Value", legend_title="Period")
    if save_path: fig.write_html(save_path)

def create_robustness_scatter_plot(analysis_df: pd.DataFrame, objective_metric: str, save_path: str):
    if analysis_df.empty: return
    fig = go.Figure()
    min_val, max_val = min(analysis_df['is_metric'].min(), analysis_df['oos_metric'].min()), max(analysis_df['is_metric'].max(), analysis_df['oos_metric'].max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='No Degradation Line', line=dict(color='grey', dash='dash')))
    fig.add_trace(go.Scatter(
        x=analysis_df['is_metric'], y=analysis_df['oos_metric'], mode='markers',
        marker=dict(size=12, color=analysis_df['degradation'], colorscale='RdYlGn_r', showscale=True, colorbar=dict(title="Degradation")),
        text=[f"IS Rank: {r}<br>IS Metric: {im:.2f}<br>OOS Metric: {om:.2f}<br>Degradation: {d:.2f}" for r, im, om, d in zip(analysis_df['rank'], analysis_df['is_metric'], analysis_df['oos_metric'], analysis_df['degradation'])],
        hoverinfo='text', name='Top IS Candidates'
    ))
    sane_candidates = analysis_df[analysis_df['oos_metric'] >= 0].copy()
    if not sane_candidates.empty:
        most_robust = sane_candidates.sort_values(by='degradation').iloc[0]
        fig.add_trace(go.Scatter(x=[most_robust['is_metric']], y=[most_robust['oos_metric']], mode='markers', marker=dict(size=18, color='cyan', symbol='star'), name='Selected Robust Params'))
    fig.update_layout(
        title='Robustness Analysis: IS vs. OOS Performance',
        xaxis_title=f'In-Sample {objective_metric}', yaxis_title=f'Out-of-Sample {objective_metric}',
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    if save_path: fig.write_html(save_path)