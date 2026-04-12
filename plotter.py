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

def create_return_calendar_heatmap(is_df: pd.DataFrame, oos_df: pd.DataFrame, title: str, save_path: Optional[str] = None):
    """
    Creates a calendar-style heatmap showing daily returns for both IS and OOS periods.
    Green for positive, red for negative, gray for zero/unchanged days.
    """
    def process_returns(df):
        if df.empty or 'total_value' not in df.columns:
            return None
        
        df_daily = df.copy()
        if not isinstance(df_daily.index, pd.DatetimeIndex):
            return None
        
        daily_values = df_daily['total_value'].resample('D').last().dropna()
        if len(daily_values) < 2:
            return None
        
        daily_returns = daily_values.pct_change() * 100
        daily_returns = daily_returns.dropna()
        
        if daily_returns.empty:
            return None
        
        df_calendar = pd.DataFrame({
            'date': daily_returns.index,
            'return': daily_returns.values,
            'year': daily_returns.index.year,
            'month': daily_returns.index.month,
            'day': daily_returns.index.day,
            'weekday': daily_returns.index.weekday
        })
        
        df_calendar['week_of_month'] = df_calendar['date'].apply(lambda x: (x.day - 1) // 7)
        df_calendar['hover'] = df_calendar.apply(
            lambda row: f"{row['date'].strftime('%Y-%m-%d')}<br>Return: {row['return']:.2f}%", 
            axis=1
        )
        
        return df_calendar
    
    is_calendar = process_returns(is_df)
    oos_calendar = process_returns(oos_df)
    
    if is_calendar is None and oos_calendar is None:
        return
    
    colorscale = [
        [0.0, '#d62728'],
        [0.45, '#ff9999'],
        [0.48, '#d3d3d3'],
        [0.52, '#d3d3d3'],
        [0.55, '#90ee90'],
        [1.0, '#2ca02c']
    ]
    
    all_months = []
    if is_calendar is not None:
        all_months.extend(is_calendar.groupby(['year', 'month']).size().index.tolist())
    if oos_calendar is not None:
        all_months.extend(oos_calendar.groupby(['year', 'month']).size().index.tolist())
    
    unique_months = sorted(set(all_months))
    n_months = len(unique_months)
    cols = 3
    rows = int(np.ceil(n_months / cols))
    
    # Calculate appropriate vertical spacing
    if rows > 1:
        max_spacing = 1.0 / (rows - 1)
        vertical_spacing = min(0.05, max_spacing * 0.9)
    else:
        vertical_spacing = 0.05
    
    subplot_titles_list = []
    for y, m in unique_months:
        month_str = pd.Timestamp(year=y, month=m, day=1).strftime('%b %Y')
        
        is_has = is_calendar is not None and ((is_calendar['year'] == y) & (is_calendar['month'] == m)).any()
        oos_has = oos_calendar is not None and ((oos_calendar['year'] == y) & (oos_calendar['month'] == m)).any()
        
        if is_has and oos_has:
            label = f"{month_str} (IS+OOS)"
        elif is_has:
            label = f"{month_str} (IS)"
        else:
            label = f"{month_str} (OOS)"
        
        subplot_titles_list.append(label)
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles_list,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.05,
        specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)]
    )
    
    for idx, (year, month) in enumerate(unique_months):
        row = idx // cols + 1
        col = idx % cols + 1
        
        month_data = None
        if is_calendar is not None:
            is_month = is_calendar[(is_calendar['year'] == year) & (is_calendar['month'] == month)]
            if not is_month.empty:
                month_data = is_month
        
        if oos_calendar is not None:
            oos_month = oos_calendar[(oos_calendar['year'] == year) & (oos_calendar['month'] == month)]
            if not oos_month.empty:
                if month_data is not None:
                    month_data = pd.concat([month_data, oos_month])
                else:
                    month_data = oos_month
        
        if month_data is None or month_data.empty:
            continue
        
        matrix = np.full((6, 7), np.nan)
        hover_matrix = np.full((6, 7), '', dtype=object)
        
        for _, row_data in month_data.iterrows():
            week = row_data['week_of_month']
            day = row_data['weekday']
            if week < 6 and day < 7:
                matrix[week, day] = row_data['return']
                hover_matrix[week, day] = row_data['hover']
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=colorscale,
                zmid=0,
                text=hover_matrix,
                hoverinfo='text',
                showscale=(idx == 0),
                colorbar=dict(title="Return %", x=1.02) if idx == 0 else None,
                xgap=2,
                ygap=2
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=list(range(7)),
            side='top',
            row=row,
            col=col
        )
        fig.update_yaxes(
            ticktext=[f'W{i+1}' for i in range(6)],
            tickvals=list(range(6)),
            row=row,
            col=col
        )
    
    # Adjust height to accommodate many rows
    height_per_row = min(250, max(150, 3000 // rows))
    
    fig.update_layout(
        title=title,
        height=height_per_row * rows,
        showlegend=False
    )

    if save_path:
        fig.write_html(save_path)


def create_regime_strategy_passport(passport_entries: List[Dict], save_path: str):
    """
    Generates the Regime Strategy Passport HTML report (enhanced dark-theme version).

    Sections:
      1. OOS Conditional Calmar heatmap — composite regime (rv × SMA200)
      2. IS  Conditional Calmar heatmap — for reference
      3. RSI(14) momentum heatmap — bull vs bear zone
      4. VIX regime heatmap (if VIX data was available)
      5. Activation Guide table — top-2 regime conditions per strategy + K-W stats
      6. Glossary — metric & term explanations (HTML footer)

    Args:
        passport_entries: output from regime_analyzer.analyze(), one entry per
                          (combination × period). Each entry contains
                          'combination', 'period', 'conditional_metrics',
                          'rsi_metrics', 'vix_metrics', 'kruskal_wallis', etc.
        save_path:        path to write the HTML file.
    """
    if not passport_entries:
        return

    # -----------------------------------------------------------------------
    # Design tokens
    # -----------------------------------------------------------------------
    BG       = '#0f172a'
    CARD     = '#1e293b'
    BORDER   = '#334155'
    TEXT     = '#f1f5f9'
    MUTED    = '#94a3b8'
    ACCENT   = '#38bdf8'
    YELLOW   = '#f59e0b'

    # -----------------------------------------------------------------------
    # Regime display labels and tooltips
    # -----------------------------------------------------------------------
    # Technical label (kept verbatim on display)
    REGIME_DISPLAY = {
        'low_vol_above':    'low_vol_above · Low Vol · Above SMA200',
        'med_vol_above':    'med_vol_above · Med Vol · Above SMA200',
        'high_vol_above':   'high_vol_above · High Vol · Above SMA200',
        'low_vol_below':    'low_vol_below · Low Vol · Below SMA200',
        'med_vol_below':    'med_vol_below · Med Vol · Below SMA200',
        'high_vol_below':   'high_vol_below · High Vol · Below SMA200',
        'rsi_bull':         'rsi_bull · RSI Bull (>50)',
        'rsi_bear':         'rsi_bear · RSI Bear (<50)',
        'vix_very_low':     'vix_very_low · VIX Very Low (<20p)',
        'vix_low':          'vix_low · VIX Low (20–50p)',
        'vix_elevated':     'vix_elevated · VIX Elevated (50–80p)',
        'vix_high':         'vix_high · VIX High (>80p)',
    }

    # Friendly plain-English label (paired with technical name)
    REGIME_FRIENDLY = {
        'low_vol_above':  'Quiet Uptrend',
        'med_vol_above':  'Trending Bull',
        'high_vol_above': 'Volatile Bull',
        'low_vol_below':  'Quiet Downtrend',
        'med_vol_below':  'Trending Bear',
        'high_vol_below': 'Crisis / Crash',
        'rsi_bull':       'Momentum Up',
        'rsi_bear':       'Momentum Down',
        'vix_very_low':   'Complacency',
        'vix_low':        'Low Fear',
        'vix_elevated':   'Elevated Fear',
        'vix_high':       'Crisis',
    }

    REGIME_TOOLTIP = {
        'low_vol_above':    '21d annualised realised vol in lowest 33rd pct (IS). Price above 200d SMA — quiet uptrend.',
        'med_vol_above':    '21d realised vol between 33rd–67th pct. Price above 200d SMA — normal bull environment.',
        'high_vol_above':   '21d realised vol above 67th pct. Price above 200d SMA — volatile but trending up.',
        'low_vol_below':    '21d realised vol below 33rd pct. Price below 200d SMA — quiet downtrend / consolidation.',
        'med_vol_below':    '21d realised vol between 33rd–67th pct. Price below 200d SMA — moderate bear.',
        'high_vol_below':   '21d realised vol above 67th pct. Price below 200d SMA — high-stress drawdown environment.',
        'rsi_bull':         '3-bar consensus: RSI(14) > 50 in ≥ 2/3 recent bars. Short-term momentum is positive.',
        'rsi_bear':         '3-bar consensus: RSI(14) ≤ 50 in ≥ 2/3 recent bars. Short-term momentum is negative.',
        'vix_very_low':     'VIX in bottom 20th pct of trailing 252-day window — very low fear/complacency.',
        'vix_low':          'VIX between 20th–50th pct — below-average volatility expectations.',
        'vix_elevated':     'VIX between 50th–80th pct — above-average fear; market hedging activity elevated.',
        'vix_high':         'VIX above 80th pct of trailing year — high stress, potential crisis environment.',
    }

    def _display(raw: str) -> str:
        """Full axis label: 'Friendly Name\ntechnical · description'."""
        friendly = REGIME_FRIENDLY.get(raw, '')
        tech     = REGIME_DISPLAY.get(raw, raw.replace('_', ' ').title())
        return f'{friendly}\n{tech}' if friendly else tech

    def _display_short(raw: str) -> str:
        """Short hover label — just the friendly name."""
        return REGIME_FRIENDLY.get(raw, REGIME_DISPLAY.get(raw, raw.replace('_', ' ').title()))

    def _regime_tip(raw: str) -> str:
        return REGIME_TOOLTIP.get(raw, '')

    def _short_combo(combo: str) -> str:
        """
        Turn raw combo string into a readable label.
        e.g. QQQ_oracle_trend_rsi_fixed_tp_sl_4hours
          → [QQQ] oracle trend rsi | fixed tp sl | 4h
        """
        KNOWN_TICKERS = {
            'QQQ', 'SPY', 'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
            'META', 'GOOG', 'IWM', 'GLD', 'TLT', 'JOINT',
        }
        TIMEFRAME_MAP = {
            '1hour': '1h', '2hours': '2h', '4hours': '4h',
            '1day': '1d', '1week': '1w', '30mins': '30m', '15mins': '15m',
        }
        parts = combo.split('_')
        ticker = ''
        if parts and parts[0].upper() in KNOWN_TICKERS:
            ticker = parts[0].upper()
            parts = parts[1:]
        # Last part may be a timeframe
        tf = ''
        if parts and parts[-1] in TIMEFRAME_MAP:
            tf = TIMEFRAME_MAP[parts[-1]]
            parts = parts[:-1]
        elif parts and parts[-1].endswith('hour') or (parts and parts[-1].endswith('hours')):
            tf = parts[-1]
            parts = parts[:-1]
        label = ' '.join(parts).replace('_', ' ')
        pieces = []
        if ticker:
            pieces.append(f'[{ticker}]')
        if label:
            pieces.append(label)
        if tf:
            pieces.append(f'| {tf}')
        return ' '.join(pieces) if pieces else combo

    # -----------------------------------------------------------------------
    # Organise data: group by combination then period
    # -----------------------------------------------------------------------
    from collections import defaultdict
    by_combo: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for entry in passport_entries:
        by_combo[entry['combination']][entry['period']] = entry

    combos = list(by_combo.keys())

    # Composite regime buckets (from OOS entries)
    all_buckets: List[str] = []
    for combo in combos:
        for bucket in by_combo[combo].get('OOS', {}).get('conditional_metrics', {}).keys():
            if bucket not in all_buckets:
                all_buckets.append(bucket)
    all_buckets = sorted(all_buckets)

    # -----------------------------------------------------------------------
    # Unified multi-dimensional scoring across composite, RSI, and VIX analyses.
    # _COMBO_SCORES: {combo → {regime_bucket → score}} covering all three analyses.
    # A strategy earns a score in every bucket it passes filters in, regardless
    # of which analysis produced that bucket.  At runtime, the strategy with the
    # highest SUM across all current conditions (composite + RSI + VIX) is selected.
    # Used to: (a) number-badge selected combos in heatmap labels,
    #          (b) color strategy names in the Activation Guide table,
    #          (c) render the unified SCORES routing block at the bottom.
    # -----------------------------------------------------------------------
    _IFELSE_PALETTE = [
        '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
        '#8b5cf6', '#06b6d4', '#f97316', '#84cc16',
        '#ec4899', '#a855f7', '#0ea5e9', '#22c55e',
    ]
    _CIRCLE_NUMS = '①②③④⑤⑥⑦⑧⑨⑩⑪⑫'

    def _regime_score(info: Dict) -> float:
        """Return score if info passes all filters, else 0."""
        if not info or info.get('insufficient_data', True):
            return 0.0
        ann = info.get('ann_return_pct') or 0.0
        dd  = info.get('max_drawdown_pct') or 0.0
        n   = info.get('n_bars', 0)
        if ann <= 0 or dd < -8.0 or abs(dd) > ann / 2 or n < 63:
            return 0.0
        return ann * (1.0 + 0.10 * (n / 252.0))

    _COMBO_SCORES: Dict[str, Dict[str, float]] = {}
    for _c in combos:
        _oos = by_combo[_c].get('OOS', {})
        _bucket_scores: Dict[str, float] = {}
        for _bucket, _info in _oos.get('conditional_metrics', {}).items():
            s = _regime_score(_info)
            if s > 0:
                _bucket_scores[_bucket] = s
        for _bucket, _info in _oos.get('rsi_metrics', {}).items():
            s = _regime_score(_info)
            if s > 0:
                _bucket_scores[_bucket] = s
        for _bucket, _info in _oos.get('vix_metrics', {}).items():
            s = _regime_score(_info)
            if s > 0:
                _bucket_scores[_bucket] = s
        if _bucket_scores:
            _COMBO_SCORES[_c] = _bucket_scores

    # Default fallback: best overall OOS Calmar
    _default_ifelse_combo = (
        max(
            combos,
            key=lambda c: by_combo[c].get('OOS', {}).get('oos_metrics', {}).get('Calmar Ratio', -99.0),
        ) if combos else None
    )

    # Order combos by total score descending; default appended if not already present
    _selected_order = sorted(
        _COMBO_SCORES.keys(),
        key=lambda c: sum(_COMBO_SCORES[c].values()),
        reverse=True,
    )
    if _default_ifelse_combo and _default_ifelse_combo not in _selected_order:
        _selected_order.append(_default_ifelse_combo)

    ifelse_combo_color: Dict[str, str] = {
        c: _IFELSE_PALETTE[i % len(_IFELSE_PALETTE)]
        for i, c in enumerate(_selected_order)
    }
    ifelse_combo_number: Dict[str, str] = {
        c: _CIRCLE_NUMS[i] if i < len(_CIRCLE_NUMS) else f'[{i+1}]'
        for i, c in enumerate(_selected_order)
    }

    # RSI buckets
    RSI_BUCKET_ORDER = ['rsi_bull', 'rsi_bear']
    has_rsi = any(
        by_combo[c].get('OOS', {}).get('rsi_metrics')
        for c in combos
    )

    # VIX buckets
    VIX_BUCKET_ORDER = ['vix_very_low', 'vix_low', 'vix_elevated', 'vix_high']
    all_vix_buckets: List[str] = []
    has_any_vix = False
    for combo in combos:
        oos_entry = by_combo[combo].get('OOS', {})
        if oos_entry.get('has_vix'):
            has_any_vix = True
            for bucket in oos_entry.get('vix_metrics', {}).keys():
                if bucket not in all_vix_buckets:
                    all_vix_buckets.append(bucket)
    # Preserve logical order
    all_vix_buckets = [b for b in VIX_BUCKET_ORDER if b in all_vix_buckets]
    has_vix_section = has_any_vix and len(all_vix_buckets) >= 2

    # -----------------------------------------------------------------------
    # Heatmap data builder
    # -----------------------------------------------------------------------
    def _build_heatmap_matrices(period: str, metrics_key: str, bucket_list: List[str],
                                coverage_key: str = 'regime_coverage'):
        z, hover_texts, y_labels = [], [], []
        for combo in combos:
            entry = by_combo[combo].get(period, {})
            cond  = entry.get(metrics_key, {})
            cov   = entry.get(coverage_key, {})
            row_z, row_hover = [], []
            for bucket in bucket_list:
                info     = cond.get(bucket)
                tip      = _regime_tip(bucket)
                cov_info = cov.get(bucket, {})
                n_bars_b = cov_info.get('n_bars', info['n_bars'] if info else 0)
                pct_b    = cov_info.get('pct', 0)
                months_b = cov_info.get('months', 0)
                cov_str  = (f'{months_b:.1f} mo · {pct_b:.0f}% of {period} period'
                            if (months_b > 0 or pct_b > 0)
                            else f'{n_bars_b} bars')

                if info is None or info.get('insufficient_data', True):
                    row_z.append(None)
                    n = info['n_bars'] if info else 0
                    row_hover.append(
                        f'<b>{_display_short(bucket)}</b>  '
                        f'<span style="color:{MUTED};font-size:10px">({bucket})</span><br>'
                        f'<span style="color:{MUTED};font-size:11px">{tip}</span><br>'
                        f'<span style="color:{MUTED};font-size:11px">Coverage: {cov_str}</span><br><br>'
                        f'<span style="color:{YELLOW}">⚠ Insufficient data — {n} bars (min 10)</span>'
                    )
                else:
                    ann_ret  = info.get('ann_return_pct') or 0.0
                    max_dd   = info.get('max_drawdown_pct') or 0.0
                    calmar   = info.get('calmar')
                    sharpe   = info.get('sharpe') or 0.0
                    win_rate = info.get('win_rate_pct') or 0.0
                    n        = info.get('n_bars', 0)
                    calmar_clean = calmar if (calmar is not None and not (isinstance(calmar, float) and np.isnan(calmar))) else None
                    c_str = f'{calmar_clean:.2f}' if calmar_clean is not None else 'N/A'
                    row_z.append(ann_ret)
                    row_hover.append(
                        f'<b>{_display_short(bucket)}</b>  '
                        f'<span style="color:{MUTED};font-size:10px">({bucket})</span><br>'
                        f'<span style="color:{MUTED};font-size:11px">{tip}</span><br>'
                        f'<span style="color:{MUTED};font-size:11px">Coverage: {cov_str}</span><br><br>'
                        f'<b>Ann. Return:</b> {ann_ret:+.1f}%'
                        f'<span style="color:{MUTED}">  annualised compound return</span><br>'
                        f'<b>Max Drawdown:</b> {max_dd:.1f}%'
                        f'<span style="color:{MUTED}">  worst peak-to-trough in this regime</span><br>'
                        f'<b>Calmar Ratio:</b> {c_str}'
                        f'<span style="color:{MUTED}">  Ann Return ÷ Max DD, capped at 20</span><br>'
                        f'<b>Sharpe Ratio:</b> {sharpe:.2f}  '
                        f'<b>Win Rate:</b> {win_rate:.1f}%  '
                        f'<b>Sample:</b> {n} bars'
                    )
            z.append(row_z)
            hover_texts.append(row_hover)
            num = ifelse_combo_number.get(combo, '')
            prefix = f'{num} ' if num else ''
            y_labels.append(f'{prefix}{_short_combo(combo)}')
        return z, hover_texts, y_labels

    def _cell_text(z_matrix):
        """Format z-values (Ann Return %) as '+23.5%' / '—'."""
        return [[
            f'{v:+.0f}%' if (v is not None and not (isinstance(v, float) and np.isnan(v))) else '—'
            for v in row
        ] for row in z_matrix]

    def _dynamic_zrange(z_matrices):
        """Symmetric colorbar range centred on 0, driven by actual data."""
        flat = [v for mat in z_matrices for row in mat for v in row if v is not None]
        if not flat:
            return -3.0, 3.0
        mx = max(abs(min(flat)), abs(max(flat)))
        mx = max(mx, 0.5)   # don't collapse to near-zero range
        return round(-mx, 1), round(mx, 1)

    # -----------------------------------------------------------------------
    # Build coverage-aware x-axis labels
    # Coverage is taken from the first available OOS entry (same market data
    # period for all combos, so one entry is representative).
    # Label format (multi-line): "Friendly Name\ntechnical · description\n~X.X mo · Y%"
    # -----------------------------------------------------------------------
    def _first_oos_coverage(cov_key: str) -> Dict[str, Dict]:
        for combo in combos:
            cov = by_combo[combo].get('OOS', {}).get(cov_key, {})
            if cov:
                return cov
        return {}

    _oos_comp_cov = _first_oos_coverage('regime_coverage')
    _oos_rsi_cov  = _first_oos_coverage('rsi_coverage')
    _oos_vix_cov  = _first_oos_coverage('vix_coverage')
    _is_comp_cov  = {
        combo: by_combo[combo].get('IS', {}).get('regime_coverage', {})
        for combo in combos
    }
    # Aggregate IS coverage (single representative)
    _is_comp_cov_agg: Dict[str, Dict] = {}
    for combo in combos:
        cov = by_combo[combo].get('IS', {}).get('regime_coverage', {})
        if cov:
            _is_comp_cov_agg = cov
            break

    def _axis_label(bucket: str, cov_dict: Dict[str, Dict]) -> str:
        friendly = REGIME_FRIENDLY.get(bucket, '')
        tech     = REGIME_DISPLAY.get(bucket, bucket.replace('_', ' ').title())
        cov_info = cov_dict.get(bucket, {})
        months   = cov_info.get('months', 0)
        pct      = cov_info.get('pct', 0)
        cov_line = f'~{months:.1f} mo · {pct:.0f}%' if (months > 0 or pct > 0) else ''
        lines    = [friendly, tech]
        if cov_line:
            lines.append(cov_line)
        return '\n'.join(filter(None, lines))

    x_labels_oos  = [_axis_label(b, _oos_comp_cov) for b in all_buckets]
    x_labels_is   = [_axis_label(b, _is_comp_cov_agg) for b in all_buckets]
    x_labels_rsi  = [_axis_label(b, _oos_rsi_cov)  for b in RSI_BUCKET_ORDER]
    x_labels_vix  = [_axis_label(b, _oos_vix_cov)  for b in all_vix_buckets]

    # -----------------------------------------------------------------------
    # Build subplot layout — RSI merged into table, not a separate heatmap
    # -----------------------------------------------------------------------
    num_combos    = len(combos)
    hm_height     = max(320, num_combos * 38 + 140)
    vix_height    = max(220, num_combos * 38 + 140)
    table_height  = max(280, num_combos * 36 + 140)

    specs  = [[{'type': 'heatmap'}], [{'type': 'heatmap'}], [{'type': 'heatmap'}]]
    titles = [
        '① OOS Ann. Return % by Volatility × Trend Regime  ← primary signal: use this to decide WHEN to trade each strategy',
        '② IS Ann. Return % by Volatility × Trend Regime  ← optimizer training window, shown for comparison only (not a live signal)',
        '③ OOS Ann. Return % by RSI(14) Momentum Zone  ← does the strategy prefer bullish or bearish short-term momentum?',
    ]
    rsi_height = max(180, num_combos * 34 + 120)
    row_heights = [hm_height, hm_height, rsi_height]

    if has_vix_section:
        specs.append([{'type': 'heatmap'}])
        titles.append('④ OOS Ann. Return % by VIX Fear Level  ← how each strategy performs under different implied-volatility environments')
        row_heights.append(vix_height)

    table_section_num = '⑤' if has_vix_section else '④'
    specs.append([{'type': 'table'}])
    titles.append(f'{table_section_num} Activation Guide  ← best deployment condition across all 3 regime dimensions (Vol×Trend / RSI / VIX). ★ = statistically validated · ~ = observed pattern')
    row_heights.append(table_height)

    n_sections = len(specs)

    fig = make_subplots(
        rows=n_sections, cols=1,
        subplot_titles=titles,
        specs=specs,
        vertical_spacing=0.10,
        row_heights=row_heights,
    )

    # -----------------------------------------------------------------------
    # Dynamic colorbar range (shared across heatmaps)
    # -----------------------------------------------------------------------
    z_oos_pre, _, _ = _build_heatmap_matrices('OOS', 'conditional_metrics', all_buckets)
    z_is_pre,  _, _ = _build_heatmap_matrices('IS',  'conditional_metrics', all_buckets)
    zmin_comp, zmax_comp = _dynamic_zrange([z_oos_pre, z_is_pre])

    z_vix_pre = []
    if has_vix_section:
        z_vix_pre, _, _ = _build_heatmap_matrices('OOS', 'vix_metrics', all_vix_buckets)
    zmin_vix, zmax_vix = _dynamic_zrange([z_vix_pre]) if z_vix_pre else (zmin_comp, zmax_comp)

    # -----------------------------------------------------------------------
    # Shared heatmap style
    # -----------------------------------------------------------------------
    # Dark diverging colorscale: avoids the bright yellow in the centre of RdYlGn
    # so white cell labels remain readable on every background shade.
    DARK_DIVERGE = [
        [0.00, '#7f1d1d'],  # deep red   (very negative)
        [0.20, '#dc2626'],  # red
        [0.40, '#b45309'],  # amber/dark orange
        [0.50, '#1e3a4c'],  # dark teal  (near zero — neutral, never bright)
        [0.60, '#166534'],  # dark green
        [0.80, '#15803d'],  # green
        [1.00, '#14532d'],  # deep green (very positive)
    ]

    HM_COMMON = dict(
        colorscale=DARK_DIVERGE,
        zmid=0,
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=11, color='white'),
        hovertemplate='%{customdata}<extra></extra>',
        showscale=False,
    )

    current_row = 1

    # -----------------------------------------------------------------------
    # OOS composite heatmap (with colorbar)
    # -----------------------------------------------------------------------
    z_oos, txt_oos, lbl_oos = _build_heatmap_matrices(
        'OOS', 'conditional_metrics', all_buckets, coverage_key='regime_coverage')
    fig.add_trace(go.Heatmap(
        z=z_oos,
        x=x_labels_oos,
        y=lbl_oos,
        text=_cell_text(z_oos),
        customdata=txt_oos,
        zmin=zmin_comp, zmax=zmax_comp,
        colorbar=dict(
            title=dict(text='Ann Ret %', font=dict(color=TEXT, size=12)),
            tickfont=dict(color=TEXT, size=11),
            ticksuffix='%',
            x=1.01, thickness=14, len=0.3,
        ),
        showscale=True,
        **{k: v for k, v in HM_COMMON.items() if k != 'showscale'},
    ), row=current_row, col=1)
    current_row += 1

    # -----------------------------------------------------------------------
    # IS composite heatmap
    # -----------------------------------------------------------------------
    z_is, txt_is, lbl_is = _build_heatmap_matrices(
        'IS', 'conditional_metrics', all_buckets, coverage_key='regime_coverage')
    fig.add_trace(go.Heatmap(
        z=z_is,
        x=x_labels_is,
        y=lbl_is,
        text=_cell_text(z_is),
        customdata=txt_is,
        zmin=zmin_comp, zmax=zmax_comp,
        **HM_COMMON,
    ), row=current_row, col=1)
    current_row += 1

    # -----------------------------------------------------------------------
    # RSI(14) momentum heatmap — bull vs bear zone
    # -----------------------------------------------------------------------
    RSI_BUCKET_ORDER_DISPLAY = ['rsi_bull', 'rsi_bear']
    z_rsi_pre, _, _ = _build_heatmap_matrices(
        'OOS', 'rsi_metrics', RSI_BUCKET_ORDER_DISPLAY, coverage_key='rsi_coverage')
    zmin_rsi, zmax_rsi = _dynamic_zrange([z_rsi_pre])
    z_rsi, txt_rsi, lbl_rsi = _build_heatmap_matrices(
        'OOS', 'rsi_metrics', RSI_BUCKET_ORDER_DISPLAY, coverage_key='rsi_coverage')
    fig.add_trace(go.Heatmap(
        z=z_rsi,
        x=x_labels_rsi,
        y=lbl_rsi,
        text=_cell_text(z_rsi),
        customdata=txt_rsi,
        zmin=zmin_rsi, zmax=zmax_rsi,
        **HM_COMMON,
    ), row=current_row, col=1)
    current_row += 1

    # -----------------------------------------------------------------------
    # VIX heatmap
    # -----------------------------------------------------------------------
    if has_vix_section:
        z_vix, txt_vix, lbl_vix = _build_heatmap_matrices(
            'OOS', 'vix_metrics', all_vix_buckets, coverage_key='vix_coverage')
        fig.add_trace(go.Heatmap(
            z=z_vix,
            x=x_labels_vix,
            y=lbl_vix,
            text=_cell_text(z_vix),
            customdata=txt_vix,
            zmin=zmin_vix, zmax=zmax_vix,
            **HM_COMMON,
        ), row=current_row, col=1)
        current_row += 1

    # -----------------------------------------------------------------------
    # Activation Guide table — all three regime dimensions
    # Columns: Strategy | Overall OOS |
    #   Best Vol×Trend | Ann Ret | Max DD |
    #   Best RSI Zone  | Ann Ret |
    #   Best VIX Level | Ann Ret |   ← conditional on VIX availability
    #   Combined Deployment Signal |
    #   KW p: Vol×Trend | RSI | VIX |
    #   Regime Matters? (any dimension validated)
    # -----------------------------------------------------------------------
    def _best_dim_cell(metrics_dict: Dict, cov_dict: Dict):
        """Find the best-performing regime bucket in any dimension by Ann Return.
        Returns: (display_label, ann_ret_str, raw_bucket_key, is_positive)
        """
        valid = {
            b: info for b, info in metrics_dict.items()
            if not info.get('insufficient_data', True)
            and info.get('ann_return_pct') is not None
        }
        if not valid:
            return 'N/A', 'N/A', None, False
        best_b    = max(valid, key=lambda b: valid[b]['ann_return_pct'])
        best_info = valid[best_b]
        ret       = best_info['ann_return_pct']
        cov       = cov_dict.get(best_b, {})
        months    = cov.get('months', 0)
        friendly  = REGIME_FRIENDLY.get(best_b, best_b.replace('_', ' ').title())
        label     = f'{friendly}\n({best_b})' + (f'\n~{months:.1f} mo' if months > 0 else '')
        sign      = '▲' if ret > 0 else '▼'
        return label, f'{sign} {ret:+.1f}%', best_b, ret > 0

    # Determine if VIX column should appear (any combo has VIX data)
    guide_has_vix = any(
        by_combo[c].get('OOS', {}).get('has_vix') for c in combos
    )

    col_defs = [
        'Strategy',
        'Overall OOS',
        '① Best\nVol×Trend Regime',
        'Ann Ret',
        'Max DD',
        '② Best\nRSI Zone',
        'Ann Ret',
    ]
    if guide_has_vix:
        col_defs += ['③ Best\nVIX Level', 'Ann Ret']
    col_defs += [
        '🧭 Combined\nDeployment Signal',
        'KW p\nVol×Trend',
        'KW p\nRSI',
    ]
    if guide_has_vix:
        col_defs.append('KW p\nVIX')
    col_defs.append('Regime\nMatters?')

    # Named column lists — avoids dict key collision for duplicate 'Ann Ret' headers
    _col_strategy:    List = []
    _col_overall:     List = []
    _col_comp_regime: List = []
    _col_comp_ret:    List = []
    _col_comp_dd:     List = []
    _col_rsi_regime:  List = []
    _col_rsi_ret:     List = []
    _col_vix_regime:  List = []
    _col_vix_ret:     List = []
    _col_signal:      List = []
    _col_kw_comp:     List = []
    _col_kw_rsi:      List = []
    _col_kw_vix:      List = []
    _col_matters:     List = []

    best_pos_flags: List[bool] = []
    any_sig_flags:  List[Optional[bool]] = []

    for combo in combos:
        oos_entry  = by_combo[combo].get('OOS', {})
        cond_comp  = oos_entry.get('conditional_metrics', {})
        cond_rsi   = oos_entry.get('rsi_metrics', {})
        cond_vix   = oos_entry.get('vix_metrics', {})
        cov_comp   = oos_entry.get('regime_coverage', {})
        cov_rsi    = oos_entry.get('rsi_coverage', {})
        cov_vix    = oos_entry.get('vix_coverage', {})
        kw_comp    = oos_entry.get('kruskal_wallis', {})
        kw_rsi_d   = oos_entry.get('kruskal_wallis_rsi', {})
        kw_vix_d   = oos_entry.get('kruskal_wallis_vix', {})

        comp_lbl, comp_ret, comp_raw, comp_pos = _best_dim_cell(cond_comp, cov_comp)
        rsi_lbl,  rsi_ret,  rsi_raw,  _        = _best_dim_cell(cond_rsi,  cov_rsi)
        vix_lbl,  vix_ret,  vix_raw,  _        = _best_dim_cell(cond_vix,  cov_vix)
        best_pos_flags.append(comp_pos)

        # Max DD for best composite bucket
        comp_best_info = cond_comp.get(comp_raw, {}) if comp_raw else {}
        comp_dd  = comp_best_info.get('max_drawdown_pct', 0) or 0
        comp_dd_str = f'{comp_dd:.1f}%' if comp_raw else 'N/A'

        # KW p-values
        kw_comp_sig = kw_comp.get('significant')
        kw_rsi_sig  = kw_rsi_d.get('significant')
        kw_vix_sig  = kw_vix_d.get('significant')
        any_sig     = any(s is True for s in [kw_comp_sig, kw_rsi_sig, kw_vix_sig])
        any_sig_flags.append(any_sig if any(
            s is not None for s in [kw_comp_sig, kw_rsi_sig, kw_vix_sig]
        ) else None)

        kw_comp_str = f"{kw_comp.get('p_value'):.3f}" if kw_comp.get('p_value') is not None else 'N/A'
        kw_rsi_str  = f"{kw_rsi_d.get('p_value'):.3f}" if kw_rsi_d.get('p_value') is not None else 'N/A'
        kw_vix_str  = f"{kw_vix_d.get('p_value'):.3f}" if kw_vix_d.get('p_value') is not None else 'N/A'

        # Combined deployment signal:
        # ★ = statistically validated (KW p<0.05), ~ = observed pattern only
        def _signal_part(raw, sig):
            if raw is None or raw == 'N/A':
                return None
            friendly = REGIME_FRIENDLY.get(raw, raw.replace('_', ' ').title())
            mark = '★' if sig else '~'
            return f'{mark} {friendly}'

        signal_parts = list(filter(None, [
            _signal_part(comp_raw, kw_comp_sig),
            _signal_part(rsi_raw,  kw_rsi_sig),
            _signal_part(vix_raw,  kw_vix_sig) if guide_has_vix else None,
        ]))
        signal_str = '\n'.join(signal_parts) if signal_parts else 'Run anytime'

        # Overall OOS label
        overall_ann    = oos_entry.get('oos_metrics', {}).get('Total Return Pct')
        overall_calmar = oos_entry.get('oos_metrics', {}).get('Calmar Ratio')
        if overall_ann is not None:
            overall_str = f'{overall_ann:.1f}%'
        elif overall_calmar is not None:
            overall_str = f'{min(overall_calmar, 20.0):.2f} (Calmar)'
        else:
            overall_str = 'N/A'

        # Regime Matters summary
        if any_sig:
            sig_summary = '✅ Yes'
        elif all(s is False for s in [kw_comp_sig, kw_rsi_sig, kw_vix_sig] if s is not None):
            sig_summary = '❌ No'
        else:
            sig_summary = '—'

        _num = ifelse_combo_number.get(combo, '')
        _lbl = f'{_num} {_short_combo(combo)}' if _num else _short_combo(combo)
        _col_strategy.append(_lbl)
        _col_overall.append(overall_str)
        _col_comp_regime.append(comp_lbl)
        _col_comp_ret.append(comp_ret)
        _col_comp_dd.append(comp_dd_str)
        _col_rsi_regime.append(rsi_lbl)
        _col_rsi_ret.append(rsi_ret)
        _col_vix_regime.append(vix_lbl)
        _col_vix_ret.append(vix_ret)
        _col_signal.append(signal_str)
        _col_kw_comp.append(kw_comp_str)
        _col_kw_rsi.append(kw_rsi_str)
        _col_kw_vix.append(kw_vix_str)
        _col_matters.append(sig_summary)

    n_tbl = len(_col_strategy)
    even_row = CARD
    odd_row  = '#162032'
    row_base = [even_row if i % 2 == 0 else odd_row for i in range(n_tbl)]
    best_regime_col = ['#14532d' if p else '#450a0a' for p in best_pos_flags]
    dd_col = ['#1e293b'] * n_tbl
    sig_col = [
        '#14532d' if s is True else ('#450a0a' if s is False else '#1e293b')
        for s in any_sig_flags
    ]
    kw_comp_col = [
        '#14532d' if (by_combo[c].get('OOS', {}).get('kruskal_wallis', {}).get('significant') is True)
        else ('#450a0a' if (by_combo[c].get('OOS', {}).get('kruskal_wallis', {}).get('significant') is False)
              else '#1e293b')
        for c in combos
    ]
    kw_rsi_col = [
        '#14532d' if (by_combo[c].get('OOS', {}).get('kruskal_wallis_rsi', {}).get('significant') is True)
        else ('#450a0a' if (by_combo[c].get('OOS', {}).get('kruskal_wallis_rsi', {}).get('significant') is False)
              else '#1e293b')
        for c in combos
    ]
    kw_vix_col = [
        '#14532d' if (by_combo[c].get('OOS', {}).get('kruskal_wallis_vix', {}).get('significant') is True)
        else ('#450a0a' if (by_combo[c].get('OOS', {}).get('kruskal_wallis_vix', {}).get('significant') is False)
              else '#1e293b')
        for c in combos
    ]

    # Strategy column: tinted background for IF-ELSE selected combos
    def _tint(hex_color: str, alpha: float = 0.30) -> str:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'

    strategy_row_col = [
        _tint(ifelse_combo_color[c]) if c in ifelse_combo_color
        else (even_row if i % 2 == 0 else odd_row)
        for i, c in enumerate(combos)
    ]

    # Build cell_colors matching col_defs order
    cell_colors = [
        strategy_row_col, # Strategy
        row_base,         # Overall OOS
        best_regime_col,  # ① Best Vol×Trend
        best_regime_col,  # Ann Ret (composite best)
        dd_col,           # Max DD
        row_base,         # ② Best RSI Zone
        row_base,         # Ann Ret (RSI)
    ]
    if guide_has_vix:
        cell_colors += [row_base, row_base]  # ③ VIX + Ann Ret
    cell_colors += [
        row_base,         # Combined Signal
        kw_comp_col,      # KW Vol×Trend
        kw_rsi_col,       # KW RSI
    ]
    if guide_has_vix:
        cell_colors.append(kw_vix_col)
    cell_colors.append(sig_col)   # Regime Matters?

    # Column widths (pixels)
    col_widths = [185, 90, 195, 90, 80, 150, 90]
    if guide_has_vix:
        col_widths += [140, 90]
    col_widths += [190, 80, 75]
    if guide_has_vix:
        col_widths.append(75)
    col_widths.append(95)

    # Assemble cell values in the same order as col_defs (no dict key collisions)
    cell_values_list = [
        _col_strategy, _col_overall,
        _col_comp_regime, _col_comp_ret, _col_comp_dd,
        _col_rsi_regime, _col_rsi_ret,
    ]
    if guide_has_vix:
        cell_values_list += [_col_vix_regime, _col_vix_ret]
    cell_values_list += [_col_signal, _col_kw_comp, _col_kw_rsi]
    if guide_has_vix:
        cell_values_list.append(_col_kw_vix)
    cell_values_list.append(_col_matters)

    fig.add_trace(go.Table(
        columnwidth=col_widths,
        header=dict(
            values=[f'<b>{c}</b>' for c in col_defs],
            fill_color='#0c1a2e',
            font=dict(color=ACCENT, size=12, family='monospace'),
            align='left',
            height=36,
            line=dict(color=BORDER, width=1),
        ),
        cells=dict(
            values=cell_values_list,
            fill_color=cell_colors,
            align='left',
            height=30,
            font=dict(color=TEXT, size=11),
            line=dict(color=BORDER, width=1),
        ),
    ), row=current_row, col=1)

    # -----------------------------------------------------------------------
    # Subplot title styling (dark theme)
    # -----------------------------------------------------------------------
    for ann in fig.layout.annotations:
        ann.update(
            font=dict(color=ACCENT, size=13, family='monospace'),
            bgcolor=CARD,
            bordercolor=BORDER,
            borderwidth=1,
            borderpad=6,
        )

    # -----------------------------------------------------------------------
    # Axis styling
    # -----------------------------------------------------------------------
    for r in range(1, current_row):
        fig.update_xaxes(
            tickangle=-45,
            tickfont=dict(color=TEXT, size=10),
            gridcolor=BORDER,
            linecolor=BORDER,
            row=r, col=1,
        )
        fig.update_yaxes(
            tickfont=dict(color=TEXT, size=10),
            gridcolor=BORDER,
            linecolor=BORDER,
            row=r, col=1,
        )

    # -----------------------------------------------------------------------
    # Overall layout
    # -----------------------------------------------------------------------
    total_height = sum(row_heights) + 140
    fig.update_layout(
        title=dict(
            text=(
                '<b>Market Regime Attribution</b><br>'
                '<span style="font-size:12px;color:#94a3b8">'
                'Each cell = Annualised Return % of that strategy while the market was in that regime condition. '
                'Hover for Max Drawdown, Calmar, Sharpe and coverage (how many months of data). '
                'Green = positive return · Red = negative · — = fewer than 10 bars recorded. '
                'Column labels show: Friendly Name / technical code / approx months of data in that regime. '
                'Scroll down for Activation Guide and methodology.'
                '</span>'
            ),
            x=0.5, xanchor='center',
            font=dict(color=TEXT, size=17),
            pad=dict(t=20, b=20),
        ),
        height=total_height,
        showlegend=False,
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        margin=dict(l=60, r=80, t=120, b=120),
        font=dict(color=TEXT),
    )

    # -----------------------------------------------------------------------
    # Methodology + Glossary HTML (appended below the Plotly chart)
    # -----------------------------------------------------------------------
    plotly_div = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={'responsive': True, 'displayModeBar': True},
    )

    # -----------------------------------------------------------------------
    # Build unified SCORES routing section (inserted at end of report).
    # Shows the pre-computed score table across composite, RSI, and VIX analyses
    # plus the select_strategy() function that sums scores across current conditions.
    # This mirrors exactly what regime_router.run_regime_switching_backtest() does.
    # -----------------------------------------------------------------------
    def _build_ifelse_html() -> str:
        if not _COMBO_SCORES:
            return ''

        # Syntax-highlight helpers
        def _kw(t):  return f'<span style="color:#94a3b8">{t}</span>'    # keyword/comment (grey)
        def _kr(t):  return f'<span style="color:#f472b6">{t}</span>'    # control keyword (pink)
        def _tx(t):  return f'<span style="color:#f1f5f9">{t}</span>'    # plain text (white)
        def _st(t):  return f'<span style="color:#34d399">{t}</span>'    # string (green)
        def _nm(t):  return f'<span style="color:#fbbf24">{t}</span>'    # number (yellow)
        def _fn(t):  return f'<span style="color:#38bdf8">{t}</span>'    # function/var (blue)

        BUCKET_LABELS: Dict[str, str] = {
            'low_vol_above':  'composite · low RV, above SMA200',
            'low_vol_below':  'composite · low RV, below SMA200',
            'med_vol_above':  'composite · med RV, above SMA200',
            'med_vol_below':  'composite · med RV, below SMA200',
            'high_vol_above': 'composite · high RV, above SMA200',
            'high_vol_below': 'composite · high RV, below SMA200',
            'rsi_bull':       'RSI · 3-day consensus > 50',
            'rsi_bear':       'RSI · 3-day consensus ≤ 50',
            'vix_very_low':   'VIX · < 20th percentile',
            'vix_low':        'VIX · 20–50th percentile',
            'vix_elevated':   'VIX · 50–80th percentile',
            'vix_high':       'VIX · > 80th percentile',
        }

        # Legend
        legend_items_html = ''
        for _c in _selected_order:
            _col  = ifelse_combo_color.get(_c, TEXT)
            _num  = ifelse_combo_number.get(_c, '')
            _tot  = sum(_COMBO_SCORES.get(_c, {}).values())
            _note = f' <span style="color:{MUTED};font-size:10px">[Σ {_tot:.1f}]</span>' if _c in _COMBO_SCORES else f' <span style="color:{MUTED}">(DEFAULT)</span>'
            legend_items_html += (
                f'<span style="display:inline-flex;align-items:center;margin:4px 12px 4px 0;">'
                f'<span style="background:{_col};color:#fff;border-radius:50%;width:22px;height:22px;'
                f'display:inline-flex;align-items:center;justify-content:center;font-size:13px;'
                f'margin-right:8px;flex-shrink:0;">{_num}</span>'
                f'<code style="color:{_col};font-size:12px">{_short_combo(_c)}{_note}</code>'
                f'</span>'
            )

        # Code block lines
        code_lines: List[str] = []
        code_lines.append(_kw('# Pre-computed OOS scores per strategy across composite, RSI and VIX analyses'))
        code_lines.append(_kw('# Filters: max_dd &gt; −8%  ·  |max_dd| ≤ ann_return / 2  ·  n ≥ 63 bars'))
        code_lines.append(_kw('# Score = ann_return × (1 + 0.10 × years)  —  0 means failed filters'))
        code_lines.append('')
        code_lines.append(_fn('SCORES') + _tx(' = {'))

        for _c in _selected_order:
            if _c not in _COMBO_SCORES:
                continue
            _col      = ifelse_combo_color.get(_c, TEXT)
            _num      = ifelse_combo_number.get(_c, '')
            _tot      = sum(_COMBO_SCORES[_c].values())
            _bkts     = sorted(_COMBO_SCORES[_c].items(), key=lambda x: x[1], reverse=True)
            code_lines.append(_kw(f'    # {_num} {_short_combo(_c)}   Σ = {_tot:.1f}'))
            code_lines.append(
                f'    <span style="color:{_col};font-weight:bold">'
                f'<span style="background:{_col}22;border-radius:3px;padding:0 5px">'
                f'&quot;{_c}&quot;</span></span>' + _tx(': {')
            )
            for _b, _s in _bkts:
                _lbl = BUCKET_LABELS.get(_b, _b)
                code_lines.append(
                    '        ' + _st(f'&quot;{_b}&quot;') + _tx(': ')
                    + _nm(f'{_s:.2f}') + _tx(',')
                    + _kw(f'  # {_lbl}')
                )
            code_lines.append(_tx('    },'))

        code_lines.append(_tx('}'))
        code_lines.append('')

        # DEFAULT
        _def_c   = _default_ifelse_combo or (combos[0] if combos else '')
        _def_col = ifelse_combo_color.get(_def_c, MUTED)
        _def_num = ifelse_combo_number.get(_def_c, '')
        code_lines.append(
            _fn('DEFAULT') + _tx(' = ')
            + f'<span style="color:{_def_col};font-weight:bold">'
              f'<span style="background:{_def_col}22;border-radius:3px;padding:0 5px">'
              f'&quot;{_def_c}&quot;</span></span>'
            + _kw('  # best overall OOS Calmar — used when no strategy passes filters')
        )
        code_lines.append('')

        # select_strategy function
        code_lines.append(
            _kw('def ') + _fn('select_strategy') + _tx('(composite, rsi, vix=None) -> str:')
        )
        code_lines.append(
            _tx('    ') + _kw('"""Sum scores across all current regime conditions → pick highest-scoring strategy."""')
        )
        code_lines.append(
            _tx('    conditions = [c ') + _kw('for') + _tx(' c ') + _kw('in')
            + _tx(' [composite, rsi, vix] ') + _kw('if') + _tx(' c]')
        )
        code_lines.append(_tx('    combined = {'))
        code_lines.append(
            _tx('        combo: sum(') + _fn('SCORES') + _tx('[combo].get(c, 0) ')
            + _kw('for') + _tx(' c ') + _kw('in') + _tx(' conditions)')
        )
        code_lines.append(_tx('        ') + _kw('for') + _tx(' combo ') + _kw('in ') + _fn('SCORES'))
        code_lines.append(_tx('    }'))
        code_lines.append(_tx('    best = max(combined, key=combined.get)'))
        code_lines.append(
            _kr('    return ') + _tx('best ')
            + _kw('if') + _tx(' combined[best] > 0 ')
            + _kw('else ') + _fn('DEFAULT')
        )

        code_html = '<br>'.join(code_lines)

        return f"""
  <!-- ── AUTO-GENERATED STRATEGY ROUTING (unified multi-dimensional) ──── -->
  <div style="background:#0c1a2e;border:2px solid {ACCENT};border-radius:10px;
              margin:28px 0 20px;padding:24px 32px;">
    <h2 style="color:{ACCENT};margin:0 0 6px;font-size:15px;letter-spacing:0.06em;text-transform:uppercase">
      Auto-Generated Strategy Routing
    </h2>
    <p style="color:{MUTED};font-size:12px;margin:0 0 18px;line-height:1.6">
      Scores are built from <strong>all three regime analyses</strong> (composite RV×SMA200, RSI, VIX).
      Every strategy earns a score in each regime bucket it passes the hard filters in:<br>
      <code style="background:#1e293b;padding:2px 8px;border-radius:4px;display:inline-block;margin:4px 0">
        max_dd &gt; −8% &nbsp;·&nbsp; |max_dd| ≤ ann_return / 2 &nbsp;·&nbsp; n_bars ≥ 63 (~3 months)
      </code><br>
      At runtime, <strong>current composite + RSI + VIX conditions are known simultaneously</strong>.
      The strategy whose scores sum highest across all active conditions is selected —
      so a strategy that works well in multiple regime dimensions beats one with a single strong match.<br>
      Circled numbers (①②…) map to the colour-coded labels in the heatmaps and Activation Guide above.
    </p>

    <!-- Legend -->
    <div style="margin-bottom:18px;padding:12px 16px;background:#1e293b;
                border-radius:6px;border:1px solid #334155;">
      <span style="color:{MUTED};font-size:11px;text-transform:uppercase;
                   letter-spacing:0.08em;display:block;margin-bottom:8px">
        Strategy Legend — ranked by total score (Σ across all regime buckets)
      </span>
      <div style="display:flex;flex-wrap:wrap;">{legend_items_html}</div>
    </div>

    <!-- Code block -->
    <pre style="background:#0f172a;border:1px solid #334155;border-radius:8px;
                padding:20px 24px;font-family:'Consolas','Fira Code',monospace;
                font-size:12.5px;line-height:1.8;overflow-x:auto;margin:0;">
<code>{code_html}</code></pre>

    <p style="color:{MUTED};font-size:11px;margin:14px 0 0;border-top:1px solid #334155;padding-top:10px">
      ⚠ When combined[best] == 0 (no strategy passed filters for any active condition) the DEFAULT
      fallback applies.  Run <code>regime_router.py --run_dir &lt;path&gt;</code> to execute the full
      regime-switching backtest using this same logic.
    </p>
  </div>"""

    ifelse_section_html = _build_ifelse_html()

    glossary_html = f"""
<div style="font-family:'Segoe UI',system-ui,sans-serif;color:{TEXT};max-width:1300px;margin:0 auto;">

  <!-- ── HOW TO USE THIS REPORT ────────────────────────────────────────── -->
  <div style="background:#0c1f2e;border:1px solid {ACCENT};border-radius:10px;
              margin:28px 0 20px;padding:24px 32px;">
    <h2 style="color:{ACCENT};margin:0 0 14px;font-size:15px;letter-spacing:0.06em;text-transform:uppercase">
      How to use this report
    </h2>
    <ol style="color:{TEXT};font-size:13px;line-height:1.9;margin:0;padding-left:20px">
      <li><b>Section ①</b> is the primary decision tool. Find a strategy row and look across the columns to
          see which market regime gives the best Annualised Return %. Greener = stronger return in that
          condition. Each column label shows the friendly name, technical code, and how many months of
          OOS data were captured in that regime.</li>
      <li><b>Section ②</b> (In-Sample) is shown for comparison only — good IS performance is expected
          because these parameters were optimised on that window. Always prefer OOS.</li>
      <li><b>Section ③</b> (RSI) tells you whether a strategy works better when the market has positive
          momentum (RSI &gt; 50) or negative momentum (RSI &lt; 50). Useful as a secondary filter.</li>
      <li><b>Section ④</b> (VIX, if available) shows how strategies behave under low-fear vs high-fear
          implied volatility. Some strategies thrive in crisis, others collapse.</li>
      <li><b>Activation Guide table</b>: shows Ann. Return % and Max Drawdown for the top-2 best regime
          conditions per strategy. The "Regime Matters?" column tells you whether the differences are
          statistically real. ✅ Yes = use the Best Regime column as a live deployment filter.
          ❌ No = strategy performs similarly across regimes — can run any time.</li>
    </ol>
    <p style="color:{MUTED};font-size:12px;margin:14px 0 0">
      <b>Hover over any cell</b> for the full breakdown: Ann. Return %, Max Drawdown %, Calmar, Sharpe,
      Win Rate, bar count and regime coverage in months. The column headers show approx coverage in months
      so you can judge statistical reliability — a bucket with only 1–2 months of data deserves less weight.
    </p>
  </div>

  <!-- ── WHY THESE INDICATORS ─────────────────────────────────────────── -->
  <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
              margin:0 0 20px;padding:24px 32px;">
    <h2 style="color:{ACCENT};margin:0 0 14px;font-size:15px;letter-spacing:0.06em;text-transform:uppercase">
      Why these three regime indicators?
    </h2>
    <p style="color:{MUTED};font-size:12px;margin:0 0 16px">
      The three indicators below were chosen because they are the most robust, cheapest to compute,
      and best-documented in academic literature for predicting whether a rule-based strategy will
      have an edge in the near-term. They answer three independent questions about market state.
    </p>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px 28px">

      <div style="border-left:3px solid {ACCENT};padding-left:14px">
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 6px">
          ① Realized Volatility × SMA200
        </h3>
        <p style="color:{MUTED};font-size:12px;margin:0 0 8px">
          <b>What it answers:</b> Is the market trending (up or down), and how turbulent is it?
        </p>
        <p style="color:{MUTED};font-size:12px;margin:0 0 8px">
          <b>Price vs 200-day SMA</b> is the single most replicated tactical signal in finance
          (Faber 2007, SSRN 962461). Above = bull regime. Below = bear/defensive regime.
          Robust OOS in equity, commodity and bond markets.
        </p>
        <p style="color:{MUTED};font-size:12px;margin:0">
          <b>21-day Realized Volatility</b> captures volatility clustering — quiet markets tend to
          stay quiet, turbulent markets stay turbulent (GARCH literature). Different strategies
          behave very differently in low-vol vs high-vol environments. Thresholds are set from the
          In-Sample period only so the OOS test is unbiased.
        </p>
      </div>

      <div style="border-left:3px solid {YELLOW};padding-left:14px">
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 6px">
          ② RSI(14) Momentum Zone
        </h3>
        <p style="color:{MUTED};font-size:12px;margin:0 0 8px">
          <b>What it answers:</b> Does the market have positive or negative short-term momentum
          right now, independent of the long-term trend?
        </p>
        <p style="color:{MUTED};font-size:12px;margin:0 0 8px">
          RSI oscillates between 0–100. A persistent RSI above 50 (bull zone) means recent gains
          are outpacing recent losses — momentum is with buyers. This is a short-term signal
          (days to weeks) that complements the longer-term SMA200.
        </p>
        <p style="color:{MUTED};font-size:12px;margin:0">
          A 3-bar consensus filter is applied (RSI must be in the zone in ≥ 2 of the last 3 bars)
          to avoid single-bar noise flips.
        </p>
      </div>

      <div style="border-left:3px solid #a78bfa;padding-left:14px">
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 6px">
          ③ VIX Percentile (if available)
        </h3>
        <p style="color:{MUTED};font-size:12px;margin:0 0 8px">
          <b>What it answers:</b> How fearful is the market relative to its own recent history?
          Is implied volatility cheap or expensive right now?
        </p>
        <p style="color:{MUTED};font-size:12px;margin:0 0 8px">
          The CBOE VIX measures the market's expected 30-day volatility priced into S&amp;P options.
          A high VIX percentile means fear is elevated vs. recent history — crisis/stress regimes.
          A low VIX percentile means complacency. Mean-reversion strategies often outperform at high
          VIX; trend-following tends to collapse.
        </p>
        <p style="color:{MUTED};font-size:12px;margin:0">
          Uses rolling 252-day percentile rank (IV Percentile) rather than raw VIX level, which makes
          it robust to secular changes in volatility regimes.
        </p>
      </div>

    </div>
    <p style="color:{MUTED};font-size:11px;margin:18px 0 0;border-top:1px solid {BORDER};padding-top:12px">
      References: Faber (2007) SSRN 962461 · Ang &amp; Timmermann (2011) NBER w17182 ·
      Shu, Yu &amp; Mulvey (2024) arXiv 2402.05272 (Statistical Jump Model) ·
      Hamilton (1989) Econometrica (Markov switching)
    </p>
  </div>

  <!-- ── GLOSSARY ──────────────────────────────────────────────────────── -->
  <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
              margin:0 0 40px;padding:24px 32px;">
    <h2 style="color:{ACCENT};margin:0 0 14px;font-size:15px;letter-spacing:0.06em;text-transform:uppercase">
      Metric glossary
    </h2>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px 40px">

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">Annualised Return % (primary metric)</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          The compound annual return the strategy earned while the market was in that regime condition.
          Computed as <code>(1 + mean_bar_return)^252 − 1</code> across all bars tagged with that regime.
          This is the primary heatmap colour — straightforward and honest.
          Positive = strategy made money in that regime. Negative = it lost money.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">Max Drawdown % (context metric)</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          The worst peak-to-trough loss recorded within that regime period. Always negative.
          Shown in the Activation Guide alongside Ann. Return so you can see the full picture:
          a 40% return with −35% drawdown is very different from a 15% return with −4% drawdown.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">Calmar Ratio (hover only)</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          Annualised return ÷ maximum drawdown in the period. Capped at 20.
          "How much did I earn per unit of worst-case loss." Shown in hover detail
          but not used as the primary colour — a Calmar of 20 could mean a great 40%/−2% strategy
          <em>or</em> a mediocre 5%/−0.25% strategy that barely moved. Ann Ret and Max DD separately
          tell a clearer story.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">Sharpe Ratio</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          Mean daily return ÷ daily return std dev × √252 (annualised).
          Measures reward per unit of day-to-day volatility.
          Above 1.0 = solid risk-adjusted return.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">In-Sample (IS) vs Out-of-Sample (OOS)</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          IS = historical window used to <em>find</em> the parameters — good performance here is expected
          and does not prove anything. OOS = data the algorithm had never "seen" — this is the
          real robustness test. Always judge strategies by OOS.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">Kruskal-Wallis test &amp; p-value</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          A statistical test that asks: <em>"Are the daily returns in regime A, B, C etc. drawn from
          different distributions, or could the differences just be random?"</em>
          <b>p &lt; 0.05</b> = the regime label genuinely predicts performance differences (95% confidence).
          <b>p ≥ 0.05</b> = regime differences are probably noise — deploy the strategy regardless of regime.
          This determines whether the "Activation Guide" is actionable for a given strategy.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">— (dash in a cell)</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          Means there were fewer than 10 bars of data in that regime bucket for this strategy —
          not enough to compute a meaningful statistic. Hover over the cell for the exact sample count.
        </p>
      </div>

      <div>
        <h3 style="color:{TEXT};font-size:13px;margin:0 0 4px">Activation Guide — "Regime Matters?"</h3>
        <p style="color:{MUTED};font-size:12px;margin:0">
          ✅ Yes = Kruskal-Wallis p &lt; 0.05. The regime filter adds real value — use the "Best Regime"
          column to decide when to trade this strategy.<br>
          ❌ No = p ≥ 0.05. The strategy's performance is similar across all regimes — no filter needed.
        </p>
      </div>

    </div>
  </div>

  {ifelse_section_html}

</div>
"""

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Market Regime Attribution</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: {BG};
      color: {TEXT};
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }}
    .plotly-container {{
      width: 100%;
    }}
  </style>
</head>
<body>
  <div class="plotly-container">
    {plotly_div}
  </div>
  {glossary_html}
</body>
</html>"""

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as fh:
            fh.write(full_html)


def create_joint_performance_summary(joint_data: List[Dict], save_path: str):
    """
    Grouped bar chart comparing IS and OOS performance across all symbols
    for a jointly-optimised strategy permutation.

    Args:
        joint_data: list of dicts, one per symbol, each containing:
            'symbol', 'combination', 'is_metrics', 'oos_metrics'
        save_path: path to write the HTML file.
    """
    if not joint_data:
        return

    metrics_to_plot = ['Calmar Ratio', 'Sharpe Ratio', 'Total Return Pct', 'Max Drawdown Pct']
    symbols = [d['symbol'] for d in joint_data]

    palette = px.colors.qualitative.Plotly
    symbol_colors = {sym: palette[i % len(palette)] for i, sym in enumerate(symbols)}

    fig = make_subplots(
        rows=len(metrics_to_plot), cols=1,
        shared_xaxes=True,
        subplot_titles=[m.replace(' Pct', ' (%)') for m in metrics_to_plot],
        vertical_spacing=0.08,
    )

    bar_width = 0.35
    x_positions = list(range(len(symbols)))

    for row_idx, metric in enumerate(metrics_to_plot, 1):
        is_vals  = [min(d['is_metrics'].get(metric, 0),  20.0) if metric == 'Calmar Ratio' else d['is_metrics'].get(metric,  0) for d in joint_data]
        oos_vals = [min(d['oos_metrics'].get(metric, 0), 20.0) if metric == 'Calmar Ratio' else d['oos_metrics'].get(metric, 0) for d in joint_data]

        show_legend = (row_idx == 1)

        fig.add_trace(go.Bar(
            name='In-Sample',
            x=symbols,
            y=is_vals,
            marker_color=[symbol_colors[s] for s in symbols],
            text=[f'{v:.2f}' for v in is_vals],
            textposition='outside',
            legendgroup='is',
            showlegend=show_legend,
            offsetgroup='is',
        ), row=row_idx, col=1)

        fig.add_trace(go.Bar(
            name='Out-of-Sample',
            x=symbols,
            y=oos_vals,
            marker_color=[symbol_colors[s] for s in symbols],
            marker_pattern_shape='/',
            text=[f'{v:.2f}' for v in oos_vals],
            textposition='outside',
            legendgroup='oos',
            showlegend=show_legend,
            offsetgroup='oos',
        ), row=row_idx, col=1)

        # Highlight the worst-performing symbol in OOS (the optimisation bottleneck)
        if oos_vals:
            worst_idx = (
                oos_vals.index(min(oos_vals)) if metric != 'Max Drawdown Pct'
                else oos_vals.index(max(oos_vals))
            )
            fig.add_annotation(
                x=symbols[worst_idx],
                y=oos_vals[worst_idx],
                text='⚠ weakest',
                showarrow=True,
                arrowhead=2,
                font=dict(color='red', size=10),
                row=row_idx, col=1,
            )

    combo_name = joint_data[0]['combination'].replace(joint_data[0]['symbol'] + '_', '') if joint_data else 'Joint'
    fig.update_layout(
        title=dict(
            text=f'<b>Joint Optimisation — Per-Symbol Performance</b><br><sup>{combo_name}</sup>',
            x=0.5,
        ),
        height=280 * len(metrics_to_plot),
        barmode='group',
        paper_bgcolor='#fafafa',
    )

    if save_path:
        fig.write_html(save_path)


# ---------------------------------------------------------------------------
# Regime-switching backtest report
# ---------------------------------------------------------------------------

_REGIME_COLORS = {
    'low_vol_above':  'rgba(144,238,144,0.25)',
    'low_vol_below':  'rgba(255,255,153,0.25)',
    'med_vol_above':  'rgba(100,200,100,0.30)',
    'med_vol_below':  'rgba(255,178,102,0.30)',
    'high_vol_above': 'rgba(255,140,0,0.35)',
    'high_vol_below': 'rgba(220,53,69,0.35)',
    'unknown':        'rgba(200,200,200,0.15)',
}


def _regime_segments(series: pd.Series) -> list:
    """Return list of (start, end, regime) from a smoothed regime series."""
    segs = []
    if series.empty:
        return segs
    current = series.iloc[0]
    start = series.index[0]
    for i in range(1, len(series)):
        if series.iloc[i] != current:
            segs.append((start, series.index[i - 1], current))
            current = series.iloc[i]
            start = series.index[i]
    segs.append((start, series.index[-1], current))
    return segs


def create_regime_switching_report(
    results: Dict[str, Dict],
    routing_table: Dict,
    save_path: str,
) -> None:
    """
    Self-contained HTML report for the regime-switching backtest.

    Sections:
      1. Routing table coloured by regime
      2. Per-symbol equity curves with regime background bands
      3. Metrics summary table (Total Return, Ann Return, Sharpe, Max DD, Calmar)

    Args:
        results:       {symbol: {'equity_df', 'metrics', 'regime_series', 'df_oos'}}
        routing_table: output of build_routing_table()
        save_path:     destination .html file path
    """
    routing = routing_table.get('routing', {})

    # --- 1. Routing table HTML -------------------------------------------
    _REGIME_ORDER = [
        'low_vol_above', 'low_vol_below',
        'med_vol_above', 'med_vol_below',
        'high_vol_above', 'high_vol_below',
        'unknown',
    ]
    all_regimes = set(_REGIME_ORDER) | set(routing.keys())
    ordered_regimes = [r for r in _REGIME_ORDER if r in all_regimes] + \
                      [r for r in sorted(all_regimes) if r not in _REGIME_ORDER]

    routing_rows = []
    for reg in ordered_regimes:
        info = routing.get(reg)
        if info:
            routing_rows.append({
                'regime':  reg,
                'combo':   info.get('combo', '—'),
                'ann_ret': info.get('ann_return_pct', '—'),
                'sharpe':  info.get('sharpe', '—'),
                'score':   info.get('score', '—'),
                'n_bars':  info.get('n_bars', '—'),
            })
        else:
            routing_rows.append({
                'regime':  reg, 'combo': '(no assignment)',
                'ann_ret': '—', 'sharpe': '—', 'score': 0.0, 'n_bars': 0,
            })

    dflt = routing_table.get('default', {})
    routing_rows.append({
        'regime':  'DEFAULT (fallback)',
        'combo':   dflt.get('combo', '—'),
        'ann_ret': '—', 'sharpe': '—', 'score': '—', 'n_bars': '—',
    })

    def _fmt(v, decimals=2):
        return f'{v:.{decimals}f}' if isinstance(v, float) else str(v)

    def _row_bg(regime):
        c = _REGIME_COLORS.get(regime, 'rgba(240,240,240,0.5)')
        return c.replace('0.25)', '0.45)').replace('0.30)', '0.45)').replace('0.35)', '0.45)')

    rt_rows_html = ''
    for row in routing_rows:
        bg = _row_bg(row['regime'])
        rt_rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:7px 12px;font-weight:bold;">{row["regime"]}</td>'
            f'<td style="padding:7px 12px;">{row["combo"]}</td>'
            f'<td style="padding:7px 12px;text-align:right;">{_fmt(row["ann_ret"], 1)}</td>'
            f'<td style="padding:7px 12px;text-align:right;">{_fmt(row["sharpe"], 3)}</td>'
            f'<td style="padding:7px 12px;text-align:right;">{_fmt(row["score"], 3)}</td>'
            f'<td style="padding:7px 12px;text-align:right;">{row["n_bars"]}</td>'
            f'</tr>\n'
        )

    routing_table_html = f"""
<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;margin-bottom:30px;">
  <thead>
    <tr style="background:#333;color:#fff;">
      <th style="padding:8px 12px;text-align:left;">Regime</th>
      <th style="padding:8px 12px;text-align:left;">Assigned Strategy</th>
      <th style="padding:8px 12px;text-align:right;">OOS Ann Ret (%)</th>
      <th style="padding:8px 12px;text-align:right;">OOS Sharpe</th>
      <th style="padding:8px 12px;text-align:right;">Score</th>
      <th style="padding:8px 12px;text-align:right;">N Bars</th>
    </tr>
  </thead>
  <tbody>
    {rt_rows_html}
  </tbody>
</table>"""

    # --- 2. Equity curve plots per symbol --------------------------------
    equity_divs = []
    for symbol, data in results.items():
        equity_df     = data.get('equity_df', pd.DataFrame())
        regime_series = data.get('regime_series', pd.Series(dtype=object))

        if equity_df.empty:
            continue

        fig = go.Figure()

        # Regime background bands
        for seg_start, seg_end, regime in _regime_segments(regime_series):
            color = _REGIME_COLORS.get(str(regime), _REGIME_COLORS['unknown'])
            fig.add_vrect(
                x0=seg_start, x1=seg_end,
                fillcolor=color, line_width=0, layer='below',
                annotation_text=str(regime),
                annotation_position='top left',
                annotation=dict(font_size=8, font_color='#555'),
            )

        # Strategy equity curve
        fig.add_trace(go.Scatter(
            x=equity_df.index,
            y=equity_df['total_value'],
            mode='lines',
            name='Regime-Switching Strategy',
            line=dict(color='#1f77b4', width=2),
        ))

        # Benchmark
        if 'benchmark_value' in equity_df.columns:
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['benchmark_value'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='#999', width=1.5, dash='dot'),
            ))

        m = data.get('metrics', {})
        subtitle = (
            f"Ann={m.get('Ann Return Pct', 0):.1f}%  "
            f"Sharpe={m.get('Sharpe Ratio', 0):.3f}  "
            f"MaxDD={m.get('Max Drawdown Pct', 0):.1f}%  "
            f"Calmar={m.get('Calmar Ratio', 0):.3f}"
        )
        fig.update_layout(
            title=dict(
                text=f'<b>{symbol} \u2014 Regime-Switching OOS Equity Curve</b>'
                     f'<br><sup>{subtitle}</sup>',
                x=0.5,
            ),
            height=420,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            paper_bgcolor='#fafafa',
            plot_bgcolor='#fff',
            legend=dict(orientation='h', y=-0.18),
        )
        equity_divs.append(fig.to_html(full_html=False, include_plotlyjs=False))

    equity_plots_html = '\n'.join(equity_divs) if equity_divs else '<p>No equity curves generated.</p>'

    # --- 3. Metrics summary table ----------------------------------------
    metric_keys   = ['Total Return Pct', 'Ann Return Pct', 'Sharpe Ratio', 'Max Drawdown Pct', 'Calmar Ratio']
    metric_labels = ['Total Return (%)', 'Ann Return (%)', 'Sharpe',        'Max DD (%)',        'Calmar']

    header_cells = ''.join(
        f'<th style="padding:8px 12px;text-align:right;">{lbl}</th>' for lbl in metric_labels
    )
    met_rows_html = ''
    for symbol, data in results.items():
        m = data.get('metrics', {})
        cells = ''.join(
            f'<td style="padding:7px 12px;text-align:right;">{m[k]:.2f}</td>'
            if isinstance(m.get(k), float) else
            '<td style="padding:7px 12px;text-align:right;">\u2014</td>'
            for k in metric_keys
        )
        met_rows_html += (
            f'<tr>'
            f'<td style="padding:7px 12px;font-weight:bold;">{symbol}</td>'
            f'{cells}</tr>\n'
        )

    metrics_table_html = f"""
<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:13px;margin-bottom:30px;">
  <thead>
    <tr style="background:#333;color:#fff;">
      <th style="padding:8px 12px;text-align:left;">Symbol</th>
      {header_cells}
    </tr>
  </thead>
  <tbody>
    {met_rows_html}
  </tbody>
</table>"""

    # --- 4. Legend key for regime colours --------------------------------
    legend_items = ''.join(
        f'<span style="display:inline-block;padding:4px 10px;border-radius:4px;'
        f'background:{c};font-family:monospace;font-size:12px;margin:3px;">{r}</span>'
        for r, c in _REGIME_COLORS.items()
    )

    # --- Assemble final HTML ---------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Regime-Switching Backtest Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
  <style>
    body  {{font-family:Arial,sans-serif;background:#f4f4f4;padding:20px;color:#222;}}
    h1    {{text-align:center;color:#333;margin-bottom:8px;}}
    h2    {{color:#444;border-bottom:2px solid #ccc;padding-bottom:5px;margin-top:40px;}}
    .wrap {{max-width:1400px;margin:0 auto;background:#fff;padding:30px;border-radius:8px;
            box-shadow:0 2px 12px rgba(0,0,0,.1);}}
    p.sub {{text-align:center;color:#666;margin-top:0;}}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Regime-Switching Backtest Report</h1>
  <p class="sub">
    Best strategy selected per market regime via OOS conditional attribution.<br>
    Score&nbsp;=&nbsp;ann_return&nbsp;&times;&nbsp;sharpe&nbsp;&times;&nbsp;sample_confidence
    &nbsp;(avoids Calmar inflation from sparse trading in a regime bucket).
  </p>

  <h2>Regime Colour Key</h2>
  <div style="margin-bottom:20px;">{legend_items}</div>

  <h2>1 \u2014 Routing Table</h2>
  {routing_table_html}

  <h2>2 \u2014 Per-Symbol Equity Curves</h2>
  {equity_plots_html}

  <h2>3 \u2014 Metrics Summary</h2>
  {metrics_table_html}
</div>
</body>
</html>
"""
    with open(save_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
