import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

compatible_timeframes = ['1 hour', '4 hours', '1 day']

# Adjusted parameter space to allow lower consensus and wider indicator ranges
param_space = {
    'macd_fast': {'type': 'int', 'low': 8, 'high': 16},
    'macd_slow': {'type': 'int', 'low': 20, 'high': 32},
    'macd_signal': {'type': 'int', 'low': 7, 'high': 11},
    'kdj_length': {'type': 'int', 'low': 9, 'high': 30},
    'kdj_smooth': {'type': 'int', 'low': 2, 'high': 5},
    'rsi_length': {'type': 'int', 'low': 6, 'high': 24},
    'lwr_length': {'type': 'int', 'low': 10, 'high': 24},
    'mtm_length': {'type': 'int', 'low': 8, 'high': 16},
    'consensus_required': {'type': 'int', 'low': 2, 'high': 5}, # Lowered minimum to 2 to ensure signals fire
    'rsi_neutral': {'type': 'float', 'low': 5.0, 'high': 15.0}, # Relaxed neutrality
    'lwr_neutral': {'type': 'float', 'low': 5.0, 'high': 15.0}  # Relaxed neutrality
}

def _rsi(series: pd.Series, period: int) -> pd.Series:
    try:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.error(f"Error in _rsi calculation: {e}")
        return pd.Series(50, index=series.index)

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    try:
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100
        return wr
    except Exception as e:
        logger.error(f"Error in _williams_r calculation: {e}")
        return pd.Series(-50, index=close.index)

def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int, smooth: int) -> tuple:
    try:
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        rsv = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
        rsv = rsv.fillna(50)
        k = rsv.ewm(alpha=1/smooth, adjust=False).mean()
        d = k.ewm(alpha=1/smooth, adjust=False).mean()
        j = 3 * k - 2 * d
        return k, d, j
    except Exception as e:
        logger.error(f"Error in _kdj calculation: {e}")
        neutral = pd.Series(50, index=close.index)
        return neutral, neutral, neutral

def strategy_function(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    kdj_length: int = 14,
    kdj_smooth: int = 3,
    rsi_length: int = 14,
    rsi_neutral: float = 15.0,
    lwr_length: int = 14,
    lwr_neutral: float = 15.0,
    mtm_length: int = 12,
    consensus_required: int = 3
) -> pd.DataFrame:
    try:
        strategy_df = df.copy()
        
        # Ensure sufficient data exists
        min_required = max(macd_slow + macd_signal, kdj_length, rsi_length, lwr_length, 24, mtm_length) + 10
        if len(strategy_df) < min_required:
            logger.warning(f"Insufficient data for Six Meridian: need {min_required} bars, have {len(strategy_df)}")
            strategy_df['position'] = 0
            return strategy_df
        
        # ===== 1. MACD =====
        ema_fast = strategy_df['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = strategy_df['close'].ewm(span=macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
        
        macd_signal_val = np.where(macd_line > signal_line, 1, np.where(macd_line < signal_line, -1, 0))
        
        # ===== 2. KDJ =====
        k, d, j = _kdj(strategy_df['high'], strategy_df['low'], strategy_df['close'], kdj_length, kdj_smooth)
        kdj_signal_val = np.where(k > d, 1, np.where(k < d, -1, 0))
        
        # ===== 3. RSI =====
        rsi = _rsi(strategy_df['close'], rsi_length)
        rsi_signal_val = np.where(rsi > (50 + rsi_neutral), 1, np.where(rsi < (50 - rsi_neutral), -1, 0))
        
        # ===== 4. Williams %R =====
        williams_r = _williams_r(strategy_df['high'], strategy_df['low'], strategy_df['close'], lwr_length)
        lwr_signal_val = np.where(williams_r > (-50 + lwr_neutral), 1, np.where(williams_r < (-50 - lwr_neutral), -1, 0))
        
        # ===== 5. BBI =====
        ma3 = strategy_df['close'].rolling(window=3).mean()
        ma6 = strategy_df['close'].rolling(window=6).mean()
        ma12 = strategy_df['close'].rolling(window=12).mean()
        ma24 = strategy_df['close'].rolling(window=24).mean()
        bbi = (ma3 + ma6 + ma12 + ma24) / 4
        bbi_signal_val = np.where(strategy_df['close'] > bbi, 1, np.where(strategy_df['close'] < bbi, -1, 0))
        
        # ===== 6. MTM =====
        mtm = strategy_df['close'] - strategy_df['close'].shift(mtm_length)
        mtm_signal_val = np.where(mtm > 0, 1, np.where(mtm < 0, -1, 0))
        
        strategy_df['macd_signal'] = macd_signal_val
        strategy_df['kdj_signal'] = kdj_signal_val
        strategy_df['rsi_signal'] = rsi_signal_val
        strategy_df['lwr_signal'] = lwr_signal_val
        strategy_df['bbi_signal'] = bbi_signal_val
        strategy_df['mtm_signal'] = mtm_signal_val
        
        # ===== Calculate Consensus =====
        bullish_count = (
            (macd_signal_val == 1).astype(int) +
            (kdj_signal_val == 1).astype(int) +
            (rsi_signal_val == 1).astype(int) +
            (lwr_signal_val == 1).astype(int) +
            (bbi_signal_val == 1).astype(int) +
            (mtm_signal_val == 1).astype(int)
        )
        
        bearish_count = (
            (macd_signal_val == -1).astype(int) +
            (kdj_signal_val == -1).astype(int) +
            (rsi_signal_val == -1).astype(int) +
            (lwr_signal_val == -1).astype(int) +
            (bbi_signal_val == -1).astype(int) +
            (mtm_signal_val == -1).astype(int)
        )
        
        strategy_df['bullish_count'] = bullish_count
        strategy_df['bearish_count'] = bearish_count
        
        long_signal = bullish_count >= consensus_required
        short_signal = bearish_count >= consensus_required
        
        raw_signal = np.where(long_signal, 1, np.nan)
        raw_signal = np.where(short_signal, -1, raw_signal)
        
        strategy_df['position'] = pd.Series(raw_signal, index=strategy_df.index).ffill().fillna(0).astype(int)
        
        warmup_period = min_required
        strategy_df.iloc[:warmup_period, strategy_df.columns.get_loc('position')] = 0
        
        # Diagnostic logging for tuning
        if logger.isEnabledFor(logging.DEBUG):
            max_bullish = int(bullish_count.max())
            max_bearish = int(bearish_count.max())
            total_long_signals = int(long_signal.sum())
            total_short_signals = int(short_signal.sum())
            logger.debug(f"Six Meridian (consensus={consensus_required}): MaxBull={max_bullish}, MaxBear={max_bearish}, "
                       f"LongSig={total_long_signals}, ShortSig={total_short_signals}")
        
        return strategy_df
        
    except Exception as e:
        logger.error(f"FATAL ERROR in six_meridian_div_swords.strategy_function: {e}", exc_info=True)
        df_copy = df.copy()
        df_copy['position'] = 0
        return df_copy