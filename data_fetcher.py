import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import pytz
from ib_insync import IB, util, Stock, Forex, Index
from ib_insync.objects import BarData

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Changed to a generic name 'market_data' and using CSVs for universal compatibility
DATA_DIR = os.path.join(SCRIPT_DIR, "market_data")
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- APPLY ASYNCIO PATCH ---
try:
    util.startLoop()
except RuntimeError:
    pass

# --- Global IB Instance (Hidden from Main) ---
_ib = IB()

def _get_data_file_path(symbol, timeframe):
    """Generates a standardized path for data files (CSV format)."""
    # Naming convention: SYMBOL_TIMEFRAME.csv (e.g., QQQ_1hour.csv)
    safe_timeframe = timeframe.replace(' ', '')
    return os.path.join(DATA_DIR, f"{symbol}_{safe_timeframe}.csv")

def initialize_data_service(host='127.0.0.1', port=7497, clientId=99):
    """
    Generic entry point to initialize the data provider.
    Currently maps to IBKR connection logic.
    """
    if not _ib.isConnected():
        try:
            logging.info(f"Initializing Data Service (Provider: IBKR) at {host}:{port}...")
            _ib.connect(host, port, clientId=clientId, timeout=10, readonly=True)
            logging.info("Data Service initialized successfully.")
            return True
        except (TimeoutError, ConnectionRefusedError) as e:
            logging.error(f"Data Service initialization failed: {e}")
            logging.error("ACTION: Please ensure the data provider (TWS/Gateway) is running.")
            return False
    else:
        logging.info("Data Service already active.")
        return True

def shutdown_data_service():
    """Generic entry point to shut down the data provider."""
    if _ib.isConnected():
        logging.info("Shutting down Data Service.")
        _ib.disconnect()

def _process_bars_to_df(bars: list, symbol: str, timeframe: str) -> pd.DataFrame:
    """Internal helper to convert IB bar objects to DataFrame."""
    if not bars:
        return pd.DataFrame()
    rows = []
    expected_attrs = ['date', 'open', 'high', 'low', 'close', 'volume']
    for b in bars:
        if isinstance(b, BarData) and all(hasattr(b, attr) for attr in expected_attrs):
            rows.append({
                "date": b.date, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume,
            })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).set_index('date').sort_index()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_market_data(symbol: str, timeframe: str = '1 hour', min_data_days: int = 365, contract_type: str = 'STK'):
    """
    Retrieves market data for a symbol.
    
    Logic:
    1. Checks local 'market_data/' folder for existing CSV.
    2. If IB is connected, attempts to fetch and append new data.
    3. Saves updated data back to CSV.
    
    This structure allows the system to run purely on CSVs if the IB connection fails 
    or if the user simply drops CSV files into the directory.
    """
    file_path = _get_data_file_path(symbol, timeframe)
    now = datetime.now(pytz.utc)
    df_local = pd.DataFrame()

    # 1. Load Local Data (CSV)
    if os.path.exists(file_path):
        try:
            df_local = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            # Ensure timezone awareness (UTC default)
            if df_local.index.tz is None:
                df_local.index = df_local.index.tz_localize(pytz.utc)
            else:
                df_local.index = df_local.index.tz_convert(pytz.utc)
            
            df_local = df_local.sort_index()
            df_local = df_local[~df_local.index.duplicated(keep='first')]
        except Exception as e:
            logging.error(f"Error loading local data from {file_path}: {e}. Re-initializing.")
            df_local = pd.DataFrame()

    latest_date = df_local.index.max() if not df_local.empty else datetime(1970, 1, 1, tzinfo=pytz.utc)

    # If IB is not connected, strictly use local data
    if not _ib.isConnected():
        if df_local.empty:
            logging.warning(f"No local data found for {symbol} and Data Service is offline.")
        return df_local

    # 2. Fetch & Update from Provider (IBKR)
    df_all_data = df_local.copy()
    
    try:
        df_new_parts = []
        
        # A. Fetch Recent Data (Forward Fill)
        if now - latest_date > timedelta(hours=1) or df_local.empty:
            logging.info(f"Local data for {symbol}-{timeframe} is stale/empty. Fetching update...")
            if contract_type.upper() == 'IND':
                contract = Index(symbol, 'CBOE', 'USD')
            elif contract_type.upper() == 'STK':
                contract = Stock(symbol, 'SMART', 'USD')
            else:
                contract = Forex(f'{symbol}USD')
            cds = _ib.reqContractDetails(contract)
            if not cds:
                logging.error(f"Contract lookup failed for {symbol}.")
                return df_local
            
            duration_recent = '30 D' if 'min' in timeframe.lower() or 'hour' in timeframe.lower() else '90 D'
            bars_recent = _ib.reqHistoricalData(
                cds[0].contract, endDateTime='', durationStr=duration_recent,
                barSizeSetting=timeframe, whatToShow='TRADES', useRTH=True, formatDate=1)

            df_temp = _process_bars_to_df(bars_recent, symbol, timeframe)
            if not df_temp.empty:
                if df_temp.index.tz is None: df_temp.index = df_temp.index.tz_localize(pytz.utc)
                else: df_temp.index = df_temp.index.tz_convert(pytz.utc)
                df_new_parts.append(df_temp)
        
        df_all_data = pd.concat([df_local] + df_new_parts)
        df_all_data = df_all_data[~df_all_data.index.duplicated(keep='first')].sort_index()

        # B. Backfill Historical Data (Backward Fill)
        fetch_needed_back_to = now - timedelta(days=min_data_days)
        current_deepest_date = df_all_data.index.min() if not df_all_data.empty else now + timedelta(days=1)
        
        while current_deepest_date > fetch_needed_back_to:
            if contract_type.upper() == 'IND':
                contract = Index(symbol, 'CBOE', 'USD')
            else:
                contract = Stock(symbol, 'SMART', 'USD')
            cds = _ib.reqContractDetails(contract)
            end_dt = current_deepest_date - timedelta(seconds=1)
            end_dt_str = end_dt.strftime('%Y%m%d %H:%M:%S') + ' UTC'
            
            duration_hist = '60 D' if 'min' in timeframe.lower() or 'hour' in timeframe.lower() else '1 Y'
            logging.info(f"Backfilling historical data for {symbol} {timeframe} ending {end_dt_str}...")

            bars_hist = _ib.reqHistoricalData(
                cds[0].contract, endDateTime=end_dt_str, durationStr=duration_hist,
                barSizeSetting=timeframe, whatToShow='TRADES', useRTH=True, formatDate=1)
            
            df_part = _process_bars_to_df(bars_hist, symbol, timeframe)
            if not df_part.empty:
                if df_part.index.tz is None: df_part.index = df_part.index.tz_localize(pytz.utc)
                else: df_part.index = df_part.index.tz_convert(pytz.utc)
                
                df_all_data = pd.concat([df_all_data, df_part])
                df_all_data = df_all_data[~df_all_data.index.duplicated(keep='first')].sort_index()
                current_deepest_date = df_all_data.index.min()
            else:
                logging.warning("No more historical data available. Stopping backfill.")
                break

    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return df_local 

    # 3. Save to CSV (Universal Format)
    if not df_all_data.empty and not df_all_data.equals(df_local):
        df_all_data.to_csv(file_path)
        logging.info(f"Data for {symbol} {timeframe} saved to {file_path}. ({len(df_all_data)} records)")
    elif not df_all_data.empty:
        logging.info(f"Data for {symbol} {timeframe} is up-to-date.")

    return df_all_data