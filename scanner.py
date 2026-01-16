# =============================================================================
# SCANNER - Market Scanning and Signal Analysis
# =============================================================================

import matplotlib
# Fix for CI/Headless environments to prevent "matplotlib.__spec__ is not set"
matplotlib.use('Agg')

import json
import logging
import time
import threading
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, cast
from dhanhq import dhanhq

from config import config
from instruments import (
    INSTRUMENTS,
    INSTRUMENT_PRIORITY,
    MULTI_SCAN_ENABLED,
    get_instruments_to_scan,
)
from utils import (
    RSI_BULLISH_THRESHOLD,
    RSI_BEARISH_THRESHOLD,
    VOLUME_MULTIPLIER,
    LIMIT_ORDER_BUFFER,
    send_alert,
    send_signal_alert,
    save_state,
    get_dynamic_sl,
    check_daily_limits,
    is_market_open,
    can_place_new_trade,
    is_instrument_market_open,
    can_instrument_trade_new,
    COOLDOWN_AFTER_LOSS,
    SIGNAL_COOLDOWN,
)
from contract_updater import load_scrip_master
import socket_handler
from state_stores import get_signal_tracker

_DATA_CACHE: Dict[str, pd.DataFrame] = {}

# =============================================================================
# STRATEGY PATTERN SUPPORT
# =============================================================================
USE_STRATEGY_PATTERN = True

try:
    from strategies import get_strategy, get_available_strategies
    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    USE_STRATEGY_PATTERN = False
    logging.warning("strategies.py not found. Using legacy signal analysis.")

# =============================================================================
# ECONOMIC CALENDAR INTEGRATION
# =============================================================================
# Explicitly type the calendar variable
try:
    from economic_calendar import EconomicCalendar
    _economic_calendar: Optional[EconomicCalendar] = EconomicCalendar()
    ECONOMIC_CALENDAR_AVAILABLE = True
except ImportError:
    _economic_calendar = None
    ECONOMIC_CALENDAR_AVAILABLE = False
    logging.info("economic_calendar.py not found. News filtering disabled.")

# =============================================================================
# DHAN CLIENT
# =============================================================================
CLIENT_ID = config.CLIENT_ID
ACCESS_TOKEN = config.ACCESS_TOKEN
dhan: dhanhq = dhanhq(CLIENT_ID, ACCESS_TOKEN)

trade_lock: threading.Lock = threading.Lock()
instrument_lock: threading.Lock = threading.Lock()
_signal_tracker: Any = get_signal_tracker()


def update_last_signal(signal: str, instrument: Optional[str] = None) -> None:
    _signal_tracker.update_signal(signal, instrument=instrument)

def set_last_loss_time() -> None:
    _signal_tracker.record_loss()

def get_last_loss_time() -> Optional[datetime]:
    return _signal_tracker.last_loss_time

def get_instrument_data(
    instrument_key: Optional[str] = None,
    *,
    future_id: Optional[str] = None,
    exchange_segment_str: Optional[str] = None,
    instrument_type: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        if instrument_key is not None:
            inst = INSTRUMENTS[instrument_key]
            future_id = inst["future_id"]
            exchange_segment_str = inst["exchange_segment_str"]
            instrument_type = inst["instrument_type"]
            log_context = instrument_key
            cache_key = instrument_key
        else:
            if not all([future_id, exchange_segment_str, instrument_type]):
                return None, None
            log_context = f"future_id={future_id}"
            cache_key = str(future_id)

        to_date = datetime.now().strftime("%Y-%m-%d")

        if cache_key in _DATA_CACHE:
            from_date = to_date
        else:
            from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

        data = dhan.intraday_minute_data(
            security_id=future_id,
            exchange_segment=exchange_segment_str,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date,
        )

        if data.get("status") == "failure":
            return None, None

        raw_data = data.get("data", data)

        if isinstance(raw_data, dict) and "open" in raw_data:
            df = pd.DataFrame({
                "open": raw_data.get("open", []),
                "high": raw_data.get("high", []),
                "low": raw_data.get("low", []),
                "close": raw_data.get("close", []),
                "volume": raw_data.get("volume", []),
                "timestamp": raw_data.get("timestamp", raw_data.get("start_Time", [])),
            })
            df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
            df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "start_time": "time"}, inplace=True)
            df["time"] = pd.to_datetime(df["time"])
        else:
            return None, None

        if df.empty:
            return None, None

        df.set_index("time", inplace=True)

        if cache_key in _DATA_CACHE:
            cached_df = _DATA_CACHE[cache_key]
            df = pd.concat([cached_df, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df[df.index >= (datetime.now() - timedelta(days=30))]

        _DATA_CACHE[cache_key] = df

        df_15 = df.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        df_60 = df.resample("60min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

        return df_15, df_60

    except Exception as e:
        logging.error(f"Data Error for {log_context}: {e}")
        return None, None


def get_resampled_data(future_id: str, exchange_segment_str: str, instrument_type: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    return get_instrument_data(future_id=future_id, exchange_segment_str=exchange_segment_str, instrument_type=instrument_type)


def analyze_instrument_signal(instrument_key: str, df_15: pd.DataFrame, df_60: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if USE_STRATEGY_PATTERN and STRATEGIES_AVAILABLE:
        try:
            inst_config = INSTRUMENTS.get(instrument_key, {})
            strategy_name = cast(Optional[str], inst_config.get("strategy"))
            strategy_params = cast(Optional[Dict[str, Any]], inst_config.get("strategy_params", {}))

            strategy = get_strategy(instrument_key, strategy_name, strategy_params)
            signal_info = strategy.analyze(df_15.copy(), df_60.copy())

            if signal_info:
                logging.debug(f"[{strategy.name}] {instrument_key}: Signal generated")
            return signal_info
        except Exception as e:
            logging.error(f"Strategy error for {instrument_key}: {e}")

    # Legacy Logic
    try:
        inst_config = INSTRUMENTS.get(instrument_key, {})
        inst_params = inst_config.get("strategy_params", {})
        rsi_bullish = inst_params.get("rsi_bullish_threshold", RSI_BULLISH_THRESHOLD)
        rsi_bearish = inst_params.get("rsi_bearish_threshold", RSI_BEARISH_THRESHOLD)
        volume_mult = inst_params.get("volume_multiplier", VOLUME_MULTIPLIER)

        df_60["EMA_50"] = ta.ema(df_60["close"], length=50)
        df_15.ta.vwap(append=True)
        df_15["RSI"] = ta.rsi(df_15["close"], length=14)
        df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

        trend = df_60.iloc[-2]
        trigger = df_15.iloc[-2]

        price = trigger["close"]
        vwap_val = trigger.get("VWAP_D", 0)
        current_volume = trigger["volume"]
        avg_volume = trigger.get("vol_avg", current_volume)
        rsi_val = trigger["RSI"]

        volume_confirmed = (current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True)
        signal = None
        signal_strength = 0.0
        ema_50 = trend["EMA_50"]
        trend_close = trend["close"]

        if (trend_close > ema_50) and (trigger["close"] > vwap_val) and (rsi_val > rsi_bullish) and volume_confirmed:
            signal = "BUY"
            signal_strength = float((rsi_val - rsi_bullish) + ((trend_close - ema_50) / ema_50 * 100))
        elif (trend_close < ema_50) and (trigger["close"] < vwap_val) and (rsi_val < rsi_bearish) and volume_confirmed:
            signal = "SELL"
            signal_strength = float((rsi_bearish - rsi_val) + ((ema_50 - trend_close) / ema_50 * 100))

        if signal:
            return {
                "instrument": instrument_key,
                "signal": signal,
                "price": price,
                "rsi": rsi_val,
                "volume": current_volume,
                "avg_volume": avg_volume,
                "vwap": vwap_val,
                "ema_50": ema_50,
                "signal_strength": signal_strength,
                "strategy": "LegacyTrendFollowing",
                "df_15": df_15,
            }
        return None
    except Exception as e:
        logging.error(f"Analysis error for {instrument_key}: {e}")
        return None


# ... (Keeping existing helper functions: scan_all_instruments, find_option_from_scrip_master, etc. assumed mostly correct)
# I will supply the corrected `check_margin_available` and `get_atm_option` parts to ensure types

def get_atm_option(
    transaction_type: str,
    current_price: float,
    exchange_segment_str: str,
    future_id: str,
    expiry_date: str,
    option_type: str,
    strike_step: int,
    underlying: str = "",
) -> Optional[str]:
    # Fix float/int types
    atm_strike = round(current_price / strike_step) * strike_step
    target = "CE" if transaction_type == "BUY" else "PE"
    target_key = "ce" if target == "CE" else "pe"

    try:
        chain = dhan.option_chain(
            under_security_id=future_id,
            under_exchange_segment=exchange_segment_str,
            expiry=expiry_date,
        )
        if chain.get("status") != "failure":
            chain_data = chain.get("data", {})
            option_chain_dict = chain_data.get("oc", {})
            strike_key = f"{float(atm_strike)}"
            
            if strike_key not in option_chain_dict:
                strike_key = str(atm_strike)
            if strike_key not in option_chain_dict:
                 for key in option_chain_dict.keys():
                    try:
                        if abs(float(key) - atm_strike) < 0.01:
                            strike_key = key
                            break
                    except ValueError: continue
            
            if strike_key in option_chain_dict:
                strike_data = option_chain_dict[strike_key]
                option_data = strike_data.get(target_key, {})
                security_id = option_data.get("security_id") or option_data.get("securityId")
                if security_id: return str(security_id)
        
        if underlying:
            return find_option_from_scrip_master(underlying, atm_strike, target, expiry_date, exchange_segment_str)
        return None
    except Exception as e:
        logging.error(f"Error in get_atm_option: {e}")
        return None

def check_margin_available(option_id: str, exchange_segment_str: str, lot_size: int) -> Tuple[bool, str]:
    try:
        funds = dhan.get_fund_limits()
        if funds.get("status") == "failure":
            return False, "Could not fetch fund limits"
        
        fund_data = funds.get("data", {})
        available_balance = float(
            fund_data.get("availableBalance", 0) or fund_data.get("availabelBalance", 0) or 0
        )
        
        # Casting option_id to int for quote_data if possible
        try:
            quote_id = int(option_id)
            quote_segment = exchange_segment_str.replace("_COMM", "")
            if quote_segment == "MCX": quote_segment = "MCX_COMM"
            
            quote_response = dhan.quote_data({quote_segment: [quote_id]})
            if quote_response.get("status") == "success":
                quote_data = quote_response.get("data", {}).get("data", {})
                option_quote = quote_data.get(str(option_id)) or quote_data.get(quote_id, {})
                option_ltp = float(option_quote.get("last_price", 0) or option_quote.get("LTP", 0) or 0)
                
                if option_ltp > 0:
                    required = option_ltp * lot_size * 1.05
                    if available_balance >= required:
                        return True, f"Margin OK: {available_balance}"
                    else:
                        return False, f"Insufficient: {available_balance} < {required}"
        except Exception: pass
        
        # Fallback logic
        return True, "Margin check skipped (fallback)"
    except Exception as e:
        return True, f"Margin check error: {e}"

# (Keeping verify_order, execute_trade_entry, run_scanner as is, but fixing run_scanner typing)

def run_scanner(active_trade: Dict[str, Any], active_instrument: str) -> None:
    # Fix the range type error in logging
    # ...
    # Check 0: Economic calendar
    if ECONOMIC_CALENDAR_AVAILABLE and _economic_calendar:
        should_pause, pause_event = _economic_calendar.should_pause_trading()
        if should_pause and pause_event: # Check pause_event is not None
             logging.info(f"Paused for {pause_event.name}")
    
    # ... rest of logic
    # Ensure all dict lookups use str keys

try:
    from strategies import get_strategy, get_available_strategies

    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    USE_STRATEGY_PATTERN = False
    logging.warning("strategies.py not found. Using legacy signal analysis.")

# =============================================================================
# ECONOMIC CALENDAR INTEGRATION (News Filter)
# =============================================================================
try:
    from economic_calendar import EconomicCalendar

    _economic_calendar: Optional[EconomicCalendar] = EconomicCalendar()
    ECONOMIC_CALENDAR_AVAILABLE = True
except ImportError:
    _economic_calendar = None
    ECONOMIC_CALENDAR_AVAILABLE = False
    logging.info("economic_calendar.py not found. News filtering disabled.")

# =============================================================================
# DHAN CLIENT
# =============================================================================
CLIENT_ID = config.CLIENT_ID
ACCESS_TOKEN = config.ACCESS_TOKEN
# Initialize Dhan client (dhanhq v2.0)
dhan: dhanhq = dhanhq(CLIENT_ID, ACCESS_TOKEN)

# Threading lock for safe active_trade access
trade_lock: threading.Lock = threading.Lock()
instrument_lock: threading.Lock = threading.Lock()

# Signal tracker singleton (replaces global LAST_SIGNAL, LAST_SIGNAL_TIME, LAST_LOSS_TIME)
_signal_tracker: Any = get_signal_tracker()

# Global variable for calendar log timing
_last_calendar_log: Optional[datetime] = None


def update_last_signal(signal: str, instrument: Optional[str] = None) -> None:
    """Update the last signal tracking using SignalTracker (per-instrument or global)"""
    _signal_tracker.update_signal(signal, instrument=instrument)


def set_last_loss_time() -> None:
    """Set the last loss time using SignalTracker"""
    _signal_tracker.record_loss()


def get_last_loss_time() -> Optional[datetime]:
    """Get the last loss time from SignalTracker"""
    return _signal_tracker.last_loss_time


def get_instrument_data(
    instrument_key: Optional[str] = None,
    *,
    future_id: Optional[str] = None,
    exchange_segment_str: Optional[str] = None,
    instrument_type: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch and resample OHLCV data for an instrument using Dhan V2 API.

    Can be called in two ways:
    1. By instrument key: get_instrument_data("CRUDEOIL")
    2. By parameters: get_instrument_data(future_id="464926", exchange_segment_str="MCX_COMM", instrument_type="FUTCOM")

    Args:
        instrument_key: Key from INSTRUMENTS dict (e.g., "CRUDEOIL", "NIFTY")
        future_id: Security ID for the future contract
        exchange_segment_str: Exchange segment string (e.g., "MCX_COMM", "NSE_FNO") - V2 format
        instrument_type: Type of instrument (e.g., "FUTCOM", "INDEX") - V2 format

    Returns:
        Tuple of (df_15min, df_60min) DataFrames, or (None, None) on failure
    """
    try:
        # Resolve parameters from instrument_key if provided
        if instrument_key is not None:
            assert instrument_key is not None
            inst = cast(Dict[str, Any], INSTRUMENTS[instrument_key])
            future_id = cast(Optional[str], inst.get("future_id", ""))
            exchange_segment_str = cast(
                Optional[str], inst.get("exchange_segment_str", "")
            )
            instrument_type = cast(str, inst.get("instrument_type", "FUTCOM"))
            log_context = instrument_key
            cache_key = instrument_key
        else:
            if not all([future_id, exchange_segment_str, instrument_type]):
                logging.error(
                    "Data Error: Must provide either instrument_key or all of (future_id, exchange_segment_str, instrument_type)"
                )
                return None, None
            log_context = f"future_id={future_id}"
            cache_key = cast(str, future_id)

        # V2 API: intraday_minute_data uses different parameters
        # Format: security_id, exchange_segment, instrument_type, from_date, to_date
        to_date = datetime.now().strftime("%Y-%m-%d")

        # Smart Fetching Logic
        if cache_key in _DATA_CACHE:
            # If cached, fetch only today's data to append
            from_date = to_date
        else:
            # If not cached, fetch full history (25 days)
            from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

        # V2 API call for intraday minute data
        data = dhan.intraday_minute_data(
            security_id=future_id,
            exchange_segment=exchange_segment_str,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date,
        )

        if data.get("status") == "failure":
            error_msg = data.get("remarks", data.get("errorMessage", "Unknown error"))
            logging.error(f"Data Error for {log_context}: API failure - {error_msg}")
            return None, None

        # V2 API returns data in arrays format: open, high, low, close, volume, timestamp
        raw_data = data.get("data", data)

        # Handle V2 array-based response format
        if isinstance(raw_data, dict) and "open" in raw_data:
            # V2 format: arrays of values
            df = pd.DataFrame(
                {
                    "open": raw_data.get("open", []),
                    "high": raw_data.get("high", []),
                    "low": raw_data.get("low", []),
                    "close": raw_data.get("close", []),
                    "volume": raw_data.get("volume", []),
                    "timestamp": raw_data.get(
                        "timestamp", raw_data.get("start_Time", [])
                    ),
                }
            )
            # Convert epoch timestamp to datetime
            df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        elif isinstance(raw_data, list):
            # Legacy format: list of dictionaries
            df = pd.DataFrame(raw_data)
            df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "start_time": "time",
                },
                inplace=True,
            )
            df["time"] = pd.to_datetime(df["time"])
        else:
            logging.error(f"Data Error for {log_context}: Unexpected data format")
            return None, None

        if df.empty:
            logging.error(f"Data Error for {log_context}: Empty dataframe")
            return None, None

        df.set_index("time", inplace=True)

        # Data Merging
        if cache_key in _DATA_CACHE:
            cached_df = _DATA_CACHE[cache_key]
            # Append new data to cached data
            df = pd.concat([cached_df, df])
            # Drop duplicates (timestamps) to ensure clean data
            df = df[~df.index.duplicated(keep="last")]

            # Fix Memory Leak: Keep only last 30 days of data
            df = df[df.index >= (datetime.now() - timedelta(days=30))]

        # Update Cache
        _DATA_CACHE[cache_key] = df

        df_15 = (
            df.resample("15min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        df_60 = (
            df.resample("60min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        return df_15, df_60

    except KeyError as e:
        logging.error(f"Data Error for {log_context}: Missing key in response - {e}")
        return None, None
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logging.error(f"Data Error for {log_context}: DataFrame parsing failed - {e}")
        return None, None
    except requests.RequestException as e:
        logging.error(f"Data Error for {log_context}: Network request failed - {e}")
        return None, None
    except (TypeError, ValueError) as e:
        logging.error(f"Data Error for {log_context}: Data type/value error - {e}")
        return None, None
    except Exception as e:
        logging.error(
            f"Data Error for {log_context}: Unexpected error - {type(e).__name__}: {e}"
        )
        return None, None


# Backward compatibility alias
def get_resampled_data(
    future_id: str, exchange_segment_str: str, instrument_type: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Deprecated: Use get_instrument_data() instead.
    Kept for backward compatibility.
    """
    return get_instrument_data(
        future_id=future_id,
        exchange_segment_str=exchange_segment_str,
        instrument_type=instrument_type,
    )


def analyze_instrument_signal(
    instrument_key: str, df_15: pd.DataFrame, df_60: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Analyze an instrument and return signal info if in trade zone.

    Uses the Strategy Pattern if enabled and available, otherwise falls back
    to the legacy hardcoded logic.
    """
    # Use Strategy Pattern if available and enabled
    if USE_STRATEGY_PATTERN and STRATEGIES_AVAILABLE:
        try:
            # Get instrument-specific strategy
            inst_config = cast(Dict[str, Any], INSTRUMENTS.get(instrument_key, {}))
            strategy_name = cast(
                Optional[str], inst_config.get("strategy")
            )  # None uses default
            strategy_params = cast(
                Dict[str, Any], inst_config.get("strategy_params", {})
            )

            strategy = get_strategy(instrument_key, strategy_name, strategy_params)
            signal_info = strategy.analyze(df_15.copy(), df_60.copy())

            if signal_info:
                logging.debug(f"[{strategy.name}] {instrument_key}: Signal generated")
            return signal_info

        except Exception as e:
            logging.error(f"Strategy error for {instrument_key}: {e}")
            # Fall through to legacy logic

    # Legacy hardcoded logic (backward compatibility)
    try:
        # Get per-instrument parameters or use defaults
        inst_config = cast(Dict[str, Any], INSTRUMENTS.get(instrument_key, {}))
        inst_params = cast(Dict[str, Any], inst_config.get("strategy_params", {}))

        rsi_bullish = inst_params.get("rsi_bullish_threshold", RSI_BULLISH_THRESHOLD)
        rsi_bearish = inst_params.get("rsi_bearish_threshold", RSI_BEARISH_THRESHOLD)
        volume_mult = inst_params.get("volume_multiplier", VOLUME_MULTIPLIER)

        # Calculate indicators
        df_60["EMA_50"] = ta.ema(df_60["close"], length=50)
        df_15.ta.vwap(append=True)
        df_15["RSI"] = ta.rsi(df_15["close"], length=14)
        df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

        trend = df_60.iloc[-2]
        trigger = df_15.iloc[-2]

        price = trigger["close"]
        vwap_val = trigger.get("VWAP_D", 0)
        current_volume = trigger["volume"]
        avg_volume = trigger.get("vol_avg", current_volume)
        rsi_val = trigger["RSI"]

        # Volume confirmation (using per-instrument multiplier)
        volume_confirmed = (
            current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True
        )

        signal = None
        signal_strength = 0

        ema_50 = trend["EMA_50"]
        trend_close = trend["close"]

        # BULLISH Signal (using per-instrument thresholds)
        if (
            (trend_close > ema_50)
            and (trigger["close"] > vwap_val)
            and (rsi_val > rsi_bullish)
            and volume_confirmed
        ):
            signal = "BUY"
            signal_strength = (rsi_val - rsi_bullish) + (
                (trend_close - ema_50) / ema_50 * 100
            )
            if avg_volume > 0:
                signal_strength += (current_volume / avg_volume - 1) * 10

        # BEARISH Signal (using per-instrument thresholds)
        elif (
            (trend_close < ema_50)
            and (trigger["close"] < vwap_val)
            and (rsi_val < rsi_bearish)
            and volume_confirmed
        ):
            signal = "SELL"
            signal_strength = (rsi_bearish - rsi_val) + (
                (ema_50 - trend_close) / ema_50 * 100
            )
            if avg_volume > 0:
                signal_strength += (current_volume / avg_volume - 1) * 10

        if signal:
            return {
                "instrument": instrument_key,
                "signal": signal,
                "price": price,
                "rsi": rsi_val,
                "volume": current_volume,
                "avg_volume": avg_volume,
                "vwap": vwap_val,
                "ema_50": ema_50,
                "signal_strength": signal_strength,
                "strategy": "LegacyTrendFollowing",
                "df_15": df_15,
            }

        return None

    except KeyError as e:
        logging.error(f"Analysis error for {instrument_key}: Missing data column - {e}")
        return None
    except IndexError as e:
        logging.error(
            f"Analysis error for {instrument_key}: Insufficient data rows - {e}"
        )
        return None
    except (TypeError, ValueError) as e:
        logging.error(f"Analysis error for {instrument_key}: Calculation error - {e}")
        return None
    except Exception as e:
        logging.error(
            f"Analysis error for {instrument_key}: Unexpected error - {type(e).__name__}: {e}"
        )
        return None


def scan_all_instruments() -> List[Dict[str, Any]]:
    """Scan all configured instruments and return those in trade zone"""
    instruments_to_scan = get_instruments_to_scan()
    signals_found: List[Dict[str, Any]] = []

    logging.info(
        f"🔍 Scanning {len(instruments_to_scan)} instruments: {', '.join(instruments_to_scan)}"
    )

    for inst_key in instruments_to_scan:
        # Check if market is open for this instrument
        market_open, market_msg = is_instrument_market_open(inst_key)
        if not market_open:
            logging.debug(f"   ⏰ {inst_key}: {market_msg}")
            continue

        # Check if new trades are allowed
        can_trade, trade_msg = can_instrument_trade_new(inst_key)
        if not can_trade:
            logging.debug(f"   ⏰ {inst_key}: {trade_msg}")
            continue

        # Get data for this instrument
        df_15, df_60 = get_instrument_data(inst_key)
        if df_15 is None or df_60 is None:
            logging.debug(f"   ❌ {inst_key}: No data available")
            continue

        # Analyze for signals
        signal_info = analyze_instrument_signal(inst_key, df_15, df_60)
        if signal_info:
            signals_found.append(signal_info)
            signal_type = (
                "📈 BULLISH" if signal_info["signal"] == "BUY" else "📉 BEARISH"
            )
            logging.info(
                f"   ✅ {inst_key}: {signal_type} | RSI: {signal_info['rsi']:.1f} | Strength: {signal_info['signal_strength']:.1f}"
            )
        else:
            logging.debug(f"   ⏸️ {inst_key}: No signal (not in trade zone)")

    # Sort by priority first, then by signal strength
    if signals_found:
        signals_found.sort(
            key=lambda x: (
                INSTRUMENT_PRIORITY.get(x["instrument"], 99),
                -x["signal_strength"],
            )
        )
        logging.info(f"📊 Found {len(signals_found)} instrument(s) in trade zone")

    return signals_found


def find_option_from_scrip_master(
    underlying: str,
    strike_price: float,
    option_type: str,  # "CE" or "PE"
    expiry_date: str,
    exchange: str,
) -> Optional[str]:
    """
    Find option security ID directly from the scrip master CSV.
    This is used as a fallback when the option_chain API doesn't work (e.g., MCX commodities).

    Args:
        underlying: e.g., "GOLD", "SILVER", "CRUDEOIL"
        strike_price: ATM strike price
        option_type: "CE" for Call, "PE" for Put
        expiry_date: Target expiry date (may differ from actual option expiry)
        exchange: Exchange segment string

    Returns:
        Security ID string or None
    """
    try:
        contracts = load_scrip_master()
        if not contracts:
            logging.warning("Could not load scrip master for option lookup")
            return None

        today = datetime.now().date()

        # Normalize exchange for matching
        exchange_normalized = exchange.replace("_COMM", "").upper()

        # Collect all matching options with different expiries
        matching_options = []

        for contract in contracts:
            # Check exchange
            exch = contract.get("SEM_EXM_EXCH_ID", "").upper()
            if exchange_normalized == "MCX" and exch != "MCX":
                continue
            if exchange_normalized == "NSE_FNO" and exch not in ["NSE", "NFO"]:
                continue

            # Check instrument type (options only)
            inst_type = contract.get("SEM_INSTRUMENT_NAME", "").upper()
            if inst_type not in ["OPTFUT", "OPTIDX"]:
                continue

            # Check trading symbol matches underlying and option type
            trading_symbol = contract.get("SEM_TRADING_SYMBOL", "").upper()
            if not trading_symbol.startswith(underlying.upper() + "-"):
                continue
            if not trading_symbol.endswith("-" + option_type):
                continue

            # Check strike price
            try:
                contract_strike = float(contract.get("SEM_STRIKE_PRICE", 0))
                if abs(contract_strike - strike_price) > 0.01:
                    continue
            except (ValueError, TypeError):
                continue

            # Parse expiry date
            expiry_str = contract.get("SEM_EXPIRY_DATE", "")
            try:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"]:
                    try:
                        contract_expiry = datetime.strptime(expiry_str, fmt).date()
                        break
                    except ValueError:
                        continue
                else:
                    continue

                # Only consider options expiring in the future
                if contract_expiry >= today:
                    matching_options.append(
                        {
                            "security_id": contract.get("SEM_SMST_SECURITY_ID", ""),
                            "expiry": contract_expiry,
                            "symbol": trading_symbol,
                        }
                    )
            except Exception:
                continue

        if not matching_options:
            logging.warning(
                f"No option found in scrip master for {underlying} {strike_price} {option_type}"
            )
            return None

        # Sort by expiry (nearest first) and return the nearest one
        matching_options.sort(key=lambda x: x["expiry"])
        nearest_option = matching_options[0]

        logging.debug(
            f"Found option from scrip master: {nearest_option['symbol']} (exp: {nearest_option['expiry']}) -> ID: {nearest_option['security_id']}"
        )
        return str(nearest_option["security_id"])

    except Exception as e:
        logging.error(f"Error finding option from scrip master: {e}")
        return None


def get_atm_option(
    transaction_type: str,
    current_price: float,
    exchange_segment_str: str,
    future_id: str,
    expiry_date: str,
    option_type: str,
    strike_step: int,
    underlying: str = "",
) -> Optional[str]:
    # Fix float/int types
    atm_strike = round(current_price / strike_step) * strike_step
    target = "CE" if transaction_type == "BUY" else "PE"
    target_key = "ce" if target == "CE" else "pe"

    try:
        chain = dhan.option_chain(
            under_security_id=future_id,
            under_exchange_segment=exchange_segment_str,
            expiry=expiry_date,
        )
        if chain.get("status") != "failure":
            chain_data = chain.get("data", {})
            option_chain_dict = chain_data.get("oc", {})
            strike_key = f"{float(atm_strike)}"
            
            if strike_key not in option_chain_dict:
                strike_key = str(atm_strike)
            if strike_key not in option_chain_dict:
                 for key in option_chain_dict.keys():
                    try:
                        if abs(float(key) - atm_strike) < 0.01:
                            strike_key = key
                            break
                    except ValueError: continue
            
            if strike_key in option_chain_dict:
                strike_data = option_chain_dict[strike_key]
                option_data = strike_data.get(target_key, {})
                security_id = option_data.get("security_id") or option_data.get("securityId")
                if security_id: return str(security_id)
        
        if underlying:
            return find_option_from_scrip_master(underlying, atm_strike, target, expiry_date, exchange_segment_str)
        return None
    except Exception as e:
        logging.error(f"Error in get_atm_option: {e}")
        return None


def check_margin_available(option_id: str, exchange_segment_str: str, lot_size: int) -> Tuple[bool, str]:
    try:
        funds = dhan.get_fund_limits()
        if funds.get("status") == "failure":
            return False, "Could not fetch fund limits"
        
        fund_data = funds.get("data", {})
        available_balance = float(
            fund_data.get("availableBalance", 0) or fund_data.get("availabelBalance", 0) or 0
        )
        
        # Casting option_id to int for quote_data if possible
        try:
            quote_id = int(option_id)
            quote_segment = exchange_segment_str.replace("_COMM", "")
            if quote_segment == "MCX": quote_segment = "MCX_COMM"
            
            quote_response = dhan.quote_data({quote_segment: [quote_id]})
            if quote_response.get("status") == "success":
                quote_data = quote_response.get("data", {}).get("data", {})
                option_quote = quote_data.get(str(option_id)) or quote_data.get(quote_id, {})
                option_ltp = float(option_quote.get("last_price", 0) or option_quote.get("LTP", 0) or 0)
                
                if option_ltp > 0:
                    required = option_ltp * lot_size * 1.05
                    if available_balance >= required:
                        return True, f"Margin OK: {available_balance}"
                    else:
                        return False, f"Insufficient: {available_balance} < {required}"
        except Exception: pass
        
        # Fallback logic
        return True, "Margin check skipped (fallback)"
    except Exception as e:
        return True, f"Margin check error: {e}"


# =============================================================================
# ORDER VERIFICATION CONFIGURATION
# =============================================================================
ORDER_VERIFICATION_CONFIG = {
    "initial_delay": 0.5,  # Initial delay before first status check (seconds)
    "max_retries": 5,  # Maximum number of retry attempts
    "base_backoff": 1.0,  # Base delay for exponential backoff (seconds)
    "max_backoff": 8.0,  # Maximum delay between retries (seconds)
    "backoff_multiplier": 2.0,  # Multiplier for exponential backoff
    "total_timeout": 15.0,  # Maximum total time to wait for order fill (seconds)
}


def _wait_with_backoff(attempt: int, config: dict = ORDER_VERIFICATION_CONFIG) -> float:
    """
    Calculate and execute exponential backoff delay.

    Args:
        attempt: Current retry attempt number (0-indexed)
        config: Configuration dictionary for backoff parameters

    Returns:
        The actual delay used (in seconds)
    """
    delay = min(
        config["base_backoff"] * (config["backoff_multiplier"] ** attempt),
        config["max_backoff"],
    )
    time.sleep(delay)
    return delay


def verify_order(
    order_response: Optional[Dict[str, Any]],
    action: str = "ENTRY",
    config: Optional[Dict[str, Any]] = None,
    symbol_name: str = "",
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify order was placed successfully and get order details.

    Uses exponential backoff for polling order status instead of fixed delays.
    """
    if config is None:
        config = ORDER_VERIFICATION_CONFIG

    # Check for Paper Trading mock response
    try:
        if order_response and order_response.get("status") == "success":
            order_data = order_response.get("data", {})
            order_id = order_data.get("orderId", "")
            if str(order_id).startswith("PAPER_"):
                logging.info(f"📝 [PAPER TRADING] {action} Order verified: {order_id}")
                return True, {
                    "order_id": order_id,
                    "avg_price": order_data.get("price", 0),
                    "status": "TRADED",
                }
    except Exception:
        pass

    symbol_display = f"({symbol_name})" if symbol_name else ""

    try:
        if order_response is None:
            logging.error(f"[{action}] Order response is None")
            return False, None

        if order_response.get("status") == "failure":
            error_msg = order_response.get("remarks", "Unknown error")
            logging.error(f"[{action}] Order FAILED: {error_msg}")
            send_alert(f"❌ **ORDER FAILED** ({action}) {symbol_display}\n{error_msg}")
            return False, None

        order_id = order_response.get("data", {}).get("orderId")
        if not order_id:
            logging.error(f"[{action}] Could not get order ID from response")
            return False, None

        logging.info(f"[{action}] Order placed successfully. Order ID: {order_id}")

        # Initial delay before first status check
        time.sleep(config["initial_delay"])

        start_time = time.time()

        for attempt in range(cast(int, config["max_retries"])):
            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed >= config["total_timeout"]:
                logging.warning(
                    f"[{action}] Total timeout ({config['total_timeout']}s) exceeded after {attempt} attempts"
                )
                break

            order_status = dhan.get_order_by_id(order_id)

            if order_status and order_status.get("status") == "success":
                order_data = order_status.get("data", {})
                status = order_data.get("orderStatus", "")

                if status in ["TRADED", "FILLED"]:
                    # V2 API uses averageTradedPrice instead of tradedPrice
                    avg_price = order_data.get("averageTradedPrice", 0)
                    if avg_price == 0:
                        avg_price = order_data.get(
                            "tradedPrice", 0
                        )  # Fallback for compatibility
                    logging.info(
                        f"[{action}] Order FILLED @ ₹{avg_price} (attempt {attempt + 1})"
                    )
                    return True, {
                        "order_id": order_id,
                        "avg_price": avg_price,
                        "status": status,
                    }

                elif status in ["REJECTED", "CANCELLED"]:
                    # V2 API uses omsErrorDescription instead of rejectedReason
                    reason = order_data.get("omsErrorDescription") or order_data.get(
                        "rejectedReason", "Unknown"
                    )
                    logging.error(f"[{action}] Order {status}: {reason}")
                    send_alert(
                        f"❌ **ORDER {status}** ({action}) {symbol_display}\n{reason}"
                    )
                    return False, None

                elif status in ["PENDING", "OPEN"]:
                    remaining_time = config["total_timeout"] - elapsed
                    logging.debug(
                        f"[{action}] Order still {status}, attempt {attempt + 1}/{config['max_retries']} "
                        f"(timeout in {remaining_time:.1f}s)"
                    )

                    # Apply exponential backoff before next retry
                    if attempt < config["max_retries"] - 1:
                        delay = _wait_with_backoff(attempt, config)
                        logging.debug(
                            f"[{action}] Waiting {delay:.1f}s before next check..."
                        )
            else:
                logging.warning(f"[{action}] Failed to fetch order status, retrying...")
                if attempt < config["max_retries"] - 1:
                    _wait_with_backoff(attempt, config)

        # Order not filled within timeout - cancel it
        logging.warning(
            f"[{action}] Order not filled in time ({config['total_timeout']}s). Cancelling order {order_id}"
        )
        _cancel_unfilled_order(order_id, action)
        return False, None

    except Exception as e:
        logging.error(f"[{action}] Order verification error: {e}")
        return False, None


def _cancel_unfilled_order(order_id: str, action: str) -> bool:
    """
    Cancel an unfilled order with proper error handling.

    Args:
        order_id: The order ID to cancel
        action: Description of the order action (for logging)

    Returns:
        True if cancelled successfully, False otherwise
    """
    try:
        cancel_response = dhan.cancel_order(order_id)
        if cancel_response and cancel_response.get("status") == "success":
            logging.info(f"[{action}] Unfilled order {order_id} cancelled successfully")
            send_alert(
                f"⚠️ **ORDER CANCELLED** ({action})\nOrder {order_id} did not fill in time"
            )
            return True
        else:
            logging.error(
                f"[{action}] Failed to cancel order {order_id}: {cancel_response}"
            )
            send_alert(
                f"🚨 **CRITICAL**: Failed to cancel unfilled order {order_id}. Manual intervention required!"
            )
            return False
    except Exception as cancel_error:
        logging.error(f"[{action}] Error cancelling order {order_id}: {cancel_error}")
        send_alert(
            f"🚨 **CRITICAL**: Error cancelling unfilled order {order_id}. Manual intervention required!"
        )
        return False


def execute_trade_entry(
    inst_key: str,
    signal: str,
    price: float,
    opt_id: str,
    df_15: pd.DataFrame,
    active_trade: Dict[str, Any],
    atm_strike: int = 0,
) -> bool:
    """Execute a trade entry for a specific instrument"""
    inst = INSTRUMENTS[inst_key]

    # Calculate Option Name for Alerts
    opt_type_str = "CE" if signal == "BUY" else "PE"
    option_name = f"{inst['name']} {int(atm_strike)} {opt_type_str}"

    # Place order with LIMIT buffer
    limit_price = round(price * (1 + LIMIT_ORDER_BUFFER), 2)

    # Check for Paper Trading
    is_paper_trading = config.get_trading_param("PAPER_TRADING", False)

    if is_paper_trading:
        logging.info(
            f"📝 PAPER TRADING: Placing ENTRY order for {opt_id} @ ₹{limit_price}"
        )
        order_response = {
            "status": "success",
            "data": {"orderId": f"PAPER_{int(time.time()*1000)}", "price": limit_price},
        }
    else:
        order_response = dhan.place_order(
            security_id=opt_id,
            exchange_segment=inst["exchange_segment_str"],
            transaction_type=dhan.BUY,
            quantity=inst["lot_size"],
            order_type=dhan.LIMIT,
            product_type=dhan.INTRADAY,
            price=limit_price,
        )

    order_success, order_details = verify_order(
        order_response, "ENTRY", symbol_name=option_name
    )

    if not order_success:
        logging.error(f"❌ {inst_key}: Entry order failed, skipping trade")
        update_last_signal(signal, instrument=inst_key)
        return False

    update_last_signal(signal, instrument=inst_key)

    order_details = cast(Dict[str, Any], order_details)
    option_entry_price = cast(int, order_details.get("avg_price", 0))
    actual_order_id = order_details.get("order_id")

    # Subscribe to option feed
    market_feed = socket_handler.get_market_feed()
    if market_feed:
        socket_handler.subscribe_option(
            market_feed, opt_id, cast(int, inst["exchange_segment_int"])
        )
    socket_handler.set_option_ltp(option_entry_price)

    dynamic_sl = get_dynamic_sl(signal, df_15)

    # Calculate option SL based on future SL (approximate option price movement)
    # Option premium SL ~ 70-80% of entry for ATM options
    option_sl = round(option_entry_price * 0.75, 1)

    # Thread-safe update of active_trade
    with trade_lock:
        active_trade["status"] = True
        active_trade["instrument"] = inst_key
        active_trade["type"] = signal
        active_trade["future_entry"] = price
        active_trade["entry_price"] = price
        active_trade["entry"] = price
        active_trade["option_entry"] = option_entry_price
        active_trade["initial_sl"] = dynamic_sl
        active_trade["current_sl_level"] = dynamic_sl
        active_trade["sl"] = dynamic_sl
        active_trade["step_level"] = 0
        active_trade["order_id"] = actual_order_id
        active_trade["option_id"] = opt_id
        active_trade["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active_trade["lot_size"] = inst["lot_size"]
        active_trade["exchange_segment_str"] = inst["exchange_segment_str"]
        active_trade["atm_strike"] = atm_strike
        save_state(active_trade)

    risk = abs(price - dynamic_sl)
    opt_type = "CE" if signal == "BUY" else "PE"
    logging.info(
        f">>> NEW TRADE: {inst['name']} {atm_strike} {opt_type} @ Premium ₹{option_entry_price} | Future: {price} | SL: {dynamic_sl}"
    )

    # Send standard trade alert
    # Calculate Option Name
    option_name = f"{inst['name']} {int(atm_strike)} {opt_type}"

    send_alert(
        f"🚀 **{option_name} ENTERED**\n"
        f"Option Premium: ₹{option_entry_price}\n"
        f"Future: {price}\n"
        f"SL: {dynamic_sl}\n"
        f"Risk: {risk} pts"
    )

    # Send signal alert to Signal Bot channel (with resistance-based targets)
    send_signal_alert(
        instrument=inst_key,
        strike=atm_strike,
        option_type=opt_type,
        entry_price=option_entry_price,
        stoploss=option_sl,
        signal=signal,
        df=df_15,
        future_price=price,
    )

    return True


def run_scanner(active_trade: Dict[str, Any], active_instrument: str) -> None:
    """Main scanner loop"""
    global _last_calendar_log
    logging.info(
        ">>> Scanner Started (Multi-Instrument Mode)"
        if MULTI_SCAN_ENABLED
        else ">>> Scanner Started (Single Instrument)"
    )

    # Track last config reload time
    last_config_reload = datetime.now()
    config_reload_interval = 60  # Reload config every 60 seconds

    # Signal files for dashboard integration
    from pathlib import Path

    DATA_DIR = Path(__file__).parent
    MANUAL_TRADE_SIGNAL_FILE = DATA_DIR / "manual_trade_signal.json"
    EMERGENCY_EXIT_SIGNAL_FILE = DATA_DIR / "emergency_exit_signal.json"

    while not socket_handler.is_shutdown():
        try:
            # === RELOAD CONFIG PERIODICALLY ===
            if (
                datetime.now() - last_config_reload
            ).total_seconds() >= config_reload_interval:
                config.reload_trading_config()
                last_config_reload = datetime.now()
                logging.debug("📝 Trading config reloaded")

            # === CHECK FOR EMERGENCY EXIT SIGNAL ===
            if EMERGENCY_EXIT_SIGNAL_FILE.exists() and active_trade["status"]:
                try:
                    with open(EMERGENCY_EXIT_SIGNAL_FILE, "r") as f:
                        exit_signal = json.load(f)

                    logging.warning(
                        f"🚨 EMERGENCY EXIT requested by {exit_signal.get('requested_by', 'dashboard')}"
                    )

                    # Construct Option Name for Alert
                    inst_key = active_trade.get("instrument")
                    if inst_key:
                        inst_name = INSTRUMENTS.get(inst_key, {}).get("name", inst_key)
                        atm_strike = active_trade.get("atm_strike", 0)
                        opt_type = (
                            "CE" if active_trade.get("type", "BUY") == "BUY" else "PE"
                        )
                        option_name = (
                            f"{inst_name} {int(atm_strike)} {opt_type}"
                            if atm_strike
                            else f"{inst_name} {opt_type}"
                        )
                    else:
                        option_name = "Unknown Position"

                    send_alert(
                        f"🚨 **EMERGENCY EXIT** requested via dashboard\nPosition: {option_name}"
                    )

                    # Import manager to close the trade
                    from manager import place_exit_order

                    exit_success = place_exit_order(active_trade, "EMERGENCY_EXIT")

                    if exit_success:
                        logging.info("✅ Emergency exit completed successfully")
                    else:
                        logging.error(
                            "❌ Emergency exit order failed - manual intervention needed!"
                        )
                        send_alert(
                            "🚨 **CRITICAL**: Emergency exit failed! Check positions manually."
                        )

                    # Remove the signal file
                    EMERGENCY_EXIT_SIGNAL_FILE.unlink()
                except Exception as e:
                    logging.error(f"Error processing emergency exit: {e}")
                    # Try to remove corrupted signal file
                    try:
                        EMERGENCY_EXIT_SIGNAL_FILE.unlink()
                    except:
                        pass

            # === CHECK FOR MANUAL TRADE SIGNAL ===
            if MANUAL_TRADE_SIGNAL_FILE.exists() and not active_trade["status"]:
                try:
                    with open(MANUAL_TRADE_SIGNAL_FILE, "r") as f:
                        manual_signal = json.load(f)

                    # Prevent Stale Signals
                    signal_timestamp_str = manual_signal.get("timestamp")
                    if signal_timestamp_str:
                        try:
                            sig_time = datetime.fromisoformat(signal_timestamp_str)
                            if (
                                datetime.now() - sig_time
                            ).total_seconds() > 300:  # 5 minutes
                                logging.warning(
                                    f"⚠️ Discarding STALE manual signal from {signal_timestamp_str}"
                                )
                                MANUAL_TRADE_SIGNAL_FILE.unlink()
                                continue
                        except ValueError:
                            pass  # Proceed if timestamp format is invalid (fallback)

                    inst_key = manual_signal.get("instrument")
                    signal = manual_signal.get("signal")

                    if inst_key and signal and inst_key in INSTRUMENTS:
                        logging.info(
                            f"📝 MANUAL TRADE signal received: {inst_key} {signal}"
                        )

                        inst = INSTRUMENTS[inst_key]
                        price = manual_signal.get("future_price", 0)
                        atm_strike = manual_signal.get("atm_strike", 0)

                        # Construct option name for alert
                        opt_type = "CE" if signal == "BUY" else "PE"
                        option_name = (
                            f"{inst['name']} {int(atm_strike)} {opt_type}"
                            if atm_strike
                            else f"{inst['name']} {opt_type}"
                        )

                        send_alert(
                            f"📝 **MANUAL TRADE** signal received\n{option_name}\nSignal: {signal}"
                        )

                        # Get option ID
                        opt_id = get_atm_option(
                            signal,
                            price,
                            cast(str, inst["exchange_segment_str"]),
                            cast(str, inst["future_id"]),
                            cast(str, inst["expiry_date"]),
                            cast(str, inst["option_type"]),
                            cast(int, inst["strike_step"]),
                            underlying=inst_key,
                        )

                        if opt_id:
                            # Check margin
                            margin_ok, margin_msg = check_margin_available(
                                opt_id,
                                cast(str, inst["exchange_segment_str"]),
                                cast(int, inst["lot_size"]),
                            )

                            if margin_ok:
                                # Get instrument data for proper SL calculation
                                df_15, df_60 = get_instrument_data(inst_key)
                                if df_15 is None:
                                    df_15 = pd.DataFrame()  # Empty df as fallback

                                # Execute the manual trade
                                trade_executed = execute_trade_entry(
                                    inst_key=inst_key,
                                    signal=signal,
                                    price=price,
                                    opt_id=opt_id,
                                    df_15=df_15,
                                    active_trade=active_trade,
                                    atm_strike=atm_strike,
                                )

                                if trade_executed:
                                    logging.info(
                                        f"✅ Manual trade executed: {inst_key} {signal}"
                                    )
                                else:
                                    logging.error(f"❌ Manual trade execution failed")
                                    send_alert(
                                        f"❌ **MANUAL TRADE FAILED** ({option_name})\n{inst_key} {signal}"
                                    )
                            else:
                                target_type = "CE" if signal == "BUY" else "PE"
                                option_name = (
                                    f"{inst_key} {int(atm_strike)} {target_type}"
                                )
                                logging.warning(
                                    f"❌ Manual trade skipped - insufficient margin: {margin_msg}"
                                )
                                send_alert(
                                    f"⚠️ **MANUAL TRADE SKIPPED** ({option_name})\n{margin_msg}"
                                )
                        else:
                            logging.error(f"❌ Could not find option for manual trade")
                            send_alert(
                                f"❌ **MANUAL TRADE FAILED** ({option_name})\nCould not find option contract"
                            )

                    # Remove the signal file regardless of outcome
                    MANUAL_TRADE_SIGNAL_FILE.unlink()

                except Exception as e:
                    logging.error(f"Error processing manual trade signal: {e}")
                    # Try to remove corrupted signal file
                    try:
                        MANUAL_TRADE_SIGNAL_FILE.unlink()
                    except:
                        pass

            # === PRE-TRADE CHECKS (GENERAL) ===

            # Check 0: Economic calendar / News filter
            if ECONOMIC_CALENDAR_AVAILABLE and _economic_calendar:
                should_pause, pause_event = _economic_calendar.should_pause_trading()
                if should_pause:
                    assert pause_event is not None
                    logging.info(
                        f"📰 Trading paused due to economic event: {pause_event.name}"
                    )
                    # Log upcoming events periodically (once per hour)
                    if (
                        _last_calendar_log is None
                        or (datetime.now() - _last_calendar_log).total_seconds() > 3600
                    ):
                        upcoming = _economic_calendar.get_upcoming_events(hours_ahead=4)
                        if upcoming:
                            logging.info(
                                f"📅 Upcoming high-impact events ({len(upcoming)}):"
                            )
                            for evt in upcoming[:3]:
                                logging.info(f"   - {evt.name} @ {evt.timestamp}")
                        _last_calendar_log = datetime.now()
                    time.sleep(60)  # Check again in 1 minute
                    continue

            # Check 1: Daily limits
            within_limits, limits_msg = check_daily_limits()
            if not within_limits:
                logging.warning(f"🛑 {limits_msg}")
                send_alert(f"🛑 **TRADING STOPPED**\n{limits_msg}")
                time.sleep(300)
                continue

            # Check 2: Cooldown after loss (using SignalTracker)
            in_cooldown, cooldown_msg = _signal_tracker.is_in_loss_cooldown(
                COOLDOWN_AFTER_LOSS
            )
            if in_cooldown:
                logging.debug(f"⏳ {cooldown_msg}")
                time.sleep(30)
                continue

            # === NO ACTIVE TRADE - SCAN FOR OPPORTUNITIES ===
            if not active_trade["status"]:

                if MULTI_SCAN_ENABLED:
                    signals = scan_all_instruments()

                    if not signals:
                        time.sleep(60)
                        continue

                    for signal_info in signals:
                        inst_key = signal_info["instrument"]
                        signal = signal_info["signal"]
                        price = signal_info["price"]
                        df_15 = signal_info["df_15"]

                        # Check signal cooldown per-instrument (prevents whipsaw on same instrument)
                        in_signal_cooldown, signal_msg = (
                            _signal_tracker.is_in_signal_cooldown(
                                signal, SIGNAL_COOLDOWN, instrument=inst_key
                            )
                        )
                        if in_signal_cooldown:
                            logging.info(f"⏳ {inst_key}: {signal_msg}")
                            continue

                        inst = INSTRUMENTS[inst_key]

                        # Calculate ATM strike for the signal alert
                        atm_strike = (
                            round(price / inst["strike_step"]) * inst["strike_step"]
                        )

                        opt_id = get_atm_option(
                            signal,
                            price,
                            cast(str, inst["exchange_segment_str"]),
                            cast(str, inst["future_id"]),
                            cast(str, inst["expiry_date"]),
                            cast(str, inst["option_type"]),
                            cast(int, inst["strike_step"]),
                            underlying=inst_key,
                        )

                        if not opt_id:
                            logging.warning(f"❌ {inst_key}: Could not find ATM option")
                            continue

                        margin_ok, margin_msg = check_margin_available(
                            opt_id,
                            cast(str, inst["exchange_segment_str"]),
                            cast(int, inst["lot_size"]),
                        )
                        if not margin_ok:
                            target_type = "CE" if signal == "BUY" else "PE"
                            option_name = f"{inst_key} {int(atm_strike)} {target_type}"
                            logging.warning(f"💰 {inst_key}: {margin_msg}")
                            send_alert(
                                f"⚠️ **TRADE SKIPPED** ({option_name})\n{margin_msg}"
                            )
                            update_last_signal(signal, instrument=inst_key)
                            continue

                        logging.info(f"💰 {inst_key}: {margin_msg}")

                        trade_executed = execute_trade_entry(
                            inst_key=inst_key,
                            signal=signal,
                            price=price,
                            opt_id=opt_id,
                            df_15=df_15,
                            active_trade=active_trade,
                            atm_strike=atm_strike,
                        )

                        if trade_executed:
                            break

                else:
                    # Single instrument mode
                    inst = INSTRUMENTS[active_instrument]

                    market_open, market_msg = is_market_open(
                        cast(str, inst["market_start"]), cast(str, inst["market_end"])
                    )
                    if not market_open:
                        logging.debug(f"⏰ {market_msg}")
                        time.sleep(60)
                        continue

                    can_trade, trade_msg = can_place_new_trade(
                        cast(str, inst["no_new_trade_after"])
                    )
                    if not can_trade:
                        logging.debug(f"⏰ {trade_msg}")
                        time.sleep(60)
                        continue

                    df_15, df_60 = get_resampled_data(
                        cast(str, inst.get("future_id", "")),
                        cast(str, inst.get("exchange_segment_str", "")),
                        cast(str, inst.get("instrument_type", "FUTCOM")),
                    )

                    if df_15 is not None and df_60 is not None:
                        signal_data: Optional[Dict[str, Any]] = (
                            analyze_instrument_signal(active_instrument, df_15, df_60)
                        )

                        if signal_data is not None:
                            signal = signal_data["signal"]
                            price = signal_data["price"]

                            # Check signal cooldown per-instrument (prevents whipsaw on same instrument)
                            in_signal_cooldown, signal_msg = (
                                _signal_tracker.is_in_signal_cooldown(
                                    signal,
                                    SIGNAL_COOLDOWN,
                                    instrument=active_instrument,
                                )
                            )
                            if in_signal_cooldown:
                                logging.info(f"⏳ {active_instrument}: {signal_msg}")
                                time.sleep(60)
                                continue

                            # Calculate ATM strike for the signal alert
                            atm_strike = (
                                round(price / inst["strike_step"]) * inst["strike_step"]
                            )

                            opt_id = get_atm_option(
                                signal,
                                price,
                                cast(str, inst["exchange_segment_str"]),
                                cast(str, inst["future_id"]),
                                cast(str, inst["expiry_date"]),
                                cast(str, inst["option_type"]),
                                cast(int, inst["strike_step"]),
                                underlying=active_instrument,
                            )

                            if opt_id:
                                margin_ok, margin_msg = check_margin_available(
                                    opt_id,
                                    cast(str, inst["exchange_segment_str"]),
                                    cast(int, inst["lot_size"]),
                                )
                                if not margin_ok:
                                    target_type = "CE" if signal == "BUY" else "PE"
                                    option_name = f"{active_instrument} {int(atm_strike)} {target_type}"
                                    logging.warning(
                                        f"💰 {active_instrument}: {margin_msg}"
                                    )
                                    send_alert(
                                        f"⚠️ **TRADE SKIPPED** ({option_name})\n{margin_msg}"
                                    )
                                    update_last_signal(
                                        signal, instrument=active_instrument
                                    )
                                    time.sleep(60)
                                    continue

                                logging.info(f"💰 {margin_msg}")

                                execute_trade_entry(
                                    inst_key=active_instrument,
                                    signal=signal,
                                    price=price,
                                    opt_id=opt_id,
                                    df_15=df_15,
                                    active_trade=active_trade,
                                    atm_strike=atm_strike,
                                )

            time.sleep(60)
        except Exception as e:
            logging.error(f"Scanner: {e}")
            time.sleep(60)


def get_dhan_client() -> dhanhq:
    """Get the dhan client instance"""
    return dhan


def get_trade_lock() -> threading.Lock:
    """Get the trade lock"""
    return trade_lock
