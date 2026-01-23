# =============================================================================
# SCANNER - Market Scanning and Signal Analysis
# =============================================================================

try:
    import matplotlib

    # Fix for CI/Headless environments to prevent "matplotlib.__spec__ is not set"
    matplotlib.use("Agg")
except Exception:
    matplotlib = None

import json
import logging
import time
import threading
import requests
import requests
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
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
    send_high_priority_alert,
    dhan_intraday_minute_data,
    dhan_place_order,
    dhan_get_positions,
    dhan_get_order_by_id,
    dhan_cancel_order,
    dhan_get_fund_limits,
    dhan_quote_data,
    dhan_option_chain,
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
from contract_updater import (
    load_scrip_master,
    find_current_month_future,
    save_contract_cache,
)
import socket_handler
from state_stores import get_signal_tracker

# Create named logger for scanner
logger = logging.getLogger("Scanner")


def _safe_format(val: Any, precision: int = 2) -> str:
    try:
        if val is None:
            return "N/A"
        return f"{float(val):.{precision}f}"
    except Exception:
        return "N/A"


_DATA_CACHE: Dict[str, Dict[str, Any]] = {}
# Last tick time for relaxed logging
LAST_TICK_TIME: Dict[str, datetime] = {}
_LAST_INSTRUMENT_STATES: Dict[str, Dict[str, bool]] = {}
LTP_HEARTBEAT_STARTED = False


def start_ltp_heartbeat(interval_seconds: int = 60) -> None:
    """Start a background daemon thread that logs a concise LTP summary every minute.

    Logs a single line like: "LTP HEARTBEAT: CRUDEOIL=5276.00@2025-12-24 09:02:00, NATGASMINI=297.60@..."
    Uses `socket_handler.INSTRUMENT_LTP` protected by `socket_handler.LTP_LOCK`.
    """
    global LTP_HEARTBEAT_STARTED
    if LTP_HEARTBEAT_STARTED:
        return

    def _heartbeat() -> None:
        while True:
            try:
                with socket_handler.LTP_LOCK:
                    ltp_map = getattr(socket_handler, "INSTRUMENT_LTP", {}) or {}
                    parts: List[str] = []
                    for inst in INSTRUMENTS.keys():
                        try:
                            entry = ltp_map.get(inst)
                            if not entry:
                                parts.append(f"{inst}=n/a")
                                continue

                            ltp_val = entry.get("ltp")
                            last_update = entry.get("last_update")

                            # Safe timestamp formatting
                            try:
                                ts = (
                                    last_update.strftime("%Y-%m-%d %H:%M:%S")
                                    if isinstance(last_update, datetime)
                                    else str(last_update)
                                )
                            except Exception:
                                ts = str(last_update)

                            # Try to coerce LTP to float for consistent formatting
                            formatted = None
                            try:
                                if ltp_val is None:
                                    formatted = f"{inst}=n/a@{ts}"
                                else:
                                    ltp_float = float(ltp_val)
                                    formatted = f"{inst}={ltp_float:.2f}@{ts}"
                            except Exception:
                                # Fallback to string representation if numeric cast fails
                                formatted = f"{inst}={ltp_val}@{ts}"

                            parts.append(formatted)
                        except Exception:
                            # Ensure one bad instrument doesn't kill the heartbeat
                            try:
                                parts.append(f"{inst}=error")
                            except Exception:
                                # last resort: skip
                                pass
                heartbeat_msg = (
                    "LTP HEARTBEAT: " + ", ".join(parts)
                    if parts
                    else "LTP HEARTBEAT: no LTP data"
                )
                logger.info(heartbeat_msg)
            except Exception:
                logger.exception("LTP heartbeat error")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_heartbeat, name="LTP-Heartbeat", daemon=True)
    thread.start()
    LTP_HEARTBEAT_STARTED = True


# Do not start heartbeat at import time to avoid circular import with `socket_handler`.
# Heartbeat will be started lazily when scanning begins.



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
            future_id = inst.get("future_id", "")  # Ensure future_id is fetched
            exchange_segment_str = inst.get(
                "exchange_segment_str", ""
            )  # Ensure exchange_segment_str is fetched
            instrument_type = inst.get(
                "instrument_type", "FUTCOM"
            )  # Ensure instrument_type is fetched
            log_context = instrument_key
            cache_key = instrument_key
        else:
            if not all([future_id, exchange_segment_str, instrument_type]):
                return None, None
            log_context = f"future_id={future_id}"
            cache_key = str(future_id)

        # Check if we have fresh cached data from live updates
        if cache_key in _DATA_CACHE:
            entry = _DATA_CACHE[cache_key]
            # Support legacy format where cache stored a DataFrame directly
            if isinstance(entry, dict) and "last_timestamp" in entry and "df" in entry:
                cached_data = entry
            elif isinstance(entry, pd.DataFrame):
                cached_data = {"df": entry, "last_timestamp": entry.index.max()}
                # Normalize cache in-place for future reads
                _DATA_CACHE[cache_key] = cached_data
            else:
                # Unknown cache format - drop it
                try:
                    del _DATA_CACHE[cache_key]
                except Exception:
                    pass
                cached_data = None

            if cached_data is not None:
                time_since_update = (
                    datetime.now() - cached_data["last_timestamp"]
                ).total_seconds()
                if (
                    cached_data["last_timestamp"].date() == datetime.now().date()
                    and time_since_update < 3600
                ):  # Cache is from today and updated within 1 hour
                    df = cached_data["df"].copy()

                    # Append latest tick from websocket buffer if available
                    import socket_handler

                    with socket_handler.LTP_LOCK:
                        latest_ltp_data = socket_handler.INSTRUMENT_LTP.get(
                            instrument_key if instrument_key else cache_key
                        )
                    if latest_ltp_data and latest_ltp_data.get("ltp", 0) > 0:
                        ltp = latest_ltp_data["ltp"]
                        last_update = latest_ltp_data["last_update"]

                        # Only append if the update is newer than the last cached data
                        if last_update > cached_data["last_timestamp"]:
                            now = last_update.replace(second=0, microsecond=0)

                            # Ensure index is DatetimeIndex
                            if not isinstance(df.index, pd.DatetimeIndex):
                                df.index = pd.to_datetime(df.index, errors="coerce")
                                df = df[~df.index.isna()]
                                if df.empty or not isinstance(
                                    df.index, pd.DatetimeIndex
                                ):
                                    logger.warning(
                                        f"Skipping live tick append for {log_context} - invalid cached index"
                                    )
                                else:
                                    if now in df.index:
                                        df.at[now, "close"] = ltp
                                        df.at[now, "high"] = max(
                                            df.at[now, "high"], ltp
                                        )
                                        df.at[now, "low"] = min(df.at[now, "low"], ltp)
                                    else:
                                        # Create a new minute candle
                                        new_row = pd.DataFrame(
                                            {
                                                "open": ltp,
                                                "high": ltp,
                                                "low": ltp,
                                                "close": ltp,
                                                "volume": 0,
                                            },
                                            index=[now],
                                        )
                                        df = pd.concat([df, new_row])
                                        df.sort_index(inplace=True)

                    logger.info(
                        f"Using cached data with live tick for {log_context} (updated {time_since_update:.0f}s ago)"
                    )
                    return df, df

        to_date = datetime.now().strftime("%Y-%m-%d")

        if cache_key in _DATA_CACHE:
            from_date = to_date
        else:
            from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

        # Enhanced data fetching with failover
        data = None
        max_retries = 2  # Primary fetch + 1 retry
        retry_delay = 2.0  # 2 seconds delay for retry

        for attempt in range(max_retries):
            try:
                data = dhan_intraday_minute_data(
                    dhan,
                    security_id=future_id,
                    exchange_segment=exchange_segment_str,
                    instrument_type=instrument_type,
                    from_date=from_date,
                    to_date=to_date,
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"⚠️ Data fetch attempt {attempt + 1} failed for {log_context}: {e}. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"❌ Data fetch failed permanently for {log_context} after {max_retries} attempts: {e}"
                    )
                    raise

        if data.get("status") == "failure":
            return None, None

        raw_data = data.get("data", data)
        print(
            f"DEBUG: Raw data type: {type(raw_data)}, has 'open' key: {'open' in raw_data if isinstance(raw_data, dict) else 'N/A'}"
        )

        if isinstance(raw_data, dict) and "open" in raw_data:
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
            # Robust timestamp parsing: handle epoch seconds or mixed formats.
            # Prefer epoch-seconds when numeric, else fallback to parsing strings.
            try:
                if pd.api.types.is_numeric_dtype(df["timestamp"]):
                    s = pd.to_datetime(
                        df["timestamp"], unit="s", errors="coerce", utc=True
                    )
                else:
                    # Try numeric conversion first (some APIs return numbers as strings)
                    numeric_ts = pd.to_numeric(df["timestamp"], errors="coerce")
                    if numeric_ts.notna().any():
                        s = pd.to_datetime(
                            numeric_ts, unit="s", errors="coerce", utc=True
                        )
                    else:
                        s = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                # Convert to local timezone and make naive datetime (drop tz info)
                s_local_str = s.dt.tz_convert("Asia/Kolkata").dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                df["timestamp"] = pd.to_datetime(s_local_str, errors="coerce")
            except Exception:
                # Fallback: best-effort parse without timezone conversion
                df["timestamp"] = pd.to_datetime(
                    df.get("timestamp", None), errors="coerce"
                )
        elif isinstance(raw_data, list):
            df = pd.DataFrame(raw_data)
            df.rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "start_time": "timestamp",
                },
                inplace=True,
            )
            # Robust parsing for legacy payloads: coerce mixed formats, convert to Asia/Kolkata
            try:
                s = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
                s_local_str = s.dt.tz_convert("Asia/Kolkata").dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                df["timestamp"] = pd.to_datetime(s_local_str, errors="coerce")
            except Exception:
                df["timestamp"] = pd.to_datetime(
                    df.get("timestamp", None), errors="coerce"
                )
        else:
            return None, None

        if df.empty:
            return None, None

        # Ensure there is a time-like column available
        if "timestamp" not in df.columns:
            # common alternatives
            for alt in ("timestamp", "start_Time", "start_time", "startTime"):
                if alt in df.columns:
                    df["timestamp"] = df[alt]
                    break
            else:
                candidates = [c for c in df.columns if "time" in c.lower()]
                if candidates:
                    df["timestamp"] = df[candidates[0]]
                else:
                    logger.error(
                        f"Data Error for {log_context}: no time-like column in payload: {list(df.columns)}"
                    )
                    return None, None

        # Removed verbose diagnostic logs (columns/sample rows) per user request.
        df.set_index("timestamp", inplace=True)
        # Removed debug prints
        # Ensure index is a valid DatetimeIndex and normalized
        try:
            # First, ensure we have a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")

            # Remove rows with invalid timestamps
            df = df[~df.index.isna()]

            if df.empty:
                logger.error(
                    f"Data Error for {log_context}: All timestamps are invalid"
                )
                return None, None

            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error(
                    f"Data Error for {log_context}: Data type/value error - index is not a valid DatetimeIndex. Index type: {type(df.index)}, dtype: {df.index.dtype}, shape: {df.shape}"
                )
                return None, None

            # Normalize timezone to local (remove tz info if present)
            df.index = (
                df.index.tz_localize(None)
                if hasattr(df.index, "tz") and df.index.tz is not None
                else df.index
            )
            df.sort_index(inplace=True)
        except Exception as e:
            logger.error(f"Data Error for {log_context}: Data type/value error - {e}")
            return None, None

        # Stale data detection - check if latest tick is older than 5 minutes
        latest_timestamp = df.index.max()
        time_diff = datetime.now() - latest_timestamp.to_pydatetime()
        if time_diff.total_seconds() > 300:  # 5 minutes
            logger.warning(
                f"⚠️ Stale data detected for {log_context}: latest tick is {time_diff.total_seconds()/60:.1f} minutes old. Skipping analysis."
            )
            return None, None

        if cache_key in _DATA_CACHE:
            # Support both legacy (DataFrame) and normalized (dict) cache entries
            cached_entry = _DATA_CACHE[cache_key]
            if isinstance(cached_entry, dict) and "df" in cached_entry:
                cached_df = cached_entry["df"]
            elif isinstance(cached_entry, pd.DataFrame):
                cached_df = cached_entry
                # Normalize stored cache
                _DATA_CACHE[cache_key] = {
                    "df": cached_df,
                    "last_timestamp": cached_df.index.max(),
                }
            else:
                cached_df = None

            if cached_df is not None:
                df = pd.concat([cached_df, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df[df.index >= (datetime.now() - timedelta(days=30))]

        # Store cache as a dict with dataframe and last timestamp (consistent structure)
        _DATA_CACHE[cache_key] = {"df": df, "last_timestamp": df.index.max()}

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

    except Exception as e:
        logger.error(f"Data Error for {log_context}: {e}")
        return None, None


def get_resampled_data(
    future_id: str, exchange_segment_str: str, instrument_type: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    return get_instrument_data(
        future_id=future_id,
        exchange_segment_str=exchange_segment_str,
        instrument_type=instrument_type,
    )


def analyze_instrument_signal(
    instrument_key: str, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
) -> Optional[Dict[str, Any]]:
    # Ensure DataFrames are prepared for time-series analysis
    for df in [df_15, df_60]:
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

    if USE_STRATEGY_PATTERN and STRATEGIES_AVAILABLE:
        try:
            inst_config = INSTRUMENTS.get(instrument_key, {})
            strategy_name = cast(Optional[str], inst_config.get("strategy"))
            strategy_params = cast(
                Optional[Dict[str, Any]], inst_config.get("strategy_params", {})
            )

            strategy = get_strategy(instrument_key, strategy_name, strategy_params)
            signal_info = strategy.analyze(df_15.copy(), df_60.copy(), **kwargs)

            if signal_info:
                logger.debug(f"[{strategy.name}] {instrument_key}: Signal generated")
            return signal_info
        except Exception as e:
            logger.error(f"Strategy error for {instrument_key}: {e}")

    # Legacy Logic
    try:
        inst_config = INSTRUMENTS.get(instrument_key, {})
        inst_params = inst_config.get("strategy_params", {})
        rsi_bullish = inst_params.get("rsi_bullish_threshold", RSI_BULLISH_THRESHOLD)
        rsi_bearish = inst_params.get("rsi_bearish_threshold", RSI_BEARISH_THRESHOLD)
        volume_mult = inst_params.get("volume_multiplier", VOLUME_MULTIPLIER)

        # EMA 50
        df_60["EMA_50"] = EMAIndicator(close=df_60["close"], window=50).ema_indicator()
        # Anchored VWAP
        df_15["tp"] = (df_15["high"] + df_15["low"] + df_15["close"]) / 3
        df_15["vp"] = df_15["tp"] * df_15["volume"]
        df_15["VWAP_D"] = (
            df_15.groupby(df_15.index.date)["vp"].cumsum()
            / df_15.groupby(df_15.index.date)["volume"].cumsum()
        )
        # RSI
        df_15["RSI"] = RSIIndicator(close=df_15["close"], window=14).rsi()
        df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

        trend = df_60.iloc[-2]
        # Use last 15m bar if it's recent (near real-time), otherwise use previous closed bar
        time_diff = datetime.now() - df_15.index.max().to_pydatetime()
        trigger_idx = -1 if time_diff.total_seconds() < 90 else -2
        trigger = df_15.iloc[trigger_idx]

        price = trigger["close"]
        vwap_val = trigger.get("VWAP_D", 0)
        current_volume = trigger["volume"]
        avg_volume = trigger.get("vol_avg", current_volume)
        rsi_val = trigger["RSI"]

        volume_confirmed = (
            current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True
        )
        signal = None
        signal_strength = 0.0
        ema_50 = trend["EMA_50"]
        trend_close = trend["close"]

        if (
            (trend_close > ema_50)
            and (trigger["close"] > vwap_val)
            and (rsi_val > rsi_bullish)
            and volume_confirmed
        ):
            signal = "BUY"
            signal_strength = float(
                (rsi_val - rsi_bullish) + ((trend_close - ema_50) / ema_50 * 100)
            )
        elif (
            (trend_close < ema_50)
            and (trigger["close"] < vwap_val)
            and (rsi_val < rsi_bearish)
            and volume_confirmed
        ):
            signal = "SELL"
            signal_strength = float(
                (rsi_bearish - rsi_val) + ((ema_50 - trend_close) / ema_50 * 100)
            )

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
        logger.error(f"Analysis error for {instrument_key}: {e}")
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
        chain = dhan_option_chain(
            dhan,
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
                    except ValueError:
                        continue

            if strike_key in option_chain_dict:
                strike_data = option_chain_dict[strike_key]
                option_data = strike_data.get(target_key, {})
                security_id = option_data.get("security_id") or option_data.get(
                    "securityId"
                )
                if security_id:
                    return str(security_id)

        if underlying:
            return find_option_from_scrip_master(
                underlying, atm_strike, target, expiry_date, exchange_segment_str
            )
        return None
    except Exception as e:
        logger.error(f"Error in get_atm_option: {e}")
        return None


def check_margin_available(
    option_id: str,
    exchange_segment_str: str,
    lot_size: int,
    instrument: Optional[str] = None,
) -> Tuple[bool, str]:
    try:
        funds = dhan_get_fund_limits(dhan)
        if funds.get("status") == "failure":
            return False, "Could not fetch fund limits"

        fund_data = funds.get("data", {})
        available_balance = float(
            fund_data.get("availableBalance", 0)
            or fund_data.get("availabelBalance", 0)
            or 0
        )

        # Casting option_id to int for quote_data if possible
        try:
            quote_id = int(option_id)
            quote_segment = exchange_segment_str.replace("_COMM", "")
            if quote_segment == "MCX":
                quote_segment = "MCX_COMM"

            quote_response = dhan_quote_data(dhan, {quote_segment: [quote_id]})
            if quote_response.get("status") == "success":
                quote_data = quote_response.get("data", {}).get("data", {})
                option_quote = quote_data.get(str(option_id)) or quote_data.get(
                    quote_id, {}
                )
                option_ltp = float(
                    option_quote.get("last_price", 0) or option_quote.get("LTP", 0) or 0
                )

                if option_ltp > 0:
                    required = option_ltp * lot_size * 1.05
                    inst_label = instrument if instrument else "UNKNOWN"
                    logger.info(
                        f"MARGIN [{inst_label}]: Required ₹{required:.0f} | Available ₹{available_balance:.0f} | Status: {'OK' if available_balance >= required else 'INSUFFICIENT'}"
                    )
                    if available_balance >= required:
                        return True, f"Margin OK: {available_balance}"
                    else:
                        return False, f"Insufficient: {available_balance} < {required}"
        except Exception:
            pass

        # Fallback logic
        return True, "Margin check skipped (fallback)"
    except Exception as e:
        return True, f"Margin check error: {e}"


# (Keeping verify_order, execute_trade_entry, run_scanner as is, but fixing run_scanner typing)





# =============================================================================
# STRATEGY PATTERN SUPPORT
# STRATEGY PATTERN SUPPORT
# =============================================================================
USE_STRATEGY_PATTERN = True

try:
    from strategies import get_strategy, get_available_strategies

    STRATEGIES_AVAILABLE = True
except ImportError:
    STRATEGIES_AVAILABLE = False
    USE_STRATEGY_PATTERN = False
    logger.warning("strategies.py not found. Using legacy signal analysis.")

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
    logger.info("economic_calendar.py not found. News filtering disabled.")

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
                logger.error(
                    "Data Error: Must provide either instrument_key or all of (future_id, exchange_segment_str, instrument_type)"
                )
                return None, None
            log_context = f"future_id={future_id}"
            cache_key = cast(str, future_id)

        # V2 API: intraday_minute_data uses different parameters
        # Format: security_id, exchange_segment, instrument_type, from_date, to_date
        to_date = datetime.now().strftime("%Y-%m-%d")

        # Always fetch full history (30 days) for merging
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        cached_df = (
            _DATA_CACHE.get(cache_key, {}).get("df")
            if cache_key in _DATA_CACHE
            else None
        )

        # V2 API call for intraday minute data
        # Use robust wrapper from utils to get retries, backoff and consistent errors
        data = dhan_intraday_minute_data(
            dhan,
            security_id=future_id,
            exchange_segment=exchange_segment_str,
            instrument_type=instrument_type,
            from_date=from_date,
            to_date=to_date,
        )

        if data.get("status") == "failure":
            error_msg = data.get("remarks", data.get("errorMessage", "Unknown error"))
            logger.warning(
                f"Data fetch failure for {log_context} (future_id: {future_id}): {error_msg} - attempting rollover/retry"
            )

            # Attempt automatic rollover if using an instrument key and contract expired
            if instrument_key is not None:
                try:
                    contracts = load_scrip_master()
                    new_contract = find_current_month_future(
                        instrument_key, exchange_segment_str, contracts
                    )
                    if new_contract:
                        new_id = new_contract.get("SEM_SMST_SECURITY_ID")
                        new_expiry = new_contract.get("SEM_EXPIRY_DATE")
                        logger.info(
                            f"Auto-rollover: Updating {instrument_key} to future_id={new_id}, expiry={new_expiry}"
                        )
                        # Update in-memory instrument config so subsequent calls use it
                        INSTRUMENTS[instrument_key]["future_id"] = new_id
                        INSTRUMENTS[instrument_key]["expiry_date"] = new_expiry

                        # Persist updated instruments to contract cache so other processes pick up new future_id
                        try:
                            save_contract_cache(INSTRUMENTS)
                            logger.info(
                                f"Persisted updated contract cache after rollover for {instrument_key}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to persist contract cache after rollover: {e}"
                            )

                        # Clear any stale cached data for this instrument before retry
                        if cache_key in _DATA_CACHE:
                            try:
                                del _DATA_CACHE[cache_key]
                                logger.debug(
                                    f"Cleared stale _DATA_CACHE for {cache_key} after rollover"
                                )
                            except Exception:
                                pass

                        # Retry data fetch once with new contract
                        data = dhan_intraday_minute_data(
                            dhan,
                            security_id=new_id,
                            exchange_segment=exchange_segment_str,
                            instrument_type=instrument_type,
                            from_date=from_date,
                            to_date=to_date,
                        )

                        if data.get("status") == "failure":
                            logger.error(
                                f"Data Error after rollover for {instrument_key} (future_id: {new_id}): {data.get('remarks', '')}"
                            )
                            return None, None
                    else:
                        logger.warning(
                            f"No rollover contract found for {instrument_key}"
                        )
                        return None, None
                except Exception as e:
                    logger.error(f"Rollover attempt failed for {instrument_key}: {e}")
                    return None, None
            else:
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
            # Manual timezone conversion to Asia/Kolkata
            import pytz

            utc_tz = pytz.UTC
            ist_tz = pytz.timezone("Asia/Kolkata")

            def convert_timestamp(ts):
                if pd.isna(ts):
                    return ts
                # Convert to UTC first if naive
                if ts.tzinfo is None:
                    ts = utc_tz.localize(ts)
                # Convert to IST
                ts = ts.astimezone(ist_tz)
                # Remove timezone info
                return ts.replace(tzinfo=None)

            df["time"] = df["time"].apply(convert_timestamp)
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
            df["time"] = (
                df["time"]
                .tz_localize("UTC")
                .tz_convert("Asia/Kolkata")
                .tz_localize(None)
            )
        else:
            logger.error(f"Data Error for {log_context}: Unexpected data format")
            return None, None

        if df.empty:
            logger.error(f"Data Error for {log_context}: Empty dataframe")
            return None, None

        if "time" not in df.columns:
            for alt in ("timestamp", "start_Time", "start_time", "startTime"):
                if alt in df.columns:
                    df["time"] = df[alt]
                    break
            else:
                candidates = [c for c in df.columns if "time" in c.lower()]
                if candidates:
                    df["time"] = df[candidates[0]]
                else:
                    logger.error(
                        f"Data Error for {log_context}: no time-like column in merged payload: {list(df.columns)}"
                    )
                    return None, None

        # Removed merged payload diagnostic logs per user request.

        df.set_index("time", inplace=True)

        # Data Merging and Caching
        new_candles = 0
        if cached_df is not None:
            old_len = len(cached_df)
            # Append new data to cached data
            df = pd.concat([cached_df, df])
            # Drop duplicates (timestamps) to ensure clean data
            df = df[~df.index.duplicated(keep="last")]
            new_candles = len(df) - old_len
        else:
            new_candles = len(df)

        # Keep rolling history of ~30 trading days
        df = df[df.index >= (datetime.now() - timedelta(days=30))]

        # Update latest candle with real-time LTP if available
        if (
            instrument_key
            and hasattr(socket_handler, "INSTRUMENT_LTP")
            and instrument_key in socket_handler.INSTRUMENT_LTP
        ):
            with socket_handler.LTP_LOCK:
                latest_time = df.index.max()
                current_minute = datetime.now().replace(second=0, microsecond=0)
                if latest_time == current_minute:
                    live_ltp = socket_handler.INSTRUMENT_LTP[instrument_key].get(
                        "ltp", 0
                    )
                    if live_ltp > 0:
                        df.loc[latest_time, "close"] = live_ltp
                        logger.debug(
                            f"Updated {instrument_key} latest candle close to live LTP: {live_ltp}"
                        )

        # Stale data detection with relaxed check for commodities (after merging)
        latest_timestamp = df.index.max()
        time_diff = datetime.now() - latest_timestamp.to_pydatetime()
        # If data is not from today, consider it stale
        if latest_timestamp.date() != datetime.now().date():
            if (
                instrument_key
                and instrument_key in LAST_TICK_TIME
                and (datetime.now() - LAST_TICK_TIME[instrument_key]).total_seconds()
                < 60
            ):
                logger.debug(
                    f"⚠️ Data from previous day for {log_context}: latest tick is {time_diff.total_seconds()/60:.1f} minutes old. Proceeding with live data."
                )
            else:
                logger.warning(
                    f"⚠️ Data from previous day for {log_context}: latest tick is {time_diff.total_seconds()/60:.1f} minutes old. Skipping analysis."
                )
                return None, None
        max_allowed_lag = (
            1800 if exchange_segment_str == "MCX_COMM" else 300
        )  # 30 mins for MCX, 5 mins for others
        if time_diff.total_seconds() > max_allowed_lag:
            if (
                instrument_key
                and instrument_key in LAST_TICK_TIME
                and (datetime.now() - LAST_TICK_TIME[instrument_key]).total_seconds()
                < 60
            ):
                logger.debug(
                    f"⚠️ Data truly stale for {log_context}: latest tick is {time_diff.total_seconds()/60:.1f} minutes old. Proceeding with live data."
                )
            else:
                logger.warning(
                    f"⚠️ Data truly stale for {log_context}: latest tick is {time_diff.total_seconds()/60:.1f} minutes old. Skipping analysis."
                )
                return None, None

        # Update Cache with DataFrame and last timestamp (only if new data is fresher)
        if (
            cache_key not in _DATA_CACHE
            or df.index.max() > _DATA_CACHE[cache_key]["last_timestamp"]
        ):
            _DATA_CACHE[cache_key] = {"df": df, "last_timestamp": df.index.max()}

        # Debug log for data freshness
        logger.debug(
            f"[{log_context}] Appended {new_candles} new minute candles, latest: {df.index.max()}"
        )

        df_15 = (
            df.resample("15min", label="left", closed="left")
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
            df.resample("60min", label="left", closed="left")
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
        logger.error(f"Data Error for {log_context}: Missing key in response - {e}")
        return None, None
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logger.error(f"Data Error for {log_context}: DataFrame parsing failed - {e}")
        return None, None
    except requests.RequestException as e:
        logger.error(f"Data Error for {log_context}: Network request failed - {e}")
        return None, None
    except (TypeError, ValueError) as e:
        logger.error(
            f"Data Error for {log_context}: Data type/value error - {e} (type: {type(e).__name__})"
        )
        return None, None
    except Exception as e:
        logger.error(
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
    instrument_key: str, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
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
            signal_info = strategy.analyze(df_15.copy(), df_60.copy(), **kwargs)

            if signal_info:
                logger.debug(f"[{strategy.name}] {instrument_key}: Signal generated")
            return signal_info

        except Exception as e:
            logger.error(f"Strategy error for {instrument_key}: {e}")
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
        df_60["EMA_50"] = EMAIndicator(close=df_60["close"], window=50).ema_indicator()
        # Anchored VWAP
        df_15["tp"] = (df_15["high"] + df_15["low"] + df_15["close"]) / 3
        df_15["vp"] = df_15["tp"] * df_15["volume"]
        df_15["VWAP_D"] = (
            df_15.groupby(df_15.index.date)["vp"].cumsum()
            / df_15.groupby(df_15.index.date)["volume"].cumsum()
        )
        # RSI
        df_15["RSI"] = RSIIndicator(close=df_15["close"], window=14).rsi()
        df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

        trend = df_60.iloc[-2]
        # Use last 15m bar if it's recent (near real-time), otherwise use previous closed bar
        time_diff = datetime.now() - df_15.index.max().to_pydatetime()
        trigger_idx = -1 if time_diff.total_seconds() < 90 else -2
        trigger = df_15.iloc[trigger_idx]

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
        logger.error(f"Analysis error for {instrument_key}: Missing data column - {e}")
        return None
    except IndexError as e:
        logger.error(
            f"Analysis error for {instrument_key}: Insufficient data rows - {e}"
        )
        return None
    except (TypeError, ValueError) as e:
        logger.error(f"Analysis error for {instrument_key}: Calculation error - {e}")
        return None
    except Exception as e:
        logger.error(
            f"Analysis error for {instrument_key}: Unexpected error - {type(e).__name__}: {e}"
        )
        return None


def scan_all_instruments() -> List[Dict[str, Any]]:
    """Scan all configured instruments and return those in trade zone"""
    # Ensure LTP heartbeat is running when scanning starts (deferred start to avoid circular import)
    try:
        start_ltp_heartbeat()
    except Exception:
        logger.exception("Failed to start LTP heartbeat")
    instruments_to_scan = get_instruments_to_scan()

    # Filter to only scan instruments that are currently in market hours
    instruments_to_scan = [
        inst for inst in instruments_to_scan if is_instrument_market_open(inst)[0]
    ]

    signals_found: List[Dict[str, Any]] = []
    scannable_count = 0

    logger.info(
        f"🔍 Scanning instruments in market hours: {', '.join(instruments_to_scan)}"
    )

    for inst_key in instruments_to_scan:
        # Initialize per-instrument state tracking
        if inst_key not in _LAST_INSTRUMENT_STATES:
            _LAST_INSTRUMENT_STATES[inst_key] = {
                "market_open": None,
                "can_trade": None,
            }

        # Since we filtered, market is open
        _LAST_INSTRUMENT_STATES[inst_key]["market_open"] = True

        # Check if new trades are allowed
        can_trade, trade_msg = can_instrument_trade_new(inst_key)
        prev_can_trade = _LAST_INSTRUMENT_STATES[inst_key].get("can_trade")
        if not can_trade:
            if prev_can_trade is not False:
                logger.info(f"SKIP [{inst_key}]: Trading restricted - {trade_msg}")
            else:
                logger.debug(
                    f"SKIP [{inst_key}]: Trading restricted - {trade_msg} (repeated)"
                )
            _LAST_INSTRUMENT_STATES[inst_key]["can_trade"] = False
            continue
        else:
            if prev_can_trade is False:
                logger.info(f"RESUME [{inst_key}]: Trading allowed")
            _LAST_INSTRUMENT_STATES[inst_key]["can_trade"] = True

        scannable_count += 1

        # Get data for this instrument
        time.sleep(1)
        df_15, df_60 = get_instrument_data(inst_key)
        if df_15 is None or df_60 is None:
            logger.info(f"SKIP [{inst_key}]: No data available")
            continue

        # Data freshness check: warn if latest candle is older than 20 minutes, but continue for swing trading
        if df_15.index.max() < datetime.now() - timedelta(minutes=20):
            logger.warning(
                f"[{inst_key}]: Data stale (latest: {df_15.index.max()}), proceeding with available data"
            )
            # Continue anyway for swing trading

        # Analyze for signals
        kwargs = {}
        if inst_key == "FINNIFTY":
            # Get BANKNIFTY data for correlation check
            try:
                banknifty_config = INSTRUMENTS.get("BANKNIFTY", {})
                if banknifty_config:
                    banknifty_df_15, banknifty_df_60 = get_instrument_data(
                        future_id=str(banknifty_config.get("future_id", "")),
                        exchange_segment_str=str(banknifty_config.get(
                            "exchange_segment_str", ""
                        )),
                        instrument_type=str(banknifty_config.get("instrument_type", "")),
                    )
                    if banknifty_df_60 is not None:
                        kwargs["banknifty_df_60"] = banknifty_df_60
            except Exception as e:
                logger.warning(
                    f"Could not get BANKNIFTY data for FINNIFTY correlation: {e}"
                )

        signal_info = analyze_instrument_signal(inst_key, df_15, df_60, **kwargs)
        if signal_info:
            signals_found.append(signal_info)
            signal_type = (
                "📈 BULLISH" if signal_info["signal"] == "BUY" else "📉 BEARISH"
            )
            # Use safe formatting to avoid exceptions when values are missing or non-numeric
            price_s = _safe_format(signal_info.get("price"), 2)
            rsi_s = _safe_format(signal_info.get("rsi"), 1)
            adx_s = _safe_format(signal_info.get("adx"), 1)
            vwap_s = _safe_format(signal_info.get("vwap"), 2)
            atr_s = _safe_format(signal_info.get("atr"), 2)
            strength_s = _safe_format(signal_info.get("signal_strength"), 1)

            logger.info(
                f"SIGNAL [{inst_key}]: {signal_type} | Price: ₹{price_s} | RSI: {rsi_s} | ADX: {adx_s} | VWAP: {vwap_s} | ATR: {atr_s} | Strength: {strength_s}"
            )
        else:
            logger.debug(f"SKIP [{inst_key}]: No signal generated")

    # Log summary of scannable instruments
    if scannable_count == 0:
        logger.info("📊 No instruments are currently in market hours")
    else:
        logger.info(f"📊 {scannable_count} instrument(s) are in market hours")

    # Sort by priority first, then by signal strength
    if signals_found:
        signals_found.sort(
            key=lambda x: (
                INSTRUMENT_PRIORITY.get(x["instrument"], 99),
                -x["signal_strength"],
            )
        )
        logger.info(f"📊 Found {len(signals_found)} instrument(s) in trade zone")

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
            logger.warning("Could not load scrip master for option lookup")
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
            logger.warning(
                f"No option found in scrip master for {underlying} {strike_price} {option_type}"
            )
            return None

        # Sort by expiry (nearest first) and return the nearest one
        matching_options.sort(key=lambda x: x["expiry"])
        nearest_option = matching_options[0]

        logger.debug(
            f"Found option from scrip master: {nearest_option['symbol']} (exp: {nearest_option['expiry']}) -> ID: {nearest_option['security_id']}"
        )
        return str(nearest_option["security_id"])

    except Exception as e:
        logger.error(f"Error finding option from scrip master: {e}")
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
    # Fix float/int types
    atm_strike = round(current_price / strike_step) * strike_step
    target = "CE" if transaction_type == "BUY" else "PE"
    target_key = "ce" if target == "CE" else "pe"

    try:
        chain = dhan_option_chain(
            dhan,
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
                    except ValueError:
                        continue

            if strike_key in option_chain_dict:
                strike_data = option_chain_dict[strike_key]
                option_data = strike_data.get(target_key, {})
                security_id = option_data.get("security_id") or option_data.get(
                    "securityId"
                )
                if security_id:
                    return str(security_id)

        if underlying:
            return find_option_from_scrip_master(
                underlying, atm_strike, target, expiry_date, exchange_segment_str
            )
        return None
    except Exception as e:
        logger.error(f"Error in get_atm_option: {e}")
        return None





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
                logger.info(f"📝 [PAPER TRADING] {action} Order verified: {order_id}")
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
            logger.error(f"[{action}] Order response is None")
            return False, None

        if order_response.get("status") == "failure":
            error_msg = order_response.get("remarks", "Unknown error")
            logger.error(f"[{action}] Order FAILED: {error_msg}")
            send_alert(f"❌ **ORDER FAILED** ({action}) {symbol_display}\n{error_msg}")
            return False, None

        order_id = order_response.get("data", {}).get("orderId")
        if not order_id:
            logger.error(f"[{action}] Could not get order ID from response")
            return False, None

        logger.info(f"[{action}] Order placed successfully. Order ID: {order_id}")

        # Initial delay before first status check
        time.sleep(config["initial_delay"])

        start_time = time.time()

        for attempt in range(cast(int, config["max_retries"])):
            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed >= config["total_timeout"]:
                logger.warning(
                    f"[{action}] Total timeout ({config['total_timeout']}s) exceeded after {attempt} attempts"
                )
                break

            order_status = dhan_get_order_by_id(dhan, order_id)

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
                    logger.info(
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
                    logger.error(f"[{action}] Order {status}: {reason}")
                    send_alert(
                        f"❌ **ORDER {status}** ({action}) {symbol_display}\n{reason}"
                    )
                    return False, None

                elif status in ["PENDING", "OPEN"]:
                    remaining_time = config["total_timeout"] - elapsed
                    logger.debug(
                        f"[{action}] Order still {status}, attempt {attempt + 1}/{config['max_retries']} "
                        f"(timeout in {remaining_time:.1f}s)"
                    )

                    # Apply exponential backoff before next retry
                    if attempt < config["max_retries"] - 1:
                        delay = _wait_with_backoff(attempt, config)
                        logger.debug(
                            f"[{action}] Waiting {delay:.1f}s before next check..."
                        )
            else:
                logger.warning(f"[{action}] Failed to fetch order status, retrying...")
                if attempt < config["max_retries"] - 1:
                    _wait_with_backoff(attempt, config)

        # Order not filled within timeout - cancel it
        logger.warning(
            f"[{action}] Order not filled in time ({config['total_timeout']}s). Cancelling order {order_id}"
        )
        _cancel_unfilled_order(order_id, action)
        return False, None

    except Exception as e:
        logger.error(f"[{action}] Order verification error: {e}")
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
        cancel_response = dhan_cancel_order(dhan, order_id)
        if cancel_response and cancel_response.get("status") == "success":
            logger.info(f"[{action}] Unfilled order {order_id} cancelled successfully")
            send_alert(
                f"⚠️ **ORDER CANCELLED** ({action})\nOrder {order_id} did not fill in time"
            )
            return True
        else:
            logger.error(
                f"[{action}] Failed to cancel order {order_id}: {cancel_response}"
            )
            send_alert(
                f"🚨 **CRITICAL**: Failed to cancel unfilled order {order_id}. Manual intervention required!"
            )
            return False
    except Exception as cancel_error:
        logger.error(f"[{action}] Error cancelling order {order_id}: {cancel_error}")
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
    signal_info: Optional[Dict[str, Any]] = None,
) -> bool:
    """Execute a trade entry for a specific instrument"""
    inst = INSTRUMENTS[inst_key]

    # Calculate Option Name for Alerts
    opt_type_str = "CE" if signal == "BUY" else "PE"
    option_name = f"{inst['name']} {int(atm_strike)} {opt_type_str}"

    # Check-Before-Entry: Verify no existing position for this instrument at broker
    try:
        positions_response = dhan_get_positions(dhan)
        if positions_response and positions_response.get("status") == "success":
            positions_data = positions_response.get("data", [])
            for position in positions_data:
                if position.get("security_id") == opt_id:
                    logger.warning(
                        f"⚠️ Position already exists for {option_name} at broker. Skipping duplicate trade entry."
                    )
                    send_high_priority_alert(
                        f"🚨 **DUPLICATE TRADE PREVENTION**\n"
                        f"Attempted to enter {option_name} but position already exists at broker.\n"
                        f"Security ID: {opt_id}\n"
                        f"Local state may be out of sync - manual verification required!"
                    )
                    return False
    except Exception as e:
        logger.error(
            f"❌ Failed to check existing positions before entry for {option_name}: {e}"
        )
        # Continue with trade entry but log the issue
        send_high_priority_alert(
            f"🚨 **POSITION CHECK FAILED**\n"
            f"Could not verify existing positions before entering {option_name}.\n"
            f"Error: {e}\n"
            f"Proceeding with trade entry - monitor closely!"
        )

    # Place order with LIMIT buffer
    limit_price = round(price * (1 + LIMIT_ORDER_BUFFER), 2)

    # Check for Paper Trading
    is_paper_trading = config.get_trading_param("PAPER_TRADING", False)

    if is_paper_trading:
        logger.info(
            f"📝 PAPER TRADING: Placing ENTRY order for {opt_id} @ ₹{limit_price}"
        )
        order_response = {
            "status": "success",
            "data": {"orderId": f"PAPER_{int(time.time()*1000)}", "price": limit_price},
        }
    else:
        order_response = dhan_place_order(
            dhan,
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
        logger.error(f"❌ {inst_key}: Entry order failed, skipping trade")
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

    # Use strategy-provided stop loss if available, otherwise calculate dynamically
    if signal_info and "stop_loss" in signal_info:
        dynamic_sl = signal_info["stop_loss"]
    else:
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
        active_trade["trailing_activated"] = False
        active_trade["exit_logic"] = (
            signal_info.get("exit_logic", {}) if signal_info else {}
        )
        save_state(active_trade)

    risk = abs(price - dynamic_sl)
    opt_type = "CE" if signal == "BUY" else "PE"
    logger.info(
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
    logger.info(
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
                logger.debug("📝 Trading config reloaded")

            # === CHECK FOR EMERGENCY EXIT SIGNAL ===
            if EMERGENCY_EXIT_SIGNAL_FILE.exists() and active_trade["status"]:
                try:
                    with open(EMERGENCY_EXIT_SIGNAL_FILE, "r") as f:
                        exit_signal = json.load(f)

                    logger.warning(
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
                        logger.info("✅ Emergency exit completed successfully")
                    else:
                        logger.error(
                            "❌ Emergency exit order failed - manual intervention needed!"
                        )
                        send_alert(
                            "🚨 **CRITICAL**: Emergency exit failed! Check positions manually."
                        )

                    # Remove the signal file
                    EMERGENCY_EXIT_SIGNAL_FILE.unlink()
                except Exception as e:
                    logger.error(f"Error processing emergency exit: {e}")
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
                                logger.warning(
                                    f"⚠️ Discarding STALE manual signal from {signal_timestamp_str}"
                                )
                                MANUAL_TRADE_SIGNAL_FILE.unlink()
                                continue
                        except ValueError:
                            pass  # Proceed if timestamp format is invalid (fallback)

                    inst_key = manual_signal.get("instrument")
                    signal = manual_signal.get("signal")

                    if inst_key and signal and inst_key in INSTRUMENTS:
                        logger.info(
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
                                inst_key,
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
                                    signal_info=None,  # Manual trade, no strategy signal info
                                )

                                if trade_executed:
                                    logger.info(
                                        f"✅ Manual trade executed: {inst_key} {signal}"
                                    )
                                else:
                                    logger.error(f"❌ Manual trade execution failed")
                                    send_alert(
                                        f"❌ **MANUAL TRADE FAILED** ({option_name})\n{inst_key} {signal}"
                                    )
                            else:
                                target_type = "CE" if signal == "BUY" else "PE"
                                option_name = (
                                    f"{inst_key} {int(atm_strike)} {target_type}"
                                )
                                logger.warning(
                                    f"❌ Manual trade skipped - insufficient margin: {margin_msg}"
                                )
                                send_alert(
                                    f"⚠️ **MANUAL TRADE SKIPPED** ({option_name})\n{margin_msg}"
                                )
                        else:
                            logger.error(f"❌ Could not find option for manual trade")
                            send_alert(
                                f"❌ **MANUAL TRADE FAILED** ({option_name})\nCould not find option contract"
                            )

                    # Remove the signal file regardless of outcome
                    MANUAL_TRADE_SIGNAL_FILE.unlink()

                except Exception as e:
                    logger.error(f"Error processing manual trade signal: {e}")
                    # Try to remove corrupted signal file
                    try:
                        MANUAL_TRADE_SIGNAL_FILE.unlink()
                    except:
                        pass

            # === PRE-TRADE CHECKS (GENERAL) ===

            # Check 0: Economic calendar / News filter
            if ECONOMIC_CALENDAR_AVAILABLE and _economic_calendar:
                enabled_instruments = config.ENABLED_INSTRUMENTS
                should_pause, pause_event = _economic_calendar.should_pause_trading(
                    enabled_instruments
                )
                if should_pause:
                    assert pause_event is not None
                    logger.info(
                        f"📰 Trading paused due to economic event: {pause_event.name}"
                    )
                    # Log upcoming events periodically (once per hour)
                    if (
                        _last_calendar_log is None
                        or (datetime.now() - _last_calendar_log).total_seconds() > 3600
                    ):
                        upcoming = _economic_calendar.get_upcoming_events(hours_ahead=4)
                        if upcoming:
                            logger.info(
                                f"📅 Upcoming high-impact events ({len(upcoming)}):"
                            )
                            for evt in upcoming[:3]:
                                logger.info(f"   - {evt.name} @ {evt.timestamp}")
                        _last_calendar_log = datetime.now()
                    time.sleep(60)  # Check again in 1 minute
                    continue

            # Check 1: Daily limits
            within_limits, limits_msg = check_daily_limits()
            if not within_limits:
                logger.warning(f"🛑 {limits_msg}")
                send_alert(f"🛑 **TRADING STOPPED**\n{limits_msg}")
                time.sleep(300)
                continue

            # Check 2: Cooldown after loss (using SignalTracker)
            in_cooldown, cooldown_msg = _signal_tracker.is_in_loss_cooldown(
                COOLDOWN_AFTER_LOSS
            )
            if in_cooldown:
                logger.debug(f"⏳ {cooldown_msg}")
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
                            logger.info(f"⏳ {inst_key}: {signal_msg}")
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
                            logger.warning(f"❌ {inst_key}: Could not find ATM option")
                            continue

                        margin_ok, margin_msg = check_margin_available(
                            opt_id,
                            cast(str, inst["exchange_segment_str"]),
                            cast(int, inst["lot_size"]),
                            inst_key,
                        )
                        if not margin_ok:
                            target_type = "CE" if signal == "BUY" else "PE"
                            option_name = f"{inst_key} {int(atm_strike)} {target_type}"
                            logger.warning(f"💰 {inst_key}: {margin_msg}")
                            send_alert(
                                f"⚠️ **TRADE SKIPPED** ({option_name})\n{margin_msg}"
                            )
                            update_last_signal(signal, instrument=inst_key)
                            continue

                        logger.info(f"💰 {inst_key}: {margin_msg}")

                        trade_executed = execute_trade_entry(
                            inst_key=inst_key,
                            signal=signal,
                            price=price,
                            opt_id=opt_id,
                            df_15=df_15,
                            active_trade=active_trade,
                            atm_strike=atm_strike,
                            signal_info=signal_info,
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
                        logger.info(
                            f"SKIP [{active_instrument}]: Market closed - {market_msg}"
                        )
                        time.sleep(60)
                        continue

                    can_trade, trade_msg = can_place_new_trade(
                        cast(str, inst["no_new_trade_after"])
                    )
                    if not can_trade:
                        logger.info(
                            f"SKIP [{active_instrument}]: Trading restricted - {trade_msg}"
                        )
                        time.sleep(60)
                        continue

                    df_15, df_60 = get_resampled_data(
                        cast(str, inst.get("future_id", "")),
                        cast(str, inst.get("exchange_segment_str", "")),
                        cast(str, inst.get("instrument_type", "FUTCOM")),
                    )

                    if df_15 is not None and df_60 is not None:
                        kwargs = {}
                        if active_instrument == "FINNIFTY":
                            # Get BANKNIFTY data for correlation check
                            try:
                                banknifty_config = INSTRUMENTS.get("BANKNIFTY", {})
                                if banknifty_config:
                                    banknifty_df_15, banknifty_df_60 = (
                                        get_instrument_data(
                                            future_id=banknifty_config.get("future_id"),
                                            exchange_segment_str=banknifty_config.get(
                                                "exchange_segment_str"
                                            ),
                                            instrument_type=banknifty_config.get(
                                                "instrument_type"
                                            ),
                                        )
                                    )
                                    if banknifty_df_60 is not None:
                                        kwargs["banknifty_df_60"] = banknifty_df_60
                            except Exception as e:
                                logger.warning(
                                    f"Could not get BANKNIFTY data for FINNIFTY correlation: {e}"
                                )

                        signal_data: Optional[Dict[str, Any]] = (
                            analyze_instrument_signal(
                                active_instrument, df_15, df_60, **kwargs
                            )
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
                                logger.info(f"⏳ {active_instrument}: {signal_msg}")
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
                                    logger.warning(
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

                                logger.info(f"💰 {margin_msg}")

                                execute_trade_entry(
                                    inst_key=active_instrument,
                                    signal=signal,
                                    price=price,
                                    opt_id=opt_id,
                                    df_15=df_15,
                                    active_trade=active_trade,
                                    atm_strike=atm_strike,
                                    signal_info=signal_data,
                                )

            time.sleep(60)
        except Exception as e:
            logger.error(f"Scanner: {e}")
            time.sleep(60)


def get_dhan_client() -> dhanhq:
    """Get the dhan client instance"""
    return dhan


def get_trade_lock() -> threading.Lock:
    """Get the trade lock"""
    return trade_lock
