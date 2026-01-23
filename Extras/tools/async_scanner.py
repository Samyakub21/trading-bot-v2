# =============================================================================
# ASYNC SCANNER - Concurrent Instrument Scanning with async/await
# =============================================================================
# Provides async versions of API calls for better performance

import asyncio
import aiohttp
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, cast
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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
    send_alert,
    is_instrument_market_open,
    can_instrument_trade_new,
)

# Dhan client for sync operations (some can't be async)
from dhanhq import dhanhq

dhan = dhanhq(config.CLIENT_ID, config.ACCESS_TOKEN)

# Thread pool for CPU-bound pandas operations
_executor = ThreadPoolExecutor(max_workers=4)


# =============================================================================
# ASYNC DATA FETCHING
# =============================================================================


async def fetch_instrument_data_async(
    instrument_key: str,
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Async wrapper to fetch resampled data for a specific instrument.
    Returns tuple of (instrument_key, df_15, df_60)
    """
    loop = asyncio.get_event_loop()

    try:
        inst = INSTRUMENTS[instrument_key]
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

        # Run blocking API call in thread pool
        data = await loop.run_in_executor(
            _executor,
            partial(
                dhan.intraday_minute_data,
                inst["future_id"],
                inst["exchange_segment_str"],
                inst["instrument_type"],
                from_date,
                to_date,
            ),
        )

        if data["status"] == "failure":
            return instrument_key, None, None

        # Process data in thread pool (pandas is CPU-bound)
        df_15, df_60 = await loop.run_in_executor(
            _executor, partial(_process_candle_data, data["data"])
        )

        return instrument_key, df_15, df_60

    except Exception as e:
        logging.error(f"Async data error for {instrument_key}: {e}")
        return instrument_key, None, None


def _process_candle_data(raw_data: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process raw candle data into 15min and 60min DataFrames"""
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

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


async def analyze_instrument_async(
    instrument_key: str, df_15: pd.DataFrame, df_60: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """
    Async wrapper for instrument signal analysis.
    """
    loop = asyncio.get_event_loop()

    try:
        # Run analysis in thread pool (pandas_ta is CPU-bound)
        result = await loop.run_in_executor(
            _executor, partial(_analyze_signal, instrument_key, df_15, df_60)
        )
        return result

    except Exception as e:
        logging.error(f"Async analysis error for {instrument_key}: {e}")
        return None


def _analyze_signal(
    instrument_key: str, df_15: pd.DataFrame, df_60: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """Analyze instrument and return signal info if in trade zone"""
    try:
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

        # Volume confirmation
        volume_confirmed = (
            current_volume >= (avg_volume * VOLUME_MULTIPLIER)
            if avg_volume > 0
            else True
        )

        signal = None
        signal_strength = 0

        ema_50 = trend["EMA_50"]
        trend_close = trend["close"]

        # BULLISH Signal
        if (
            (trend_close > ema_50)
            and (trigger["close"] > vwap_val)
            and (rsi_val > RSI_BULLISH_THRESHOLD)
            and volume_confirmed
        ):
            signal = "BUY"
            signal_strength = (rsi_val - RSI_BULLISH_THRESHOLD) + (
                (trend_close - ema_50) / ema_50 * 100
            )
            if avg_volume > 0:
                signal_strength += (current_volume / avg_volume - 1) * 10

        # BEARISH Signal
        elif (
            (trend_close < ema_50)
            and (trigger["close"] < vwap_val)
            and (rsi_val < RSI_BEARISH_THRESHOLD)
            and volume_confirmed
        ):
            signal = "SELL"
            signal_strength = (RSI_BEARISH_THRESHOLD - rsi_val) + (
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
                "df_15": df_15,
            }

        return None

    except Exception as e:
        logging.error(f"Analysis error for {instrument_key}: {e}")
        return None


# =============================================================================
# ASYNC SCANNING
# =============================================================================


async def scan_all_instruments_async() -> List[Dict[str, Any]]:
    """
    Asynchronously scan all configured instruments in parallel.
    Returns list of signal dictionaries sorted by priority and strength.
    """
    instruments_to_scan = get_instruments_to_scan()
    signals_found = []

    logging.info(
        f"🔍 Async scanning {len(instruments_to_scan)} instruments: {', '.join(instruments_to_scan)}"
    )

    # Filter instruments by market hours first
    scannable_instruments = []
    for inst_key in instruments_to_scan:
        market_open, market_msg = is_instrument_market_open(inst_key)
        if not market_open:
            logging.debug(f"   ⏰ {inst_key}: {market_msg}")
            continue

        can_trade, trade_msg = can_instrument_trade_new(inst_key)
        if not can_trade:
            logging.debug(f"   ⏰ {inst_key}: {trade_msg}")
            continue

        scannable_instruments.append(inst_key)

    if not scannable_instruments:
        logging.info("   No instruments currently tradeable")
        return []

    # Fetch all data concurrently
    fetch_tasks = [
        fetch_instrument_data_async(inst_key) for inst_key in scannable_instruments
    ]

    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Analyze instruments with valid data
    analysis_tasks = []
    for fetch_result in fetch_results:
        if isinstance(fetch_result, Exception):
            logging.error(f"Fetch exception: {fetch_result}")
            continue

        fetch_result = cast(
            Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]], fetch_result
        )
        inst_key, df_15, df_60 = fetch_result
        if df_15 is None or df_60 is None:
            logging.debug(f"   ❌ {inst_key}: No data available")
            continue

        analysis_tasks.append(analyze_instrument_async(inst_key, df_15, df_60))

    # Run all analyses concurrently
    analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

    # Collect valid signals
    for analysis_result in analysis_results:
        if isinstance(analysis_result, Exception):
            logging.error(f"Analysis exception: {analysis_result}")
            continue

        analysis_result = cast(Optional[Dict[str, Any]], analysis_result)
        if analysis_result is not None:
            signals_found.append(analysis_result)
            signal_type = (
                "📈 BULLISH" if analysis_result["signal"] == "BUY" else "📉 BEARISH"
            )
            logging.info(
                f"   ✅ {analysis_result['instrument']}: {signal_type} | RSI: {analysis_result['rsi']:.1f} | Strength: {analysis_result['signal_strength']:.1f}"
            )

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


async def get_option_chain_async(
    exchange_segment_str: str, future_id: str, expiry_date: str, option_type: str
) -> Optional[Dict]:
    """Async wrapper for option chain fetch"""
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            _executor,
            partial(
                dhan.option_chain,
                exchange_segment_str,
                future_id,
                expiry_date,
                option_type,
            ),
        )
        return result
    except Exception as e:
        logging.error(f"Async option chain error: {e}")
        return None


async def check_margin_async(
    option_id: str, exchange_segment_str: str, lot_size: int
) -> Tuple[bool, str]:
    """Async wrapper for margin check"""
    loop = asyncio.get_event_loop()

    try:
        # Fetch funds and margin in parallel
        funds_task = loop.run_in_executor(_executor, dhan.get_fund_limits)
        margin_task = loop.run_in_executor(
            _executor,
            partial(
                dhan.margin_calculator,
                security_id=option_id,
                exchange_segment=exchange_segment_str,
                transaction_type="BUY",
                quantity=lot_size,
                product_type="INTRADAY",
                price=0,
            ),
        )

        funds, margin_response = await asyncio.gather(funds_task, margin_task)

        if funds.get("status") == "failure":
            return False, "Could not fetch fund limits"

        fund_data = funds.get("data", {})
        available_balance = float(fund_data.get("availabelBalance", 0))

        if margin_response.get("status") == "success":
            required_margin = float(
                margin_response.get("data", {}).get("totalMargin", 0)
            )

            if available_balance >= required_margin:
                return (
                    True,
                    f"Margin OK: Available ₹{available_balance:.2f} >= Required ₹{required_margin:.2f}",
                )
            else:
                return (
                    False,
                    f"Insufficient margin: Available ₹{available_balance:.2f} < Required ₹{required_margin:.2f}",
                )
        else:
            if available_balance >= 10000:
                return (
                    True,
                    f"Balance OK: ₹{available_balance:.2f} (margin calc unavailable)",
                )
            else:
                return False, f"Low balance: ₹{available_balance:.2f}"

    except Exception as e:
        logging.error(f"Async margin check error: {e}")
        return True, f"Margin check failed: {e} (proceeding with caution)"


# =============================================================================
# ASYNC RUNNER
# =============================================================================


def run_async_scan() -> List[Dict[str, Any]]:
    """
    Run async scan from synchronous code.
    Creates event loop if needed.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context, create new loop
            import nest_asyncio

            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(scan_all_instruments_async())


async def scan_with_timeout(timeout: float = 30.0) -> List[Dict[str, Any]]:
    """
    Run scan with timeout protection.
    """
    try:
        return await asyncio.wait_for(scan_all_instruments_async(), timeout=timeout)
    except asyncio.TimeoutError:
        logging.warning(f"⚠️ Async scan timed out after {timeout}s")
        return []


# =============================================================================
# BATCH OPERATIONS
# =============================================================================


async def fetch_multiple_option_chains(
    instruments: List[str],
) -> Dict[str, Optional[Dict]]:
    """
    Fetch option chains for multiple instruments in parallel.
    """
    tasks = []

    for inst_key in instruments:
        inst = INSTRUMENTS.get(inst_key)
        if inst:
            tasks.append(
                get_option_chain_async(
                    str(inst["exchange_segment_str"]),
                    str(inst["future_id"]),
                    str(inst["expiry_date"]),
                    str(inst["option_type"]),
                )
            )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        inst_key: (
            cast(Optional[Dict], result) if not isinstance(result, Exception) else None
        )
        for inst_key, result in zip(instruments, results)
    }


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================


class ScanMetrics:
    """Track scanning performance metrics"""

    def __init__(self):
        self.total_scans = 0
        self.successful_scans = 0
        self.failed_scans = 0
        self.total_time_ms = 0
        self.last_scan_time_ms = 0

    def record_scan(self, duration_ms: float, success: bool):
        """Record a scan result"""
        self.total_scans += 1
        self.total_time_ms += duration_ms
        self.last_scan_time_ms = duration_ms

        if success:
            self.successful_scans += 1
        else:
            self.failed_scans += 1

    @property
    def avg_scan_time_ms(self) -> float:
        """Average scan time in milliseconds"""
        if self.total_scans == 0:
            return 0
        return self.total_time_ms / self.total_scans

    @property
    def success_rate(self) -> float:
        """Success rate as percentage"""
        if self.total_scans == 0:
            return 0
        return (self.successful_scans / self.total_scans) * 100


# Global metrics instance
scan_metrics = ScanMetrics()


async def timed_scan() -> Tuple[List[Dict[str, Any]], float]:
    """
    Run scan and return results with timing.
    """
    import time

    start = time.perf_counter()

    try:
        results = await scan_all_instruments_async()
        elapsed_ms = (time.perf_counter() - start) * 1000
        scan_metrics.record_scan(elapsed_ms, True)
        return results, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        scan_metrics.record_scan(elapsed_ms, False)
        logging.error(f"Timed scan failed: {e}")
        return [], elapsed_ms
