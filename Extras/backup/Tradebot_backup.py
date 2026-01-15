import time
import threading
import logging
import json
import os
import pandas as pd
import pandas_ta as ta
import requests
import math
import asyncio
from datetime import datetime, timedelta
from dhanhq import dhanhq, marketfeed
from config import config  # Import configuration loader

# --- 1. CONFIGURATION ---
# Credentials are now loaded from config.py (environment variables or credentials.json)
CLIENT_ID = config.CLIENT_ID
ACCESS_TOKEN = config.ACCESS_TOKEN

# =============================================================================
# INSTRUMENT CONFIGURATIONS - Add/modify instruments here
# =============================================================================
INSTRUMENTS = {
    "CRUDEOIL": {
        "name": "CRUDE OIL",
        "exchange_segment_int": 5,  # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "464926",  # <--- UPDATE for current month future
        "lot_size": 10,
        "strike_step": 50,
        "expiry_date": "2026-01-16",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",  # MCX Crude trading hours
        "market_end": "23:30",
        "no_new_trade_after": "23:00",  # Stop new entries 30 min before close
    },
    "NATURALGAS": {
        "name": "NATURAL GAS",
        "exchange_segment_int": 5,  # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "465123",  # <--- UPDATE with actual ID
        "lot_size": 1250,
        "strike_step": 5,
        "expiry_date": "2026-01-27",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
    },
    "GOLD": {
        "name": "GOLD",
        "exchange_segment_int": 5,  # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "465200",  # <--- UPDATE with actual ID
        "lot_size": 10,
        "strike_step": 100,
        "expiry_date": "2026-02-05",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
    },
    "SILVER": {
        "name": "SILVER",
        "exchange_segment_int": 5,  # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "465300",  # <--- UPDATE with actual ID
        "lot_size": 30,
        "strike_step": 500,
        "expiry_date": "2026-02-05",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
    },
    "NIFTY": {
        "name": "NIFTY 50",
        "exchange_segment_int": 2,  # marketfeed.NSE_FNO
        "exchange_segment_str": "NSE_FNO",
        "future_id": "13",  # Nifty underlying ID
        "lot_size": 25,
        "strike_step": 50,
        "expiry_date": "2026-01-16",
        "instrument_type": "INDEX",
        "option_type": "OPTIDX",
        "market_start": "09:15",  # NSE trading hours
        "market_end": "15:30",
        "no_new_trade_after": "15:00",
    },
    "BANKNIFTY": {
        "name": "BANK NIFTY",
        "exchange_segment_int": 2,  # marketfeed.NSE_FNO
        "exchange_segment_str": "NSE_FNO",
        "future_id": "25",  # BankNifty underlying ID
        "lot_size": 15,
        "strike_step": 100,
        "expiry_date": "2026-01-15",
        "instrument_type": "INDEX",
        "option_type": "OPTIDX",
        "market_start": "09:15",
        "market_end": "15:30",
        "no_new_trade_after": "15:00",
    },
}

# =============================================================================
# SELECT ACTIVE INSTRUMENTS FOR SCANNING
# =============================================================================
# Set to None to scan ALL instruments, or provide a list like ["CRUDEOIL", "GOLD"]
SCAN_INSTRUMENTS = None  # None = scan all, or ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "NIFTY", "BANKNIFTY"]

# Default instrument (used when starting fresh or for state files)
DEFAULT_INSTRUMENT = "CRUDEOIL"

# Multi-instrument scanning mode
MULTI_SCAN_ENABLED = True  # Set to False to use single-instrument mode

# Priority order for instrument selection when multiple signals found
# Higher priority instruments are preferred (1=highest)
INSTRUMENT_PRIORITY = {
    "CRUDEOIL": 1,
    "GOLD": 2,
    "SILVER": 3,
    "NATURALGAS": 4,
    "NIFTY": 5,
    "BANKNIFTY": 6,
}


# Get list of instruments to scan
def get_instruments_to_scan():
    """Returns the list of instrument keys to scan"""
    if SCAN_INSTRUMENTS:
        return [k for k in SCAN_INSTRUMENTS if k in INSTRUMENTS]
    return list(INSTRUMENTS.keys())


# Current active instrument (will change when trade is taken)
ACTIVE_INSTRUMENT = DEFAULT_INSTRUMENT

# Load active instrument config (will be updated dynamically)
INSTRUMENT = INSTRUMENTS[ACTIVE_INSTRUMENT]
EXCHANGE_SEGMENT_INT = INSTRUMENT["exchange_segment_int"]
EXCHANGE_SEGMENT_STR = INSTRUMENT["exchange_segment_str"]
FUTURE_ID = INSTRUMENT["future_id"]
LOT_SIZE = INSTRUMENT["lot_size"]
STRIKE_STEP = INSTRUMENT["strike_step"]
EXPIRY_DATE = INSTRUMENT["expiry_date"]
INSTRUMENT_TYPE = INSTRUMENT["instrument_type"]
OPTION_TYPE = INSTRUMENT["option_type"]
MARKET_START = INSTRUMENT["market_start"]
MARKET_END = INSTRUMENT["market_end"]
NO_NEW_TRADE_AFTER = INSTRUMENT["no_new_trade_after"]

# State files - now instrument-aware with multi-instrument support
STATE_FILE = "trade_state_active.json"  # Single state file for active trade
DAILY_PNL_FILE = "daily_pnl_combined.json"  # Combined daily P&L tracking
TRADE_HISTORY_FILE = "trade_history_combined.json"  # Combined historical trade log
# Telegram credentials from config
TELEGRAM_TOKEN = config.TELEGRAM_TOKEN
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID

# =============================================================================
# RISK MANAGEMENT CONFIG
# =============================================================================
MAX_DAILY_LOSS = 5000  # Maximum loss per day in INR (stop trading after this)
MAX_TRADES_PER_DAY = 5  # Maximum number of trades per day
COOLDOWN_AFTER_LOSS = 300  # Wait 5 minutes after a loss before next trade
SIGNAL_COOLDOWN = 900  # 15 minutes cooldown for same direction signal (avoid whipsaw)
AUTO_SQUARE_OFF_BUFFER = (
    5  # Minutes before market close to auto square-off (avoid broker penalty)
)

# SIGNAL STRENGTH CONFIG
RSI_BULLISH_THRESHOLD = (
    60  # RSI must be above this for bullish signal (stronger filter)
)
RSI_BEARISH_THRESHOLD = (
    40  # RSI must be below this for bearish signal (stronger filter)
)
VOLUME_MULTIPLIER = 1.2  # Volume must be 1.2x average for signal confirmation

# GLOBAL VARS
LATEST_LTP = 0  # Future/Underlying LTP
OPTION_LTP = 0  # Option Premium LTP
LAST_TICK_TIME = datetime.now()
LAST_OPTION_TICK_TIME = datetime.now()
LAST_LOSS_TIME = None  # Track time of last loss for cooldown
LAST_SIGNAL = None  # Track last signal direction ("BUY" or "SELL")
LAST_SIGNAL_TIME = None  # Track time of last signal for whipsaw prevention
dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)

# Multi-instrument LTP storage
INSTRUMENT_LTP = {}  # {instrument_key: {"ltp": price, "last_update": datetime}}

# Threading lock for safe active_trade access (Fix: Global Variable Trap)
trade_lock = threading.Lock()
instrument_lock = threading.Lock()  # Lock for instrument switching

# Limit order buffer (Fix: Market Order Slippage)
LIMIT_ORDER_BUFFER = 0.01  # 1% buffer for limit orders

# Socket reconnection control
SOCKET_RECONNECT_EVENT = threading.Event()  # Signal socket to reconnect
SOCKET_HEALTHY = threading.Event()  # Indicates socket is receiving data
SHUTDOWN_EVENT = threading.Event()  # Signal graceful shutdown
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)

# --- 2. HELPERS ---


def is_market_open():
    """Check if market is currently open for trading"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")

    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False, "Market closed (Weekend)"

    if current_time < MARKET_START:
        return False, f"Market not open yet (opens at {MARKET_START})"

    if current_time > MARKET_END:
        return False, f"Market closed (closed at {MARKET_END})"

    return True, "Market Open"


def can_place_new_trade():
    """Check if new trades are allowed (time restriction before market close)"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")

    if current_time > NO_NEW_TRADE_AFTER:
        return False, f"No new trades after {NO_NEW_TRADE_AFTER}"

    return True, "New trades allowed"


def load_daily_pnl():
    """Load daily P&L data, reset if it's a new day"""
    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(DAILY_PNL_FILE):
        try:
            with open(DAILY_PNL_FILE, "r") as f:
                data = json.load(f)
                # Reset if it's a new day
                if data.get("date") != today:
                    return {
                        "date": today,
                        "pnl": 0,
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                    }
                return data
        except:
            pass

    return {"date": today, "pnl": 0, "trades": 0, "wins": 0, "losses": 0}


def save_daily_pnl(data):
    """Save daily P&L data"""
    with open(DAILY_PNL_FILE, "w") as f:
        json.dump(data, f)


def load_trade_history():
    """Load historical trade data"""
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return []


def save_trade_to_history(trade_data):
    """Append trade to historical log"""
    history = load_trade_history()
    history.append(trade_data)

    with open(TRADE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def get_performance_stats(days=None):
    """Calculate performance statistics

    Args:
        days: Number of days to analyze (None for all time)

    Returns:
        dict: Performance metrics
    """
    history = load_trade_history()

    if not history:
        return None

    # Filter by date if specified
    if days:
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        history = [t for t in history if t.get("exit_time", "") >= cutoff_date]

    if not history:
        return None

    total_trades = len(history)
    wins = [t for t in history if t["pnl"] > 0]
    losses = [t for t in history if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in history)
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_win = (gross_profit / len(wins)) if wins else 0
    avg_loss = (gross_loss / len(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # R-multiple stats
    r_multiples = [t.get("r_multiple", 0) for t in history if "r_multiple" in t]
    avg_r = (sum(r_multiples) / len(r_multiples)) if r_multiples else 0

    # Best and worst trades
    best_trade = max(history, key=lambda x: x["pnl"])["pnl"] if history else 0
    worst_trade = min(history, key=lambda x: x["pnl"])["pnl"] if history else 0

    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in history:
        if trade["pnl"] > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

    return {
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_r_multiple": avg_r,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
        "expectancy": (avg_win * win_rate / 100) - (avg_loss * (100 - win_rate) / 100),
    }


def display_performance_report(days=None):
    """Display formatted performance report"""
    stats = get_performance_stats(days)

    if not stats:
        period_str = f"last {days} days" if days else "all time"
        logging.info(f"📊 No trade history available for {period_str}")
        return

    period_str = f"Last {days} Days" if days else "All Time"

    logging.info("=" * 60)
    logging.info(f"📊 PERFORMANCE REPORT - {period_str}")
    logging.info("=" * 60)
    logging.info(
        f"Total Trades: {stats['total_trades']} | Wins: {stats['wins']} | Losses: {stats['losses']}"
    )
    logging.info(
        f"Win Rate: {stats['win_rate']:.1f}% | Avg R-Multiple: {stats['avg_r_multiple']:.2f}R"
    )
    logging.info("-" * 60)
    logging.info(f"Total P&L: ₹{stats['total_pnl']:.2f}")
    logging.info(
        f"Gross Profit: ₹{stats['gross_profit']:.2f} | Gross Loss: ₹{stats['gross_loss']:.2f}"
    )
    logging.info(f"Profit Factor: {stats['profit_factor']:.2f}")
    logging.info("-" * 60)
    logging.info(
        f"Avg Win: ₹{stats['avg_win']:.2f} | Avg Loss: ₹{stats['avg_loss']:.2f}"
    )
    logging.info(
        f"Best Trade: ₹{stats['best_trade']:.2f} | Worst Trade: ₹{stats['worst_trade']:.2f}"
    )
    logging.info("-" * 60)
    logging.info(
        f"Max Consecutive Wins: {stats['max_consecutive_wins']} | Losses: {stats['max_consecutive_losses']}"
    )
    logging.info(f"Expectancy per Trade: ₹{stats['expectancy']:.2f}")
    logging.info("=" * 60)


def check_daily_limits():
    """Check if daily loss limit or trade limit is reached"""
    daily_pnl = load_daily_pnl()

    if daily_pnl["pnl"] <= -MAX_DAILY_LOSS:
        return False, f"Daily loss limit reached: ₹{daily_pnl['pnl']}"

    if daily_pnl["trades"] >= MAX_TRADES_PER_DAY:
        return False, f"Max trades per day reached: {daily_pnl['trades']}"

    return True, f"Daily P&L: ₹{daily_pnl['pnl']} | Trades: {daily_pnl['trades']}"


def check_cooldown():
    """Check if we're in cooldown period after a loss"""
    global LAST_LOSS_TIME

    if LAST_LOSS_TIME is None:
        return True, "No cooldown"

    elapsed = (datetime.now() - LAST_LOSS_TIME).total_seconds()
    if elapsed < COOLDOWN_AFTER_LOSS:
        remaining = int(COOLDOWN_AFTER_LOSS - elapsed)
        return False, f"Cooldown active: {remaining}s remaining"

    return True, "Cooldown complete"


def check_signal_cooldown(signal):
    """Check if we should skip this signal due to recent same-direction signal (whipsaw prevention)"""
    global LAST_SIGNAL, LAST_SIGNAL_TIME

    if LAST_SIGNAL is None or LAST_SIGNAL_TIME is None:
        return True, "No signal cooldown"

    # Only apply cooldown for same direction signals
    if signal == LAST_SIGNAL:
        elapsed = (datetime.now() - LAST_SIGNAL_TIME).total_seconds()
        if elapsed < SIGNAL_COOLDOWN:
            remaining = int(SIGNAL_COOLDOWN - elapsed)
            return False, f"Same signal ({signal}) cooldown: {remaining}s remaining"

    return True, "Signal allowed"


def update_last_signal(signal):
    """Update the last signal tracking"""
    global LAST_SIGNAL, LAST_SIGNAL_TIME
    LAST_SIGNAL = signal
    LAST_SIGNAL_TIME = datetime.now()


def switch_active_instrument(instrument_key):
    """Switch the active instrument configuration with validation"""
    global ACTIVE_INSTRUMENT, INSTRUMENT, EXCHANGE_SEGMENT_INT, EXCHANGE_SEGMENT_STR
    global FUTURE_ID, LOT_SIZE, STRIKE_STEP, EXPIRY_DATE, INSTRUMENT_TYPE, OPTION_TYPE
    global MARKET_START, MARKET_END, NO_NEW_TRADE_AFTER, MARKET_FEED

    if instrument_key not in INSTRUMENTS:
        logging.error(f"Unknown instrument: {instrument_key}")
        return False

    # Validate instrument has required fields
    required_fields = [
        "future_id",
        "lot_size",
        "strike_step",
        "expiry_date",
        "exchange_segment_str",
        "exchange_segment_int",
    ]
    inst = INSTRUMENTS[instrument_key]
    for field in required_fields:
        if field not in inst or inst[field] is None:
            logging.error(
                f"Instrument {instrument_key} missing required field: {field}"
            )
            return False

    # Check if we have recent data for this instrument (in multi-instrument mode)
    if MULTI_SCAN_ENABLED and instrument_key in INSTRUMENT_LTP:
        last_update = INSTRUMENT_LTP[instrument_key].get("last_update")
        if last_update:
            data_age = (datetime.now() - last_update).total_seconds()
            if data_age > 60:
                logging.warning(f"⚠️ {instrument_key} data is {data_age:.0f}s old")

    with instrument_lock:
        ACTIVE_INSTRUMENT = instrument_key
        INSTRUMENT = INSTRUMENTS[instrument_key]
        EXCHANGE_SEGMENT_INT = INSTRUMENT["exchange_segment_int"]
        EXCHANGE_SEGMENT_STR = INSTRUMENT["exchange_segment_str"]
        FUTURE_ID = INSTRUMENT["future_id"]
        LOT_SIZE = INSTRUMENT["lot_size"]
        STRIKE_STEP = INSTRUMENT["strike_step"]
        EXPIRY_DATE = INSTRUMENT["expiry_date"]
        INSTRUMENT_TYPE = INSTRUMENT["instrument_type"]
        OPTION_TYPE = INSTRUMENT["option_type"]
        MARKET_START = INSTRUMENT["market_start"]
        MARKET_END = INSTRUMENT["market_end"]
        NO_NEW_TRADE_AFTER = INSTRUMENT["no_new_trade_after"]

    # Subscribe to the new instrument's feed if not in multi-scan mode
    if not MULTI_SCAN_ENABLED and MARKET_FEED:
        try:
            new_sub = [(EXCHANGE_SEGMENT_INT, str(FUTURE_ID), marketfeed.Ticker)]
            MARKET_FEED.subscribe_symbols(new_sub)
            logging.debug(f"Subscribed to {instrument_key} feed")
        except Exception as e:
            logging.warning(f"Could not subscribe to {instrument_key} feed: {e}")

    logging.info(
        f"🔄 Switched active instrument to: {INSTRUMENT['name']} ({instrument_key})"
    )
    return True


def is_instrument_market_open(instrument_key):
    """Check if market is open for a specific instrument"""
    if instrument_key not in INSTRUMENTS:
        return False, "Unknown instrument"

    inst = INSTRUMENTS[instrument_key]
    now = datetime.now()
    current_time = now.strftime("%H:%M")

    # Check if it's a weekday
    if now.weekday() >= 5:
        return False, "Weekend"

    if current_time < inst["market_start"]:
        return False, f"Not open yet (opens {inst['market_start']})"

    if current_time > inst["market_end"]:
        return False, f"Closed (closed at {inst['market_end']})"

    return True, "Market Open"


def can_instrument_trade_new(instrument_key):
    """Check if new trades allowed for a specific instrument"""
    if instrument_key not in INSTRUMENTS:
        return False, "Unknown instrument"

    inst = INSTRUMENTS[instrument_key]
    now = datetime.now()
    current_time = now.strftime("%H:%M")

    if current_time > inst["no_new_trade_after"]:
        return False, f"No new trades after {inst['no_new_trade_after']}"

    return True, "New trades allowed"


def get_instrument_data(instrument_key):
    """Fetch resampled data for a specific instrument"""
    try:
        inst = INSTRUMENTS[instrument_key]
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

        data = dhan.intraday_minute_data(
            inst["future_id"],
            inst["exchange_segment_str"],
            inst["instrument_type"],
            from_date,
            to_date,
        )

        if data["status"] == "failure":
            return None, None

        df = pd.DataFrame(data["data"])
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
        df.set_index("time", inplace=True)

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
        logging.error(f"Data Error for {instrument_key}: {e}")
        return None, None


def analyze_instrument_signal(instrument_key, df_15, df_60):
    """Analyze an instrument and return signal info if in trade zone"""
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
        signal_strength = 0  # Higher = stronger signal

        # Calculate signal strength based on how strongly conditions are met
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
            # Signal strength: distance from thresholds + volume excess
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
            # Signal strength: distance from thresholds + volume excess
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
                "df_15": df_15,  # Include for SL calculation
            }

        return None

    except Exception as e:
        logging.error(f"Analysis error for {instrument_key}: {e}")
        return None


def scan_all_instruments():
    """Scan all configured instruments and return those in trade zone"""
    instruments_to_scan = get_instruments_to_scan()
    signals_found = []

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


def check_margin_available(option_id):
    """Check if sufficient margin/funds are available for the trade"""
    try:
        # Get fund limits
        funds = dhan.get_fund_limits()

        if funds.get("status") == "failure":
            logging.error(
                f"Failed to fetch fund limits: {funds.get('remarks', 'Unknown error')}"
            )
            return False, "Could not fetch fund limits"

        fund_data = funds.get("data", {})
        available_balance = float(fund_data.get("availabelBalance", 0))

        # Get margin required for this order
        # For options, we need to check margin requirement
        margin_response = dhan.margin_calculator(
            security_id=option_id,
            exchange_segment=EXCHANGE_SEGMENT_STR,
            transaction_type="BUY",
            quantity=LOT_SIZE,
            product_type="INTRADAY",
            price=0,
        )

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
            # If margin calculator fails, do a simple balance check
            # Assume minimum 10000 for safety
            if available_balance >= 10000:
                return (
                    True,
                    f"Balance OK: ₹{available_balance:.2f} (margin calc unavailable)",
                )
            else:
                return False, f"Low balance: ₹{available_balance:.2f}"

    except Exception as e:
        logging.error(f"Margin check error: {e}")
        # Don't block trade on margin check error, but log warning
        return True, f"Margin check failed: {e} (proceeding with caution)"


def should_auto_square_off():
    """Check if we need to auto square-off before market close"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")

    # Parse market end time
    market_end_hour, market_end_min = map(int, MARKET_END.split(":"))
    square_off_hour = market_end_hour
    square_off_min = market_end_min - AUTO_SQUARE_OFF_BUFFER

    # Handle minute underflow
    if square_off_min < 0:
        square_off_min += 60
        square_off_hour -= 1

    square_off_time = f"{square_off_hour:02d}:{square_off_min:02d}"

    if current_time >= square_off_time:
        return True, f"Auto square-off time reached ({square_off_time})"

    return False, f"Square-off at {square_off_time}"


def update_daily_pnl(pnl_amount, is_win):
    """Update daily P&L after a trade closes"""
    daily_data = load_daily_pnl()
    daily_data["pnl"] += pnl_amount
    daily_data["trades"] += 1
    if is_win:
        daily_data["wins"] += 1
    else:
        daily_data["losses"] += 1
    save_daily_pnl(daily_data)
    return daily_data


def verify_order(order_response, action="ENTRY"):
    """Verify order was placed successfully and get order details"""
    try:
        if order_response is None:
            logging.error(f"[{action}] Order response is None")
            return False, None

        if order_response.get("status") == "failure":
            error_msg = order_response.get("remarks", "Unknown error")
            logging.error(f"[{action}] Order FAILED: {error_msg}")
            send_alert(f"❌ **ORDER FAILED** ({action})\n{error_msg}")
            return False, None

        order_id = order_response.get("data", {}).get("orderId")
        if order_id:
            logging.info(f"[{action}] Order placed successfully. Order ID: {order_id}")

            # Wait briefly and check order status
            time.sleep(1)
            order_status = dhan.get_order_by_id(order_id)

            if order_status and order_status.get("status") == "success":
                order_data = order_status.get("data", {})
                status = order_data.get("orderStatus", "")

                if status in ["TRADED", "FILLED"]:
                    avg_price = order_data.get("tradedPrice", 0)
                    logging.info(f"[{action}] Order FILLED @ ₹{avg_price}")
                    return True, {
                        "order_id": order_id,
                        "avg_price": avg_price,
                        "status": status,
                    }
                elif status in ["REJECTED", "CANCELLED"]:
                    reason = order_data.get("rejectedReason", "Unknown")
                    logging.error(f"[{action}] Order {status}: {reason}")
                    send_alert(f"❌ **ORDER {status}** ({action})\n{reason}")
                    return False, None
                elif status in ["PENDING", "OPEN"]:
                    logging.warning(f"[{action}] Order still {status}, waiting...")
                    # Wait a bit more for market order to fill
                    time.sleep(2)
                    order_status = dhan.get_order_by_id(order_id)
                    if order_status and order_status.get("status") == "success":
                        order_data = order_status.get("data", {})
                        if order_data.get("orderStatus") in ["TRADED", "FILLED"]:
                            avg_price = order_data.get("tradedPrice", 0)
                            return True, {
                                "order_id": order_id,
                                "avg_price": avg_price,
                                "status": "FILLED",
                            }

                    # ⚠️ ZOMBIE ORDER FIX: Cancel the order if it's still pending
                    logging.warning(
                        f"[{action}] Order not filled in time. Cancelling order {order_id} to prevent zombie trade"
                    )
                    try:
                        cancel_response = dhan.cancel_order(order_id)
                        if (
                            cancel_response
                            and cancel_response.get("status") == "success"
                        ):
                            logging.info(
                                f"[{action}] Unfilled order {order_id} cancelled successfully"
                            )
                            send_alert(
                                f"⚠️ **ORDER CANCELLED** ({action})\nOrder {order_id} did not fill in time and was cancelled to prevent zombie trade"
                            )
                        else:
                            logging.error(
                                f"[{action}] Failed to cancel order {order_id}: {cancel_response}"
                            )
                            send_alert(
                                f"🚨 **CRITICAL**: Failed to cancel unfilled order {order_id}. Manual intervention required!"
                            )
                    except Exception as cancel_error:
                        logging.error(
                            f"[{action}] Error cancelling order {order_id}: {cancel_error}"
                        )
                        send_alert(
                            f"🚨 **CRITICAL**: Error cancelling unfilled order {order_id}. Manual intervention required!"
                        )

                    return False, None

        logging.error(f"[{action}] Could not get order ID from response")
        return False, None

    except Exception as e:
        logging.error(f"[{action}] Order verification error: {e}")
        return False, None


def get_atm_option(transaction_type, current_price):
    """
    transaction_type: "BUY" for Bullish (returns CE), "SELL" for Bearish (returns PE)
    """
    try:
        atm_strike = round(current_price / STRIKE_STEP) * STRIKE_STEP
        # If view is BUY (Bullish) -> Get CE
        # If view is SELL (Bearish) -> Get PE
        target = "CE" if transaction_type == "BUY" else "PE"

        chain = dhan.option_chain(
            EXCHANGE_SEGMENT_STR, FUTURE_ID, EXPIRY_DATE, OPTION_TYPE
        )
        if chain["status"] == "failure":
            logging.error(
                f"Option chain fetch failed: {chain.get('remarks', 'Unknown error')}"
            )
            return None

        for item in chain["data"]:
            if item["strike_price"] == atm_strike and item["dr_option_type"] == target:
                logging.debug(
                    f"Found ATM option: Strike {atm_strike} {target} -> ID: {item['security_id']}"
                )
                return item["security_id"]

        logging.warning(f"No ATM option found for strike {atm_strike} {target}")
        return None
    except Exception as e:
        logging.error(f"Error in get_atm_option: {e}")
        return None


def get_resampled_data():
    try:
        to_date = datetime.now().strftime("%Y-%m-%d")
        # Change days=5 to days=25 or days=30 to ensure valid 60min EMA_50
        from_date = (datetime.now() - timedelta(days=25)).strftime("%Y-%m-%d")

        data = dhan.intraday_minute_data(
            FUTURE_ID, EXCHANGE_SEGMENT_STR, INSTRUMENT_TYPE, from_date, to_date
        )
        if data["status"] == "failure":
            return None, None

        df = pd.DataFrame(data["data"])
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
        df.set_index("time", inplace=True)

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
        logging.error(f"Data Error: {e}")
        return None, None


# --- 3. SOCKET ---


def on_ticks(instance, ticks):
    global LATEST_LTP, OPTION_LTP, LAST_TICK_TIME, LAST_OPTION_TICK_TIME, INSTRUMENT_LTP
    if "LTP" in ticks:
        security_id = str(ticks.get("security_id", ""))
        ltp = float(ticks["LTP"])

        # Mark socket as healthy on any tick
        SOCKET_HEALTHY.set()

        # Check if this tick is for the option or the future
        if active_trade.get("status") and security_id == str(
            active_trade.get("option_id")
        ):
            OPTION_LTP = ltp
            LAST_OPTION_TICK_TIME = datetime.now()
        elif security_id == str(FUTURE_ID):
            LATEST_LTP = ltp
            LAST_TICK_TIME = datetime.now()
        else:
            # Check if it's any of our monitored instrument futures
            for inst_key, inst in INSTRUMENTS.items():
                if security_id == str(inst["future_id"]):
                    INSTRUMENT_LTP[inst_key] = {
                        "ltp": ltp,
                        "last_update": datetime.now(),
                    }
                    # Update main LTP if it's the active instrument
                    if inst_key == ACTIVE_INSTRUMENT:
                        LATEST_LTP = ltp
                        LAST_TICK_TIME = datetime.now()
                    break


def subscribe_option(feed, option_id):
    """Subscribe to option feed for premium tracking"""
    try:
        sub_instruments = [(EXCHANGE_SEGMENT_INT, str(option_id), marketfeed.Ticker)]
        feed.subscribe_symbols(sub_instruments)
        logging.info(f"📊 Subscribed to option feed: {option_id}")
    except Exception as e:
        logging.error(f"Failed to subscribe to option: {e}")


def unsubscribe_option(feed, option_id):
    """Unsubscribe from option feed"""
    try:
        unsub_instruments = [(EXCHANGE_SEGMENT_INT, str(option_id), marketfeed.Ticker)]
        feed.unsubscribe_symbols(unsub_instruments)
        logging.info(f"📊 Unsubscribed from option feed: {option_id}")
    except Exception as e:
        logging.error(f"Failed to unsubscribe from option: {e}")


# Global feed reference for subscribing/unsubscribing
MARKET_FEED = None


def get_all_instrument_subscriptions():
    """Get subscription list for all instruments to scan in multi-instrument mode"""
    subscriptions = []
    instruments_to_scan = (
        get_instruments_to_scan() if MULTI_SCAN_ENABLED else [ACTIVE_INSTRUMENT]
    )

    for inst_key in instruments_to_scan:
        inst = INSTRUMENTS[inst_key]
        subscriptions.append(
            (inst["exchange_segment_int"], str(inst["future_id"]), marketfeed.Ticker)
        )

    return subscriptions


def socket_heartbeat_monitor():
    """Separate thread to monitor socket health and trigger reconnection"""
    global SOCKET_HEALTHY

    logging.info(">>> Heartbeat Monitor Started")

    while not SHUTDOWN_EVENT.is_set():
        # Wait for either socket health signal or timeout
        socket_ok = SOCKET_HEALTHY.wait(timeout=30)

        if SHUTDOWN_EVENT.is_set():
            break

        if not socket_ok:
            # No tick received in 30 seconds
            logging.warning(
                "⚠️ HEARTBEAT FAILED - No tick data for 30s. Triggering reconnection..."
            )
            SOCKET_RECONNECT_EVENT.set()
        else:
            # Clear the flag for next cycle
            SOCKET_HEALTHY.clear()

        time.sleep(1)  # Small delay between checks


def start_socket():
    global MARKET_FEED, LAST_TICK_TIME
    logging.info(">>> Socket Connecting...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Subscribe to all instruments in multi-scan mode
    instruments = get_all_instrument_subscriptions()
    version = "v2"

    logging.info(f"📡 Subscribing to {len(instruments)} instrument feed(s)")
    for inst in instruments:
        logging.debug(f"   -> Exchange: {inst[0]}, Security: {inst[1]}")

    MARKET_FEED = marketfeed.DhanFeed(CLIENT_ID, ACCESS_TOKEN, instruments, version)

    # Start heartbeat monitor in separate thread
    heartbeat_thread = threading.Thread(target=socket_heartbeat_monitor, daemon=True)
    heartbeat_thread.start()

    while not SHUTDOWN_EVENT.is_set():
        try:
            # Check if reconnection is requested
            if SOCKET_RECONNECT_EVENT.is_set():
                logging.info("🔄 Reconnecting socket...")
                try:
                    MARKET_FEED.close_connection()
                except Exception as e:
                    logging.debug(f"Error closing connection: {e}")

                time.sleep(2)
                instruments = (
                    get_all_instrument_subscriptions()
                )  # Refresh subscription list
                MARKET_FEED = marketfeed.DhanFeed(
                    CLIENT_ID, ACCESS_TOKEN, instruments, version
                )
                SOCKET_RECONNECT_EVENT.clear()
                logging.info("✅ Socket reconnected successfully")

            MARKET_FEED.run_forever()
            response = MARKET_FEED.get_data()
            if response and "LTP" in response:
                on_ticks(MARKET_FEED, response)
        except Exception as e:
            logging.error(f"Socket Error: {e}")
            time.sleep(5)

    # Graceful shutdown
    logging.info("🔌 Socket shutting down...")
    try:
        MARKET_FEED.close_connection()
    except:
        pass


# --- 4. STATE ---


def save_state(data):
    """Transaction-safe state saving with backup"""
    temp_file = STATE_FILE + ".tmp"
    backup_file = STATE_FILE + ".bak"

    try:
        # Write to temp file first
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)

        # Backup existing state file
        if os.path.exists(STATE_FILE):
            try:
                os.replace(STATE_FILE, backup_file)
            except Exception as e:
                logging.debug(f"Could not create backup: {e}")

        # Atomic rename temp to actual
        os.replace(temp_file, STATE_FILE)

    except Exception as e:
        logging.error(f"Failed to save state: {e}")
        # Try direct write as fallback
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(data, f)
        except Exception as e2:
            logging.error(f"Fallback save also failed: {e2}")


def load_state():
    """Load state with fallback to backup"""
    # Try main state file first
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading state file: {e}")

    # Try backup file if main fails
    backup_file = STATE_FILE + ".bak"
    if os.path.exists(backup_file):
        try:
            logging.info("Loading state from backup file...")
            with open(backup_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading backup state: {e}")

    return {
        "status": False,
        "type": None,
        "option_id": None,
        "entry": 0,
        "sl": 0,
        "initial_sl": 0,
        "step_level": 0,
        "instrument": None,  # Track which instrument the trade is on
        "lot_size": None,  # Store lot size for the trade
        "exchange_segment_str": None,  # Store exchange segment for the trade
    }


active_trade = load_state()


def send_alert(msg):
    """Send Telegram alert with retry logic"""
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        if response.status_code != 200:
            logging.debug(f"Telegram alert failed: {response.status_code}")
    except requests.exceptions.Timeout:
        logging.debug("Telegram alert timeout")
    except Exception as e:
        logging.debug(f"Telegram alert error: {e}")


def get_dynamic_sl(action, df, buffer=2):
    try:
        if len(df) < 3:
            return 20
        last_two = df.iloc[-3:-1]
        current_cmp = df.iloc[-1]["close"]

        if action == "BUY":  # Bullish Trade SL
            swing_low = last_two["low"].min()
            sl_price = swing_low - buffer
            if (current_cmp - sl_price) < 5:
                sl_price = current_cmp - 10

        else:  # Bearish Trade SL ("SELL" Signal)
            swing_high = last_two["high"].max()
            sl_price = swing_high + buffer
            if (sl_price - current_cmp) < 5:
                sl_price = current_cmp + 10

        return int(sl_price)
    except Exception as e:
        print(f"[Error] SL Calc Failed: {e}")
        return 20


# --- 5. SCANNER (MULTI-INSTRUMENT SUPPORT) ---


def run_scanner():
    global active_trade, OPTION_LTP, MARKET_FEED
    logging.info(
        ">>> Scanner Started (Multi-Instrument Mode)"
        if MULTI_SCAN_ENABLED
        else ">>> Scanner Started (Single Instrument)"
    )

    while not SHUTDOWN_EVENT.is_set():
        try:
            # === PRE-TRADE CHECKS (GENERAL) ===

            # Check 1: Daily limits
            within_limits, limits_msg = check_daily_limits()
            if not within_limits:
                logging.warning(f"🛑 {limits_msg}")
                send_alert(f"🛑 **TRADING STOPPED**\n{limits_msg}")
                time.sleep(300)  # Check again in 5 minutes
                continue

            # Check 2: Cooldown after loss
            cooldown_ok, cooldown_msg = check_cooldown()
            if not cooldown_ok:
                logging.debug(f"⏳ {cooldown_msg}")
                time.sleep(30)
                continue

            # === NO ACTIVE TRADE - SCAN FOR OPPORTUNITIES ===
            if not active_trade["status"]:

                if MULTI_SCAN_ENABLED:
                    # Multi-instrument scanning mode
                    signals = scan_all_instruments()

                    if not signals:
                        time.sleep(60)
                        continue

                    # Try each signal in priority order until one succeeds
                    for signal_info in signals:
                        inst_key = signal_info["instrument"]
                        signal = signal_info["signal"]
                        price = signal_info["price"]
                        df_15 = signal_info["df_15"]

                        # Check signal cooldown
                        signal_ok, signal_msg = check_signal_cooldown(signal)
                        if not signal_ok:
                            logging.info(f"⏳ {inst_key}: {signal_msg}")
                            continue

                        # Switch to this instrument
                        if not switch_active_instrument(inst_key):
                            continue

                        # Get ATM option for this instrument
                        opt_id = get_atm_option(signal, price)

                        if not opt_id:
                            logging.warning(f"❌ {inst_key}: Could not find ATM option")
                            continue

                        # Check margin
                        margin_ok, margin_msg = check_margin_available(opt_id)
                        if not margin_ok:
                            logging.warning(f"💰 {inst_key}: {margin_msg}")
                            send_alert(
                                f"⚠️ **TRADE SKIPPED** ({inst_key})\n{margin_msg}"
                            )
                            update_last_signal(signal)
                            continue

                        logging.info(f"💰 {inst_key}: {margin_msg}")

                        # Execute trade
                        trade_executed = execute_trade_entry(
                            inst_key=inst_key,
                            signal=signal,
                            price=price,
                            opt_id=opt_id,
                            df_15=df_15,
                        )

                        if trade_executed:
                            break  # Trade placed, stop trying other instruments

                else:
                    # Single instrument mode (original behavior)
                    # Check market hours for active instrument
                    market_open, market_msg = is_market_open()
                    if not market_open:
                        logging.debug(f"⏰ {market_msg}")
                        time.sleep(60)
                        continue

                    # Check new trade time window
                    can_trade, trade_msg = can_place_new_trade()
                    if not can_trade:
                        logging.debug(f"⏰ {trade_msg}")
                        time.sleep(60)
                        continue

                    df_15, df_60 = get_resampled_data()

                    if df_15 is not None and df_60 is not None:
                        signal_info = analyze_instrument_signal(
                            ACTIVE_INSTRUMENT, df_15, df_60
                        )

                        if signal_info:
                            signal = signal_info["signal"]
                            price = signal_info["price"]

                            # Check signal cooldown
                            signal_ok, signal_msg = check_signal_cooldown(signal)
                            if not signal_ok:
                                logging.info(f"⏳ {signal_msg}")
                                time.sleep(60)
                                continue

                            opt_id = get_atm_option(signal, price)

                            if opt_id:
                                # Check margin
                                margin_ok, margin_msg = check_margin_available(opt_id)
                                if not margin_ok:
                                    logging.warning(f"💰 {margin_msg}")
                                    send_alert(f"⚠️ **TRADE SKIPPED**\n{margin_msg}")
                                    update_last_signal(signal)
                                    time.sleep(60)
                                    continue

                                logging.info(f"💰 {margin_msg}")

                                execute_trade_entry(
                                    inst_key=ACTIVE_INSTRUMENT,
                                    signal=signal,
                                    price=price,
                                    opt_id=opt_id,
                                    df_15=df_15,
                                )

            time.sleep(60)
        except Exception as e:
            logging.error(f"Scanner: {e}")
            time.sleep(60)


def execute_trade_entry(inst_key, signal, price, opt_id, df_15):
    """Execute a trade entry for a specific instrument"""
    global active_trade, OPTION_LTP, MARKET_FEED

    inst = INSTRUMENTS[inst_key]

    # Place order with LIMIT buffer
    limit_price = round(price * (1 + LIMIT_ORDER_BUFFER), 2)

    order_response = dhan.place_order(
        security_id=opt_id,
        exchange_segment=inst["exchange_segment_str"],
        transaction_type=dhan.BUY,
        quantity=inst["lot_size"],
        order_type=dhan.LIMIT,
        product_type=dhan.INTRADAY,
        price=limit_price,
    )

    # Verify order was filled
    order_success, order_details = verify_order(order_response, "ENTRY")

    if not order_success:
        logging.error(f"❌ {inst_key}: Entry order failed, skipping trade")
        update_last_signal(signal)
        return False

    # Update last signal tracking
    update_last_signal(signal)

    # Get actual entry price from order
    option_entry_price = order_details.get("avg_price", 0)
    actual_order_id = order_details.get("order_id")

    # Subscribe to option feed for premium tracking
    if MARKET_FEED:
        subscribe_option(MARKET_FEED, opt_id)
    OPTION_LTP = option_entry_price

    dynamic_sl = get_dynamic_sl(signal, df_15)

    # Thread-safe update of active_trade
    with trade_lock:
        active_trade["status"] = True
        active_trade["instrument"] = inst_key  # Store which instrument is being traded
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
        active_trade["lot_size"] = inst["lot_size"]  # Store lot size for this trade
        active_trade["exchange_segment_str"] = inst["exchange_segment_str"]
        save_state(active_trade)

    risk = abs(price - dynamic_sl)
    opt_type = "CALL" if signal == "BUY" else "PUT"
    logging.info(
        f">>> NEW TRADE: {inst['name']} {opt_type} @ Premium ₹{option_entry_price} | Future: {price} | SL: {dynamic_sl}"
    )

    send_alert(
        f"🚀 **{inst['name']} {opt_type} ENTERED**\n"
        f"Option Premium: ₹{option_entry_price}\n"
        f"Future: {price}\n"
        f"SL: {dynamic_sl}\n"
        f"Risk: {risk} pts"
    )

    return True


# --- 6. MANAGER (UPDATED WITH OPTION PREMIUM TRACKING) ---


def close_trade(exit_reason, exit_price_future, exit_price_option):
    """Close trade and calculate actual P&L based on option premium"""
    global active_trade, LAST_LOSS_TIME, OPTION_LTP, MARKET_FEED

    # Get trade instrument info (use stored values or fall back to global)
    trade_instrument = active_trade.get("instrument", ACTIVE_INSTRUMENT)
    trade_lot_size = active_trade.get("lot_size", LOT_SIZE)
    trade_exchange_segment = active_trade.get(
        "exchange_segment_str", EXCHANGE_SEGMENT_STR
    )

    option_entry = active_trade.get("option_entry", 0)
    option_exit = exit_price_option

    # Calculate actual P&L in rupees (option premium difference * lot size)
    pnl_per_lot = (option_exit - option_entry) * trade_lot_size

    is_win = pnl_per_lot > 0

    # Calculate R-multiple achieved
    risk_unit = abs(active_trade.get("entry", 0) - active_trade.get("initial_sl", 0))
    if risk_unit > 0:
        if active_trade["type"] == "BUY":
            profit_points = exit_price_future - active_trade.get("entry", 0)
        else:
            profit_points = active_trade.get("entry", 0) - exit_price_future
        r_multiple = profit_points / risk_unit
    else:
        r_multiple = 0

    # Update daily P&L
    daily_data = update_daily_pnl(pnl_per_lot, is_win)

    # Log trade to historical database
    trade_record = {
        "instrument": trade_instrument,
        "entry_time": active_trade.get(
            "entry_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
        "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_type": active_trade["type"],
        "option_type": "CALL" if active_trade["type"] == "BUY" else "PUT",
        "future_entry": active_trade.get("entry", 0),
        "future_exit": exit_price_future,
        "option_entry": option_entry,
        "option_exit": option_exit,
        "initial_sl": active_trade.get("initial_sl", 0),
        "final_sl": active_trade.get("sl", 0),
        "max_step_level": active_trade.get("step_level", 0),
        "pnl": pnl_per_lot,
        "r_multiple": r_multiple,
        "exit_reason": exit_reason,
        "lot_size": trade_lot_size,
    }
    save_trade_to_history(trade_record)

    # Set cooldown if loss
    if not is_win:
        LAST_LOSS_TIME = datetime.now()

    # Unsubscribe from option feed
    if MARKET_FEED and active_trade.get("option_id"):
        unsubscribe_option(MARKET_FEED, active_trade["option_id"])

    # Reset option LTP
    OPTION_LTP = 0

    trade_type = active_trade["type"]
    opt_type = "CALL" if trade_type == "BUY" else "PUT"

    # Get instrument name for display
    inst_name = INSTRUMENTS.get(trade_instrument, {}).get("name", trade_instrument)

    result_emoji = "✅" if is_win else "❌"
    pnl_sign = "+" if pnl_per_lot > 0 else ""

    logging.info(f"{result_emoji} TRADE CLOSED: {inst_name} - {exit_reason}")
    logging.info(f"   Option Entry: ₹{option_entry} | Exit: ₹{option_exit}")
    logging.info(
        f"   P&L: {pnl_sign}₹{pnl_per_lot:.2f} | R-Multiple: {r_multiple:.2f}R"
    )
    logging.info(
        f"   Daily P&L: ₹{daily_data['pnl']:.2f} | Trades: {daily_data['trades']}"
    )

    send_alert(
        f"{result_emoji} **{inst_name} {opt_type} CLOSED** - {exit_reason}\n"
        f"Option Entry: ₹{option_entry}\n"
        f"Option Exit: ₹{option_exit}\n"
        f"**P&L: {pnl_sign}₹{pnl_per_lot:.2f} ({r_multiple:.2f}R)**\n"
        f"Daily P&L: ₹{daily_data['pnl']:.2f}"
    )

    # Thread-safe update of active_trade (Fix: Global Variable Trap)
    with trade_lock:
        active_trade["status"] = False
        active_trade["instrument"] = None
        save_state(active_trade)


def run_manager():
    global active_trade, LATEST_LTP, OPTION_LTP, LAST_TICK_TIME, LAST_OPTION_TICK_TIME, MARKET_FEED
    logging.info(">>> Manager Started (Step Ladder Active)")

    while not SHUTDOWN_EVENT.is_set():
        # Check for data feed lag
        if (datetime.now() - LAST_TICK_TIME).total_seconds() > 10 and active_trade[
            "status"
        ]:
            logging.warning("⚠️ FUTURE DATA FEED LAG DETECTED - WATCH MANUALLY")

        if active_trade["status"] and OPTION_LTP > 0:
            if (datetime.now() - LAST_OPTION_TICK_TIME).total_seconds() > 10:
                logging.warning("⚠️ OPTION DATA FEED LAG DETECTED")

        # === AUTO SQUARE-OFF CHECK ===
        if active_trade["status"]:
            # Get trade-specific values
            trade_instrument = active_trade.get("instrument", ACTIVE_INSTRUMENT)
            trade_lot_size = active_trade.get("lot_size", LOT_SIZE)
            trade_exchange_segment = active_trade.get(
                "exchange_segment_str", EXCHANGE_SEGMENT_STR
            )
            trade_inst_config = INSTRUMENTS.get(trade_instrument, INSTRUMENT)

            # Check auto square-off using the trade's instrument config
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            market_end = trade_inst_config.get("market_end", MARKET_END)
            market_end_hour, market_end_min = map(int, market_end.split(":"))
            square_off_hour = market_end_hour
            square_off_min = market_end_min - AUTO_SQUARE_OFF_BUFFER
            if square_off_min < 0:
                square_off_min += 60
                square_off_hour -= 1
            square_off_time = f"{square_off_hour:02d}:{square_off_min:02d}"

            should_square_off = current_time >= square_off_time
            square_off_msg = f"Auto square-off time reached ({square_off_time})"

            if should_square_off:
                logging.warning(f"⏰ {square_off_msg}")
                option_ltp = (
                    OPTION_LTP
                    if OPTION_LTP > 0
                    else active_trade.get("option_entry", 0)
                )

                # Place exit order using trade's specific exchange segment
                squareoff_limit_price = round(option_ltp * (1 - LIMIT_ORDER_BUFFER), 2)

                exit_response = dhan.place_order(
                    security_id=active_trade["option_id"],
                    exchange_segment=trade_exchange_segment,
                    transaction_type=dhan.SELL,
                    quantity=trade_lot_size,
                    order_type=dhan.LIMIT,
                    product_type=dhan.INTRADAY,
                    price=squareoff_limit_price,
                )

                exit_success, exit_details = verify_order(
                    exit_response, "EXIT-AUTO-SQUAREOFF"
                )
                exit_price = (
                    exit_details.get("avg_price", option_ltp)
                    if exit_success
                    else option_ltp
                )

                close_trade(
                    f"AUTO SQUARE-OFF ({square_off_msg})", LATEST_LTP, exit_price
                )
                time.sleep(1)
                continue

        if LATEST_LTP != 0 and active_trade["status"]:
            # Get trade-specific values
            trade_lot_size = active_trade.get("lot_size", LOT_SIZE)
            trade_exchange_segment = active_trade.get(
                "exchange_segment_str", EXCHANGE_SEGMENT_STR
            )

            ltp = LATEST_LTP  # Future/Underlying price for SL tracking
            option_ltp = (
                OPTION_LTP if OPTION_LTP > 0 else active_trade.get("option_entry", 0)
            )
            trade_type = active_trade["type"]  # "BUY" or "SELL"

            # --- 1. SL HIT CHECK (Based on Future Price) ---
            sl_hit = False
            if trade_type == "BUY":
                if ltp <= active_trade["sl"]:
                    sl_hit = True
            elif trade_type == "SELL":
                if ltp >= active_trade["sl"]:
                    sl_hit = True

            if sl_hit:
                logging.info(f"🛑 SL HIT @ Future: {ltp}")

                # Place exit order and verify (Fix: Using LIMIT order with buffer)
                exit_limit_price = round(
                    option_ltp * (1 - LIMIT_ORDER_BUFFER), 2
                )  # 1% buffer below LTP for sell

                exit_response = dhan.place_order(
                    security_id=active_trade["option_id"],
                    exchange_segment=trade_exchange_segment,
                    transaction_type=dhan.SELL,
                    quantity=trade_lot_size,
                    order_type=dhan.LIMIT,
                    product_type=dhan.INTRADAY,
                    price=exit_limit_price,
                )

                exit_success, exit_details = verify_order(exit_response, "EXIT-SL")

                if exit_success:
                    exit_price = exit_details.get("avg_price", option_ltp)
                else:
                    exit_price = (
                        option_ltp  # Use last known LTP if order verification fails
                    )
                    logging.warning(
                        "⚠️ Exit order verification failed, using last known option LTP"
                    )

                close_trade("SL HIT", ltp, exit_price)
                continue

            # --- 2. PROFIT CALCULATION (Based on Future for R-multiple, Option for actual P&L) ---
            risk_unit = abs(active_trade["entry"] - active_trade["initial_sl"])
            if risk_unit == 0:
                risk_unit = 1

            # Calculate R-multiple based on future price movement
            if trade_type == "BUY":
                current_profit_future = ltp - active_trade["entry"]
            else:  # SELL
                current_profit_future = active_trade["entry"] - ltp

            current_r = current_profit_future / risk_unit

            # Calculate actual option P&L for display
            option_entry = active_trade.get("option_entry", 0)
            option_pnl = (option_ltp - option_entry) * trade_lot_size

            # Jackpot Exit (1:5)
            if current_r >= 5.0:
                logging.info(f">>> 🎯 1:5 TARGET HIT! Option P&L: ₹{option_pnl:.2f}")

                # Fix: Using LIMIT order with buffer for target exit
                target_exit_price = round(option_ltp * (1 - LIMIT_ORDER_BUFFER), 2)

                exit_response = dhan.place_order(
                    security_id=active_trade["option_id"],
                    exchange_segment=trade_exchange_segment,
                    transaction_type=dhan.SELL,
                    quantity=trade_lot_size,
                    order_type=dhan.LIMIT,
                    product_type=dhan.INTRADAY,
                    price=target_exit_price,
                )

                exit_success, exit_details = verify_order(exit_response, "EXIT-TARGET")
                exit_price = (
                    exit_details.get("avg_price", option_ltp)
                    if exit_success
                    else option_ltp
                )

                close_trade("1:5 TARGET HIT", ltp, exit_price)
                continue

            # Trailing Steps
            lock_r = 0
            msg = ""

            if current_r >= 4.0 and active_trade["step_level"] < 4:
                lock_r = 3.0
                active_trade["step_level"] = 4
                msg = "🚀 **Step 3 (1:4)**"
            elif current_r >= 3.0 and active_trade["step_level"] < 3:
                lock_r = 2.0
                active_trade["step_level"] = 3
                msg = "🚀 **Step 2 (1:3)**"
            elif current_r >= 2.0 and active_trade["step_level"] < 2:
                lock_r = 1.0
                active_trade["step_level"] = 2
                msg = "✅ **Step 1 (1:2)**"

            if lock_r > 0:
                if trade_type == "BUY":
                    new_sl = active_trade["entry"] + (lock_r * risk_unit)
                else:  # SELL
                    new_sl = active_trade["entry"] - (lock_r * risk_unit)

                # Thread-safe update of active_trade (Fix: Global Variable Trap)
                with trade_lock:
                    active_trade["sl"] = new_sl
                    save_state(active_trade)
                send_alert(
                    f"{msg}\n🔒 Locking {lock_r}R\nNew SL: {new_sl}\nOption P&L: ₹{option_pnl:.2f}"
                )

        time.sleep(0.5)


if __name__ == "__main__":
    # Log active instrument details
    logging.info("=" * 60)
    logging.info(f"🚀 TRADING BOT STARTED")
    logging.info("=" * 60)

    # Multi-instrument mode info
    if MULTI_SCAN_ENABLED:
        instruments_to_scan = get_instruments_to_scan()
        logging.info(f"🔄 MODE: Multi-Instrument Scanning ENABLED")
        logging.info(f"📊 Scanning {len(instruments_to_scan)} instruments:")
        for i, inst_key in enumerate(instruments_to_scan, 1):
            inst = INSTRUMENTS[inst_key]
            priority = INSTRUMENT_PRIORITY.get(inst_key, 99)
            logging.info(f"   {i}. {inst['name']} ({inst_key}) - Priority: {priority}")
    else:
        logging.info(f"📊 MODE: Single Instrument")
        logging.info(
            f"📊 Active Instrument: {INSTRUMENT['name']} ({ACTIVE_INSTRUMENT})"
        )
        logging.info(f"📈 Exchange: {EXCHANGE_SEGMENT_STR}")
        logging.info(f"🔢 Future ID: {FUTURE_ID}")
        logging.info(f"📦 Lot Size: {LOT_SIZE}")
        logging.info(f"📅 Expiry: {EXPIRY_DATE}")
        logging.info(f"🎯 Strike Step: {STRIKE_STEP}")

    logging.info("-" * 60)

    # Show all instrument market hours
    if MULTI_SCAN_ENABLED:
        logging.info("⏰ Market Hours per Instrument:")
        mcx_shown = False
        nse_shown = False
        for inst_key in get_instruments_to_scan():
            inst = INSTRUMENTS[inst_key]
            if inst["exchange_segment_str"] == "MCX" and not mcx_shown:
                logging.info(
                    f"   MCX: {inst['market_start']} - {inst['market_end']} | No trades after {inst['no_new_trade_after']}"
                )
                mcx_shown = True
            elif inst["exchange_segment_str"] == "NSE_FNO" and not nse_shown:
                logging.info(
                    f"   NSE: {inst['market_start']} - {inst['market_end']} | No trades after {inst['no_new_trade_after']}"
                )
                nse_shown = True
    else:
        logging.info(f"⏰ Market Hours: {MARKET_START} - {MARKET_END}")
        logging.info(f"🚫 No New Trades After: {NO_NEW_TRADE_AFTER}")
        # Calculate auto square-off time
        market_end_hour, market_end_min = map(int, MARKET_END.split(":"))
        sq_min = market_end_min - AUTO_SQUARE_OFF_BUFFER
        sq_hour = market_end_hour
        if sq_min < 0:
            sq_min += 60
            sq_hour -= 1
        logging.info(f"🔄 Auto Square-Off At: {sq_hour:02d}:{sq_min:02d}")

    logging.info("-" * 60)
    logging.info(f"💰 Max Daily Loss: ₹{MAX_DAILY_LOSS}")
    logging.info(f"📊 Max Trades/Day: {MAX_TRADES_PER_DAY}")
    logging.info(f"⏳ Cooldown After Loss: {COOLDOWN_AFTER_LOSS}s")
    logging.info(f"🔁 Signal Cooldown (Whipsaw): {SIGNAL_COOLDOWN}s")
    logging.info("-" * 60)
    logging.info(f"📈 RSI Bullish Threshold: > {RSI_BULLISH_THRESHOLD}")
    logging.info(f"📉 RSI Bearish Threshold: < {RSI_BEARISH_THRESHOLD}")
    logging.info(f"📊 Volume Multiplier: {VOLUME_MULTIPLIER}x avg")
    logging.info("-" * 60)

    # Load and display daily stats
    daily_stats = load_daily_pnl()
    logging.info(
        f"📈 Today's Stats: P&L: ₹{daily_stats['pnl']} | Trades: {daily_stats['trades']} | W:{daily_stats['wins']} L:{daily_stats['losses']}"
    )
    logging.info("=" * 60)

    # Display historical performance (last 30 days and all time)
    display_performance_report(days=30)
    time.sleep(1)
    display_performance_report()  # All time stats

    # Check market status at startup
    if MULTI_SCAN_ENABLED:
        logging.info("🏪 Market Status (per instrument):")
        for inst_key in get_instruments_to_scan():
            market_open, market_msg = is_instrument_market_open(inst_key)
            status_icon = "✅" if market_open else "⏸️"
            logging.info(f"   {status_icon} {inst_key}: {market_msg}")
    else:
        market_open, market_msg = is_market_open()
        logging.info(f"🏪 Market Status: {market_msg}")

    # Check daily limits at startup
    within_limits, limits_msg = check_daily_limits()
    if not within_limits:
        logging.warning(f"⚠️ {limits_msg}")

    t1 = threading.Thread(target=start_socket)
    t1.start()
    time.sleep(5)
    t2 = threading.Thread(target=run_scanner, daemon=True)
    t2.start()
    t3 = threading.Thread(target=run_manager, daemon=True)
    t3.start()

    def graceful_shutdown(reason="User request"):
        """Handle graceful shutdown with open position management"""
        global active_trade, OPTION_LTP, LATEST_LTP

        logging.info(f"🛑 Initiating graceful shutdown: {reason}")

        # Signal all threads to stop
        SHUTDOWN_EVENT.set()

        # Check if there's an open trade
        if active_trade.get("status"):
            logging.warning("⚠️ OPEN POSITION DETECTED during shutdown!")
            trade_instrument = active_trade.get("instrument", ACTIVE_INSTRUMENT)
            trade_lot_size = active_trade.get("lot_size", LOT_SIZE)
            trade_exchange_segment = active_trade.get(
                "exchange_segment_str", EXCHANGE_SEGMENT_STR
            )

            send_alert(
                f"🚨 **BOT SHUTDOWN WITH OPEN POSITION**\n"
                f"Instrument: {trade_instrument}\n"
                f"Type: {active_trade.get('type')}\n"
                f"Entry: ₹{active_trade.get('option_entry', 0)}\n"
                f"Current SL: {active_trade.get('sl', 0)}\n"
                f"⚠️ MANUAL INTERVENTION MAY BE REQUIRED"
            )

            # Attempt to close position
            user_input = (
                input(
                    "\n⚠️ Open position detected! Close position before shutdown? (y/n): "
                )
                .strip()
                .lower()
            )

            if user_input == "y":
                logging.info("Attempting to close open position...")
                option_ltp = (
                    OPTION_LTP
                    if OPTION_LTP > 0
                    else active_trade.get("option_entry", 0)
                )

                try:
                    exit_limit_price = round(option_ltp * (1 - LIMIT_ORDER_BUFFER), 2)

                    exit_response = dhan.place_order(
                        security_id=active_trade["option_id"],
                        exchange_segment=trade_exchange_segment,
                        transaction_type=dhan.SELL,
                        quantity=trade_lot_size,
                        order_type=dhan.LIMIT,
                        product_type=dhan.INTRADAY,
                        price=exit_limit_price,
                    )

                    exit_success, exit_details = verify_order(
                        exit_response, "EXIT-SHUTDOWN"
                    )

                    if exit_success:
                        exit_price = exit_details.get("avg_price", option_ltp)
                        close_trade(f"SHUTDOWN ({reason})", LATEST_LTP, exit_price)
                        logging.info("✅ Position closed successfully")
                    else:
                        logging.error(
                            "❌ Failed to close position. Manual intervention required!"
                        )
                        send_alert(
                            "🚨 **FAILED TO CLOSE POSITION ON SHUTDOWN** - Manual intervention required!"
                        )
                except Exception as e:
                    logging.error(f"Error closing position: {e}")
                    send_alert(f"🚨 **ERROR CLOSING POSITION**: {e}")
            else:
                logging.warning("⚠️ Position left open. State saved for recovery.")
                send_alert(
                    "⚠️ **BOT SHUTDOWN** - Position left open. State saved for recovery on restart."
                )

        # Save final state
        save_state(active_trade)

        # Wait for threads to finish
        logging.info("Waiting for threads to finish...")
        time.sleep(3)

        logging.info("=" * 60)
        logging.info("🛑 TRADING BOT STOPPED")
        logging.info("=" * 60)

    try:
        while not SHUTDOWN_EVENT.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        graceful_shutdown("User interrupted (Ctrl+C)")
    except Exception as e:
        logging.error(f"Bot error: {e}")
        graceful_shutdown(f"Error: {e}")
