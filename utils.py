# =============================================================================
# UTILITY FUNCTIONS - State Management, Alerts, Helpers
# =============================================================================

import os
import json
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config import config
from instruments import (
    INSTRUMENTS, DEFAULT_INSTRUMENT, MULTI_SCAN_ENABLED,
    get_instruments_to_scan
)

# =============================================================================
# CONFIGURATION (loaded from config.py - supports env vars and config files)
# =============================================================================
TELEGRAM_TOKEN = config.TELEGRAM_TOKEN
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID

# State files (configurable via trading_config.json or env vars)
STATE_FILE = config.STATE_FILE
DAILY_PNL_FILE = config.DAILY_PNL_FILE
TRADE_HISTORY_FILE = config.TRADE_HISTORY_FILE

# =============================================================================
# RISK MANAGEMENT CONFIG (configurable via trading_config.json or env vars)
# =============================================================================
MAX_DAILY_LOSS = config.MAX_DAILY_LOSS
MAX_TRADES_PER_DAY = config.MAX_TRADES_PER_DAY
COOLDOWN_AFTER_LOSS = config.COOLDOWN_AFTER_LOSS
SIGNAL_COOLDOWN = config.SIGNAL_COOLDOWN
AUTO_SQUARE_OFF_BUFFER = config.AUTO_SQUARE_OFF_BUFFER

# SIGNAL STRENGTH CONFIG (configurable via trading_config.json or env vars)
RSI_BULLISH_THRESHOLD = config.RSI_BULLISH_THRESHOLD
RSI_BEARISH_THRESHOLD = config.RSI_BEARISH_THRESHOLD
VOLUME_MULTIPLIER = config.VOLUME_MULTIPLIER

# Limit order buffer (configurable via trading_config.json or env vars)
LIMIT_ORDER_BUFFER = config.LIMIT_ORDER_BUFFER

# =============================================================================
# GLOBAL STATE VARIABLES
# =============================================================================
LAST_LOSS_TIME = None       # Track time of last loss for cooldown
LAST_SIGNAL = None          # Track last signal direction ("BUY" or "SELL")
LAST_SIGNAL_TIME = None     # Track time of last signal for whipsaw prevention


# =============================================================================
# TELEGRAM ALERTS
# =============================================================================
def send_alert(msg: str) -> None:
    """Send Telegram alert with retry logic"""
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            params={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
        if response.status_code != 200:
            logging.debug(f"Telegram alert failed: {response.status_code}")
    except requests.exceptions.Timeout:
        logging.debug("Telegram alert timeout")
    except Exception as e:
        logging.debug(f"Telegram alert error: {e}")


# =============================================================================
# STATE MANAGEMENT
# =============================================================================
def save_state(data: Dict[str, Any]) -> None:
    """Transaction-safe state saving with backup"""
    temp_file = STATE_FILE + '.tmp'
    backup_file = STATE_FILE + '.bak'
    
    try:
        # Write to temp file first
        with open(temp_file, 'w') as f:
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
            with open(STATE_FILE, 'w') as f:
                json.dump(data, f)
        except Exception as e2:
            logging.error(f"Fallback save also failed: {e2}")


def load_state() -> Dict[str, Any]:
    """Load state with fallback to backup"""
    # Try main state file first
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading state file: {e}")
    
    # Try backup file if main fails
    backup_file = STATE_FILE + '.bak'
    if os.path.exists(backup_file):
        try:
            logging.info("Loading state from backup file...")
            with open(backup_file, 'r') as f:
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
        "instrument": None,
        "lot_size": None,
        "exchange_segment_str": None
    }


# =============================================================================
# DAILY P&L MANAGEMENT
# =============================================================================
def load_daily_pnl() -> Dict[str, Any]:
    """Load daily P&L data, reset if it's a new day"""
    today = datetime.now().strftime("%Y-%m-%d")
    
    if os.path.exists(DAILY_PNL_FILE):
        try:
            with open(DAILY_PNL_FILE, 'r') as f:
                data = json.load(f)
                if data.get("date") != today:
                    return {"date": today, "pnl": 0, "trades": 0, "wins": 0, "losses": 0}
                return data
        except:
            pass
    
    return {"date": today, "pnl": 0, "trades": 0, "wins": 0, "losses": 0}


def save_daily_pnl(data: Dict[str, Any]) -> None:
    """Save daily P&L data"""
    with open(DAILY_PNL_FILE, 'w') as f:
        json.dump(data, f)


def update_daily_pnl(pnl_amount: float, is_win: bool) -> Dict[str, Any]:
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


# =============================================================================
# TRADE HISTORY
# =============================================================================
def load_trade_history() -> List[Dict[str, Any]]:
    """Load historical trade data"""
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []


def save_trade_to_history(trade_data: Dict[str, Any]) -> None:
    """Append trade to historical log"""
    history = load_trade_history()
    history.append(trade_data)
    
    with open(TRADE_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# =============================================================================
# PERFORMANCE STATISTICS
# =============================================================================
def get_performance_stats(days: Optional[int] = None) -> Optional[Dict[str, Any]]:
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
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        history = [t for t in history if t.get('exit_time', '') >= cutoff_date]
    
    if not history:
        return None
    
    total_trades = len(history)
    wins = [t for t in history if t['pnl'] > 0]
    losses = [t for t in history if t['pnl'] <= 0]
    
    total_pnl = sum(t['pnl'] for t in history)
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_win = (gross_profit / len(wins)) if wins else 0
    avg_loss = (gross_loss / len(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    # R-multiple stats
    r_multiples = [t.get('r_multiple', 0) for t in history if 'r_multiple' in t]
    avg_r = (sum(r_multiples) / len(r_multiples)) if r_multiples else 0
    
    # Best and worst trades
    best_trade = max(history, key=lambda x: x['pnl'])['pnl'] if history else 0
    worst_trade = min(history, key=lambda x: x['pnl'])['pnl'] if history else 0
    
    # Consecutive wins/losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0
    
    for trade in history:
        if trade['pnl'] > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
    
    return {
        'total_trades': total_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_r_multiple': avg_r,
        'best_trade': best_trade,
        'worst_trade': worst_trade,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'expectancy': (avg_win * win_rate/100) - (avg_loss * (100-win_rate)/100)
    }


def display_performance_report(days: Optional[int] = None) -> None:
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
    logging.info(f"Total Trades: {stats['total_trades']} | Wins: {stats['wins']} | Losses: {stats['losses']}")
    logging.info(f"Win Rate: {stats['win_rate']:.1f}% | Avg R-Multiple: {stats['avg_r_multiple']:.2f}R")
    logging.info("-" * 60)
    logging.info(f"Total P&L: ₹{stats['total_pnl']:.2f}")
    logging.info(f"Gross Profit: ₹{stats['gross_profit']:.2f} | Gross Loss: ₹{stats['gross_loss']:.2f}")
    logging.info(f"Profit Factor: {stats['profit_factor']:.2f}")
    logging.info("-" * 60)
    logging.info(f"Avg Win: ₹{stats['avg_win']:.2f} | Avg Loss: ₹{stats['avg_loss']:.2f}")
    logging.info(f"Best Trade: ₹{stats['best_trade']:.2f} | Worst Trade: ₹{stats['worst_trade']:.2f}")
    logging.info("-" * 60)
    logging.info(f"Max Consecutive Wins: {stats['max_consecutive_wins']} | Losses: {stats['max_consecutive_losses']}")
    logging.info(f"Expectancy per Trade: ₹{stats['expectancy']:.2f}")
    logging.info("=" * 60)


# =============================================================================
# MARKET HOURS & TIME CHECKS
# =============================================================================
def is_market_open(market_start: str, market_end: str) -> Tuple[bool, str]:
    """Check if market is currently open for trading"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False, "Market closed (Weekend)"
    
    if current_time < market_start:
        return False, f"Market not open yet (opens at {market_start})"
    
    if current_time > market_end:
        return False, f"Market closed (closed at {market_end})"
    
    return True, "Market Open"


def can_place_new_trade(no_new_trade_after: str) -> Tuple[bool, str]:
    """Check if new trades are allowed (time restriction before market close)"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    if current_time > no_new_trade_after:
        return False, f"No new trades after {no_new_trade_after}"
    
    return True, "New trades allowed"


def is_instrument_market_open(instrument_key: str) -> Tuple[bool, str]:
    """Check if market is open for a specific instrument"""
    if instrument_key not in INSTRUMENTS:
        return False, "Unknown instrument"
    
    inst = INSTRUMENTS[instrument_key]
    return is_market_open(inst["market_start"], inst["market_end"])


def can_instrument_trade_new(instrument_key: str) -> Tuple[bool, str]:
    """Check if new trades allowed for a specific instrument"""
    if instrument_key not in INSTRUMENTS:
        return False, "Unknown instrument"
    
    inst = INSTRUMENTS[instrument_key]
    return can_place_new_trade(inst["no_new_trade_after"])


def should_auto_square_off(market_end: str) -> Tuple[bool, str]:
    """Check if we need to auto square-off before market close"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    
    # Parse market end time
    market_end_hour, market_end_min = map(int, market_end.split(":"))
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


# =============================================================================
# RISK & COOLDOWN CHECKS
# =============================================================================
def check_daily_limits() -> Tuple[bool, str]:
    """Check if daily loss limit or trade limit is reached"""
    daily_pnl = load_daily_pnl()
    
    if daily_pnl["pnl"] <= -MAX_DAILY_LOSS:
        return False, f"Daily loss limit reached: ₹{daily_pnl['pnl']}"
    
    if daily_pnl["trades"] >= MAX_TRADES_PER_DAY:
        return False, f"Max trades per day reached: {daily_pnl['trades']}"
    
    return True, f"Daily P&L: ₹{daily_pnl['pnl']} | Trades: {daily_pnl['trades']}"


def check_cooldown(last_loss_time: Optional[datetime]) -> Tuple[bool, str]:
    """Check if we're in cooldown period after a loss"""
    if last_loss_time is None:
        return True, "No cooldown"
    
    elapsed = (datetime.now() - last_loss_time).total_seconds()
    if elapsed < COOLDOWN_AFTER_LOSS:
        remaining = int(COOLDOWN_AFTER_LOSS - elapsed)
        return False, f"Cooldown active: {remaining}s remaining"
    
    return True, "Cooldown complete"


def check_signal_cooldown(
    signal: str,
    last_signal: Optional[str],
    last_signal_time: Optional[datetime]
) -> Tuple[bool, str]:
    """Check if we should skip this signal due to recent same-direction signal"""
    if last_signal is None or last_signal_time is None:
        return True, "No signal cooldown"
    
    # Only apply cooldown for same direction signals
    if signal == last_signal:
        elapsed = (datetime.now() - last_signal_time).total_seconds()
        if elapsed < SIGNAL_COOLDOWN:
            remaining = int(SIGNAL_COOLDOWN - elapsed)
            return False, f"Same signal ({signal}) cooldown: {remaining}s remaining"
    
    return True, "Signal allowed"


# =============================================================================
# STOP LOSS CALCULATION
# =============================================================================
def get_dynamic_sl(action: str, df: pd.DataFrame, buffer: int = 2) -> int:
    """Calculate dynamic stop loss based on swing high/low"""
    try:
        if len(df) < 3:
            return 20
        
        last_two = df.iloc[-3:-1]
        current_cmp = df.iloc[-1]['close']
        
        if action == "BUY":  # Bullish Trade SL
            swing_low = last_two['low'].min()
            sl_price = swing_low - buffer
            if (current_cmp - sl_price) < 5:
                sl_price = current_cmp - 10
        else:  # Bearish Trade SL ("SELL" Signal)
            swing_high = last_two['high'].max()
            sl_price = swing_high + buffer
            if (sl_price - current_cmp) < 5:
                sl_price = current_cmp + 10
        
        return int(sl_price)
    except Exception as e:
        print(f"[Error] SL Calc Failed: {e}")
        return 20
