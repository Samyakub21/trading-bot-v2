# =============================================================================
# SCANNER - Market Scanning and Signal Analysis
# =============================================================================

import logging
import time
import threading
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dhanhq import dhanhq

from config import config
from instruments import (
    INSTRUMENTS, INSTRUMENT_PRIORITY, MULTI_SCAN_ENABLED,
    get_instruments_to_scan
)
from utils import (
    RSI_BULLISH_THRESHOLD, RSI_BEARISH_THRESHOLD, VOLUME_MULTIPLIER,
    LIMIT_ORDER_BUFFER, send_alert, save_state, get_dynamic_sl,
    check_daily_limits, is_market_open, can_place_new_trade, 
    is_instrument_market_open, can_instrument_trade_new,
    COOLDOWN_AFTER_LOSS, SIGNAL_COOLDOWN
)
import socket_handler
from state_stores import get_signal_tracker

# =============================================================================
# DHAN CLIENT
# =============================================================================
CLIENT_ID = config.CLIENT_ID
ACCESS_TOKEN = config.ACCESS_TOKEN
dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)

# Threading lock for safe active_trade access
trade_lock = threading.Lock()
instrument_lock = threading.Lock()

# Signal tracker singleton (replaces global LAST_SIGNAL, LAST_SIGNAL_TIME, LAST_LOSS_TIME)
_signal_tracker = get_signal_tracker()


def update_last_signal(signal: str) -> None:
    """Update the last signal tracking using SignalTracker"""
    _signal_tracker.update_signal(signal)


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
    instrument_type: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch and resample OHLCV data for an instrument.
    
    Can be called in two ways:
    1. By instrument key: get_instrument_data("CRUDEOIL")
    2. By parameters: get_instrument_data(future_id="464926", exchange_segment_str="MCX", instrument_type="FUTURES")
    
    Args:
        instrument_key: Key from INSTRUMENTS dict (e.g., "CRUDEOIL", "NIFTY")
        future_id: Security ID for the future contract
        exchange_segment_str: Exchange segment string (e.g., "MCX", "NSE_FNO")
        instrument_type: Type of instrument (e.g., "FUTURES", "INDEX")
    
    Returns:
        Tuple of (df_15min, df_60min) DataFrames, or (None, None) on failure
    """
    try:
        # Resolve parameters from instrument_key if provided
        if instrument_key is not None:
            inst = INSTRUMENTS[instrument_key]
            future_id = inst["future_id"]
            exchange_segment_str = inst["exchange_segment_str"]
            instrument_type = inst["instrument_type"]
            log_context = instrument_key
        else:
            if not all([future_id, exchange_segment_str, instrument_type]):
                logging.error("Data Error: Must provide either instrument_key or all of (future_id, exchange_segment_str, instrument_type)")
                return None, None
            log_context = f"future_id={future_id}"
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d')
        
        data = dhan.intraday_minute_data(future_id, exchange_segment_str, instrument_type, from_date, to_date)
        
        if data['status'] == 'failure':
            return None, None
        
        df = pd.DataFrame(data['data'])
        df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume','start_time':'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        df_15 = df.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        df_60 = df.resample('60min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        
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
        logging.error(f"Data Error for {log_context}: Unexpected error - {type(e).__name__}: {e}")
        return None, None


# Backward compatibility alias
def get_resampled_data(
    future_id: str,
    exchange_segment_str: str,
    instrument_type: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Deprecated: Use get_instrument_data() instead.
    Kept for backward compatibility.
    """
    return get_instrument_data(
        future_id=future_id,
        exchange_segment_str=exchange_segment_str,
        instrument_type=instrument_type
    )


def analyze_instrument_signal(
    instrument_key: str,
    df_15: pd.DataFrame,
    df_60: pd.DataFrame
) -> Optional[Dict[str, Any]]:
    """Analyze an instrument and return signal info if in trade zone"""
    try:
        # Calculate indicators
        df_60['EMA_50'] = ta.ema(df_60['close'], length=50)
        df_15.ta.vwap(append=True)
        df_15['RSI'] = ta.rsi(df_15['close'], length=14)
        df_15['vol_avg'] = df_15['volume'].rolling(window=20).mean()
        
        trend = df_60.iloc[-2]
        trigger = df_15.iloc[-2]
        
        price = trigger['close']
        vwap_val = trigger.get('VWAP_D', 0)
        current_volume = trigger['volume']
        avg_volume = trigger.get('vol_avg', current_volume)
        rsi_val = trigger['RSI']
        
        # Volume confirmation
        volume_confirmed = current_volume >= (avg_volume * VOLUME_MULTIPLIER) if avg_volume > 0 else True
        
        signal = None
        signal_strength = 0
        
        ema_50 = trend['EMA_50']
        trend_close = trend['close']
        
        # BULLISH Signal
        if (trend_close > ema_50) and (trigger['close'] > vwap_val) and (rsi_val > RSI_BULLISH_THRESHOLD) and volume_confirmed:
            signal = "BUY"
            signal_strength = (rsi_val - RSI_BULLISH_THRESHOLD) + ((trend_close - ema_50) / ema_50 * 100)
            if avg_volume > 0:
                signal_strength += (current_volume / avg_volume - 1) * 10
        
        # BEARISH Signal
        elif (trend_close < ema_50) and (trigger['close'] < vwap_val) and (rsi_val < RSI_BEARISH_THRESHOLD) and volume_confirmed:
            signal = "SELL"
            signal_strength = (RSI_BEARISH_THRESHOLD - rsi_val) + ((ema_50 - trend_close) / ema_50 * 100)
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
        
    except KeyError as e:
        logging.error(f"Analysis error for {instrument_key}: Missing data column - {e}")
        return None
    except IndexError as e:
        logging.error(f"Analysis error for {instrument_key}: Insufficient data rows - {e}")
        return None
    except (TypeError, ValueError) as e:
        logging.error(f"Analysis error for {instrument_key}: Calculation error - {e}")
        return None
    except Exception as e:
        logging.error(f"Analysis error for {instrument_key}: Unexpected error - {type(e).__name__}: {e}")
        return None


def scan_all_instruments() -> List[Dict[str, Any]]:
    """Scan all configured instruments and return those in trade zone"""
    instruments_to_scan = get_instruments_to_scan()
    signals_found: List[Dict[str, Any]] = []
    
    logging.info(f"🔍 Scanning {len(instruments_to_scan)} instruments: {', '.join(instruments_to_scan)}")
    
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
            signal_type = "📈 BULLISH" if signal_info["signal"] == "BUY" else "📉 BEARISH"
            logging.info(f"   ✅ {inst_key}: {signal_type} | RSI: {signal_info['rsi']:.1f} | Strength: {signal_info['signal_strength']:.1f}")
        else:
            logging.debug(f"   ⏸️ {inst_key}: No signal (not in trade zone)")
    
    # Sort by priority first, then by signal strength
    if signals_found:
        signals_found.sort(key=lambda x: (INSTRUMENT_PRIORITY.get(x["instrument"], 99), -x["signal_strength"]))
        logging.info(f"📊 Found {len(signals_found)} instrument(s) in trade zone")
    
    return signals_found


def get_atm_option(
    transaction_type: str,
    current_price: float,
    exchange_segment_str: str,
    future_id: str,
    expiry_date: str,
    option_type: str,
    strike_step: int
) -> Optional[str]:
    """
    transaction_type: "BUY" for Bullish (returns CE), "SELL" for Bearish (returns PE)
    """
    try:
        atm_strike = round(current_price / strike_step) * strike_step
        target = "CE" if transaction_type == "BUY" else "PE"
        
        chain = dhan.option_chain(exchange_segment_str, future_id, expiry_date, option_type)
        if chain['status'] == 'failure':
            logging.error(f"Option chain fetch failed: {chain.get('remarks', 'Unknown error')}")
            return None

        for item in chain['data']:
            if item['strike_price'] == atm_strike and item['dr_option_type'] == target:
                logging.debug(f"Found ATM option: Strike {atm_strike} {target} -> ID: {item['security_id']}")
                return item['security_id']
        
        logging.warning(f"No ATM option found for strike {atm_strike} {target}")
        return None
    except KeyError as e:
        logging.error(f"Error in get_atm_option: Missing key in response - {e}")
        return None
    except requests.RequestException as e:
        logging.error(f"Error in get_atm_option: Network request failed - {e}")
        return None
    except (TypeError, ValueError) as e:
        logging.error(f"Error in get_atm_option: Calculation error - {e}")
        return None
    except Exception as e:
        logging.error(f"Error in get_atm_option: Unexpected error - {type(e).__name__}: {e}")
        return None


def check_margin_available(
    option_id: str,
    exchange_segment_str: str,
    lot_size: int
) -> Tuple[bool, str]:
    """Check if sufficient margin/funds are available for the trade"""
    try:
        funds = dhan.get_fund_limits()
        
        if funds.get('status') == 'failure':
            logging.error(f"Failed to fetch fund limits: {funds.get('remarks', 'Unknown error')}")
            return False, "Could not fetch fund limits"
        
        fund_data = funds.get('data', {})
        available_balance = float(fund_data.get('availabelBalance', 0))
        
        margin_response = dhan.margin_calculator(
            security_id=option_id,
            exchange_segment=exchange_segment_str,
            transaction_type="BUY",
            quantity=lot_size,
            product_type="INTRADAY",
            price=0
        )
        
        if margin_response.get('status') == 'success':
            required_margin = float(margin_response.get('data', {}).get('totalMargin', 0))
            
            if available_balance >= required_margin:
                return True, f"Margin OK: Available ₹{available_balance:.2f} >= Required ₹{required_margin:.2f}"
            else:
                return False, f"Insufficient margin: Available ₹{available_balance:.2f} < Required ₹{required_margin:.2f}"
        else:
            if available_balance >= 10000:
                return True, f"Balance OK: ₹{available_balance:.2f} (margin calc unavailable)"
            else:
                return False, f"Low balance: ₹{available_balance:.2f}"
                
    except KeyError as e:
        logging.error(f"Margin check error: Missing key in response - {e}")
        return True, f"Margin check failed: Missing data (proceeding with caution)"
    except requests.RequestException as e:
        logging.error(f"Margin check error: Network request failed - {e}")
        return True, f"Margin check failed: Network error (proceeding with caution)"
    except (TypeError, ValueError) as e:
        logging.error(f"Margin check error: Data parsing failed - {e}")
        return True, f"Margin check failed: Invalid data (proceeding with caution)"
    except Exception as e:
        logging.error(f"Margin check error: Unexpected error - {type(e).__name__}: {e}")
        return True, f"Margin check failed: {e} (proceeding with caution)"


# =============================================================================
# ORDER VERIFICATION CONFIGURATION
# =============================================================================
ORDER_VERIFICATION_CONFIG = {
    "initial_delay": 0.5,       # Initial delay before first status check (seconds)
    "max_retries": 5,           # Maximum number of retry attempts
    "base_backoff": 1.0,        # Base delay for exponential backoff (seconds)
    "max_backoff": 8.0,         # Maximum delay between retries (seconds)
    "backoff_multiplier": 2.0,  # Multiplier for exponential backoff
    "total_timeout": 15.0,      # Maximum total time to wait for order fill (seconds)
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
        config["max_backoff"]
    )
    time.sleep(delay)
    return delay


def verify_order(
    order_response: Optional[Dict[str, Any]],
    action: str = "ENTRY",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify order was placed successfully and get order details.
    
    Uses exponential backoff for polling order status instead of fixed delays.
    
    Args:
        order_response: Response from dhan.place_order()
        action: Description of the order action (for logging)
        config: Optional custom configuration for timeouts/retries
    
    Returns:
        Tuple of (success: bool, details: dict or None)
    """
    if config is None:
        config = ORDER_VERIFICATION_CONFIG
    
    try:
        if order_response is None:
            logging.error(f"[{action}] Order response is None")
            return False, None
        
        if order_response.get('status') == 'failure':
            error_msg = order_response.get('remarks', 'Unknown error')
            logging.error(f"[{action}] Order FAILED: {error_msg}")
            send_alert(f"❌ **ORDER FAILED** ({action})\n{error_msg}")
            return False, None
        
        order_id = order_response.get('data', {}).get('orderId')
        if not order_id:
            logging.error(f"[{action}] Could not get order ID from response")
            return False, None
        
        logging.info(f"[{action}] Order placed successfully. Order ID: {order_id}")
        
        # Initial delay before first status check
        time.sleep(config["initial_delay"])
        
        start_time = time.time()
        
        for attempt in range(config["max_retries"]):
            # Check total timeout
            elapsed = time.time() - start_time
            if elapsed >= config["total_timeout"]:
                logging.warning(f"[{action}] Total timeout ({config['total_timeout']}s) exceeded after {attempt} attempts")
                break
            
            order_status = dhan.get_order_by_id(order_id)
            
            if order_status and order_status.get('status') == 'success':
                order_data = order_status.get('data', {})
                status = order_data.get('orderStatus', '')
                
                if status in ['TRADED', 'FILLED']:
                    avg_price = order_data.get('tradedPrice', 0)
                    logging.info(f"[{action}] Order FILLED @ ₹{avg_price} (attempt {attempt + 1})")
                    return True, {"order_id": order_id, "avg_price": avg_price, "status": status}
                
                elif status in ['REJECTED', 'CANCELLED']:
                    reason = order_data.get('rejectedReason', 'Unknown')
                    logging.error(f"[{action}] Order {status}: {reason}")
                    send_alert(f"❌ **ORDER {status}** ({action})\n{reason}")
                    return False, None
                
                elif status in ['PENDING', 'OPEN']:
                    remaining_time = config["total_timeout"] - elapsed
                    logging.debug(f"[{action}] Order still {status}, attempt {attempt + 1}/{config['max_retries']} "
                                  f"(timeout in {remaining_time:.1f}s)")
                    
                    # Apply exponential backoff before next retry
                    if attempt < config["max_retries"] - 1:
                        delay = _wait_with_backoff(attempt, config)
                        logging.debug(f"[{action}] Waiting {delay:.1f}s before next check...")
            else:
                logging.warning(f"[{action}] Failed to fetch order status, retrying...")
                if attempt < config["max_retries"] - 1:
                    _wait_with_backoff(attempt, config)
        
        # Order not filled within timeout - cancel it
        logging.warning(f"[{action}] Order not filled in time ({config['total_timeout']}s). Cancelling order {order_id}")
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
        if cancel_response and cancel_response.get('status') == 'success':
            logging.info(f"[{action}] Unfilled order {order_id} cancelled successfully")
            send_alert(f"⚠️ **ORDER CANCELLED** ({action})\nOrder {order_id} did not fill in time")
            return True
        else:
            logging.error(f"[{action}] Failed to cancel order {order_id}: {cancel_response}")
            send_alert(f"🚨 **CRITICAL**: Failed to cancel unfilled order {order_id}. Manual intervention required!")
            return False
    except Exception as cancel_error:
        logging.error(f"[{action}] Error cancelling order {order_id}: {cancel_error}")
        send_alert(f"🚨 **CRITICAL**: Error cancelling unfilled order {order_id}. Manual intervention required!")
        return False


def execute_trade_entry(
    inst_key: str,
    signal: str,
    price: float,
    opt_id: str,
    df_15: pd.DataFrame,
    active_trade: Dict[str, Any]
) -> bool:
    """Execute a trade entry for a specific instrument"""
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
        price=limit_price
    )
    
    order_success, order_details = verify_order(order_response, "ENTRY")
    
    if not order_success:
        logging.error(f"❌ {inst_key}: Entry order failed, skipping trade")
        update_last_signal(signal)
        return False
    
    update_last_signal(signal)
    
    option_entry_price = order_details.get("avg_price", 0)
    actual_order_id = order_details.get("order_id")
    
    # Subscribe to option feed
    market_feed = socket_handler.get_market_feed()
    if market_feed:
        socket_handler.subscribe_option(market_feed, opt_id, inst["exchange_segment_int"])
    socket_handler.set_option_ltp(option_entry_price)
    
    dynamic_sl = get_dynamic_sl(signal, df_15)
    
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
        save_state(active_trade)
    
    risk = abs(price - dynamic_sl)
    opt_type = "CALL" if signal == "BUY" else "PUT"
    logging.info(f">>> NEW TRADE: {inst['name']} {opt_type} @ Premium ₹{option_entry_price} | Future: {price} | SL: {dynamic_sl}")
    
    send_alert(
        f"🚀 **{inst['name']} {opt_type} ENTERED**\n"
        f"Option Premium: ₹{option_entry_price}\n"
        f"Future: {price}\n"
        f"SL: {dynamic_sl}\n"
        f"Risk: {risk} pts"
    )
    
    return True


def run_scanner(active_trade: Dict[str, Any], active_instrument: str) -> None:
    """Main scanner loop"""
    logging.info(">>> Scanner Started (Multi-Instrument Mode)" if MULTI_SCAN_ENABLED else ">>> Scanner Started (Single Instrument)")
    
    while not socket_handler.is_shutdown():
        try:
            # === PRE-TRADE CHECKS (GENERAL) ===
            
            # Check 1: Daily limits
            within_limits, limits_msg = check_daily_limits()
            if not within_limits:
                logging.warning(f"🛑 {limits_msg}")
                send_alert(f"🛑 **TRADING STOPPED**\n{limits_msg}")
                time.sleep(300)
                continue
            
            # Check 2: Cooldown after loss (using SignalTracker)
            in_cooldown, cooldown_msg = _signal_tracker.is_in_loss_cooldown(COOLDOWN_AFTER_LOSS)
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
                        
                        # Check signal cooldown (using SignalTracker)
                        in_signal_cooldown, signal_msg = _signal_tracker.is_in_signal_cooldown(signal, SIGNAL_COOLDOWN)
                        if in_signal_cooldown:
                            logging.info(f"⏳ {inst_key}: {signal_msg}")
                            continue
                        
                        inst = INSTRUMENTS[inst_key]
                        opt_id = get_atm_option(
                            signal, price,
                            inst["exchange_segment_str"],
                            inst["future_id"],
                            inst["expiry_date"],
                            inst["option_type"],
                            inst["strike_step"]
                        )
                        
                        if not opt_id:
                            logging.warning(f"❌ {inst_key}: Could not find ATM option")
                            continue
                        
                        margin_ok, margin_msg = check_margin_available(opt_id, inst["exchange_segment_str"], inst["lot_size"])
                        if not margin_ok:
                            logging.warning(f"💰 {inst_key}: {margin_msg}")
                            send_alert(f"⚠️ **TRADE SKIPPED** ({inst_key})\n{margin_msg}")
                            update_last_signal(signal)
                            continue
                        
                        logging.info(f"💰 {inst_key}: {margin_msg}")
                        
                        trade_executed = execute_trade_entry(
                            inst_key=inst_key,
                            signal=signal,
                            price=price,
                            opt_id=opt_id,
                            df_15=df_15,
                            active_trade=active_trade
                        )
                        
                        if trade_executed:
                            break
                        
                else:
                    # Single instrument mode
                    inst = INSTRUMENTS[active_instrument]
                    
                    market_open, market_msg = is_market_open(inst["market_start"], inst["market_end"])
                    if not market_open:
                        logging.debug(f"⏰ {market_msg}")
                        time.sleep(60)
                        continue
                    
                    can_trade, trade_msg = can_place_new_trade(inst["no_new_trade_after"])
                    if not can_trade:
                        logging.debug(f"⏰ {trade_msg}")
                        time.sleep(60)
                        continue
                    
                    df_15, df_60 = get_resampled_data(inst["future_id"], inst["exchange_segment_str"], inst["instrument_type"])

                    if df_15 is not None and df_60 is not None:
                        signal_info = analyze_instrument_signal(active_instrument, df_15, df_60)
                        
                        if signal_info:
                            signal = signal_info["signal"]
                            price = signal_info["price"]
                            
                            # Check signal cooldown (using SignalTracker)
                            in_signal_cooldown, signal_msg = _signal_tracker.is_in_signal_cooldown(signal, SIGNAL_COOLDOWN)
                            if in_signal_cooldown:
                                logging.info(f"⏳ {signal_msg}")
                                time.sleep(60)
                                continue
                            
                            opt_id = get_atm_option(
                                signal, price,
                                inst["exchange_segment_str"],
                                inst["future_id"],
                                inst["expiry_date"],
                                inst["option_type"],
                                inst["strike_step"]
                            )
                            
                            if opt_id:
                                margin_ok, margin_msg = check_margin_available(opt_id, inst["exchange_segment_str"], inst["lot_size"])
                                if not margin_ok:
                                    logging.warning(f"💰 {margin_msg}")
                                    send_alert(f"⚠️ **TRADE SKIPPED**\n{margin_msg}")
                                    update_last_signal(signal)
                                    time.sleep(60)
                                    continue
                                
                                logging.info(f"💰 {margin_msg}")
                                
                                execute_trade_entry(
                                    inst_key=active_instrument,
                                    signal=signal,
                                    price=price,
                                    opt_id=opt_id,
                                    df_15=df_15,
                                    active_trade=active_trade
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
