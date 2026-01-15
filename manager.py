# =============================================================================
# MANAGER - Trade Management and Trailing Stop Loss
# =============================================================================

import logging
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from dhanhq import dhanhq

from config import config
from instruments import INSTRUMENTS, DEFAULT_INSTRUMENT
from utils import (
    LIMIT_ORDER_BUFFER, AUTO_SQUARE_OFF_BUFFER,
    send_alert, save_state, update_daily_pnl, save_trade_to_history
)
import socket_handler
import scanner

# =============================================================================
# DHAN CLIENT
# =============================================================================
CLIENT_ID = config.CLIENT_ID
ACCESS_TOKEN = config.ACCESS_TOKEN
# Initialize Dhan client (dhanhq v2.0)
dhan: dhanhq = dhanhq(CLIENT_ID, ACCESS_TOKEN)

# =============================================================================
# POLL FALLBACK CONFIGURATION
# =============================================================================
POLL_FALLBACK_ENABLED = config.get_trading_param('POLL_FALLBACK_ENABLED', True)
POLL_FALLBACK_THRESHOLD_SECONDS = config.get_trading_param('POLL_FALLBACK_THRESHOLD', 5)
_last_poll_time: datetime = datetime.now()
_poll_cooldown_seconds = config.get_trading_param('POLL_COOLDOWN', 2)


def _get_ltp_via_rest(
    security_id: str,
    exchange_segment_str: str
) -> Optional[float]:
    """
    Fallback: Get LTP via REST API when WebSocket data is stale.
    
    Uses dhan.quote_data() to fetch current price directly.
    
    Args:
        security_id: The security ID to fetch price for
        exchange_segment_str: Exchange segment (e.g., 'MCX_COMM', 'NSE_FNO')
        
    Returns:
        Current LTP or None if fetch fails
    """
    global _last_poll_time
    
    # Rate limit REST polls
    if (datetime.now() - _last_poll_time).total_seconds() < _poll_cooldown_seconds:
        return None
    
    try:
        # Map exchange segment to quote format
        quote_segment = exchange_segment_str.replace('_COMM', '')  # MCX_COMM -> MCX
        if quote_segment == 'NSE_FNO':
            quote_segment = 'NSE_FNO'
        elif quote_segment == 'MCX':
            quote_segment = 'MCX_COMM'  # Quote API uses MCX_COMM
        
        quote_response = dhan.quote_data({quote_segment: [int(security_id)]})
        _last_poll_time = datetime.now()
        
        if quote_response.get('status') == 'success':
            quote_data = quote_response.get('data', {}).get('data', {})
            security_quote = quote_data.get(str(security_id), quote_data.get(int(security_id), {}))
            
            if security_quote and 'last_price' in security_quote:
                ltp = float(security_quote['last_price'])
                logging.debug(f"📡 REST Poll: {security_id} LTP = {ltp}")
                return ltp
            elif security_quote and 'ltp' in security_quote:
                ltp = float(security_quote['ltp'])
                logging.debug(f"📡 REST Poll: {security_id} LTP = {ltp}")
                return ltp
        
        logging.debug(f"REST quote_data response: {quote_response}")
        return None
        
    except Exception as e:
        logging.warning(f"REST poll failed for {security_id}: {e}")
        return None


def get_ltp_with_fallback(
    active_trade: Dict[str, Any],
    active_instrument: str
) -> Tuple[float, float, bool]:
    """
    Get LTP values with REST API fallback when WebSocket data is stale.
    
    This is the critical poll fallback mechanism to ensure SL triggers
    even when WebSocket hangs without disconnecting.
    
    Args:
        active_trade: Active trade dictionary
        active_instrument: Current trading instrument
        
    Returns:
        Tuple of (future_ltp, option_ltp, used_fallback)
    """
    # Get WebSocket values
    latest_ltp = socket_handler.get_latest_ltp()
    option_ltp = socket_handler.get_option_ltp()
    last_tick_time = socket_handler.get_last_tick_time()
    last_option_tick_time = socket_handler.get_last_option_tick_time()
    
    used_fallback = False
    
    # Check if fallback is enabled
    if not POLL_FALLBACK_ENABLED:
        return latest_ltp, option_ltp, used_fallback
    
    # Check if we have an active trade
    if not active_trade.get("status"):
        return latest_ltp, option_ltp, used_fallback
    
    trade_instrument = active_trade.get("instrument", active_instrument)
    trade_exchange_segment = active_trade.get("exchange_segment_str", INSTRUMENTS[trade_instrument]["exchange_segment_str"])
    
    # Check if future data is stale (> 5 seconds)
    future_stale_seconds = (datetime.now() - last_tick_time).total_seconds()
    if future_stale_seconds > POLL_FALLBACK_THRESHOLD_SECONDS:
        logging.warning(f"⚠️ Future data stale ({future_stale_seconds:.1f}s), triggering REST fallback")
        
        # Get future LTP via REST
        future_id = str(INSTRUMENTS[trade_instrument]["future_id"])
        rest_ltp = _get_ltp_via_rest(future_id, trade_exchange_segment)
        
        if rest_ltp and rest_ltp > 0:
            latest_ltp = rest_ltp
            used_fallback = True
            logging.info(f"📡 Using REST fallback for future: {latest_ltp}")
    
    # Check if option data is stale (> 5 seconds) when we have an active position
    if option_ltp > 0:
        option_stale_seconds = (datetime.now() - last_option_tick_time).total_seconds()
        if option_stale_seconds > POLL_FALLBACK_THRESHOLD_SECONDS:
            logging.warning(f"⚠️ Option data stale ({option_stale_seconds:.1f}s), triggering REST fallback")
            
            # Get option LTP via REST
            option_id = active_trade.get("option_id")
            if option_id:
                rest_option_ltp = _get_ltp_via_rest(str(option_id), trade_exchange_segment)
                
                if rest_option_ltp and rest_option_ltp > 0:
                    option_ltp = rest_option_ltp
                    used_fallback = True
                    logging.info(f"📡 Using REST fallback for option: {option_ltp}")
    
    return latest_ltp, option_ltp, used_fallback


def close_trade(
    exit_reason: str,
    exit_price_future: float,
    exit_price_option: float,
    active_trade: Dict[str, Any]
) -> None:
    """Close trade and calculate actual P&L based on option premium"""
    
    # Get trade instrument info
    trade_instrument = active_trade.get("instrument", DEFAULT_INSTRUMENT)
    trade_lot_size = active_trade.get("lot_size", INSTRUMENTS[trade_instrument]["lot_size"])
    trade_exchange_segment = active_trade.get("exchange_segment_str", INSTRUMENTS[trade_instrument]["exchange_segment_str"])
    
    option_entry = active_trade.get("option_entry", 0)
    option_exit = exit_price_option
    
    # Calculate actual P&L in rupees
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
        'instrument': trade_instrument,
        'entry_time': active_trade.get('entry_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'trade_type': active_trade["type"],
        'option_type': "CALL" if active_trade["type"] == "BUY" else "PUT",
        'future_entry': active_trade.get("entry", 0),
        'future_exit': exit_price_future,
        'option_entry': option_entry,
        'option_exit': option_exit,
        'initial_sl': active_trade.get("initial_sl", 0),
        'final_sl': active_trade.get("sl", 0),
        'max_step_level': active_trade.get("step_level", 0),
        'pnl': pnl_per_lot,
        'r_multiple': r_multiple,
        'exit_reason': exit_reason,
        'lot_size': trade_lot_size
    }
    save_trade_to_history(trade_record)
    
    # Set cooldown if loss
    if not is_win:
        scanner.set_last_loss_time()
    
    # Unsubscribe from option feed
    market_feed = socket_handler.get_market_feed()
    if market_feed and active_trade.get("option_id"):
        inst = INSTRUMENTS.get(trade_instrument, {})
        exchange_segment_int = inst.get("exchange_segment_int", 5)
        socket_handler.unsubscribe_option(market_feed, active_trade["option_id"], exchange_segment_int)
    
    # Reset option LTP
    socket_handler.reset_option_ltp()
    
    trade_type = active_trade["type"]
    opt_type_str = "CE" if trade_type == "BUY" else "PE"
    
    # Get instrument name for display
    inst_name = INSTRUMENTS.get(trade_instrument, {}).get("name", trade_instrument)
    
    # Construct Option Name for Display
    atm_strike = active_trade.get("atm_strike", 0)
    option_name = f"{inst_name} {int(atm_strike)} {opt_type_str}" if atm_strike else f"{inst_name} {opt_type_str}"

    result_emoji = "✅" if is_win else "❌"
    pnl_sign = "+" if pnl_per_lot > 0 else ""
    
    logging.info(f"{result_emoji} TRADE CLOSED: {option_name} - {exit_reason}")
    logging.info(f"   Option Entry: ₹{option_entry} | Exit: ₹{option_exit}")
    logging.info(f"   P&L: {pnl_sign}₹{pnl_per_lot:.2f} | R-Multiple: {r_multiple:.2f}R")
    logging.info(f"   Daily P&L: ₹{daily_data['pnl']:.2f} | Trades: {daily_data['trades']}")
    
    send_alert(
        f"{result_emoji} **{option_name} CLOSED** - {exit_reason}\n"
        f"Option Entry: ₹{option_entry}\n"
        f"Option Exit: ₹{option_exit}\n"
        f"**P&L: {pnl_sign}₹{pnl_per_lot:.2f} ({r_multiple:.2f}R)**\n"
        f"Daily P&L: ₹{daily_data['pnl']:.2f}"
    )
    
    # Thread-safe update of active_trade
    trade_lock = scanner.get_trade_lock()
    with trade_lock:
        active_trade["status"] = False
        active_trade["instrument"] = None
        save_state(active_trade)


def run_manager(active_trade: Dict[str, Any], active_instrument: str) -> None:
    """Main manager loop for trade management"""
    logging.info(">>> Manager Started (Step Ladder Active)")

    while not socket_handler.is_shutdown():
        # Get current LTP values with REST API fallback for stale data
        latest_ltp, option_ltp, used_fallback = get_ltp_with_fallback(active_trade, active_instrument)
        last_tick_time = socket_handler.get_last_tick_time()
        last_option_tick_time = socket_handler.get_last_option_tick_time()
        
        # Log if fallback was used (only when there's an active trade)
        if used_fallback and active_trade["status"]:
            logging.info("📡 REST Poll Fallback activated for price verification")
        
        # Check for data feed lag (warning only - fallback handles SL)
        if (datetime.now() - last_tick_time).total_seconds() > 10 and active_trade["status"]:
            logging.warning("⚠️ FUTURE DATA FEED LAG DETECTED - REST fallback active")
        
        if active_trade["status"] and option_ltp > 0:
            if (datetime.now() - last_option_tick_time).total_seconds() > 10:
                logging.warning("⚠️ OPTION DATA FEED LAG DETECTED - REST fallback active")

        # === AUTO SQUARE-OFF CHECK ===
        if active_trade["status"]:
            trade_instrument = active_trade.get("instrument", active_instrument)
            trade_lot_size = active_trade.get("lot_size", INSTRUMENTS[trade_instrument]["lot_size"])
            trade_exchange_segment = active_trade.get("exchange_segment_str", INSTRUMENTS[trade_instrument]["exchange_segment_str"])
            trade_inst_config = INSTRUMENTS.get(trade_instrument, INSTRUMENTS[active_instrument])
            
            # Construct Option Name for Alert
            inst_name = trade_inst_config.get("name", trade_instrument)
            atm_strike = active_trade.get("atm_strike", 0)
            opt_type = "CE" if active_trade.get("type", "BUY") == "BUY" else "PE"
            option_name = f"{inst_name} {int(atm_strike)} {opt_type}" if atm_strike else f"{inst_name} {opt_type}"
            
            # Check auto square-off
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            market_end = trade_inst_config.get("market_end", "23:30")
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
                opt_ltp = option_ltp if option_ltp > 0 else active_trade.get("option_entry", 0)
                
                squareoff_limit_price = round(opt_ltp * (1 - LIMIT_ORDER_BUFFER), 2)
                
                # Check for Paper Trading
                is_paper_trading = config.get_trading_param("PAPER_TRADING", False)
                
                if is_paper_trading:
                    logging.info(f"📝 PAPER TRADING: Placing SQUARE-OFF EXIT order @ ₹{squareoff_limit_price}")
                    exit_response = {
                        "status": "success", 
                        "data": {
                            "orderId": f"PAPER_EXIT_SQ_{int(time.time()*1000)}",
                            "price": squareoff_limit_price
                        }
                    }
                else:
                    exit_response = dhan.place_order(
                        security_id=active_trade["option_id"],
                        exchange_segment=trade_exchange_segment,
                        transaction_type=dhan.SELL,
                        quantity=trade_lot_size,
                        order_type=dhan.LIMIT,
                        product_type=dhan.INTRADAY,
                        price=squareoff_limit_price
                    )
                
                exit_success, exit_details = scanner.verify_order(exit_response, "EXIT-AUTO-SQUAREOFF", symbol_name=option_name)
                exit_price = exit_details.get("avg_price", opt_ltp) if exit_success else opt_ltp
                
                close_trade(f"AUTO SQUARE-OFF ({square_off_msg})", latest_ltp, exit_price, active_trade)
                time.sleep(1)
                continue

        if latest_ltp != 0 and active_trade["status"]:
            trade_lot_size = active_trade.get("lot_size", INSTRUMENTS.get(active_trade.get("instrument", active_instrument), {}).get("lot_size", 10))
            trade_exchange_segment = active_trade.get("exchange_segment_str", INSTRUMENTS.get(active_trade.get("instrument", active_instrument), {}).get("exchange_segment_str", "MCX"))
            
            ltp = latest_ltp
            opt_ltp = option_ltp if option_ltp > 0 else active_trade.get("option_entry", 0)
            trade_type = active_trade["type"]

            # --- 1. SL HIT CHECK ---
            sl_hit = False
            if trade_type == "BUY":
                if ltp <= active_trade["sl"]:
                    sl_hit = True
            elif trade_type == "SELL":
                if ltp >= active_trade["sl"]:
                    sl_hit = True

            if sl_hit:
                logging.info(f"🛑 SL HIT @ Future: {ltp}")
                
                exit_limit_price = round(opt_ltp * (1 - LIMIT_ORDER_BUFFER), 2)
                
                # Check for Paper Trading
                is_paper_trading = config.get_trading_param("PAPER_TRADING", False)
                
                if is_paper_trading:
                    logging.info(f"📝 PAPER TRADING: Placing SL EXIT order @ ₹{exit_limit_price}")
                    exit_response = {
                        "status": "success", 
                        "data": {
                            "orderId": f"PAPER_EXIT_SL_{int(time.time()*1000)}",
                            "price": exit_limit_price
                        }
                    }
                else:
                    exit_response = dhan.place_order(
                        security_id=active_trade["option_id"],
                        exchange_segment=trade_exchange_segment,
                        transaction_type=dhan.SELL,
                        quantity=trade_lot_size,
                        order_type=dhan.LIMIT,
                        product_type=dhan.INTRADAY,
                        price=exit_limit_price
                    )
                
                exit_success, exit_details = scanner.verify_order(exit_response, "EXIT-SL", symbol_name=option_name)
                
                if exit_success:
                    exit_price = exit_details.get("avg_price", opt_ltp)
                else:
                    exit_price = opt_ltp
                    logging.warning("⚠️ Exit order verification failed, using last known option LTP")
                
                close_trade("SL HIT", ltp, exit_price, active_trade)
                continue

            # --- 2. PROFIT CALCULATION ---
            risk_unit = abs(active_trade["entry"] - active_trade["initial_sl"])
            if risk_unit == 0:
                risk_unit = 1

            if trade_type == "BUY":
                current_profit_future = ltp - active_trade["entry"]
            else:
                current_profit_future = active_trade["entry"] - ltp

            current_r = current_profit_future / risk_unit
            
            option_entry = active_trade.get("option_entry", 0)
            option_pnl = (opt_ltp - option_entry) * trade_lot_size

            # Jackpot Exit (1:5)
            if current_r >= 5.0:
                logging.info(f">>> 🎯 1:5 TARGET HIT! Option P&L: ₹{option_pnl:.2f}")
                
                target_exit_price = round(opt_ltp * (1 - LIMIT_ORDER_BUFFER), 2)
                
                # Check for Paper Trading
                is_paper_trading = config.get_trading_param("PAPER_TRADING", False)
                
                if is_paper_trading:
                    logging.info(f"📝 PAPER TRADING: Placing TARGET EXIT order @ ₹{target_exit_price}")
                    exit_response = {
                        "status": "success", 
                        "data": {
                            "orderId": f"PAPER_EXIT_TRGT_{int(time.time()*1000)}",
                            "price": target_exit_price
                        }
                    }
                else:
                    exit_response = dhan.place_order(
                        security_id=active_trade["option_id"], 
                        exchange_segment=trade_exchange_segment, 
                        transaction_type=dhan.SELL, 
                        quantity=trade_lot_size, 
                        order_type=dhan.LIMIT, 
                        product_type=dhan.INTRADAY, 
                        price=target_exit_price
                    )
                
                exit_success, exit_details = scanner.verify_order(exit_response, "EXIT-TARGET", symbol_name=option_name)
                exit_price = exit_details.get("avg_price", opt_ltp) if exit_success else opt_ltp
                
                close_trade("1:5 TARGET HIT", ltp, exit_price, active_trade)
                continue

            # Trailing Steps
            lock_r = 0
            msg = ""
            trade_lock = scanner.get_trade_lock()

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
                else:
                    new_sl = active_trade["entry"] - (lock_r * risk_unit)

                with trade_lock:
                    active_trade["sl"] = new_sl
                    save_state(active_trade)
                
                # Construct Option Name for Alert
                inst_name = INSTRUMENTS.get(active_trade.get("instrument", ""), {}).get("name", active_trade.get("instrument", ""))
                atm_strike = active_trade.get("atm_strike", 0)
                opt_type = "CE" if active_trade["type"] == "BUY" else "PE"
                option_name = f"{inst_name} {int(atm_strike)} {opt_type}" if atm_strike else f"{inst_name} {opt_type}"
                
                send_alert(f"{msg}\n**{option_name}**\n🔒 Locking {lock_r}R\nNew SL: {new_sl}\nOption P&L: ₹{option_pnl:.2f}")

        time.sleep(0.5)


def place_exit_order(active_trade: Dict[str, Any], exit_reason: str = "MANUAL") -> bool:
    """Place an exit order for manual/shutdown scenarios (V2 API)"""
    option_ltp = socket_handler.get_option_ltp()
    latest_ltp = socket_handler.get_latest_ltp()
    
    # -------------------------------------------------------------------------
    # SAFE SHUTDOWN LOGIC: Force fresh price fetch
    # -------------------------------------------------------------------------
    is_critical = any(k in exit_reason for k in ["SHUTDOWN", "EMERGENCY", "User", "Manual"])
    
    if is_critical:
        logging.info(f"🚨 CRITICAL EXIT ({exit_reason}): Forcing fresh price fetch via REST...")
        try:
            trade_exchange_segment = active_trade.get("exchange_segment_str", "MCX_COMM")
            option_id = active_trade.get("option_id")
            
            if option_id:
                # Ensure correct segment format for Quote API
                quote_segment = trade_exchange_segment
                if quote_segment == 'MCX': quote_segment = 'MCX_COMM'
                
                # Setup 1-second timeout manual poll
                quote_response = dhan.quote_data({quote_segment: [int(option_id)]})
                
                if quote_response.get('status') == 'success':
                    q_data = quote_response.get('data', {}).get('data', {})
                    sec_data = q_data.get(str(option_id)) or q_data.get(int(option_id))
                    
                    if sec_data:
                        fresh_ltp = float(sec_data.get('last_price', sec_data.get('ltp', 0)))
                        if fresh_ltp > 0:
                            logging.info(f"✅ Fresh REST Price: {fresh_ltp} (Socket was: {option_ltp})")
                            option_ltp = fresh_ltp
        except Exception as e:
            logging.error(f"⚠️ Failed to fetch fresh price during shutdown: {e}")
            # Fallback to socket_handler.get_option_ltp() value
            
    trade_lot_size = active_trade.get("lot_size", 10)
    # V2 API: Smart fallback for exchange segment
    trade_exchange_segment = active_trade.get("exchange_segment_str")
    if not trade_exchange_segment:
        # Fallback: Try to look up from INSTRUMENTS config
        inst_key = active_trade.get("instrument")
        if inst_key and inst_key in INSTRUMENTS:
            trade_exchange_segment = INSTRUMENTS[inst_key].get("exchange_segment_str", "MCX_COMM")
        else:
            trade_exchange_segment = "MCX_COMM"
    
    opt_ltp = option_ltp if option_ltp > 0 else active_trade.get("option_entry", 0)
    
    exit_limit_price = round(opt_ltp * (1 - LIMIT_ORDER_BUFFER), 2)
    
    # Check for Paper Trading
    is_paper_trading = config.get_trading_param("PAPER_TRADING", False)
    
    # Construct Option Name
    inst_name = INSTRUMENTS.get(active_trade.get("instrument", ""), {}).get("name", active_trade.get("instrument", ""))
    atm_strike = active_trade.get("atm_strike", 0)
    opt_type = "CE" if active_trade.get("type", "BUY") == "BUY" else "PE"
    option_name = f"{inst_name} {int(atm_strike)} {opt_type}" if atm_strike else f"{inst_name} {opt_type}"
    
    if is_paper_trading:
        logging.info(f"📝 PAPER TRADING: Placing EXIT order ({exit_reason}) @ ₹{exit_limit_price}")
        exit_response = {
            "status": "success", 
            "data": {
                "orderId": f"PAPER_EXIT_{int(time.time()*1000)}",
                "price": exit_limit_price
            }
        }
    else:
        exit_response = dhan.place_order(
            security_id=active_trade["option_id"],
            exchange_segment=trade_exchange_segment,
            transaction_type=dhan.SELL,
            quantity=trade_lot_size,
            order_type=dhan.LIMIT,
            product_type=dhan.INTRADAY,
            price=exit_limit_price
        )
    
    exit_success, exit_details = scanner.verify_order(exit_response, f"EXIT-{exit_reason}", symbol_name=option_name)
    
    if exit_success:
        exit_price = exit_details.get("avg_price", opt_ltp)
        close_trade(exit_reason, latest_ltp, exit_price, active_trade)
        return True
    else:
        logging.error(f"❌ Failed to close position: {exit_reason}")
        return False
