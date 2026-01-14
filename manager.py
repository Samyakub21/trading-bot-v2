# =============================================================================
# MANAGER - Trade Management and Trailing Stop Loss
# =============================================================================

import logging
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional
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
dhan = dhanhq(CLIENT_ID, ACCESS_TOKEN)


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
    opt_type = "CALL" if trade_type == "BUY" else "PUT"
    
    # Get instrument name for display
    inst_name = INSTRUMENTS.get(trade_instrument, {}).get("name", trade_instrument)
    
    result_emoji = "✅" if is_win else "❌"
    pnl_sign = "+" if pnl_per_lot > 0 else ""
    
    logging.info(f"{result_emoji} TRADE CLOSED: {inst_name} - {exit_reason}")
    logging.info(f"   Option Entry: ₹{option_entry} | Exit: ₹{option_exit}")
    logging.info(f"   P&L: {pnl_sign}₹{pnl_per_lot:.2f} | R-Multiple: {r_multiple:.2f}R")
    logging.info(f"   Daily P&L: ₹{daily_data['pnl']:.2f} | Trades: {daily_data['trades']}")
    
    send_alert(
        f"{result_emoji} **{inst_name} {opt_type} CLOSED** - {exit_reason}\n"
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
        # Get current LTP values
        latest_ltp = socket_handler.get_latest_ltp()
        option_ltp = socket_handler.get_option_ltp()
        last_tick_time = socket_handler.get_last_tick_time()
        last_option_tick_time = socket_handler.get_last_option_tick_time()
        
        # Check for data feed lag
        if (datetime.now() - last_tick_time).total_seconds() > 10 and active_trade["status"]:
            logging.warning("⚠️ FUTURE DATA FEED LAG DETECTED - WATCH MANUALLY")
        
        if active_trade["status"] and option_ltp > 0:
            if (datetime.now() - last_option_tick_time).total_seconds() > 10:
                logging.warning("⚠️ OPTION DATA FEED LAG DETECTED")

        # === AUTO SQUARE-OFF CHECK ===
        if active_trade["status"]:
            trade_instrument = active_trade.get("instrument", active_instrument)
            trade_lot_size = active_trade.get("lot_size", INSTRUMENTS[trade_instrument]["lot_size"])
            trade_exchange_segment = active_trade.get("exchange_segment_str", INSTRUMENTS[trade_instrument]["exchange_segment_str"])
            trade_inst_config = INSTRUMENTS.get(trade_instrument, INSTRUMENTS[active_instrument])
            
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
                
                exit_response = dhan.place_order(
                    security_id=active_trade["option_id"],
                    exchange_segment=trade_exchange_segment,
                    transaction_type=dhan.SELL,
                    quantity=trade_lot_size,
                    order_type=dhan.LIMIT,
                    product_type=dhan.INTRADAY,
                    price=squareoff_limit_price
                )
                
                exit_success, exit_details = scanner.verify_order(exit_response, "EXIT-AUTO-SQUAREOFF")
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
                
                exit_response = dhan.place_order(
                    security_id=active_trade["option_id"],
                    exchange_segment=trade_exchange_segment,
                    transaction_type=dhan.SELL,
                    quantity=trade_lot_size,
                    order_type=dhan.LIMIT,
                    product_type=dhan.INTRADAY,
                    price=exit_limit_price
                )
                
                exit_success, exit_details = scanner.verify_order(exit_response, "EXIT-SL")
                
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
                
                exit_response = dhan.place_order(
                    security_id=active_trade["option_id"], 
                    exchange_segment=trade_exchange_segment, 
                    transaction_type=dhan.SELL, 
                    quantity=trade_lot_size, 
                    order_type=dhan.LIMIT, 
                    product_type=dhan.INTRADAY, 
                    price=target_exit_price
                )
                
                exit_success, exit_details = scanner.verify_order(exit_response, "EXIT-TARGET")
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
                send_alert(f"{msg}\n🔒 Locking {lock_r}R\nNew SL: {new_sl}\nOption P&L: ₹{option_pnl:.2f}")

        time.sleep(0.5)


def place_exit_order(active_trade: Dict[str, Any], exit_reason: str = "MANUAL") -> bool:
    """Place an exit order for manual/shutdown scenarios (V2 API)"""
    option_ltp = socket_handler.get_option_ltp()
    latest_ltp = socket_handler.get_latest_ltp()
    
    trade_lot_size = active_trade.get("lot_size", 10)
    # V2 API: Default to MCX_COMM for commodity options
    trade_exchange_segment = active_trade.get("exchange_segment_str", "MCX_COMM")
    
    opt_ltp = option_ltp if option_ltp > 0 else active_trade.get("option_entry", 0)
    
    exit_limit_price = round(opt_ltp * (1 - LIMIT_ORDER_BUFFER), 2)
    
    exit_response = dhan.place_order(
        security_id=active_trade["option_id"],
        exchange_segment=trade_exchange_segment,
        transaction_type=dhan.SELL,
        quantity=trade_lot_size,
        order_type=dhan.LIMIT,
        product_type=dhan.INTRADAY,
        price=exit_limit_price
    )
    
    exit_success, exit_details = scanner.verify_order(exit_response, f"EXIT-{exit_reason}")
    
    if exit_success:
        exit_price = exit_details.get("avg_price", opt_ltp)
        close_trade(exit_reason, latest_ltp, exit_price, active_trade)
        return True
    else:
        logging.error(f"❌ Failed to close position: {exit_reason}")
        return False
