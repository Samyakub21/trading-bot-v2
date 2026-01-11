# =============================================================================
# SOCKET HANDLER - WebSocket Management for Market Data
# =============================================================================

import logging
import time
import threading
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dhanhq import marketfeed

from instruments import INSTRUMENTS, MULTI_SCAN_ENABLED, get_instruments_to_scan

# =============================================================================
# GLOBAL SOCKET STATE
# =============================================================================
MARKET_FEED = None
LATEST_LTP = 0              # Future/Underlying LTP
OPTION_LTP = 0              # Option Premium LTP
LAST_TICK_TIME = datetime.now()
LAST_OPTION_TICK_TIME = datetime.now()
INSTRUMENT_LTP = {}         # {instrument_key: {"ltp": price, "last_update": datetime}}

# Socket events
SOCKET_RECONNECT_EVENT = threading.Event()
SOCKET_HEALTHY = threading.Event()
SHUTDOWN_EVENT = threading.Event()


def get_all_instrument_subscriptions(active_instrument: str) -> List[Tuple[int, str, Any]]:
    """Get subscription list for all instruments to scan in multi-instrument mode"""
    subscriptions: List[Tuple[int, str, Any]] = []
    instruments_to_scan = get_instruments_to_scan() if MULTI_SCAN_ENABLED else [active_instrument]
    
    for inst_key in instruments_to_scan:
        inst = INSTRUMENTS[inst_key]
        subscriptions.append((inst["exchange_segment_int"], str(inst["future_id"]), marketfeed.Ticker))
    
    return subscriptions


def on_ticks(instance: Any, ticks: Dict[str, Any], active_instrument: str, active_trade: Dict[str, Any]) -> None:
    """Handle incoming tick data"""
    global LATEST_LTP, OPTION_LTP, LAST_TICK_TIME, LAST_OPTION_TICK_TIME
    
    if 'LTP' in ticks:
        security_id = str(ticks.get('security_id', ''))
        ltp = float(ticks['LTP'])
        
        # Mark socket as healthy on any tick
        SOCKET_HEALTHY.set()
        
        # Check if this tick is for the option or the future
        if active_trade.get("status") and security_id == str(active_trade.get("option_id")):
            OPTION_LTP = ltp
            LAST_OPTION_TICK_TIME = datetime.now()
        elif security_id == str(INSTRUMENTS[active_instrument]["future_id"]):
            LATEST_LTP = ltp
            LAST_TICK_TIME = datetime.now()
        else:
            # Check if it's any of our monitored instrument futures
            for inst_key, inst in INSTRUMENTS.items():
                if security_id == str(inst["future_id"]):
                    INSTRUMENT_LTP[inst_key] = {
                        "ltp": ltp,
                        "last_update": datetime.now()
                    }
                    # Update main LTP if it's the active instrument
                    if inst_key == active_instrument:
                        LATEST_LTP = ltp
                        LAST_TICK_TIME = datetime.now()
                    break


def subscribe_option(feed: Any, option_id: str, exchange_segment_int: int) -> None:
    """Subscribe to option feed for premium tracking"""
    try:
        sub_instruments = [(exchange_segment_int, str(option_id), marketfeed.Ticker)]
        feed.subscribe_symbols(sub_instruments)
        logging.info(f"📊 Subscribed to option feed: {option_id}")
    except Exception as e:
        logging.error(f"Failed to subscribe to option: {e}")


def unsubscribe_option(feed: Any, option_id: str, exchange_segment_int: int) -> None:
    """Unsubscribe from option feed"""
    try:
        unsub_instruments = [(exchange_segment_int, str(option_id), marketfeed.Ticker)]
        feed.unsubscribe_symbols(unsub_instruments)
        logging.info(f"📊 Unsubscribed from option feed: {option_id}")
    except Exception as e:
        logging.error(f"Failed to unsubscribe from option: {e}")


def socket_heartbeat_monitor() -> None:
    """Separate thread to monitor socket health and trigger reconnection"""
    
    logging.info(">>> Heartbeat Monitor Started")
    
    while not SHUTDOWN_EVENT.is_set():
        # Wait for either socket health signal or timeout
        socket_ok = SOCKET_HEALTHY.wait(timeout=30)
        
        if SHUTDOWN_EVENT.is_set():
            break
        
        if not socket_ok:
            # No tick received in 30 seconds
            logging.warning("⚠️ HEARTBEAT FAILED - No tick data for 30s. Triggering reconnection...")
            SOCKET_RECONNECT_EVENT.set()
        else:
            # Clear the flag for next cycle
            SOCKET_HEALTHY.clear()
        
        time.sleep(1)  # Small delay between checks


def start_socket(client_id: str, access_token: str, active_instrument: str, active_trade: Dict[str, Any]) -> None:
    """Start WebSocket connection for market data"""
    global MARKET_FEED
    
    logging.info(">>> Socket Connecting...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Subscribe to all instruments in multi-scan mode
    instruments = get_all_instrument_subscriptions(active_instrument)
    version = "v2"
    
    logging.info(f"📡 Subscribing to {len(instruments)} instrument feed(s)")
    for inst in instruments:
        logging.debug(f"   -> Exchange: {inst[0]}, Security: {inst[1]}")
    
    MARKET_FEED = marketfeed.DhanFeed(client_id, access_token, instruments, version)
    
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
                instruments = get_all_instrument_subscriptions(active_instrument)
                MARKET_FEED = marketfeed.DhanFeed(client_id, access_token, instruments, version)
                SOCKET_RECONNECT_EVENT.clear()
                logging.info("✅ Socket reconnected successfully")
            
            MARKET_FEED.run_forever()
            response = MARKET_FEED.get_data()
            if response and 'LTP' in response:
                on_ticks(MARKET_FEED, response, active_instrument, active_trade)
        except Exception as e:
            logging.error(f"Socket Error: {e}")
            time.sleep(5)
    
    # Graceful shutdown
    logging.info("🔌 Socket shutting down...")
    try:
        MARKET_FEED.close_connection()
    except:
        pass


def get_market_feed() -> Optional[Any]:
    """Get the current market feed instance"""
    return MARKET_FEED


def get_latest_ltp() -> float:
    """Get the latest LTP"""
    return LATEST_LTP


def get_option_ltp() -> float:
    """Get the option LTP"""
    return OPTION_LTP


def get_last_tick_time() -> datetime:
    """Get the last tick time"""
    return LAST_TICK_TIME


def get_last_option_tick_time() -> datetime:
    """Get the last option tick time"""
    return LAST_OPTION_TICK_TIME


def set_option_ltp(value: float) -> None:
    """Set the option LTP"""
    global OPTION_LTP
    OPTION_LTP = value


def reset_option_ltp() -> None:
    """Reset option LTP to 0"""
    global OPTION_LTP
    OPTION_LTP = 0


def shutdown_socket() -> None:
    """Signal socket shutdown"""
    SHUTDOWN_EVENT.set()


def is_shutdown() -> bool:
    """Check if shutdown is requested"""
    return SHUTDOWN_EVENT.is_set()
