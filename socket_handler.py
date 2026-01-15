# =============================================================================
# SOCKET HANDLER - WebSocket Management for Market Data
# =============================================================================

import logging
import time
import threading
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dhanhq import marketfeed

from instruments import INSTRUMENTS, MULTI_SCAN_ENABLED, get_instruments_to_scan
from config import config

# =============================================================================
# CONFIGURATION & STATE
# =============================================================================
CLIENT_ID = config.CLIENT_ID
ACCESS_TOKEN = config.ACCESS_TOKEN
DATA_DIR = Path(__file__).parent / 'data'  # Force 'data' folder

MARKET_FEED = None
LATEST_LTP = 0
OPTION_LTP = 0
LAST_TICK_TIME = datetime.now()
LAST_OPTION_TICK_TIME = datetime.now()
INSTRUMENT_LTP = {}
MESSAGE_COUNT = 0  # Track total messages received

# Events
SOCKET_RECONNECT_EVENT = threading.Event()
SOCKET_HEALTHY = threading.Event()
SHUTDOWN_EVENT = threading.Event()

def get_all_instrument_subscriptions(active_instrument: str) -> List[Tuple[int, str, Any]]:
    """Get subscription list for all instruments"""
    subscriptions: List[Tuple[int, str, Any]] = []
    instruments_to_scan = get_instruments_to_scan() if MULTI_SCAN_ENABLED else [active_instrument]
    
    for inst_key in instruments_to_scan:
        inst = INSTRUMENTS[inst_key]
        subscriptions.append((inst["exchange_segment_int"], str(inst["future_id"]), marketfeed.Ticker))
    
    return subscriptions

def on_ticks(instance: Any, ticks: Dict[str, Any], active_instrument: str, active_trade: Dict[str, Any]) -> None:
    """Handle incoming tick data"""
    global LATEST_LTP, OPTION_LTP, LAST_TICK_TIME, LAST_OPTION_TICK_TIME, MESSAGE_COUNT
    
    # Increment counter for dashboard
    MESSAGE_COUNT += 1

    if 'LTP' in ticks:
        security_id = str(ticks.get('security_id', ''))
        ltp = float(ticks['LTP'])
        
        SOCKET_HEALTHY.set()
        
        if active_trade.get("status") and security_id == str(active_trade.get("option_id")):
            OPTION_LTP = ltp
            LAST_OPTION_TICK_TIME = datetime.now()
        elif security_id == str(INSTRUMENTS[active_instrument]["future_id"]):
            LATEST_LTP = ltp
            LAST_TICK_TIME = datetime.now()
        else:
            for inst_key, inst in INSTRUMENTS.items():
                if security_id == str(inst["future_id"]):
                    INSTRUMENT_LTP[inst_key] = {
                        "ltp": ltp,
                        "last_update": datetime.now()
                    }
                    if inst_key == active_instrument:
                        LATEST_LTP = ltp
                        LAST_TICK_TIME = datetime.now()
                    break

def write_socket_status():
    """Write REAL-TIME status to JSON for dashboard"""
    try:
        # Check if connected (Feed exists + Not shutting down)
        connected = MARKET_FEED is not None and not SHUTDOWN_EVENT.is_set()
        
        status = {
            "connected": connected,
            "last_message_time": LAST_TICK_TIME.isoformat(),
            "latency_ms": 0, # Placeholder
            "messages_received": MESSAGE_COUNT,
            "errors": 0,
            "reconnect_count": 0,
            "subscribed_symbols": list(INSTRUMENTS.keys()),
            # Extract simple LTPs for dashboard display
            "last_prices": {k: {"ltp": v.get("ltp", 0)} for k, v in INSTRUMENT_LTP.items()}
        }
        
        # Ensure directory exists
        DATA_DIR.mkdir(exist_ok=True)
        
        with open(DATA_DIR / "websocket_status.json", "w") as f:
            json.dump(status, f)
            
    except Exception as e:
        logging.error(f"Failed to write socket status: {e}")

def socket_heartbeat_monitor() -> None:
    """Monitor health and update dashboard status file"""
    logging.info(">>> Heartbeat Monitor Started")
    
    while not SHUTDOWN_EVENT.is_set():
        # Update the dashboard file every second
        write_socket_status()

        socket_ok = SOCKET_HEALTHY.wait(timeout=config.HEARTBEAT_TIMEOUT_SECONDS)
        
        if SHUTDOWN_EVENT.is_set():
            break
        
        if not socket_ok:
            logging.warning("⚠️ HEARTBEAT FAILED - No tick data. Reconnecting...")
            SOCKET_RECONNECT_EVENT.set()
        else:
            SOCKET_HEALTHY.clear()
        
        time.sleep(1)

def start_socket(client_id: str, access_token: str, active_instrument: str, active_trade: Dict[str, Any]) -> None:
    """Start WebSocket connection"""
    global MARKET_FEED
    
    logging.info(">>> Socket Connecting...")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    instruments = get_all_instrument_subscriptions(active_instrument)
    MARKET_FEED = marketfeed.DhanFeed(client_id, access_token, instruments, "v2")
    
    # Start heartbeat monitor (which also writes status file)
    heartbeat_thread = threading.Thread(target=socket_heartbeat_monitor, daemon=True)
    heartbeat_thread.start()
    
    retry_delay = 2
    
    while not SHUTDOWN_EVENT.is_set():
        try:
            if SOCKET_RECONNECT_EVENT.is_set():
                logging.info("🔄 Reconnecting socket...")
                try: MARKET_FEED.close_connection()
                except: pass
                time.sleep(retry_delay)
                MARKET_FEED = marketfeed.DhanFeed(client_id, access_token, instruments, "v2")
                SOCKET_RECONNECT_EVENT.clear()
            
            MARKET_FEED.run_forever()
            response = MARKET_FEED.get_data()
            if response and 'LTP' in response:
                on_ticks(MARKET_FEED, response, active_instrument, active_trade)
                retry_delay = 2
        except Exception as e:
            logging.error(f"Connection failed: {e}. Retrying...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
    
    try: MARKET_FEED.close_connection()
    except: pass

# ... Keep existing getters (get_market_feed, etc.) ...
def get_market_feed() -> Optional[Any]: return MARKET_FEED
def get_latest_ltp() -> float: return LATEST_LTP
def get_option_ltp() -> float: return OPTION_LTP
def get_last_tick_time() -> datetime: return LAST_TICK_TIME
def get_last_option_tick_time() -> datetime: return LAST_OPTION_TICK_TIME
def set_option_ltp(value: float) -> None: global OPTION_LTP; OPTION_LTP = value
def reset_option_ltp() -> None: global OPTION_LTP; OPTION_LTP = 0
def shutdown_socket() -> None: SHUTDOWN_EVENT.set()
def is_shutdown() -> bool: return SHUTDOWN_EVENT.is_set()
def should_process_tick(now_ms: int) -> bool: return (now_ms - LAST_TICK_TIME.timestamp() * 1000) >= config.MIN_TICK_INTERVAL_MS


def subscribe_option(market_feed, security_id: str, exchange_segment: int) -> None:
    """Subscribe to a specific option contract"""
    if market_feed:
        try:
            market_feed.subscribe_symbols([(exchange_segment, str(security_id))])
            logging.info(f"✅ Subscribed to option: {security_id}")
        except Exception as e:
            logging.error(f"Failed to subscribe to option {security_id}: {e}")

def unsubscribe_option(market_feed, security_id: str, exchange_segment: int) -> None:
    """Unsubscribe from a specific option contract"""
    if market_feed:
        try:
            market_feed.unsubscribe_symbols([(exchange_segment, str(security_id))])
            logging.info(f"Unsubscribed from option: {security_id}")
        except Exception as e:
            logging.error(f"Failed to unsubscribe from option {security_id}: {e}")