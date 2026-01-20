# =============================================================================
# TRADING BOT - Main Entry Point
# =============================================================================
# A multi-instrument options trading bot for Dhan HQ
# Supports: CRUDEOIL, NATURALGAS, GOLD, SILVER, NIFTY, BANKNIFTY
# =============================================================================

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, cast

from config import config
from config import (
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_BACKUP_COUNT,
    LOG_FILE_PATH,
)
from instruments import (
    INSTRUMENTS,
    DEFAULT_INSTRUMENT,
    MULTI_SCAN_ENABLED,
    INSTRUMENT_PRIORITY,
    get_instruments_to_scan,
)
from utils import (
    load_state,
    save_state,
    load_daily_pnl,
    check_daily_limits,
    display_performance_report,
    send_alert,
    is_instrument_market_open,
    is_market_open,
    MAX_DAILY_LOSS,
    MAX_TRADES_PER_DAY,
    COOLDOWN_AFTER_LOSS,
    SIGNAL_COOLDOWN,
    RSI_BULLISH_THRESHOLD,
    RSI_BEARISH_THRESHOLD,
    VOLUME_MULTIPLIER,
    AUTO_SQUARE_OFF_BUFFER,
    LIMIT_ORDER_BUFFER,
)
import socket_handler
import scanner
import manager

# New modules for improved architecture
from state_stores import (
    get_market_data_store,
    get_signal_tracker,
    get_socket_state,
    get_trade_state_manager,
    MarketDataStore,
    SignalTracker,
)
from contract_updater import auto_update_instruments_on_startup, schedule_daily_update
from position_reconciliation import reconcile_on_startup, run_periodic_reconciliation


def setup_logging():
    """Configure structured logging for the trading bot."""
    import logging
    from logging.handlers import RotatingFileHandler

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger level
    log_level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    root_logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG level with rotation)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Log initial setup
    logging.info("🔧 Logging system initialized")
    logging.info(f"📁 Log file: {LOG_FILE_PATH}")
    logging.info(f"🐛 Debug mode: {'ENABLED' if config.DEBUG_MODE else 'DISABLED'}")


# Module-level logger for import-time logging
logger = logging.getLogger(__name__)


# Heartbeat / Dead Man's Switch
try:
    from heartbeat import (
        start_heartbeat,
        stop_heartbeat,
        increment_scanner_cycle,
        record_heartbeat_error,
        check_stop_signal,
    )

    HEARTBEAT_AVAILABLE = True
except ImportError:
    HEARTBEAT_AVAILABLE = False
    logger.warning("⚠️ Heartbeat module not available - dead man's switch disabled")

# End of Day Report
try:
    from eod_report import (
        schedule_eod_report,
        stop_eod_report,
        generate_and_send_eod_report,
    )

    EOD_REPORT_AVAILABLE = True
except ImportError:
    EOD_REPORT_AVAILABLE = False
    logger.warning("⚠️ EOD Report module not available - daily reports disabled")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# Configure StreamHandler to use UTF-8 encoding for Windows console
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and hasattr(handler.stream, "fileno"):
        stream_name = getattr(handler.stream, "name", None)
        if stream_name in ["<stdout>", "<stderr>"]:
            try:
                handler.setStream(
                    open(
                        handler.stream.fileno(),
                        mode="w",
                        encoding="utf-8",
                        buffering=1,
                        closefd=False,
                    )
                )
            except (OSError, AttributeError):
                pass


# =============================================================================
# TRADING BOT CLASS
# =============================================================================
@dataclass
class TradeBotConfig:
    """Configuration settings for the trading bot"""

    client_id: str
    access_token: str
    max_daily_loss: float = MAX_DAILY_LOSS
    max_trades_per_day: int = MAX_TRADES_PER_DAY
    cooldown_after_loss: int = COOLDOWN_AFTER_LOSS
    signal_cooldown: int = SIGNAL_COOLDOWN
    rsi_bullish_threshold: float = RSI_BULLISH_THRESHOLD
    rsi_bearish_threshold: float = RSI_BEARISH_THRESHOLD
    volume_multiplier: float = VOLUME_MULTIPLIER
    auto_square_off_buffer: int = AUTO_SQUARE_OFF_BUFFER
    limit_order_buffer: float = LIMIT_ORDER_BUFFER
    auto_close_on_shutdown: bool = True  # New config option


class TradingBot:
    """
    Main Trading Bot class that encapsulates all trading state and operations.

    Attributes:
        config: Bot configuration settings
        active_trade: Current trade state dictionary
        active_instrument: Currently active trading instrument
        threads: List of running threads
        is_running: Flag indicating if bot is running
    """

    def __init__(self, bot_config: TradeBotConfig):
        """
        Initialize the trading bot with configuration.

        Args:
            bot_config: TradeBotConfig instance with credentials and settings
        """
        self.logger = logging.getLogger("TradingBot")
        self.config = bot_config
        self.active_trade: Dict[str, Any] = load_state()
        self.active_instrument: str = (
            self.active_trade.get("instrument") or DEFAULT_INSTRUMENT
        )

        # Initialize state stores (replacing globals)
        self.market_data_store = get_market_data_store()
        self.signal_tracker = get_signal_tracker()
        self.socket_state = get_socket_state()
        self.threads: List[threading.Thread] = []
        self.is_running: bool = False
        self._lock = threading.Lock()

    @property
    def client_id(self) -> str:
        """Get client ID from config"""
        return self.config.client_id

    @property
    def access_token(self) -> str:
        """Get access token from config"""
        return self.config.access_token

    def graceful_shutdown(self, reason: str = "User request") -> None:
        """
        Handle graceful shutdown with open position management.

        Args:
            reason: Reason for shutdown
        """
        self.logger.info(f"🛑 Initiating graceful shutdown: {reason}")

        # Signal all threads to stop
        socket_handler.shutdown_socket()
        self.is_running = False

        # Check if there's an open trade
        with self._lock:
            if self.active_trade.get("status"):
                self.logger.warning("⚠️ OPEN POSITION DETECTED during shutdown!")
                trade_instrument = self.active_trade.get(
                    "instrument", self.active_instrument
                )

                send_alert(
                    f"🚨 **BOT SHUTDOWN WITH OPEN POSITION**\n"
                    f"Instrument: {trade_instrument}\n"
                    f"Type: {self.active_trade.get('type')}\n"
                    f"Entry: ₹{self.active_trade.get('option_entry', 0)}\n"
                    f"Current SL: {self.active_trade.get('sl', 0)}\n"
                    f"⚠️ MANUAL INTERVENTION MAY BE REQUIRED"
                )

                # Check config instead of input
                if self.config.auto_close_on_shutdown:
                    logging.info("Auto-closing open position due to shutdown...")

                    try:
                        success = manager.place_exit_order(
                            self.active_trade, f"SHUTDOWN ({reason})"
                        )
                        if success:
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
            save_state(self.active_trade)

        # Wait for threads to finish
        logging.info("Waiting for threads to finish...")
        time.sleep(3)

        # Stop heartbeat
        if HEARTBEAT_AVAILABLE:
            stop_heartbeat()

        # Stop EOD report scheduler
        if EOD_REPORT_AVAILABLE:
            try:
                stop_eod_report()
                logging.info("EOD Report scheduler stopped")
            except Exception as e:
                logging.debug(f"Error stopping EOD report: {e}")

        # Send shutdown alert (only if not already sent for open position)
        if not self.active_trade.get("status"):
            send_alert(
                f"🛑 **TRADING BOT STOPPED**\n"
                f"Reason: {reason}\n"
                f"No open positions."
            )

        logging.info("=" * 60)
        logging.info("🛑 TRADING BOT STOPPED")
        logging.info("=" * 60)

    def log_startup_info(self) -> None:
        """Log startup information including configuration and market status"""
        logging.info("=" * 60)
        logging.info("🚀 TRADING BOT STARTED")
        logging.info("=" * 60)

        # Multi-instrument mode info
        if MULTI_SCAN_ENABLED:
            instruments_to_scan = get_instruments_to_scan()
            logging.info("🔄 MODE: Multi-Instrument Scanning ENABLED")
            logging.info(f"📊 Scanning {len(instruments_to_scan)} instruments:")
            for i, inst_key in enumerate(instruments_to_scan, 1):
                inst = INSTRUMENTS[inst_key]
                priority = INSTRUMENT_PRIORITY.get(inst_key, 99)
                logging.info(
                    f"   {i}. {inst['name']} ({inst_key}) - Priority: {priority}"
                )
        else:
            inst = INSTRUMENTS[self.active_instrument]
            logging.info("📊 MODE: Single Instrument")
            logging.info(
                f"📊 Active Instrument: {inst['name']} ({self.active_instrument})"
            )
            logging.info(f"📈 Exchange: {inst['exchange_segment_str']}")
            logging.info(f"🔢 Future ID: {inst['future_id']}")
            logging.info(f"📦 Lot Size: {inst['lot_size']}")
            logging.info(f"📅 Expiry: {inst['expiry_date']}")
            logging.info(f"🎯 Strike Step: {inst['strike_step']}")

        logging.info("-" * 60)

        # Show market hours
        if MULTI_SCAN_ENABLED:
            logging.info("⏰ Market Hours per Instrument:")
            mcx_shown = False
            nse_shown = False
            for inst_key in get_instruments_to_scan():
                inst = INSTRUMENTS[inst_key]
                if inst["exchange_segment_str"] == "MCX_COMM" and not mcx_shown:
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
            inst = INSTRUMENTS[self.active_instrument]
            # FIX: Cast to str before split
            market_start = str(inst.get("market_start", "09:00"))
            market_end = str(inst.get("market_end", "23:30"))

            logging.info(f"⏰ Market Hours: {market_start} - {market_end}")
            logging.info(f"🚫 No New Trades After: {inst.get('no_new_trade_after')}")

            # FIX: Ensure we are splitting a string
            market_end_hour, market_end_min = map(int, market_end.split(":"))
            sq_min = market_end_min - self.config.auto_square_off_buffer
            sq_hour = market_end_hour
            if sq_min < 0:
                sq_min += 60
                sq_hour -= 1
            logging.info(f"🔄 Auto Square-Off At: {sq_hour:02d}:{sq_min:02d}")

        logging.info("-" * 60)
        logging.info(f"💰 Max Daily Loss: ₹{self.config.max_daily_loss}")
        logging.info(f"📊 Max Trades/Day: {self.config.max_trades_per_day}")
        logging.info(f"⏳ Cooldown After Loss: {self.config.cooldown_after_loss}s")
        logging.info(f"🔁 Signal Cooldown (Whipsaw): {self.config.signal_cooldown}s")
        logging.info("-" * 60)
        logging.info(f"📈 RSI Bullish Threshold: > {self.config.rsi_bullish_threshold}")
        logging.info(f"📉 RSI Bearish Threshold: < {self.config.rsi_bearish_threshold}")
        logging.info(f"📊 Volume Multiplier: {self.config.volume_multiplier}x avg")
        logging.info("-" * 60)

        # Load and display daily stats
        daily_stats = load_daily_pnl()
        logging.info(
            f"📈 Today's Stats: P&L: ₹{daily_stats['pnl']} | Trades: {daily_stats['trades']} | W:{daily_stats['wins']} L:{daily_stats['losses']}"
        )
        logging.info("=" * 60)

        # Display historical performance
        display_performance_report(days=30)
        time.sleep(1)
        display_performance_report()

        # Perform database backup on startup
        try:
            from database import DatabaseManager

            db_manager = DatabaseManager()
            db_manager.backup_database()
        except Exception as e:
            logging.warning(f"Startup backup failed: {e}")

        # Check market status at startup
        if MULTI_SCAN_ENABLED:
            logging.info("🏪 Market Status (per instrument):")
            for inst_key in get_instruments_to_scan():
                # FIX: Explicit cast to str
                market_open, market_msg = is_instrument_market_open(str(inst_key))
                # FIX: Explicit cast to str
                market_open, market_msg = is_instrument_market_open(str(inst_key))
                status_icon = "✅" if market_open else "⏸️"
                logging.info(f"   {status_icon} {inst_key}: {market_msg}")
        else:
            inst = INSTRUMENTS[self.active_instrument]
            # FIX: Explicit cast to str
            start_str = str(inst.get("market_start", "09:00"))
            end_str = str(inst.get("market_end", "23:30"))
            market_open, market_msg = is_market_open(start_str, end_str)
            # FIX: Explicit cast to str
            start_str = str(inst.get("market_start", "09:00"))
            end_str = str(inst.get("market_end", "23:30"))
            market_open, market_msg = is_market_open(start_str, end_str)
            logging.info(f"🏪 Market Status: {market_msg}")

        # Check daily limits at startup
        within_limits, limits_msg = check_daily_limits()
        if not within_limits:
            logging.warning(f"⚠️ {limits_msg}")

    def _start_socket_thread(self) -> threading.Thread:
        """Start the WebSocket thread for market data"""
        thread = threading.Thread(
            target=socket_handler.start_socket,
            args=(
                self.client_id,
                self.access_token,
                self.active_instrument,
                self.active_trade,
            ),
            name="SocketThread",
        )
        thread.start()
        return thread

    def _start_scanner_thread(self) -> threading.Thread:
        """Start the market scanner thread"""
        thread = threading.Thread(
            target=scanner.run_scanner,
            args=(self.active_trade, self.active_instrument),
            daemon=True,
            name="ScannerThread",
        )
        thread.start()
        return thread

    def _start_manager_thread(self) -> threading.Thread:
        """Start the trade manager thread"""
        thread = threading.Thread(
            target=manager.run_manager,
            args=(self.active_trade, self.active_instrument),
            daemon=True,
            name="ManagerThread",
        )
        thread.start()
        return thread

    def _start_reconciliation_thread(self) -> threading.Thread:
        """Start the position reconciliation thread"""
        thread = run_periodic_reconciliation(self.active_trade, interval_seconds=300)
        return thread

    def start(self) -> None:
        """Start the trading bot and all worker threads"""
        # Start heartbeat (dead man's switch)
        if HEARTBEAT_AVAILABLE:
            start_heartbeat()
            logging.info("💓 Heartbeat monitor started")

        # Auto-update instrument contracts on startup
        logging.info("🔄 Checking for contract updates...")
        try:
            global INSTRUMENTS
            INSTRUMENTS = auto_update_instruments_on_startup(INSTRUMENTS)
            logging.info("✅ Instrument contracts updated")
        except Exception as e:
            logging.warning(f"⚠️ Contract update failed (using existing): {e}")

        # Run startup position reconciliation
        logging.info("🔍 Running position reconciliation...")
        reconcile_ok = reconcile_on_startup(self.active_trade)
        if not reconcile_ok:
            logging.warning("⚠️ Position reconciliation had issues - verify manually")

        self.log_startup_info()
        self.is_running = True

        # Start socket thread
        socket_thread = self._start_socket_thread()
        self.threads.append(socket_thread)

        logging.info("Waiting for market data feed...")
        for _ in range(15):  # Wait up to 15 seconds
            if socket_handler.get_latest_ltp() > 0:
                logging.info("✅ Market data received")
                break
            time.sleep(1)
        else:
            logging.warning("⚠️ Starting scanner without live data (LTP is 0)")

        # Start scanner thread
        scanner_thread = self._start_scanner_thread()
        self.threads.append(scanner_thread)

        # Start manager thread
        manager_thread = self._start_manager_thread()
        self.threads.append(manager_thread)

        # Start reconciliation thread
        reconciliation_thread = self._start_reconciliation_thread()
        self.threads.append(reconciliation_thread)

        # Schedule daily contract updates (before market open)
        schedule_daily_update(INSTRUMENTS, update_time="08:55")

        # Schedule End of Day Report (at 11:35 PM)
        if EOD_REPORT_AVAILABLE:
            try:
                schedule_eod_report()
                logging.info("📊 End of Day Report scheduled for 23:35")
            except Exception as e:
                logging.warning(f"⚠️ Failed to schedule EOD report: {e}")

        # Send startup alert
        if MULTI_SCAN_ENABLED:
            instruments_list = ", ".join(get_instruments_to_scan())
            send_alert(
                f"🚀 **TRADING BOT STARTED**\n"
                f"Mode: Multi-Instrument Scanning\n"
                f"Instruments: {instruments_list}\n"
                f"All systems initialized ✅"
            )
        else:
            inst = INSTRUMENTS[self.active_instrument]
            send_alert(
                f"🚀 **TRADING BOT STARTED**\n"
                f"Mode: Single Instrument\n"
                f"Instrument: {inst['name']} ({self.active_instrument})\n"
                f"All systems initialized ✅"
            )

    def run(self) -> None:
        """Main run loop - blocks until shutdown"""
        try:
            while self.is_running and not socket_handler.is_shutdown():
                # Check for external stop signal (from dashboard panic button)
                if HEARTBEAT_AVAILABLE:
                    should_stop, reason = check_stop_signal()
                    if should_stop:
                        logging.warning(f"⚠️ External stop signal received: {reason}")
                        self.graceful_shutdown(f"External stop: {reason}")
                        break

                time.sleep(1)
        except KeyboardInterrupt:
            self.graceful_shutdown("User interrupted (Ctrl+C)")
        except Exception as e:
            logging.error(f"Bot error: {e}")
            if HEARTBEAT_AVAILABLE:
                record_heartbeat_error(str(e))
            self.graceful_shutdown(f"Error: {e}")

    def get_trade_state(self) -> Dict[str, Any]:
        """Thread-safe access to current trade state"""
        with self._lock:
            return self.active_trade.copy()

    def update_trade_state(self, updates: Dict[str, Any]) -> None:
        """Thread-safe update of trade state"""
        with self._lock:
            self.active_trade.update(updates)
            save_state(self.active_trade)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def create_bot() -> TradingBot:
    """Factory function to create a configured TradingBot instance"""
    bot_config = TradeBotConfig(
        client_id=config.CLIENT_ID, access_token=config.ACCESS_TOKEN
    )
    return TradingBot(bot_config)


if __name__ == "__main__":
    # Initialize logging system
    setup_logging()

    # Create and start the trading bot
    bot = create_bot()
    bot.start()
    bot.run()
