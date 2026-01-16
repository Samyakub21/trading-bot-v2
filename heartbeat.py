# =============================================================================
# HEARTBEAT MONITOR - Dead Man's Switch for Trading Bot
# =============================================================================
"""
Implements a heartbeat system to detect when the bot freezes.
The bot updates heartbeat.json every few seconds. An external monitor
(cron job or separate process) checks this file and restarts the service
if it's stale.

Usage:
    # In the main bot:
    from heartbeat import start_heartbeat, stop_heartbeat
    start_heartbeat()

    # External monitor (run via cron or systemd):
    python heartbeat.py --monitor
"""

import json
import logging
import os
import sys
import time
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent
HEARTBEAT_FILE = DATA_DIR / "heartbeat.json"
BOT_CONTROL_FILE = DATA_DIR / "bot_control.json"

# Heartbeat settings
HEARTBEAT_INTERVAL = 10  # Update heartbeat every 10 seconds
HEARTBEAT_TIMEOUT = 60  # Consider bot dead if heartbeat older than 60 seconds

# Alert settings (loaded from credentials if available)
try:
    from utils import send_alert, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    TELEGRAM_TOKEN = None
    TELEGRAM_CHAT_ID = None


# =============================================================================
# HEARTBEAT DATA
# =============================================================================


class HeartbeatData:
    """Represents heartbeat status data."""

    def __init__(
        self,
        timestamp: datetime,
        status: str = "running",
        active_trade: bool = False,
        instrument: Optional[str] = None,
        last_tick_time: Optional[datetime] = None,
        scanner_cycle: int = 0,
        memory_mb: float = 0,
        threads_active: int = 0,
        error_count: int = 0,
        last_error: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.status = status
        self.active_trade = active_trade
        self.instrument = instrument
        self.last_tick_time = last_tick_time
        self.scanner_cycle = scanner_cycle
        self.memory_mb = memory_mb
        self.threads_active = threads_active
        self.error_count = error_count
        self.last_error = last_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "active_trade": self.active_trade,
            "instrument": self.instrument,
            "last_tick_time": (
                self.last_tick_time.isoformat() if self.last_tick_time else None
            ),
            "scanner_cycle": self.scanner_cycle,
            "memory_mb": round(self.memory_mb, 2),
            "threads_active": self.threads_active,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "pid": os.getpid(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeartbeatData":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=data.get("status", "unknown"),
            active_trade=data.get("active_trade", False),
            instrument=data.get("instrument"),
            last_tick_time=(
                datetime.fromisoformat(data["last_tick_time"])
                if data.get("last_tick_time")
                else None
            ),
            scanner_cycle=data.get("scanner_cycle", 0),
            memory_mb=data.get("memory_mb", 0),
            threads_active=data.get("threads_active", 0),
            error_count=data.get("error_count", 0),
            last_error=data.get("last_error"),
        )

    def age_seconds(self) -> float:
        """Get age of heartbeat in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()

    def is_stale(self, timeout: int = HEARTBEAT_TIMEOUT) -> bool:
        """Check if heartbeat is stale (older than timeout)."""
        return self.age_seconds() > timeout


# =============================================================================
# HEARTBEAT WRITER (Used by Trading Bot)
# =============================================================================


class HeartbeatWriter:
    """
    Writes heartbeat data to file periodically.
    Run this in the trading bot process.
    """

    def __init__(self, interval: int = HEARTBEAT_INTERVAL):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._scanner_cycle = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0
        except Exception:
            return 0

    def _get_active_threads(self) -> int:
        """Get number of active threads."""
        return threading.active_count()

    def _write_heartbeat(self) -> None:
        """Write heartbeat data to file."""
        try:
            # Try to get trade state
            active_trade = False
            instrument = None
            last_tick_time = None

            try:
                from state_stores import get_trade_state_manager

                trade_manager = get_trade_state_manager()
                if trade_manager:
                    # FIX: 'is_active' is likely a bool property or variable, not a method
                    # Use getattr to be safe, or just access it without ()
                    active_trade = getattr(trade_manager, "is_active", False)
                    # If it IS a method, this will return the method object which is truthy
                    # If it's a bool, it returns the bool.
                    # If the error was "bool not callable", it was definitely a bool.
                    if callable(active_trade):
                        active_trade = active_trade()

                    if active_trade:
                        instrument = trade_manager.get_instrument()
            except Exception:
                pass

            try:
                import socket_handler

                last_tick = socket_handler.get_last_tick_time()
                if last_tick:
                    last_tick_time = last_tick
            except Exception:
                pass

            with self._lock:
                heartbeat = HeartbeatData(
                    timestamp=datetime.now(),
                    status="running",
                    active_trade=active_trade,
                    instrument=instrument,
                    last_tick_time=last_tick_time,
                    scanner_cycle=self._scanner_cycle,
                    memory_mb=self._get_memory_usage(),
                    threads_active=self._get_active_threads(),
                    error_count=self._error_count,
                    last_error=self._last_error,
                )

            with open(HEARTBEAT_FILE, "w") as f:
                json.dump(heartbeat.to_dict(), f, indent=2)

        except Exception as e:
            logging.debug(f"Heartbeat write error: {e}")

    def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            self._write_heartbeat()
            time.sleep(self.interval)

    def start(self) -> None:
        """Start the heartbeat writer."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="HeartbeatWriter"
        )
        self._thread.start()
        logging.info(f"💓 Heartbeat started (interval: {self.interval}s)")

    def stop(self) -> None:
        """Stop the heartbeat writer."""
        self._running = False

        # Write final "stopped" heartbeat
        try:
            heartbeat = HeartbeatData(
                timestamp=datetime.now(),
                status="stopped",
                scanner_cycle=self._scanner_cycle,
                error_count=self._error_count,
            )
            with open(HEARTBEAT_FILE, "w") as f:
                json.dump(heartbeat.to_dict(), f, indent=2)
        except Exception:
            pass

        logging.info("💔 Heartbeat stopped")

    def increment_scanner_cycle(self) -> None:
        """Increment scanner cycle counter."""
        with self._lock:
            self._scanner_cycle += 1

    def record_error(self, error: str) -> None:
        """Record an error for heartbeat."""
        with self._lock:
            self._error_count += 1
            self._last_error = str(error)[:200]  # Truncate long errors


# =============================================================================
# SINGLETON WRITER INSTANCE
# =============================================================================

_heartbeat_writer: Optional[HeartbeatWriter] = None
_writer_lock = threading.Lock()


def get_heartbeat_writer() -> HeartbeatWriter:
    """Get the singleton HeartbeatWriter instance."""
    global _heartbeat_writer
    if _heartbeat_writer is None:
        with _writer_lock:
            if _heartbeat_writer is None:
                _heartbeat_writer = HeartbeatWriter()
    return _heartbeat_writer


def start_heartbeat() -> None:
    """Start the heartbeat writer."""
    get_heartbeat_writer().start()


def stop_heartbeat() -> None:
    """Stop the heartbeat writer."""
    if _heartbeat_writer:
        _heartbeat_writer.stop()


def increment_scanner_cycle() -> None:
    """Increment scanner cycle (call from scanner loop)."""
    if _heartbeat_writer:
        _heartbeat_writer.increment_scanner_cycle()


def record_heartbeat_error(error: str) -> None:
    """Record an error in heartbeat."""
    if _heartbeat_writer:
        _heartbeat_writer.record_error(error)


# =============================================================================
# HEARTBEAT MONITOR (External Process / Cron Job)
# =============================================================================


class HeartbeatMonitor:
    """
    Monitors the heartbeat file and takes action if it's stale.
    Run this as a separate process or cron job.
    """

    def __init__(
        self,
        timeout: int = HEARTBEAT_TIMEOUT,
        check_interval: int = 30,
        restart_command: Optional[str] = None,
    ):
        self.timeout = timeout
        self.check_interval = check_interval
        self.restart_command = restart_command or self._get_default_restart_command()
        self._last_alert_time: Optional[datetime] = None
        self._alert_cooldown = 300  # 5 minutes between alerts

    def _get_default_restart_command(self) -> str:
        """Get default restart command based on OS."""
        if sys.platform == "win32":
            return f'python "{DATA_DIR / "Tradebot.py"}"'
        else:
            # Linux/Mac - use systemd or direct restart
            return f'systemctl restart trading-bot || python3 "{DATA_DIR / "Tradebot.py"}" &'

    def read_heartbeat(self) -> Optional[HeartbeatData]:
        """Read heartbeat data from file."""
        try:
            if not HEARTBEAT_FILE.exists():
                return None

            with open(HEARTBEAT_FILE, "r") as f:
                data = json.load(f)

            return HeartbeatData.from_dict(data)
        except Exception as e:
            logging.error(f"Failed to read heartbeat: {e}")
            return None

    def send_alert_message(self, message: str) -> None:
        """Send alert via Telegram."""
        # Check cooldown
        now = datetime.now()
        if self._last_alert_time:
            if (now - self._last_alert_time).total_seconds() < self._alert_cooldown:
                return

        self._last_alert_time = now

        if ALERTS_AVAILABLE:
            try:
                send_alert(message)
            except Exception as e:
                logging.error(f"Failed to send alert: {e}")
        else:
            # Fallback to direct Telegram API call
            try:
                import requests

                if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                    requests.get(
                        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                        params={
                            "chat_id": TELEGRAM_CHAT_ID,
                            "text": message,
                            "parse_mode": "Markdown",
                        },
                        timeout=10,
                    )
            except Exception:
                pass

        logging.warning(f"ALERT: {message}")

    def restart_bot(self) -> bool:
        """Attempt to restart the trading bot."""
        logging.warning("Attempting to restart trading bot...")

        try:
            if sys.platform == "win32":
                # Windows - start new process
                subprocess.Popen(
                    self.restart_command,
                    shell=True,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            else:
                # Linux - use subprocess
                subprocess.Popen(
                    self.restart_command, shell=True, start_new_session=True
                )

            logging.info("Restart command executed")
            return True
        except Exception as e:
            logging.error(f"Failed to restart bot: {e}")
            return False

    def check_heartbeat(self) -> Tuple[bool, str]:
        """
        Check heartbeat status.

        Returns:
            Tuple of (is_healthy, message)
        """
        heartbeat = self.read_heartbeat()

        if heartbeat is None:
            return False, "No heartbeat file found - bot may not be running"

        if heartbeat.status == "stopped":
            return False, "Bot status is 'stopped'"

        if heartbeat.is_stale(self.timeout):
            age = int(heartbeat.age_seconds())
            return False, f"Heartbeat is stale ({age}s old, timeout: {self.timeout}s)"

        # Check for other issues
        if heartbeat.error_count > 10:
            return False, f"High error count: {heartbeat.error_count}"

        return (
            True,
            f"OK (age: {int(heartbeat.age_seconds())}s, cycles: {heartbeat.scanner_cycle})",
        )

    def run_once(self, auto_restart: bool = True) -> bool:
        """
        Run a single health check.

        Args:
            auto_restart: Whether to attempt restart if unhealthy

        Returns:
            True if healthy, False if unhealthy
        """
        is_healthy, message = self.check_heartbeat()

        if is_healthy:
            logging.info(f"✅ Heartbeat check: {message}")
            return True

        logging.error(f"❌ Heartbeat check failed: {message}")

        # Send alert
        self.send_alert_message(
            f"🚨 **TRADING BOT HEARTBEAT FAILED**\n"
            f"Status: {message}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'Attempting restart...' if auto_restart else 'Manual restart required!'}"
        )

        # Attempt restart if enabled
        if auto_restart:
            time.sleep(5)  # Wait a bit before restart

            if self.restart_bot():
                self.send_alert_message(
                    "🔄 **BOT RESTART INITIATED**\n" "Check status in a few minutes."
                )
            else:
                self.send_alert_message(
                    "🚨 **BOT RESTART FAILED**\n" "Manual intervention required!"
                )

        return False

    def run_loop(self, auto_restart: bool = True) -> None:
        """
        Run continuous monitoring loop.

        Args:
            auto_restart: Whether to attempt restart if unhealthy
        """
        logging.info(
            f"🔍 Heartbeat monitor started (check interval: {self.check_interval}s)"
        )

        while True:
            try:
                self.run_once(auto_restart=auto_restart)
            except Exception as e:
                logging.error(f"Monitor error: {e}")

            time.sleep(self.check_interval)


# =============================================================================
# BOT CONTROL (Stop Bot via File)
# =============================================================================


def write_stop_signal(reason: str = "Manual stop") -> None:
    """Write a stop signal that the bot should pick up."""
    data = {
        "action": "STOP",
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }
    with open(BOT_CONTROL_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Stop signal written: {reason}")


def check_stop_signal() -> Tuple[bool, str]:
    """
    Check if a stop signal has been written.

    Returns:
        Tuple of (should_stop, reason)
    """
    try:
        if BOT_CONTROL_FILE.exists():
            with open(BOT_CONTROL_FILE, "r") as f:
                data = json.load(f)

            if data.get("action") == "STOP":
                reason = data.get("reason", "Unknown")
                # Remove the file after reading
                BOT_CONTROL_FILE.unlink()
                return True, reason
    except Exception as e:
        logging.warning(f"Error checking stop signal: {e}")

    return False, ""


def clear_stop_signal() -> None:
    """Clear any existing stop signal."""
    try:
        if BOT_CONTROL_FILE.exists():
            BOT_CONTROL_FILE.unlink()
    except Exception:
        pass


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """Command-line interface for heartbeat monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Trading Bot Heartbeat Monitor")
    parser.add_argument(
        "--monitor", "-m", action="store_true", help="Run continuous monitoring loop"
    )
    parser.add_argument(
        "--check", "-c", action="store_true", help="Run a single health check"
    )
    parser.add_argument(
        "--status", "-s", action="store_true", help="Show current heartbeat status"
    )
    parser.add_argument("--stop", action="store_true", help="Send stop signal to bot")
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=HEARTBEAT_TIMEOUT,
        help=f"Heartbeat timeout in seconds (default: {HEARTBEAT_TIMEOUT})",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-restart", action="store_true", help="Disable automatic restart"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    monitor = HeartbeatMonitor(timeout=args.timeout, check_interval=args.interval)

    if args.status:
        # Show current status
        heartbeat = monitor.read_heartbeat()
        if heartbeat:
            print("\n" + "=" * 50)
            print("HEARTBEAT STATUS")
            print("=" * 50)
            print(f"Timestamp:     {heartbeat.timestamp}")
            print(f"Age:           {int(heartbeat.age_seconds())}s")
            print(f"Status:        {heartbeat.status}")
            print(f"Active Trade:  {heartbeat.active_trade}")
            print(f"Instrument:    {heartbeat.instrument or 'N/A'}")
            print(f"Scanner Cycle: {heartbeat.scanner_cycle}")
            print(f"Memory:        {heartbeat.memory_mb:.1f} MB")
            print(f"Threads:       {heartbeat.threads_active}")
            print(f"Errors:        {heartbeat.error_count}")
            if heartbeat.last_error:
                print(f"Last Error:    {heartbeat.last_error}")
            print("=" * 50)

            is_healthy, msg = monitor.check_heartbeat()
            status_icon = "✅" if is_healthy else "❌"
            print(f"\nHealth: {status_icon} {msg}")
        else:
            print("No heartbeat file found.")

    elif args.check:
        # Single check
        is_healthy = monitor.run_once(auto_restart=not args.no_restart)
        sys.exit(0 if is_healthy else 1)

    elif args.monitor:
        # Continuous monitoring
        monitor.run_loop(auto_restart=not args.no_restart)

    elif args.stop:
        # Send stop signal
        write_stop_signal("Manual stop via CLI")
        print("Stop signal sent to bot.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
