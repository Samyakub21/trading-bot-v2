# =============================================================================
# STATE STORES - Encapsulated State Management Classes
# =============================================================================
# Replaces global variables with thread-safe class instances

import threading
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# MARKET DATA STORE
# =============================================================================
class MarketDataStore:
    """
    Thread-safe store for market data (LTP values and tick times).
    Replaces global variables: LATEST_LTP, OPTION_LTP, LAST_TICK_TIME, etc.
    """

    _instance: Optional["MarketDataStore"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._data_lock = threading.RLock()

        # Future/Underlying data
        self._latest_ltp: float = 0.0
        self._last_tick_time: datetime = datetime.now()

        # Option data
        self._option_ltp: float = 0.0
        self._last_option_tick_time: datetime = datetime.now()

        # Multi-instrument LTP tracking
        self._instrument_ltp: Dict[str, Dict[str, Any]] = {}

        self._initialized = True

    # --- Future LTP ---
    @property
    def latest_ltp(self) -> float:
        with self._data_lock:
            return self._latest_ltp

    @latest_ltp.setter
    def latest_ltp(self, value: float):
        with self._data_lock:
            self._latest_ltp = value
            self._last_tick_time = datetime.now()

    @property
    def last_tick_time(self) -> datetime:
        with self._data_lock:
            return self._last_tick_time

    # --- Option LTP ---
    @property
    def option_ltp(self) -> float:
        with self._data_lock:
            return self._option_ltp

    @option_ltp.setter
    def option_ltp(self, value: float):
        with self._data_lock:
            self._option_ltp = value
            self._last_option_tick_time = datetime.now()

    @property
    def last_option_tick_time(self) -> datetime:
        with self._data_lock:
            return self._last_option_tick_time

    def reset_option_ltp(self):
        """Reset option LTP when trade closes"""
        with self._data_lock:
            self._option_ltp = 0.0

    # --- Multi-instrument LTP ---
    def get_instrument_ltp(self, instrument_key: str) -> Optional[Dict[str, Any]]:
        """Get LTP data for a specific instrument"""
        with self._data_lock:
            return self._instrument_ltp.get(instrument_key)

    def set_instrument_ltp(self, instrument_key: str, ltp: float):
        """Set LTP for a specific instrument"""
        with self._data_lock:
            self._instrument_ltp[instrument_key] = {
                "ltp": ltp,
                "last_update": datetime.now(),
            }

    def get_all_instrument_ltps(self) -> Dict[str, Dict[str, Any]]:
        """Get all instrument LTP data"""
        with self._data_lock:
            return self._instrument_ltp.copy()

    # --- Data freshness checks ---
    def is_data_stale(self, threshold_seconds: int = 10) -> bool:
        """Check if future data feed is stale"""
        with self._data_lock:
            elapsed = (datetime.now() - self._last_tick_time).total_seconds()
            return elapsed > threshold_seconds

    def is_option_data_stale(self, threshold_seconds: int = 10) -> bool:
        """Check if option data feed is stale"""
        with self._data_lock:
            elapsed = (datetime.now() - self._last_option_tick_time).total_seconds()
            return elapsed > threshold_seconds

    def reset(self):
        """Reset all data (for testing or reconnection)"""
        with self._data_lock:
            self._latest_ltp = 0.0
            self._option_ltp = 0.0
            self._last_tick_time = datetime.now()
            self._last_option_tick_time = datetime.now()
            self._instrument_ltp.clear()


# =============================================================================
# SIGNAL TRACKER
# =============================================================================
class SignalTracker:
    """
    Thread-safe tracker for signal state and cooldowns.
    Replaces global variables: LAST_SIGNAL, LAST_SIGNAL_TIME, LAST_LOSS_TIME
    """

    _instance: Optional["SignalTracker"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._data_lock = threading.RLock()

        # Per-instrument signal tracking: {instrument: {"signal": str, "time": datetime}}
        self._instrument_signals: Dict[str, Dict[str, Any]] = {}

        # Legacy global signal tracking (for backward compatibility)
        self._last_signal: Optional[str] = None
        self._last_signal_time: Optional[datetime] = None

        # Loss tracking for cooldown
        self._last_loss_time: Optional[datetime] = None

        self._initialized = True

    # --- Signal tracking ---
    @property
    def last_signal(self) -> Optional[str]:
        with self._data_lock:
            return self._last_signal

    @property
    def last_signal_time(self) -> Optional[datetime]:
        with self._data_lock:
            return self._last_signal_time

    def update_signal(self, signal: str, instrument: Optional[str] = None):
        """Update last signal after a trade attempt (per-instrument if specified)"""
        with self._data_lock:
            # Update global tracking (backward compatibility)
            self._last_signal = signal
            self._last_signal_time = datetime.now()

            # Update per-instrument tracking
            if instrument:
                self._instrument_signals[instrument] = {
                    "signal": signal,
                    "time": datetime.now(),
                }

    # --- Loss tracking ---
    @property
    def last_loss_time(self) -> Optional[datetime]:
        with self._data_lock:
            return self._last_loss_time

    def record_loss(self):
        """Record a losing trade for cooldown"""
        with self._data_lock:
            self._last_loss_time = datetime.now()

    def clear_loss_cooldown(self):
        """Clear loss cooldown (for testing)"""
        with self._data_lock:
            self._last_loss_time = None

    # --- Cooldown checks ---
    def is_in_loss_cooldown(self, cooldown_seconds: int) -> tuple[bool, str]:
        """Check if we're in cooldown period after a loss"""
        with self._data_lock:
            if self._last_loss_time is None:
                return False, "No cooldown"

            elapsed = (datetime.now() - self._last_loss_time).total_seconds()
            if elapsed < cooldown_seconds:
                remaining = int(cooldown_seconds - elapsed)
                return True, f"Cooldown active: {remaining}s remaining"

            return False, "Cooldown complete"

    def is_in_signal_cooldown(
        self, signal: str, cooldown_seconds: int, instrument: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Check if same-direction signal is in cooldown for a specific instrument.

        Signal cooldown is per-instrument to prevent whipsaw on the same instrument,
        but allows trading different instruments with the same signal direction.
        """
        with self._data_lock:
            # Per-instrument cooldown (preferred)
            if instrument and instrument in self._instrument_signals:
                inst_data = self._instrument_signals[instrument]
                last_signal = inst_data.get("signal")
                last_time = inst_data.get("time")

                if last_signal == signal and last_time:
                    elapsed = (datetime.now() - last_time).total_seconds()
                    if elapsed < cooldown_seconds:
                        remaining = int(cooldown_seconds - elapsed)
                        return (
                            True,
                            f"Same signal ({signal}) cooldown: {remaining}s remaining",
                        )

                return False, "Signal allowed"

            # Fallback to global cooldown (for backward compatibility / single-instrument mode)
            if self._last_signal is None or self._last_signal_time is None:
                return False, "No signal cooldown"

            # Only apply cooldown for same direction signals
            if signal == self._last_signal:
                elapsed = (datetime.now() - self._last_signal_time).total_seconds()
                if elapsed < cooldown_seconds:
                    remaining = int(cooldown_seconds - elapsed)
                    return (
                        True,
                        f"Same signal ({signal}) cooldown: {remaining}s remaining",
                    )

            return False, "Signal allowed"

    def reset(self):
        """Reset all tracking (for testing)"""
        with self._data_lock:
            self._last_signal = None
            self._last_signal_time = None
            self._last_loss_time = None
            self._instrument_signals.clear()


# =============================================================================
# SOCKET STATE
# =============================================================================
class SocketState:
    """
    Thread-safe state management for WebSocket connection.
    Replaces global events and connection tracking.
    """

    _instance: Optional["SocketState"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._data_lock = threading.RLock()

        # Market feed reference
        self._market_feed: Any = None

        # Socket events
        self._shutdown_event = threading.Event()
        self._reconnect_event = threading.Event()
        self._healthy_event = threading.Event()

        self._initialized = True

    # --- Market feed ---
    @property
    def market_feed(self) -> Any:
        with self._data_lock:
            return self._market_feed

    @market_feed.setter
    def market_feed(self, value: Any):
        with self._data_lock:
            self._market_feed = value

    # --- Shutdown ---
    def shutdown(self):
        """Signal shutdown"""
        self._shutdown_event.set()

    def is_shutdown(self) -> bool:
        """Check if shutdown requested"""
        return self._shutdown_event.is_set()

    def clear_shutdown(self):
        """Clear shutdown (for restart)"""
        self._shutdown_event.clear()

    # --- Reconnection ---
    def request_reconnect(self):
        """Request socket reconnection"""
        self._reconnect_event.set()

    def is_reconnect_requested(self) -> bool:
        """Check if reconnection requested"""
        return self._reconnect_event.is_set()

    def clear_reconnect(self):
        """Clear reconnection request"""
        self._reconnect_event.clear()

    # --- Health ---
    def mark_healthy(self):
        """Mark socket as healthy (received tick)"""
        self._healthy_event.set()

    def is_healthy(self) -> bool:
        """Check if socket is healthy"""
        return self._healthy_event.is_set()

    def clear_healthy(self):
        """Clear healthy status for next check cycle"""
        self._healthy_event.clear()

    def wait_for_health(self, timeout: float = 30.0) -> bool:
        """Wait for health signal with timeout"""
        return self._healthy_event.wait(timeout=timeout)

    def reset(self):
        """Reset all state (for testing)"""
        with self._data_lock:
            self._market_feed = None
        self._shutdown_event.clear()
        self._reconnect_event.clear()
        self._healthy_event.clear()


# =============================================================================
# TRADE STATE
# =============================================================================
@dataclass
class TradeState:
    """
    Encapsulated trade state with thread-safe access.
    Replaces the active_trade dictionary pattern.
    """

    status: bool = False
    trade_type: Optional[str] = None  # "BUY" or "SELL"
    instrument: Optional[str] = None

    # Entry details
    entry: float = 0.0
    future_entry: float = 0.0
    option_entry: float = 0.0
    entry_time: Optional[str] = None

    # Stop loss
    initial_sl: float = 0.0
    sl: float = 0.0
    step_level: int = 0

    # Option details
    option_id: Optional[str] = None
    order_id: Optional[str] = None

    # Instrument details
    lot_size: Optional[int] = None
    exchange_segment_str: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "status": self.status,
            "type": self.trade_type,
            "instrument": self.instrument,
            "entry": self.entry,
            "future_entry": self.future_entry,
            "option_entry": self.option_entry,
            "entry_time": self.entry_time,
            "initial_sl": self.initial_sl,
            "sl": self.sl,
            "step_level": self.step_level,
            "option_id": self.option_id,
            "order_id": self.order_id,
            "lot_size": self.lot_size,
            "exchange_segment_str": self.exchange_segment_str,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeState":
        """Create from dictionary"""
        return cls(
            status=data.get("status", False),
            trade_type=data.get("type"),
            instrument=data.get("instrument"),
            entry=data.get("entry", 0.0),
            future_entry=data.get("future_entry", 0.0),
            option_entry=data.get("option_entry", 0.0),
            entry_time=data.get("entry_time"),
            initial_sl=data.get("initial_sl", 0.0),
            sl=data.get("sl", 0.0),
            step_level=data.get("step_level", 0),
            option_id=data.get("option_id"),
            order_id=data.get("order_id"),
            lot_size=data.get("lot_size"),
            exchange_segment_str=data.get("exchange_segment_str"),
        )

    def reset(self):
        """Reset trade state after trade closes"""
        self.status = False
        self.trade_type = None
        self.instrument = None
        self.entry = 0.0
        self.future_entry = 0.0
        self.option_entry = 0.0
        self.entry_time = None
        self.initial_sl = 0.0
        self.sl = 0.0
        self.step_level = 0
        self.option_id = None
        self.order_id = None
        self.lot_size = None
        self.exchange_segment_str = None


class TradeStateManager:
    """
    Thread-safe manager for TradeState.
    Provides locking and state persistence.
    """

    _instance: Optional["TradeStateManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._data_lock = threading.RLock()
        self._state = TradeState()
        self._initialized = True

    @property
    def state(self) -> TradeState:
        """Get current trade state (read-only snapshot)"""
        with self._data_lock:
            return TradeState.from_dict(self._state.to_dict())

    def update(self, **kwargs):
        """Thread-safe update of trade state fields"""
        with self._data_lock:
            for key, value in kwargs.items():
                if key == "type":
                    key = "trade_type"
                if hasattr(self._state, key):
                    setattr(self._state, key, value)

    def load_from_dict(self, data: Dict[str, Any]):
        """Load state from dictionary"""
        with self._data_lock:
            self._state = TradeState.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Get state as dictionary"""
        with self._data_lock:
            return self._state.to_dict()

    def reset(self):
        """Reset trade state"""
        with self._data_lock:
            self._state.reset()

    @property
    def is_active(self) -> bool:
        """Check if there's an active trade"""
        with self._data_lock:
            return self._state.status

    def get_lock(self) -> threading.RLock:
        """Get the data lock for complex operations"""
        return self._data_lock


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================
def get_market_data_store() -> MarketDataStore:
    """Get the singleton MarketDataStore instance"""
    return MarketDataStore()


def get_signal_tracker() -> SignalTracker:
    """Get the singleton SignalTracker instance"""
    return SignalTracker()


def get_socket_state() -> SocketState:
    """Get the singleton SocketState instance"""
    return SocketState()


def get_trade_state_manager() -> TradeStateManager:
    """Get the singleton TradeStateManager instance"""
    return TradeStateManager()
