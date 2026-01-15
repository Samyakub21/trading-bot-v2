"""
Configuration loader for Trading Bot
Loads credentials from environment variables or config file
Loads trading parameters from trading_config.json or environment variables
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# DEFAULT TRADING CONFIGURATION
# =============================================================================
# Data directory for runtime files
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_TRADING_CONFIG = {
    # Risk Management
    "MAX_DAILY_LOSS": 2000,  # Maximum loss per day in INR
    "MAX_TRADES_PER_DAY": 5,  # Maximum number of trades per day
    "COOLDOWN_AFTER_LOSS": 300,  # Wait 5 minutes after a loss before next trade
    "SIGNAL_COOLDOWN": 900,  # 15 minutes cooldown for same direction signal
    "AUTO_SQUARE_OFF_BUFFER": 5,  # Minutes before market close to auto square-off
    # Signal Strength Thresholds
    "RSI_BULLISH_THRESHOLD": 60,  # RSI must be above this for bullish signal
    "RSI_BEARISH_THRESHOLD": 40,  # RSI must be below this for bearish signal
    "VOLUME_MULTIPLIER": 1.2,  # Volume must be 1.2x average for signal confirmation
    # Order Execution
    "LIMIT_ORDER_BUFFER": 0.01,  # 1% buffer for limit orders
    # Paper Trading / Dry Run
    "PAPER_TRADING": True,  # Simulate trades without placing orders
    # Database Configuration
    "USE_DATABASE": True,  # Use SQLite database instead of JSON files
    "DATABASE_PATH": str(DATA_DIR / "trading_bot.db"),  # SQLite database file path
    # Poll Fallback Configuration
    "POLL_FALLBACK_ENABLED": True,  # Enable REST API fallback for stale data
    "POLL_FALLBACK_THRESHOLD": 5,  # Seconds before triggering REST fallback
    "POLL_COOLDOWN": 2,  # Minimum seconds between REST polls
    # Instrument Configuration
    "ENABLED_INSTRUMENTS": [
        "CRUDEOIL",
        "NATURALGAS",
        "GOLD",
        "SILVER",
        "NIFTY",
        "BANKNIFTY",
    ],
    "INSTRUMENT_PRIORITY": {
        "CRUDEOIL": 3,
        "GOLD": 5,
        "SILVER": 6,  # Lowest priority
        "NATURALGAS": 4,
        "NIFTY": 1,  # Highest priority
        "BANKNIFTY": 2,
    },
    # Per-Instrument Custom Settings (overrides global settings)
    "PER_INSTRUMENT_SETTINGS": {
        # Example: Custom settings for specific instruments
        # "CRUDEOIL": {
        #     "use_custom": True,
        #     "rsi_bullish": 58,
        #     "rsi_bearish": 42,
        #     "volume_multiplier": 1.3
        # }
    },
    # State Files (fallback when database is disabled)
    "STATE_FILE": str(DATA_DIR / "trade_state_active.json"),
    "DAILY_PNL_FILE": str(DATA_DIR / "daily_pnl_combined.json"),
    "TRADE_HISTORY_FILE": str(DATA_DIR / "trade_history_combined.json"),
}


class Config:
    """Configuration class that loads credentials and trading params from environment or config files"""

    def __init__(self):
        # --- Credentials ---
        # Try to load from environment variables first (recommended for servers)
        self.CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
        self.ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        self.SIGNAL_BOT_TOKEN = os.getenv("SIGNAL_BOT_TOKEN")

        # If not in environment, try to load from credentials.json
        if not all(
            [
                self.CLIENT_ID,
                self.ACCESS_TOKEN,
                self.TELEGRAM_TOKEN,
                self.TELEGRAM_CHAT_ID,
            ]
        ):
            self._load_credentials_from_file()

        # Validate that we have credentials
        if not all(
            [
                self.CLIENT_ID,
                self.ACCESS_TOKEN,
                self.TELEGRAM_TOKEN,
                self.TELEGRAM_CHAT_ID,
            ]
        ):
            raise ValueError(
                "Missing credentials! Please either:\n"
                "1. Set environment variables: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID\n"
                "2. Create a credentials.json file (see credentials.example.json)\n"
            )

        # Clean whitespace
        self.CLIENT_ID = str(self.CLIENT_ID).strip()
        self.ACCESS_TOKEN = str(self.ACCESS_TOKEN).strip()
        self.TELEGRAM_TOKEN = str(self.TELEGRAM_TOKEN).strip()
        self.TELEGRAM_CHAT_ID = str(self.TELEGRAM_CHAT_ID).strip()
        if self.SIGNAL_BOT_TOKEN:
            self.SIGNAL_BOT_TOKEN = str(self.SIGNAL_BOT_TOKEN).strip()

        # --- Trading Configuration ---
        self._trading_config = self._load_trading_config()

    def _load_credentials_from_file(self):
        """Load credentials from credentials.json file"""
        credentials_file = Path(__file__).parent / "credentials.json"

        if not credentials_file.exists():
            return

        try:
            with open(credentials_file, "r") as f:
                creds = json.load(f)

            self.CLIENT_ID = creds.get("CLIENT_ID", self.CLIENT_ID)
            self.ACCESS_TOKEN = creds.get("ACCESS_TOKEN", self.ACCESS_TOKEN)
            self.TELEGRAM_TOKEN = creds.get("TELEGRAM_TOKEN", self.TELEGRAM_TOKEN)
            self.TELEGRAM_CHAT_ID = creds.get("TELEGRAM_CHAT_ID", self.TELEGRAM_CHAT_ID)
            self.SIGNAL_BOT_TOKEN = creds.get("SIGNAL_BOT_TOKEN", self.SIGNAL_BOT_TOKEN)
        except Exception as e:
            print(f"Warning: Could not load credentials.json: {e}")

    def _load_trading_config(self) -> Dict[str, Any]:
        """
        Load trading configuration with priority:
        1. Environment variables (highest priority)
        2. trading_config.json file
        3. Default values (lowest priority)
        """
        # Start with defaults
        config = DEFAULT_TRADING_CONFIG.copy()

        # Try to load from trading_config.json
        config_file = Path(__file__).parent / "trading_config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                config.update(file_config)
                print(f"Loaded trading config from {config_file}")
            except Exception as e:
                print(f"Warning: Could not load trading_config.json: {e}")

        # Override with environment variables if set
        env_mappings = {
            "MAX_DAILY_LOSS": ("TRADING_MAX_DAILY_LOSS", float),
            "MAX_TRADES_PER_DAY": ("TRADING_MAX_TRADES_PER_DAY", int),
            "COOLDOWN_AFTER_LOSS": ("TRADING_COOLDOWN_AFTER_LOSS", int),
            "SIGNAL_COOLDOWN": ("TRADING_SIGNAL_COOLDOWN", int),
            "AUTO_SQUARE_OFF_BUFFER": ("TRADING_AUTO_SQUARE_OFF_BUFFER", int),
            "RSI_BULLISH_THRESHOLD": ("TRADING_RSI_BULLISH_THRESHOLD", float),
            "RSI_BEARISH_THRESHOLD": ("TRADING_RSI_BEARISH_THRESHOLD", float),
            "VOLUME_MULTIPLIER": ("TRADING_VOLUME_MULTIPLIER", float),
            "LIMIT_ORDER_BUFFER": ("TRADING_LIMIT_ORDER_BUFFER", float),
        }

        for config_key, (env_var, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    config[config_key] = type_func(env_value)
                except ValueError:
                    print(f"Warning: Invalid value for {env_var}: {env_value}")

        return config

    def get_trading_param(self, key: str, default: Any = None) -> Any:
        """Get a trading configuration parameter"""
        return self._trading_config.get(key, default)

    # --- Trading Config Properties (for easy access) ---
    @property
    def MAX_DAILY_LOSS(self) -> float:
        return self._trading_config["MAX_DAILY_LOSS"]

    @property
    def MAX_TRADES_PER_DAY(self) -> int:
        return self._trading_config["MAX_TRADES_PER_DAY"]

    @property
    def COOLDOWN_AFTER_LOSS(self) -> int:
        return self._trading_config["COOLDOWN_AFTER_LOSS"]

    @property
    def SIGNAL_COOLDOWN(self) -> int:
        return self._trading_config["SIGNAL_COOLDOWN"]

    @property
    def AUTO_SQUARE_OFF_BUFFER(self) -> int:
        return self._trading_config["AUTO_SQUARE_OFF_BUFFER"]

    @property
    def RSI_BULLISH_THRESHOLD(self) -> float:
        return self._trading_config["RSI_BULLISH_THRESHOLD"]

    @property
    def RSI_BEARISH_THRESHOLD(self) -> float:
        return self._trading_config["RSI_BEARISH_THRESHOLD"]

    @property
    def VOLUME_MULTIPLIER(self) -> float:
        return self._trading_config["VOLUME_MULTIPLIER"]

    @property
    def LIMIT_ORDER_BUFFER(self) -> float:
        return self._trading_config["LIMIT_ORDER_BUFFER"]

    @property
    def STATE_FILE(self) -> str:
        return self._trading_config["STATE_FILE"]

    @property
    def DAILY_PNL_FILE(self) -> str:
        return self._trading_config["DAILY_PNL_FILE"]

    @property
    def TRADE_HISTORY_FILE(self) -> str:
        return self._trading_config["TRADE_HISTORY_FILE"]

    # --- Database Configuration ---
    @property
    def USE_DATABASE(self) -> bool:
        return self._trading_config.get("USE_DATABASE", True)

    @property
    def DATABASE_PATH(self) -> str:
        return self._trading_config.get(
            "DATABASE_PATH", str(DATA_DIR / "trading_bot.db")
        )

    # --- Poll Fallback Configuration ---
    @property
    def POLL_FALLBACK_ENABLED(self) -> bool:
        return self._trading_config.get("POLL_FALLBACK_ENABLED", True)

    @property
    def POLL_FALLBACK_THRESHOLD(self) -> int:
        return self._trading_config.get("POLL_FALLBACK_THRESHOLD", 5)

    @property
    def POLL_COOLDOWN(self) -> int:
        return self._trading_config.get("POLL_COOLDOWN", 2)

    # --- Instrument Configuration ---
    @property
    def ENABLED_INSTRUMENTS(self) -> list:
        """List of enabled instruments for scanning/trading"""
        return self._trading_config.get(
            "ENABLED_INSTRUMENTS",
            ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "NIFTY", "BANKNIFTY"],
        )

    @property
    def INSTRUMENT_PRIORITY(self) -> dict:
        """Priority order for instruments (1=highest)"""
        return self._trading_config.get(
            "INSTRUMENT_PRIORITY",
            {
                "CRUDEOIL": 1,
                "GOLD": 2,
                "SILVER": 3,
                "NATURALGAS": 4,
                "NIFTY": 5,
                "BANKNIFTY": 6,
            },
        )

    @property
    def PER_INSTRUMENT_SETTINGS(self) -> dict:
        """Custom per-instrument signal settings"""
        return self._trading_config.get("PER_INSTRUMENT_SETTINGS", {})

    def get_instrument_settings(self, instrument: str) -> dict:
        """
        Get signal settings for a specific instrument.
        Returns custom settings if defined, otherwise global settings.

        Args:
            instrument: Instrument key (e.g., 'CRUDEOIL')

        Returns:
            Dictionary with rsi_bullish, rsi_bearish, volume_multiplier
        """
        per_inst = self.PER_INSTRUMENT_SETTINGS.get(instrument, {})

        if per_inst.get("use_custom", False):
            return {
                "rsi_bullish": per_inst.get("rsi_bullish", self.RSI_BULLISH_THRESHOLD),
                "rsi_bearish": per_inst.get("rsi_bearish", self.RSI_BEARISH_THRESHOLD),
                "volume_multiplier": per_inst.get(
                    "volume_multiplier", self.VOLUME_MULTIPLIER
                ),
            }

        # Return global settings
        return {
            "rsi_bullish": self.RSI_BULLISH_THRESHOLD,
            "rsi_bearish": self.RSI_BEARISH_THRESHOLD,
            "volume_multiplier": self.VOLUME_MULTIPLIER,
        }

    def get_enabled_instruments_sorted(self) -> list:
        """
        Get enabled instruments sorted by priority.

        Returns:
            List of instrument keys sorted by priority (highest first)
        """
        enabled = self.ENABLED_INSTRUMENTS
        priority = self.INSTRUMENT_PRIORITY

        return sorted(enabled, key=lambda x: priority.get(x, 999))

    # --- Socket/Network Configuration ---
    @property
    def HEARTBEAT_TIMEOUT_SECONDS(self) -> int:
        """Timeout in seconds before socket reconnection is triggered"""
        return 30

    @property
    def RECONNECT_DELAY_SECONDS(self) -> int:
        """Delay in seconds before attempting socket reconnection"""
        return 5

    @property
    def MIN_TICK_INTERVAL_MS(self) -> int:
        """Minimum interval in milliseconds between tick processing"""
        return 100

    def reload_trading_config(self) -> Dict[str, Any]:
        """
        Reload trading configuration from file.
        Call this to pick up live config changes from the dashboard.

        Returns:
            The updated trading config dictionary
        """
        self._trading_config = self._load_trading_config()
        return self._trading_config

    def get_fresh_config(self) -> Dict[str, Any]:
        """
        Get a fresh copy of trading config from file without caching.
        Useful for checking if config has changed.

        Returns:
            Fresh trading config dictionary
        """
        return self._load_trading_config()


# Create a singleton instance
config = Config()
