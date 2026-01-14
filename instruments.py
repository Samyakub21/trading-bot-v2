# =============================================================================
# INSTRUMENT CONFIGURATIONS
# =============================================================================
"""
Instrument configurations with per-instrument strategy parameters.
Each instrument can have custom RSI thresholds, volume multipliers, etc.
"""

INSTRUMENTS = {
    "CRUDEOIL": {
        "name": "CRUDE OIL",
        "exchange_segment_int": 5,              # marketfeed.MCX_COMM (V2)
        "exchange_segment_str": "MCX_COMM",     # V2 API: MCX_COMM for commodities
        "future_id": "464926",                  # <--- UPDATE for current month future
        "lot_size": 10,
        "strike_step": 50,
        "expiry_date": "2026-01-16",
        "instrument_type": "FUTCOM",            # V2 API: FUTCOM for commodity futures
        "option_type": "OPTFUT",
        "market_start": "09:00",                # MCX Crude trading hours
        "market_end": "23:30",
        "no_new_trade_after": "23:00",          # Stop new entries 30 min before close
        # Per-instrument strategy parameters
        "strategy": "TrendFollowing",
        "strategy_params": {
            "rsi_bullish_threshold": 58,        # Crude works well with slightly lower threshold
            "rsi_bearish_threshold": 42,
            "volume_multiplier": 1.2,
            "ema_length": 50,
        },
    },
    "NATURALGAS": {
        "name": "NATURAL GAS",
        "exchange_segment_int": 5,              # marketfeed.MCX_COMM (V2)
        "exchange_segment_str": "MCX_COMM",     # V2 API: MCX_COMM for commodities
        "future_id": "465123",                  # <--- UPDATE with actual ID
        "lot_size": 1250,
        "strike_step": 5,
        "expiry_date": "2026-01-27",
        "instrument_type": "FUTCOM",            # V2 API: FUTCOM for commodity futures
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
        # Per-instrument strategy parameters (NatGas is volatile - use momentum)
        "strategy": "MomentumBreakout",
        "strategy_params": {
            "rsi_min_bullish": 55,
            "rsi_max_bearish": 45,
            "volume_multiplier": 1.5,           # Need strong volume for NatGas
            "lookback_period": 15,
            "breakout_threshold": 0.008,        # Higher threshold for volatile NatGas
        },
    },
    "GOLD": {
        "name": "GOLD",
        "exchange_segment_int": 5,              # marketfeed.MCX_COMM (V2)
        "exchange_segment_str": "MCX_COMM",     # V2 API: MCX_COMM for commodities
        "future_id": "465200",                  # <--- UPDATE with actual ID
        "lot_size": 10,
        "strike_step": 100,
        "expiry_date": "2026-02-05",
        "instrument_type": "FUTCOM",            # V2 API: FUTCOM for commodity futures
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
        # Per-instrument strategy parameters (Gold - stable trend follower)
        "strategy": "TrendFollowing",
        "strategy_params": {
            "rsi_bullish_threshold": 60,
            "rsi_bearish_threshold": 40,
            "volume_multiplier": 1.1,           # Gold has consistent volume
            "ema_length": 50,
        },
    },
    "SILVER": {
        "name": "SILVER",
        "exchange_segment_int": 5,              # marketfeed.MCX_COMM (V2)
        "exchange_segment_str": "MCX_COMM",     # V2 API: MCX_COMM for commodities
        "future_id": "465300",                  # <--- UPDATE with actual ID
        "lot_size": 30,
        "strike_step": 500,
        "expiry_date": "2026-02-05",
        "instrument_type": "FUTCOM",            # V2 API: FUTCOM for commodity futures
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
        # Per-instrument strategy parameters (Silver - more volatile than gold)
        "strategy": "MomentumBreakout",
        "strategy_params": {
            "rsi_min_bullish": 55,
            "rsi_max_bearish": 45,
            "volume_multiplier": 1.3,
            "lookback_period": 20,
            "breakout_threshold": 0.006,
        },
    },
    "NIFTY": {
        "name": "NIFTY 50",
        "exchange_segment_int": 2,              # marketfeed.NSE_FNO
        "exchange_segment_str": "NSE_FNO",      # V2 API: NSE_FNO for F&O
        "future_id": "13",                      # Nifty underlying ID
        "lot_size": 25,
        "strike_step": 50,
        "expiry_date": "2026-01-16",
        "instrument_type": "INDEX",            # V2 API: INDEX for index underlying
        "option_type": "OPTIDX",               # V2 API: OPTIDX for index options
        "market_start": "09:15",                # NSE trading hours
        "market_end": "15:30",
        "no_new_trade_after": "15:00",
        # Per-instrument strategy parameters (Nifty - lower RSI works better)
        "strategy": "TrendFollowing",
        "strategy_params": {
            "rsi_bullish_threshold": 55,        # Nifty trends at lower RSI levels
            "rsi_bearish_threshold": 45,
            "volume_multiplier": 1.2,
            "ema_length": 50,
        },
    },
    "BANKNIFTY": {
        "name": "BANK NIFTY",
        "exchange_segment_int": 2,              # marketfeed.NSE_FNO
        "exchange_segment_str": "NSE_FNO",      # V2 API: NSE_FNO for F&O
        "future_id": "25",                      # BankNifty underlying ID
        "lot_size": 15,
        "strike_step": 100,
        "expiry_date": "2026-01-15",
        "instrument_type": "INDEX",            # V2 API: INDEX for index underlying
        "option_type": "OPTIDX",               # V2 API: OPTIDX for index options
        "market_start": "09:15",
        "market_end": "15:30",
        "no_new_trade_after": "15:00",
        # Per-instrument strategy parameters (BankNifty - volatile index)
        "strategy": "TrendFollowing",
        "strategy_params": {
            "rsi_bullish_threshold": 58,
            "rsi_bearish_threshold": 42,
            "volume_multiplier": 1.3,           # BankNifty needs volume confirmation
            "ema_length": 50,
        },
    },
}

# =============================================================================
# SELECT ACTIVE INSTRUMENTS FOR SCANNING
# =============================================================================
# Set to None to scan ALL instruments, or provide a list like ["CRUDEOIL", "GOLD"]
SCAN_INSTRUMENTS = None  # None = scan all, or ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "NIFTY", "BANKNIFTY"]

# Default instrument (used when starting fresh or for state files)
DEFAULT_INSTRUMENT = "CRUDEOIL"

# Multi-instrument scanning mode
MULTI_SCAN_ENABLED = True  # Set to False to use single-instrument mode

# Priority order for instrument selection when multiple signals found
# Higher priority instruments are preferred (1=highest)
INSTRUMENT_PRIORITY = {
    "CRUDEOIL": 1,
    "GOLD": 2,
    "SILVER": 3,
    "NATURALGAS": 4,
    "NIFTY": 5,
    "BANKNIFTY": 6,
}


def get_instruments_to_scan():
    """Returns the list of instrument keys to scan"""
    if SCAN_INSTRUMENTS:
        return [k for k in SCAN_INSTRUMENTS if k in INSTRUMENTS]
    return list(INSTRUMENTS.keys())
