# =============================================================================
# INSTRUMENT CONFIGURATIONS
# =============================================================================

INSTRUMENTS = {
    "CRUDEOIL": {
        "name": "CRUDE OIL",
        "exchange_segment_int": 5,              # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "464926",                  # <--- UPDATE for current month future
        "lot_size": 10,
        "strike_step": 50,
        "expiry_date": "2026-01-16",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",                # MCX Crude trading hours
        "market_end": "23:30",
        "no_new_trade_after": "23:00",          # Stop new entries 30 min before close
    },
    "NATURALGAS": {
        "name": "NATURAL GAS",
        "exchange_segment_int": 5,              # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "465123",                  # <--- UPDATE with actual ID
        "lot_size": 1250,
        "strike_step": 5,
        "expiry_date": "2026-01-27",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
    },
    "GOLD": {
        "name": "GOLD",
        "exchange_segment_int": 5,              # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "465200",                  # <--- UPDATE with actual ID
        "lot_size": 10,
        "strike_step": 100,
        "expiry_date": "2026-02-05",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
    },
    "SILVER": {
        "name": "SILVER",
        "exchange_segment_int": 5,              # marketfeed.MCX
        "exchange_segment_str": "MCX",
        "future_id": "465300",                  # <--- UPDATE with actual ID
        "lot_size": 30,
        "strike_step": 500,
        "expiry_date": "2026-02-05",
        "instrument_type": "FUTURES",
        "option_type": "OPTFUT",
        "market_start": "09:00",
        "market_end": "23:30",
        "no_new_trade_after": "23:00",
    },
    "NIFTY": {
        "name": "NIFTY 50",
        "exchange_segment_int": 2,              # marketfeed.NSE_FNO
        "exchange_segment_str": "NSE_FNO",
        "future_id": "13",                      # Nifty underlying ID
        "lot_size": 25,
        "strike_step": 50,
        "expiry_date": "2026-01-16",
        "instrument_type": "INDEX",
        "option_type": "OPTIDX",
        "market_start": "09:15",                # NSE trading hours
        "market_end": "15:30",
        "no_new_trade_after": "15:00",
    },
    "BANKNIFTY": {
        "name": "BANK NIFTY",
        "exchange_segment_int": 2,              # marketfeed.NSE_FNO
        "exchange_segment_str": "NSE_FNO",
        "future_id": "25",                      # BankNifty underlying ID
        "lot_size": 15,
        "strike_step": 100,
        "expiry_date": "2026-01-15",
        "instrument_type": "INDEX",
        "option_type": "OPTIDX",
        "market_start": "09:15",
        "market_end": "15:30",
        "no_new_trade_after": "15:00",
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
