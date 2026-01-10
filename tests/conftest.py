"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_dhan_client():
    """Create mock Dhan client for API testing"""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.BUY = "BUY"
    mock.SELL = "SELL"
    mock.LIMIT = "LIMIT"
    mock.MARKET = "MARKET"
    mock.INTRADAY = "INTRADAY"
    
    return mock


@pytest.fixture
def sample_instruments():
    """Sample instrument configuration"""
    return {
        "CRUDEOIL": {
            "name": "CRUDE OIL",
            "exchange_segment_int": 5,
            "exchange_segment_str": "MCX",
            "future_id": "464926",
            "lot_size": 10,
            "strike_step": 50,
            "expiry_date": "2026-01-16",
            "instrument_type": "FUTURES",
            "option_type": "OPTFUT",
            "market_start": "09:00",
            "market_end": "23:30",
            "no_new_trade_after": "23:00",
        }
    }
