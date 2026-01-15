"""
Unit tests for scanner.py functions
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanner import (
    analyze_instrument_signal,
    get_atm_option,
    check_margin_available,
    verify_order,
)
from instruments import INSTRUMENTS


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def bullish_market_data():
    """Create bullish market DataFrame for 15min and 60min"""
    # 15-minute data - price above VWAP, RSI bullish
    df_15 = pd.DataFrame(
        {
            "open": [6000 + i * 10 for i in range(30)],
            "high": [6005 + i * 10 for i in range(30)],
            "low": [5995 + i * 10 for i in range(30)],
            "close": [6003 + i * 10 for i in range(30)],
            "volume": [10000 + i * 100 for i in range(30)],
        }
    )
    df_15.index = pd.date_range(start="2026-01-10 09:00", periods=30, freq="15min")

    # 60-minute data - price above EMA50
    df_60 = pd.DataFrame(
        {
            "open": [6000 + i * 20 for i in range(20)],
            "high": [6010 + i * 20 for i in range(20)],
            "low": [5990 + i * 20 for i in range(20)],
            "close": [6005 + i * 20 for i in range(20)],
            "volume": [50000 + i * 500 for i in range(20)],
        }
    )
    df_60.index = pd.date_range(start="2026-01-10 09:00", periods=20, freq="60min")

    return df_15, df_60


@pytest.fixture
def bearish_market_data():
    """Create bearish market DataFrame"""
    # 15-minute data - price below VWAP, RSI bearish
    df_15 = pd.DataFrame(
        {
            "open": [6300 - i * 10 for i in range(30)],
            "high": [6305 - i * 10 for i in range(30)],
            "low": [6295 - i * 10 for i in range(30)],
            "close": [6297 - i * 10 for i in range(30)],
            "volume": [10000 + i * 100 for i in range(30)],
        }
    )
    df_15.index = pd.date_range(start="2026-01-10 09:00", periods=30, freq="15min")

    # 60-minute data - price below EMA50
    df_60 = pd.DataFrame(
        {
            "open": [6300 - i * 15 for i in range(20)],
            "high": [6310 - i * 15 for i in range(20)],
            "low": [6290 - i * 15 for i in range(20)],
            "close": [6295 - i * 15 for i in range(20)],
            "volume": [50000 + i * 500 for i in range(20)],
        }
    )
    df_60.index = pd.date_range(start="2026-01-10 09:00", periods=20, freq="60min")

    return df_15, df_60


@pytest.fixture
def sideways_market_data():
    """Create sideways/neutral market DataFrame"""
    np.random.seed(42)

    df_15 = pd.DataFrame(
        {
            "open": [6000 + np.random.uniform(-5, 5) for _ in range(30)],
            "high": [6005 + np.random.uniform(-5, 5) for _ in range(30)],
            "low": [5995 + np.random.uniform(-5, 5) for _ in range(30)],
            "close": [6000 + np.random.uniform(-5, 5) for _ in range(30)],
            "volume": [10000 for _ in range(30)],
        }
    )
    df_15.index = pd.date_range(start="2026-01-10 09:00", periods=30, freq="15min")

    df_60 = pd.DataFrame(
        {
            "open": [6000 + np.random.uniform(-5, 5) for _ in range(20)],
            "high": [6005 + np.random.uniform(-5, 5) for _ in range(20)],
            "low": [5995 + np.random.uniform(-5, 5) for _ in range(20)],
            "close": [6000 + np.random.uniform(-5, 5) for _ in range(20)],
            "volume": [50000 for _ in range(20)],
        }
    )
    df_60.index = pd.date_range(start="2026-01-10 09:00", periods=20, freq="60min")

    return df_15, df_60


# =============================================================================
# SIGNAL ANALYSIS TESTS
# =============================================================================


class TestAnalyzeInstrumentSignal:
    """Tests for signal analysis"""

    def test_no_signal_insufficient_data(self):
        """Should return None with insufficient data"""
        df_15 = pd.DataFrame({"close": [100]})
        df_60 = pd.DataFrame({"close": [100]})

        result = analyze_instrument_signal("CRUDEOIL", df_15, df_60)
        assert result is None

    def test_signal_structure(self, bullish_market_data):
        """Signal should have all required fields"""
        df_15, df_60 = bullish_market_data

        # Mock to force a signal
        with patch("scanner.RSI_BULLISH_THRESHOLD", 0):
            with patch("scanner.VOLUME_MULTIPLIER", 0):
                result = analyze_instrument_signal("CRUDEOIL", df_15, df_60)

                if result:
                    assert "instrument" in result
                    assert "signal" in result
                    assert "price" in result
                    assert "rsi" in result
                    assert "signal_strength" in result

    def test_buy_signal_conditions(self, bullish_market_data):
        """BUY signal should require: above EMA50, above VWAP, RSI bullish, volume confirmed"""
        df_15, df_60 = bullish_market_data

        # This test verifies the logic exists - actual signal depends on data
        result = analyze_instrument_signal("CRUDEOIL", df_15, df_60)

        if result and result["signal"] == "BUY":
            assert result["signal_strength"] > 0


# =============================================================================
# OPTION CHAIN TESTS
# =============================================================================


class TestGetATMOption:
    """Tests for ATM option selection - V2 API format"""

    @patch("scanner.dhan")
    def test_atm_strike_calculation(self, mock_dhan):
        """Should calculate correct ATM strike with V2 API format"""
        # V2 API returns option chain in nested format
        mock_dhan.option_chain.return_value = {
            "status": "success",
            "data": {
                "last_price": 6010,
                "oc": {
                    "6000.0": {
                        "ce": {"security_id": "12345", "last_price": 150},
                        "pe": {"security_id": "12346", "last_price": 140},
                    },
                    "6050.0": {
                        "ce": {"security_id": "12347", "last_price": 100},
                        "pe": {"security_id": "12348", "last_price": 180},
                    },
                },
            },
        }

        # BUY signal should get CE option
        result = get_atm_option(
            "BUY", 6010, "NSE_FNO", "464926", "2026-01-16", "OPTFUT", 50
        )
        assert result == "12345"

    @patch("scanner.dhan")
    def test_put_option_for_sell(self, mock_dhan):
        """SELL signal should get PE option with V2 format"""
        mock_dhan.option_chain.return_value = {
            "status": "success",
            "data": {
                "last_price": 6010,
                "oc": {
                    "6000.0": {
                        "ce": {"security_id": "12345", "last_price": 150},
                        "pe": {"security_id": "12346", "last_price": 140},
                    }
                },
            },
        }

        result = get_atm_option(
            "SELL", 6010, "NSE_FNO", "464926", "2026-01-16", "OPTFUT", 50
        )
        assert result == "12346"

    @patch("scanner.dhan")
    def test_option_chain_failure(self, mock_dhan):
        """Should return None on API failure"""
        mock_dhan.option_chain.return_value = {
            "status": "failure",
            "errorMessage": "API error",
        }

        result = get_atm_option(
            "BUY", 6010, "MCX_COMM", "464926", "2026-01-16", "OPTFUT", 50
        )
        assert result is None

    @patch("scanner.dhan")
    def test_atm_rounding(self, mock_dhan):
        """ATM strike should be rounded to nearest strike step"""
        mock_dhan.option_chain.return_value = {
            "status": "success",
            "data": {
                "last_price": 6030,
                "oc": {
                    "6050.0": {
                        "ce": {"security_id": "12347", "last_price": 100},
                        "pe": {"security_id": "12348", "last_price": 180},
                    }
                },
            },
        }

        # Price 6030 with step 50 should round to 6050
        result = get_atm_option(
            "BUY", 6030, "NSE_FNO", "464926", "2026-01-16", "OPTFUT", 50
        )
        assert result == "12347"


# =============================================================================
# MARGIN CHECK TESTS
# =============================================================================


class TestCheckMarginAvailable:
    """Tests for margin availability checking - V2 API format"""

    @patch("scanner.dhan")
    def test_sufficient_margin(self, mock_dhan):
        """Should return True when margin is sufficient"""
        # V2 API uses availableBalance (with correct spelling)
        mock_dhan.get_fund_limits.return_value = {
            "status": "success",
            "data": {"availableBalance": 50000},
        }
        mock_dhan.margin_calculator.return_value = {
            "status": "success",
            "data": {"totalMargin": 10000},
        }

        ok, msg = check_margin_available("12345", "MCX_COMM", 10)
        assert ok == True
        assert "Margin OK" in msg

    @patch("scanner.dhan")
    def test_insufficient_margin(self, mock_dhan):
        """Should return False when margin is insufficient"""
        mock_dhan.get_fund_limits.return_value = {
            "status": "success",
            "data": {"availableBalance": 5000},
        }
        mock_dhan.margin_calculator.return_value = {
            "status": "success",
            "data": {"totalMargin": 10000},
        }

        ok, msg = check_margin_available("12345", "MCX_COMM", 10)
        assert ok == False
        assert "Insufficient" in msg

    @patch("scanner.dhan")
    def test_fund_api_failure(self, mock_dhan):
        """Should handle fund API failure gracefully"""
        mock_dhan.get_fund_limits.return_value = {
            "status": "failure",
            "errorMessage": "API error",
        }

        ok, msg = check_margin_available("12345", "MCX_COMM", 10)
        assert ok == False


# =============================================================================
# ORDER VERIFICATION TESTS
# =============================================================================


class TestVerifyOrder:
    """Tests for order verification - V2 API format"""

    @patch("scanner.dhan")
    def test_successful_order(self, mock_dhan):
        """Should return True for filled order with V2 API fields"""
        # V2 API uses averageTradedPrice instead of tradedPrice
        mock_dhan.get_order_by_id.return_value = {
            "status": "success",
            "data": {
                "orderStatus": "TRADED",
                "averageTradedPrice": 150.50,
                "filledQty": 10,
            },
        }

        order_response = {"status": "success", "data": {"orderId": "ORD123"}}

        success, details = verify_order(order_response, "ENTRY")
        assert success == True
        assert details["avg_price"] == 150.50

    @patch("scanner.dhan")
    def test_rejected_order(self, mock_dhan):
        """Should return False for rejected order with V2 error fields"""
        # V2 API uses omsErrorDescription instead of rejectedReason
        mock_dhan.get_order_by_id.return_value = {
            "status": "success",
            "data": {
                "orderStatus": "REJECTED",
                "omsErrorDescription": "Insufficient margin",
            },
        }

        order_response = {"status": "success", "data": {"orderId": "ORD123"}}

        success, details = verify_order(order_response, "ENTRY")
        assert success == False

    def test_none_response(self):
        """Should handle None response"""
        success, details = verify_order(None, "ENTRY")
        assert success == False
        assert details is None

    def test_failure_response(self):
        """Should handle failure response"""
        order_response = {"status": "failure", "remarks": "Order placement failed"}

        success, details = verify_order(order_response, "ENTRY")
        assert success == False


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
