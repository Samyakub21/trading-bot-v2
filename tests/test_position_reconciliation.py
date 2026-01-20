"""
Unit tests for position_reconciliation.py
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import asdict

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_instruments():
    """Mock instruments configuration - V2 API format"""
    return {
        "CRUDEOIL": {
            "name": "CRUDE OIL",
            "exchange_segment_int": 5,
            "exchange_segment_str": "MCX_COMM",  # V2 API format
            "future_id": "464926",
            "lot_size": 10,
        }
    }


@pytest.fixture
def active_trade_with_position():
    """Active trade state with open position - V2 format"""
    return {
        "status": True,
        "type": "BUY",
        "instrument": "CRUDEOIL",
        "entry": 6000,
        "initial_sl": 5980,
        "sl": 5980,
        "option_id": "12345",
        "option_entry": 150,
        "lot_size": 10,
        "exchange_segment_str": "MCX_COMM",  # V2 API format
    }


@pytest.fixture
def empty_trade_state():
    """Empty/no position trade state"""
    return {
        "status": False,
        "type": None,
        "instrument": None,
        "option_id": None,
    }


@pytest.fixture
def mock_broker_position():
    """Mock broker position response - V2 API format"""
    # V2 uses buyAvg/sellAvg instead of averagePrice
    return {
        "securityId": "12345",
        "tradingSymbol": "CRUDEOIL26JAN6000CE",
        "exchangeSegment": "MCX_COMM",
        "netQty": 10,
        "buyAvg": 150.0,
        "sellAvg": 0.0,
        "costPrice": 150.0,
        "unrealizedProfit": 200.0,
        "productType": "INTRADAY",
    }


@pytest.fixture
def mock_dhan_positions_response(mock_broker_position):
    """Mock Dhan API positions response"""
    return {"status": "success", "data": [mock_broker_position]}


# =============================================================================
# BROKER POSITION DATA CLASS TESTS
# =============================================================================


class TestBrokerPosition:
    """Tests for BrokerPosition dataclass"""

    def test_from_dhan_response_long_position(self, mock_broker_position):
        """Should correctly parse LONG position from Dhan response"""
        from position_reconciliation import BrokerPosition

        pos = BrokerPosition.from_dhan_response(mock_broker_position)

        assert pos.security_id == "12345"
        assert pos.trading_symbol == "CRUDEOIL26JAN6000CE"
        assert pos.position_type == "LONG"
        assert pos.quantity == 10
        assert pos.average_price == 150.0

    def test_from_dhan_response_short_position(self, mock_broker_position):
        """Should correctly parse SHORT position from Dhan response"""
        from position_reconciliation import BrokerPosition

        mock_broker_position["netQty"] = -10
        pos = BrokerPosition.from_dhan_response(mock_broker_position)

        assert pos.position_type == "SHORT"
        assert pos.quantity == 10  # Absolute value


# =============================================================================
# RECONCILIATION STATUS TESTS
# =============================================================================


class TestReconciliationStatus:
    """Tests for ReconciliationStatus enum"""

    def test_all_statuses_exist(self):
        """Should have all required reconciliation statuses"""
        from position_reconciliation import ReconciliationStatus

        assert ReconciliationStatus.MATCHED.value == "matched"
        assert ReconciliationStatus.MISMATCH_LOCAL_ONLY.value == "local_only"
        assert ReconciliationStatus.MISMATCH_BROKER_ONLY.value == "broker_only"
        assert ReconciliationStatus.MISMATCH_DETAILS.value == "details_mismatch"
        assert ReconciliationStatus.ERROR.value == "error"


# =============================================================================
# FETCH BROKER POSITIONS TESTS
# =============================================================================


class TestFetchBrokerPositions:
    """Tests for fetching positions from broker"""

    @patch("position_reconciliation.dhan")
    def test_fetch_positions_success(self, mock_dhan, mock_dhan_positions_response):
        """Should successfully fetch and parse broker positions"""
        from position_reconciliation import fetch_broker_positions

        mock_dhan.get_positions.return_value = mock_dhan_positions_response

        success, positions = fetch_broker_positions()

        assert success is True
        assert len(positions) == 1
        assert positions[0].security_id == "12345"

    @patch("position_reconciliation.dhan")
    def test_fetch_positions_failure(self, mock_dhan):
        """Should handle API failure gracefully"""
        from position_reconciliation import fetch_broker_positions

        mock_dhan.get_positions.return_value = {
            "status": "failure",
            "remarks": "API error",
        }

        success, positions = fetch_broker_positions()

        assert success is False
        assert positions == []

    @patch("position_reconciliation.dhan")
    def test_fetch_positions_exception(self, mock_dhan):
        """Should handle exceptions gracefully"""
        from position_reconciliation import fetch_broker_positions

        mock_dhan.get_positions.side_effect = Exception("Network error")

        success, positions = fetch_broker_positions()

        assert success is False
        assert positions == []

    @patch("position_reconciliation.dhan")
    def test_fetch_positions_filters_zero_quantity(self, mock_dhan):
        """Should filter out positions with zero quantity"""
        from position_reconciliation import fetch_broker_positions

        # V2 API uses buyAvg/sellAvg instead of averagePrice
        mock_dhan.get_positions.return_value = {
            "status": "success",
            "data": [
                {"securityId": "12345", "netQty": 10, "buyAvg": 100, "sellAvg": 0},
                {
                    "securityId": "12346",
                    "netQty": 0,
                    "buyAvg": 100,
                    "sellAvg": 0,
                },  # Zero qty
            ],
        }

        success, positions = fetch_broker_positions()

        assert success is True
        assert len(positions) == 1


# =============================================================================
# RECONCILIATION LOGIC TESTS
# =============================================================================


class TestReconcilePositions:
    """Tests for main reconciliation logic"""

    @patch("position_reconciliation.fetch_broker_positions")
    def test_reconcile_matched_with_position(
        self, mock_fetch, active_trade_with_position, mock_instruments
    ):
        """Should return MATCHED when local and broker positions match"""
        from position_reconciliation import (
            reconcile_positions,
            ReconciliationStatus,
            BrokerPosition,
        )

        broker_pos = BrokerPosition(
            security_id="12345",
            trading_symbol="CRUDEOIL26JAN6000CE",
            exchange_segment="MCX_COMM",  # V2 API format
            position_type="LONG",
            quantity=10,
            average_price=150.0,
            unrealized_pnl=0,
            product_type="INTRADAY",
        )
        mock_fetch.return_value = (True, [broker_pos])

        with patch("position_reconciliation.INSTRUMENTS", mock_instruments):
            result = reconcile_positions(active_trade_with_position)

        assert result.status == ReconciliationStatus.MATCHED

    @patch("position_reconciliation.fetch_broker_positions")
    def test_reconcile_matched_no_positions(self, mock_fetch, empty_trade_state):
        """Should return MATCHED when both local and broker have no positions"""
        from position_reconciliation import reconcile_positions, ReconciliationStatus

        mock_fetch.return_value = (True, [])

        result = reconcile_positions(empty_trade_state)

        assert result.status == ReconciliationStatus.MATCHED
        assert "No positions" in result.message

    @patch("position_reconciliation.fetch_broker_positions")
    @patch("position_reconciliation.send_alert")
    def test_reconcile_local_only_mismatch(
        self, mock_alert, mock_fetch, active_trade_with_position
    ):
        """Should detect when local has position but broker doesn't"""
        from position_reconciliation import reconcile_positions, ReconciliationStatus

        mock_fetch.return_value = (True, [])  # Broker has no positions

        result = reconcile_positions(active_trade_with_position)

        assert result.status == ReconciliationStatus.MISMATCH_LOCAL_ONLY
        assert len(result.discrepancies) > 0
        mock_alert.assert_called_once()

    @patch("position_reconciliation.fetch_broker_positions")
    @patch("position_reconciliation.send_alert")
    def test_reconcile_broker_only_mismatch(
        self, mock_alert, mock_fetch, empty_trade_state, mock_instruments
    ):
        """Should detect when broker has position but local doesn't"""
        from position_reconciliation import (
            reconcile_positions,
            ReconciliationStatus,
            BrokerPosition,
        )

        broker_pos = BrokerPosition(
            security_id="12345",
            trading_symbol="CRUDEOIL26JAN6000CE",
            exchange_segment="MCX_COMM",  # V2 API format
            position_type="LONG",
            quantity=10,
            average_price=150.0,
            unrealized_pnl=0,
            product_type="INTRADAY",
        )
        mock_fetch.return_value = (True, [broker_pos])

        with patch("position_reconciliation.INSTRUMENTS", mock_instruments):
            result = reconcile_positions(empty_trade_state)

        assert result.status == ReconciliationStatus.MISMATCH_BROKER_ONLY

    @patch("position_reconciliation.fetch_broker_positions")
    def test_reconcile_quantity_mismatch(
        self, mock_fetch, active_trade_with_position, mock_instruments
    ):
        """Should detect quantity mismatch between local and broker"""
        from position_reconciliation import (
            reconcile_positions,
            ReconciliationStatus,
            BrokerPosition,
        )

        # Broker shows different quantity
        broker_pos = BrokerPosition(
            security_id="12345",
            trading_symbol="CRUDEOIL26JAN6000CE",
            exchange_segment="MCX_COMM",  # V2 API format
            position_type="LONG",
            quantity=20,  # Different from local lot_size of 10
            average_price=150.0,
            unrealized_pnl=0,
            product_type="INTRADAY",
        )
        mock_fetch.return_value = (True, [broker_pos])

        with patch("position_reconciliation.INSTRUMENTS", mock_instruments):
            result = reconcile_positions(active_trade_with_position)

        assert result.status == ReconciliationStatus.MISMATCH_DETAILS
        assert "Quantity mismatch" in result.message

    @patch("position_reconciliation.fetch_broker_positions")
    def test_reconcile_error_on_fetch_failure(
        self, mock_fetch, active_trade_with_position
    ):
        """Should return ERROR status when fetch fails"""
        from position_reconciliation import reconcile_positions, ReconciliationStatus

        mock_fetch.return_value = (False, [])

        result = reconcile_positions(active_trade_with_position)

        assert result.status == ReconciliationStatus.ERROR


# =============================================================================
# AUTO-FIX TESTS
# =============================================================================


class TestAutoFixMismatch:
    """Tests for auto-fix functionality"""

    @patch("position_reconciliation.save_state")
    @patch("position_reconciliation.send_alert")
    def test_auto_fix_local_only_clears_state(
        self, mock_alert, mock_save, active_trade_with_position
    ):
        """Should clear local state when position closed at broker"""
        from position_reconciliation import (
            auto_fix_mismatch,
            ReconciliationResult,
            ReconciliationStatus,
        )

        result = ReconciliationResult(
            status=ReconciliationStatus.MISMATCH_LOCAL_ONLY,
            message="Position closed at broker",
        )

        fixed = auto_fix_mismatch(result, active_trade_with_position)

        assert fixed is True
        assert active_trade_with_position["status"] is False
        mock_save.assert_called_once()

    @patch("position_reconciliation.send_alert")
    def test_auto_fix_broker_only_returns_false(self, mock_alert, empty_trade_state):
        """Should not auto-fix when broker has untracked position"""
        from position_reconciliation import (
            auto_fix_mismatch,
            ReconciliationResult,
            ReconciliationStatus,
        )

        result = ReconciliationResult(
            status=ReconciliationStatus.MISMATCH_BROKER_ONLY,
            message="Untracked position at broker",
        )

        fixed = auto_fix_mismatch(result, empty_trade_state)

        assert fixed is False
        mock_alert.assert_called_once()

    @patch("position_reconciliation.save_state")
    @patch("position_reconciliation.send_alert")
    def test_auto_fix_quantity_mismatch_updates_local(
        self, mock_alert, mock_save, active_trade_with_position
    ):
        """Should update local quantity to match broker"""
        from position_reconciliation import (
            auto_fix_mismatch,
            ReconciliationResult,
            ReconciliationStatus,
            BrokerPosition,
        )

        broker_pos = BrokerPosition(
            security_id="12345",
            trading_symbol="CRUDEOIL26JAN6000CE",
            exchange_segment="MCX_COMM",  # V2 API format
            position_type="LONG",
            quantity=20,
            average_price=150.0,
            unrealized_pnl=0,
            product_type="INTRADAY",
        )

        result = ReconciliationResult(
            status=ReconciliationStatus.MISMATCH_DETAILS,
            message="Quantity mismatch",
            broker_positions=[broker_pos],
        )

        fixed = auto_fix_mismatch(result, active_trade_with_position)

        assert fixed is True
        assert active_trade_with_position["lot_size"] == 20

    def test_auto_fix_matched_returns_true(self, active_trade_with_position):
        """Should return True for already matched status"""
        from position_reconciliation import (
            auto_fix_mismatch,
            ReconciliationResult,
            ReconciliationStatus,
        )

        result = ReconciliationResult(status=ReconciliationStatus.MATCHED)

        fixed = auto_fix_mismatch(result, active_trade_with_position)

        assert fixed is True


# =============================================================================
# TRADE VERIFICATION TESTS
# =============================================================================


class TestTradeVerification:
    """Tests for trade entry/exit verification"""

    @patch("position_reconciliation.fetch_broker_positions")
    @patch("position_reconciliation.time.sleep")
    def test_verify_entry_success(self, mock_sleep, mock_fetch):
        """Should verify successful trade entry"""
        from position_reconciliation import verify_trade_entry, BrokerPosition

        broker_pos = BrokerPosition(
            security_id="12345",
            trading_symbol="CRUDEOIL26JAN6000CE",
            exchange_segment="MCX_COMM",  # V2 API format
            position_type="LONG",
            quantity=10,
            average_price=150.0,
            unrealized_pnl=0,
            product_type="INTRADAY",
        )
        mock_fetch.return_value = (True, [broker_pos])

        success, message = verify_trade_entry("ORDER123", "12345", 10)

        assert success is True
        assert "verified" in message.lower()

    @patch("position_reconciliation.fetch_broker_positions")
    @patch("position_reconciliation.time.sleep")
    def test_verify_entry_position_not_found(self, mock_sleep, mock_fetch):
        """Should fail when position not found after entry"""
        from position_reconciliation import verify_trade_entry

        mock_fetch.return_value = (True, [])

        success, message = verify_trade_entry("ORDER123", "12345", 10)

        assert success is False
        assert "not found" in message.lower()

    @patch("position_reconciliation.fetch_broker_positions")
    @patch("position_reconciliation.time.sleep")
    def test_verify_entry_quantity_mismatch(self, mock_sleep, mock_fetch):
        """Should fail when quantity doesn't match"""
        from position_reconciliation import verify_trade_entry, BrokerPosition

        broker_pos = BrokerPosition(
            security_id="12345",
            trading_symbol="CRUDEOIL26JAN6000CE",
            exchange_segment="MCX_COMM",  # V2 API format
            position_type="LONG",
            quantity=5,  # Less than expected
            average_price=150.0,
            unrealized_pnl=0,
            product_type="INTRADAY",
        )
        mock_fetch.return_value = (True, [broker_pos])

        success, message = verify_trade_entry("ORDER123", "12345", 10)

        assert success is False
        assert "mismatch" in message.lower()


# =============================================================================
# STARTUP RECONCILIATION TESTS
# =============================================================================


class TestStartupReconciliation:
    """Tests for startup reconciliation"""

    @patch("position_reconciliation.reconcile_positions")
    def test_startup_reconcile_pass(self, mock_reconcile, active_trade_with_position):
        """Should return True when reconciliation passes"""
        from position_reconciliation import (
            reconcile_on_startup,
            ReconciliationResult,
            ReconciliationStatus,
        )

        mock_reconcile.return_value = ReconciliationResult(
            status=ReconciliationStatus.MATCHED, message="All matched"
        )

        result = reconcile_on_startup(active_trade_with_position)

        assert result is True

    @patch("position_reconciliation.reconcile_positions")
    @patch("position_reconciliation.auto_fix_mismatch")
    def test_startup_reconcile_attempts_fix(
        self, mock_fix, mock_reconcile, active_trade_with_position
    ):
        """Should attempt auto-fix on mismatch"""
        from position_reconciliation import (
            reconcile_on_startup,
            ReconciliationResult,
            ReconciliationStatus,
        )

        mock_reconcile.return_value = ReconciliationResult(
            status=ReconciliationStatus.MISMATCH_LOCAL_ONLY, message="Local only"
        )
        mock_fix.return_value = True

        result = reconcile_on_startup(active_trade_with_position)

        mock_fix.assert_called_once()
        assert result is True
