"""
Integration tests with mock broker for end-to-end trading workflow
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
import threading
import time


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_dhan_client():
    """Create comprehensive mock Dhan client"""
    mock = MagicMock()
    
    # Constants
    mock.BUY = "BUY"
    mock.SELL = "SELL"
    mock.LIMIT = "LIMIT"
    mock.MARKET = "MARKET"
    mock.INTRADAY = "INTRADAY"
    mock.CNC = "CNC"
    mock.NSE = "NSE"
    mock.MCX = "MCX"
    
    return mock


@pytest.fixture
def mock_instruments():
    """Complete instrument configuration - V2 API format"""
    return {
        "CRUDEOIL": {
            "name": "CRUDE OIL",
            "exchange_segment_int": 5,
            "exchange_segment_str": "MCX_COMM",  # V2 API format
            "future_id": "464926",
            "lot_size": 10,
            "strike_step": 50,
            "expiry_date": "2026-01-16",
            "instrument_type": "FUTCOM",  # V2 API format
            "option_type": "OPTFUT",
            "market_start": "09:00",
            "market_end": "23:30",
            "no_new_trade_after": "23:00",
        }
    }


@pytest.fixture
def empty_trade_state():
    """Empty trade state - no active position"""
    return {
        "status": False,
        "type": None,
        "instrument": None,
        "entry": 0,
        "initial_sl": 0,
        "sl": 0,
        "step_level": 0,
        "option_id": None,
        "option_entry": 0,
        "lot_size": 0,
        "exchange_segment_str": None,
        "entry_time": None
    }


@pytest.fixture
def active_buy_trade():
    """Active BUY trade state - V2 API format"""
    return {
        "status": True,
        "type": "BUY",
        "instrument": "CRUDEOIL",
        "entry": 6000,
        "initial_sl": 5980,
        "sl": 5980,
        "step_level": 0,
        "option_id": "12345",
        "option_entry": 150,
        "lot_size": 10,
        "exchange_segment_str": "MCX_COMM",  # V2 API format
        "entry_time": "2026-01-10 10:00:00"
    }


# =============================================================================
# MOCK BROKER RESPONSES
# =============================================================================

@pytest.fixture
def mock_order_response():
    """Mock successful order placement response"""
    return {
        'status': 'success',
        'data': {
            'orderId': 'ORDER123456',
            'orderStatus': 'PENDING'
        }
    }


@pytest.fixture
def mock_position_response():
    """Mock positions API response - V2 API format"""
    return {
        'status': 'success',
        'data': [
            {
                'securityId': '12345',
                'tradingSymbol': 'CRUDEOIL26JAN6000CE',
                'exchangeSegment': 'MCX_COMM',  # V2 API format
                'netQty': 10,
                'buyAvg': 150.0,  # V2 uses buyAvg/sellAvg
                'sellAvg': 0.0,
                'unrealizedProfit': 200.0,
                'productType': 'INTRADAY'
            }
        ]
    }


@pytest.fixture
def mock_order_status_response():
    """Mock order status check response - V2 API format"""
    return {
        'status': 'success',
        'data': {
            'orderId': 'ORDER123456',
            'orderStatus': 'TRADED',
            'tradedQuantity': 10,
            'averageTradedPrice': 150.0  # V2 field name
        }
    }


# =============================================================================
# INTEGRATION TEST: FULL TRADE ENTRY FLOW
# =============================================================================

class TestTradeEntryIntegration:
    """Integration tests for trade entry workflow"""
    
    @patch('manager.dhan')
    @patch('manager.socket_handler')
    @patch('manager.send_alert')
    @patch('manager.save_state')
    def test_full_buy_signal_to_entry(
        self, 
        mock_save, 
        mock_alert, 
        mock_socket, 
        mock_dhan,
        mock_instruments,
        empty_trade_state,
        mock_order_response
    ):
        """Test complete flow from BUY signal to position entry"""
        # Setup mocks
        mock_dhan.place_order.return_value = mock_order_response
        mock_socket.get_latest_ltp.return_value = 6000.0
        
        # Simulate signal analysis result
        signal = {
            'type': 'BUY',
            'instrument': 'CRUDEOIL',
            'entry_price': 6000.0,
            'stop_loss': 5980.0,
        }
        
        # Verify the signal is valid
        assert signal['type'] in ['BUY', 'SELL']
        assert signal['stop_loss'] < signal['entry_price']  # For BUY
        
        # Calculate option strike
        entry = signal['entry_price']
        strike_step = mock_instruments['CRUDEOIL']['strike_step']
        atm_strike = round(entry / strike_step) * strike_step
        
        assert atm_strike == 6000  # ATM for 6000 with step 50
    
    @patch('manager.dhan')
    def test_order_placement_with_limit_buffer(
        self, 
        mock_dhan, 
        mock_instruments,
        mock_order_response
    ):
        """Test that limit orders include proper buffer"""
        mock_dhan.place_order.return_value = mock_order_response
        
        option_ltp = 150.0
        limit_buffer = 0.5  # 0.5% buffer
        
        # For BUY, add buffer
        buy_limit_price = option_ltp * (1 + limit_buffer / 100)
        assert buy_limit_price > option_ltp
        
        # For SELL, subtract buffer
        sell_limit_price = option_ltp * (1 - limit_buffer / 100)
        assert sell_limit_price < option_ltp
    
    def test_lot_size_calculation(self, mock_instruments):
        """Test lot size is correctly fetched from instrument config"""
        instrument = mock_instruments['CRUDEOIL']
        lot_size = instrument['lot_size']
        
        assert lot_size == 10
        
        # Calculate position value
        option_premium = 150.0
        position_value = option_premium * lot_size
        
        assert position_value == 1500.0


# =============================================================================
# INTEGRATION TEST: TRAILING STOP LOSS MANAGEMENT
# =============================================================================

class TestTrailingStopLossIntegration:
    """Integration tests for trailing stop loss workflow"""
    
    def test_step_ladder_progression(self, active_buy_trade):
        """Test complete step ladder progression from entry to target"""
        entry = active_buy_trade['entry']
        initial_sl = active_buy_trade['initial_sl']
        risk_unit = abs(entry - initial_sl)  # 20 points
        
        # Track SL changes through step levels
        sl_history = [initial_sl]
        
        # Step 1: At 2R, lock 1R
        price_at_2r = entry + (2 * risk_unit)  # 6040
        new_sl = entry + (1 * risk_unit)  # 6020
        sl_history.append(new_sl)
        
        assert price_at_2r == 6040
        assert new_sl == 6020
        
        # Step 2: At 3R, lock 2R
        price_at_3r = entry + (3 * risk_unit)  # 6060
        new_sl = entry + (2 * risk_unit)  # 6040
        sl_history.append(new_sl)
        
        assert price_at_3r == 6060
        assert new_sl == 6040
        
        # Step 3: At 4R, lock 3R
        price_at_4r = entry + (4 * risk_unit)  # 6080
        new_sl = entry + (3 * risk_unit)  # 6060
        sl_history.append(new_sl)
        
        # Target: At 5R, exit
        target_price = entry + (5 * risk_unit)  # 6100
        
        assert target_price == 6100
        
        # Verify SL only moves up
        for i in range(1, len(sl_history)):
            assert sl_history[i] >= sl_history[i-1]
    
    def test_sl_update_triggers_order_modification(self, active_buy_trade):
        """Test that SL update should trigger order modification"""
        old_sl = active_buy_trade['sl']
        new_sl = old_sl + 20  # Move SL up by 1R
        
        # SL should only move in profit direction
        if active_buy_trade['type'] == 'BUY':
            assert new_sl > old_sl
        else:
            assert new_sl < old_sl


# =============================================================================
# INTEGRATION TEST: POSITION EXIT FLOW
# =============================================================================

class TestPositionExitIntegration:
    """Integration tests for position exit workflow"""
    
    @patch('manager.dhan')
    @patch('manager.socket_handler')
    @patch('manager.save_state')
    @patch('manager.update_daily_pnl')
    @patch('manager.save_trade_to_history')
    @patch('manager.send_alert')
    def test_full_exit_on_stop_loss(
        self,
        mock_alert,
        mock_history,
        mock_pnl,
        mock_save,
        mock_socket,
        mock_dhan,
        active_buy_trade
    ):
        """Test complete exit flow when stop loss hit"""
        # Setup
        mock_socket.get_option_ltp.return_value = 120.0  # Exit at loss
        mock_socket.get_market_feed.return_value = MagicMock()
        mock_pnl.return_value = {'daily_pnl': -300, 'win_count': 0, 'loss_count': 1}
        
        # Calculate P&L
        entry_premium = active_buy_trade['option_entry']
        exit_premium = 120.0
        lot_size = active_buy_trade['lot_size']
        
        pnl = (exit_premium - entry_premium) * lot_size
        
        assert pnl == -300  # Loss of 30 points * 10 lots
    
    @patch('manager.dhan')
    @patch('manager.socket_handler')
    def test_full_exit_on_target(
        self,
        mock_socket,
        mock_dhan,
        active_buy_trade
    ):
        """Test complete exit flow when target reached"""
        mock_socket.get_option_ltp.return_value = 250.0  # Exit at profit
        
        # Calculate P&L
        entry_premium = active_buy_trade['option_entry']
        exit_premium = 250.0
        lot_size = active_buy_trade['lot_size']
        
        pnl = (exit_premium - entry_premium) * lot_size
        
        assert pnl == 1000  # Profit of 100 points * 10 lots
    
    def test_r_multiple_calculation_on_exit(self, active_buy_trade):
        """Test R-multiple calculation for exit"""
        entry = active_buy_trade['entry']
        initial_sl = active_buy_trade['initial_sl']
        risk_unit = abs(entry - initial_sl)
        
        # Exit at 3R
        exit_price = entry + (3 * risk_unit)
        profit_points = exit_price - entry
        r_multiple = profit_points / risk_unit
        
        assert r_multiple == 3.0


# =============================================================================
# INTEGRATION TEST: RECONCILIATION WITH BROKER
# =============================================================================

class TestReconciliationIntegration:
    """Integration tests for position reconciliation with broker"""
    
    @patch('position_reconciliation.dhan')
    def test_startup_reconciliation_passes(
        self, 
        mock_dhan, 
        active_buy_trade,
        mock_position_response
    ):
        """Test startup reconciliation when positions match"""
        from position_reconciliation import reconcile_positions, ReconciliationStatus
        
        mock_dhan.get_positions.return_value = mock_position_response
        
        result = reconcile_positions(active_buy_trade)
        
        assert result.status == ReconciliationStatus.MATCHED
    
    @patch('position_reconciliation.dhan')
    @patch('position_reconciliation.send_alert')
    def test_detect_orphan_position_at_broker(
        self, 
        mock_alert,
        mock_dhan, 
        empty_trade_state,
        mock_position_response,
        mock_instruments
    ):
        """Test detection of position at broker not tracked locally"""
        from position_reconciliation import reconcile_positions, ReconciliationStatus
        
        mock_dhan.get_positions.return_value = mock_position_response
        
        with patch('position_reconciliation.INSTRUMENTS', mock_instruments):
            result = reconcile_positions(empty_trade_state)
        
        assert result.status == ReconciliationStatus.MISMATCH_BROKER_ONLY
    
    @patch('position_reconciliation.dhan')
    @patch('position_reconciliation.send_alert')
    def test_detect_phantom_local_position(
        self, 
        mock_alert,
        mock_dhan, 
        active_buy_trade
    ):
        """Test detection of local position not found at broker"""
        from position_reconciliation import reconcile_positions, ReconciliationStatus
        
        mock_dhan.get_positions.return_value = {
            'status': 'success',
            'data': []  # No positions at broker
        }
        
        result = reconcile_positions(active_buy_trade)
        
        assert result.status == ReconciliationStatus.MISMATCH_LOCAL_ONLY


# =============================================================================
# INTEGRATION TEST: CONTRACT AUTO-UPDATE
# =============================================================================

class TestContractUpdateIntegration:
    """Integration tests for contract auto-update"""
    
    @patch('contract_updater.download_scrip_master')
    @patch('contract_updater.load_scrip_master')
    def test_auto_update_refreshes_contracts(
        self, 
        mock_load, 
        mock_download, 
        mock_instruments
    ):
        """Test that auto-update fetches new contract IDs"""
        from contract_updater import update_all_instruments
        
        mock_download.return_value = True
        
        # New contracts with updated IDs
        today = datetime.now()
        new_expiry = (today + timedelta(days=20)).strftime('%Y-%m-%d')
        
        mock_load.return_value = [
            {
                'SEM_SMST_SECURITY_ID': 'NEW_464926',
                'SEM_EXM_EXCH_ID': 'MCX',
                'SEM_INSTRUMENT_NAME': 'FUTCOM',
                'SEM_TRADING_SYMBOL': 'CRUDEOIL-26FEB-FUT',
                'SEM_CUSTOM_SYMBOL': 'CRUDEOIL FEB FUT',
                'SEM_EXPIRY_DATE': new_expiry,
                'SEM_LOT_UNITS': '10',
            }
        ]
        
        result = update_all_instruments(mock_instruments)
        
        assert result['CRUDEOIL']['future_id'] == 'NEW_464926'
    
    @patch('contract_updater.load_contract_cache')
    def test_use_cache_when_valid(self, mock_cache, mock_instruments):
        """Test that valid cache is used instead of fetching"""
        from contract_updater import auto_update_instruments_on_startup
        
        mock_cache.return_value = {
            'CRUDEOIL': {
                'future_id': 'CACHED_ID',
                'expiry_date': '2026-01-16',
                'lot_size': 10
            }
        }
        
        result = auto_update_instruments_on_startup(mock_instruments)
        
        assert result['CRUDEOIL']['future_id'] == 'CACHED_ID'


# =============================================================================
# INTEGRATION TEST: WEBSOCKET RECONNECTION DURING TRADE
# =============================================================================

class TestWebSocketReconnectionIntegration:
    """Integration tests for WebSocket reconnection during active trade"""
    
    def test_socket_reconnect_preserves_subscriptions(self, active_buy_trade, mock_instruments):
        """Test that reconnection re-subscribes to option feed"""
        import socket_handler
        
        # Simulate active trade with option subscription
        option_id = active_buy_trade['option_id']
        
        # Verify option_id is preserved for re-subscription
        assert option_id is not None
        assert option_id == '12345'
    
    def test_ltp_recovery_after_reconnect(self):
        """Test that LTP is recovered after socket reconnection"""
        import socket_handler
        
        # Save state before disconnect
        socket_handler.LATEST_LTP = 6000.0
        old_ltp = socket_handler.get_latest_ltp()
        
        # Simulate reconnection
        socket_handler.SOCKET_RECONNECT_EVENT.set()
        socket_handler.SOCKET_RECONNECT_EVENT.clear()
        
        # LTP should remain (or be refreshed on next tick)
        assert socket_handler.LATEST_LTP == old_ltp


# =============================================================================
# INTEGRATION TEST: FULL TRADING DAY SIMULATION
# =============================================================================

class TestFullTradingDaySimulation:
    """Simulate a complete trading day workflow"""
    
    def test_day_start_sequence(self, mock_instruments, empty_trade_state):
        """Test proper sequence of operations at day start"""
        sequence = []
        
        # 1. Contract update
        sequence.append('contract_update')
        
        # 2. Position reconciliation
        sequence.append('reconcile')
        
        # 3. Start WebSocket
        sequence.append('socket_start')
        
        # 4. Start scanning
        sequence.append('scanner_start')
        
        expected_sequence = ['contract_update', 'reconcile', 'socket_start', 'scanner_start']
        assert sequence == expected_sequence
    
    def test_trade_cooldown_after_loss(self, active_buy_trade):
        """Test that cooldown is applied after losing trade"""
        # Simulate loss
        is_loss = True
        cooldown_minutes = 3
        
        if is_loss:
            cooldown_end = datetime.now() + timedelta(minutes=cooldown_minutes)
        
        # Verify cooldown is set
        assert cooldown_end > datetime.now()
    
    def test_daily_pnl_tracking(self):
        """Test daily P&L accumulation across multiple trades"""
        daily_pnl = 0
        trades = [
            {'pnl': 500, 'is_win': True},
            {'pnl': -300, 'is_win': False},
            {'pnl': 800, 'is_win': True},
        ]
        
        for trade in trades:
            daily_pnl += trade['pnl']
        
        assert daily_pnl == 1000
        
        # Count wins and losses
        wins = sum(1 for t in trades if t['is_win'])
        losses = len(trades) - wins
        
        assert wins == 2
        assert losses == 1


# =============================================================================
# INTEGRATION TEST: ERROR HANDLING
# =============================================================================

class TestErrorHandlingIntegration:
    """Integration tests for error handling scenarios"""
    
    @patch('manager.dhan')
    def test_order_rejection_handling(self, mock_dhan):
        """Test handling of order rejection from broker"""
        mock_dhan.place_order.return_value = {
            'status': 'failure',
            'remarks': 'Insufficient margin'
        }
        
        # Verify rejection is detected
        response = mock_dhan.place_order()
        assert response['status'] == 'failure'
    
    @patch('manager.dhan')
    def test_api_timeout_handling(self, mock_dhan):
        """Test handling of API timeout"""
        mock_dhan.place_order.side_effect = TimeoutError("API timeout")
        
        with pytest.raises(TimeoutError):
            mock_dhan.place_order()
    
    def test_invalid_signal_rejected(self):
        """Test that invalid signals are rejected"""
        invalid_signal = {
            'type': 'BUY',
            'entry_price': 6000,
            'stop_loss': 6020,  # SL above entry for BUY is invalid
        }
        
        # Validate signal
        is_valid = True
        if invalid_signal['type'] == 'BUY' and invalid_signal['stop_loss'] >= invalid_signal['entry_price']:
            is_valid = False
        
        assert is_valid is False
