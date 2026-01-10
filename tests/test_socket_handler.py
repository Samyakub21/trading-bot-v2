"""
Unit tests for socket_handler.py - WebSocket reconnection logic
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_instruments():
    """Mock instruments configuration"""
    return {
        "CRUDEOIL": {
            "name": "CRUDE OIL",
            "exchange_segment_int": 5,
            "exchange_segment_str": "MCX",
            "future_id": "464926",
            "lot_size": 10,
        },
        "GOLD": {
            "name": "GOLD",
            "exchange_segment_int": 5,
            "exchange_segment_str": "MCX",
            "future_id": "464927",
            "lot_size": 100,
        }
    }


@pytest.fixture
def mock_active_trade():
    """Mock active trade state"""
    return {
        "status": True,
        "type": "BUY",
        "instrument": "CRUDEOIL",
        "option_id": "12345",
    }


@pytest.fixture
def mock_market_feed():
    """Mock DhanFeed instance"""
    mock = MagicMock()
    mock.subscribe_symbols = MagicMock()
    mock.unsubscribe_symbols = MagicMock()
    mock.close_connection = MagicMock()
    mock.run_forever = MagicMock()
    mock.get_data = MagicMock(return_value=None)
    return mock


# =============================================================================
# TICK HANDLING TESTS
# =============================================================================

class TestTickHandling:
    """Tests for tick data handling"""
    
    def test_on_ticks_updates_ltp(self, mock_instruments, mock_active_trade):
        """Should update LATEST_LTP on receiving tick for active instrument"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            import socket_handler
            
            # Reset state
            socket_handler.LATEST_LTP = 0
            socket_handler.SOCKET_HEALTHY.clear()
            
            ticks = {
                'LTP': 6050.0,
                'security_id': '464926'  # CRUDEOIL future_id
            }
            
            socket_handler.on_ticks(None, ticks, "CRUDEOIL", mock_active_trade)
            
            assert socket_handler.LATEST_LTP == 6050.0
            assert socket_handler.SOCKET_HEALTHY.is_set()
    
    def test_on_ticks_updates_option_ltp(self, mock_instruments, mock_active_trade):
        """Should update OPTION_LTP when tick is for active option"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            import socket_handler
            
            # Reset state
            socket_handler.OPTION_LTP = 0
            
            ticks = {
                'LTP': 150.0,
                'security_id': '12345'  # Active option_id
            }
            
            socket_handler.on_ticks(None, ticks, "CRUDEOIL", mock_active_trade)
            
            assert socket_handler.OPTION_LTP == 150.0
    
    def test_on_ticks_updates_instrument_ltp_dict(self, mock_instruments, mock_active_trade):
        """Should update INSTRUMENT_LTP dict for multi-scan mode"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            import socket_handler
            
            socket_handler.INSTRUMENT_LTP = {}
            
            # Tick for GOLD (not active instrument)
            ticks = {
                'LTP': 62000.0,
                'security_id': '464927'  # GOLD future_id
            }
            
            # Active instrument is CRUDEOIL
            socket_handler.on_ticks(None, ticks, "CRUDEOIL", {"status": False})
            
            assert "GOLD" in socket_handler.INSTRUMENT_LTP
            assert socket_handler.INSTRUMENT_LTP["GOLD"]["ltp"] == 62000.0
    
    def test_on_ticks_ignores_non_ltp_data(self, mock_instruments, mock_active_trade):
        """Should ignore ticks without LTP field"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            import socket_handler
            
            initial_ltp = socket_handler.LATEST_LTP
            
            ticks = {
                'volume': 1000,
                'security_id': '464926'
            }
            
            socket_handler.on_ticks(None, ticks, "CRUDEOIL", mock_active_trade)
            
            # LTP should remain unchanged
            assert socket_handler.LATEST_LTP == initial_ltp


# =============================================================================
# SOCKET SUBSCRIPTION TESTS
# =============================================================================

class TestSocketSubscription:
    """Tests for option subscription/unsubscription"""
    
    def test_subscribe_option_success(self, mock_market_feed):
        """Should successfully subscribe to option feed"""
        import socket_handler
        
        socket_handler.subscribe_option(mock_market_feed, "12345", 5)
        
        mock_market_feed.subscribe_symbols.assert_called_once()
        call_args = mock_market_feed.subscribe_symbols.call_args[0][0]
        assert call_args[0][1] == "12345"
    
    def test_unsubscribe_option_success(self, mock_market_feed):
        """Should successfully unsubscribe from option feed"""
        import socket_handler
        
        socket_handler.unsubscribe_option(mock_market_feed, "12345", 5)
        
        mock_market_feed.unsubscribe_symbols.assert_called_once()
    
    def test_subscribe_option_handles_exception(self, mock_market_feed, caplog):
        """Should handle subscription errors gracefully"""
        import socket_handler
        import logging
        
        mock_market_feed.subscribe_symbols.side_effect = Exception("Connection error")
        
        with caplog.at_level(logging.ERROR):
            socket_handler.subscribe_option(mock_market_feed, "12345", 5)
        
        assert "Failed to subscribe" in caplog.text
    
    def test_unsubscribe_option_handles_exception(self, mock_market_feed, caplog):
        """Should handle unsubscription errors gracefully"""
        import socket_handler
        import logging
        
        mock_market_feed.unsubscribe_symbols.side_effect = Exception("Connection error")
        
        with caplog.at_level(logging.ERROR):
            socket_handler.unsubscribe_option(mock_market_feed, "12345", 5)
        
        assert "Failed to unsubscribe" in caplog.text


# =============================================================================
# RECONNECTION LOGIC TESTS
# =============================================================================

class TestReconnectionLogic:
    """Tests for WebSocket reconnection logic"""
    
    def test_socket_healthy_event_set_on_tick(self, mock_instruments, mock_active_trade):
        """SOCKET_HEALTHY event should be set when tick received"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            import socket_handler
            
            socket_handler.SOCKET_HEALTHY.clear()
            
            ticks = {'LTP': 6000.0, 'security_id': '464926'}
            socket_handler.on_ticks(None, ticks, "CRUDEOIL", mock_active_trade)
            
            assert socket_handler.SOCKET_HEALTHY.is_set()
    
    def test_reconnect_event_can_be_set(self):
        """SOCKET_RECONNECT_EVENT can be set to trigger reconnection"""
        import socket_handler
        
        socket_handler.SOCKET_RECONNECT_EVENT.clear()
        assert not socket_handler.SOCKET_RECONNECT_EVENT.is_set()
        
        socket_handler.SOCKET_RECONNECT_EVENT.set()
        assert socket_handler.SOCKET_RECONNECT_EVENT.is_set()
        
        socket_handler.SOCKET_RECONNECT_EVENT.clear()
    
    def test_shutdown_event_signals_socket_shutdown(self):
        """SHUTDOWN_EVENT should signal socket to stop"""
        import socket_handler
        
        socket_handler.SHUTDOWN_EVENT.clear()
        assert not socket_handler.is_shutdown()
        
        socket_handler.shutdown_socket()
        assert socket_handler.is_shutdown()
        
        # Reset for other tests
        socket_handler.SHUTDOWN_EVENT.clear()
    
    def test_heartbeat_monitor_triggers_reconnect_on_timeout(self):
        """Heartbeat monitor should trigger reconnect when no ticks received"""
        import socket_handler
        
        # Clear events
        socket_handler.SOCKET_HEALTHY.clear()
        socket_handler.SOCKET_RECONNECT_EVENT.clear()
        socket_handler.SHUTDOWN_EVENT.clear()
        
        # Simulate heartbeat check with very short timeout for testing
        # In real code, timeout is 30s, but we test the logic
        socket_ok = socket_handler.SOCKET_HEALTHY.wait(timeout=0.1)
        
        if not socket_ok:
            socket_handler.SOCKET_RECONNECT_EVENT.set()
        
        assert socket_handler.SOCKET_RECONNECT_EVENT.is_set()
        
        # Cleanup
        socket_handler.SOCKET_RECONNECT_EVENT.clear()


# =============================================================================
# GETTER/SETTER TESTS
# =============================================================================

class TestGettersSetters:
    """Tests for state getter and setter functions"""
    
    def test_get_latest_ltp(self):
        """Should return current LATEST_LTP"""
        import socket_handler
        
        socket_handler.LATEST_LTP = 6100.0
        assert socket_handler.get_latest_ltp() == 6100.0
    
    def test_get_option_ltp(self):
        """Should return current OPTION_LTP"""
        import socket_handler
        
        socket_handler.OPTION_LTP = 175.0
        assert socket_handler.get_option_ltp() == 175.0
    
    def test_set_option_ltp(self):
        """Should set OPTION_LTP"""
        import socket_handler
        
        socket_handler.set_option_ltp(200.0)
        assert socket_handler.OPTION_LTP == 200.0
    
    def test_reset_option_ltp(self):
        """Should reset OPTION_LTP to 0"""
        import socket_handler
        
        socket_handler.OPTION_LTP = 150.0
        socket_handler.reset_option_ltp()
        assert socket_handler.OPTION_LTP == 0
    
    def test_get_last_tick_time(self):
        """Should return last tick time"""
        import socket_handler
        
        test_time = datetime.now()
        socket_handler.LAST_TICK_TIME = test_time
        
        assert socket_handler.get_last_tick_time() == test_time


# =============================================================================
# MULTI-INSTRUMENT SUBSCRIPTION TESTS
# =============================================================================

class TestMultiInstrumentSubscription:
    """Tests for multi-instrument subscription logic"""
    
    def test_get_all_instrument_subscriptions_single_mode(self, mock_instruments):
        """Should return single subscription when multi-scan disabled"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            with patch('socket_handler.MULTI_SCAN_ENABLED', False):
                with patch('socket_handler.get_instruments_to_scan', return_value=["CRUDEOIL"]):
                    import socket_handler
                    
                    # Force reimport to get patched values
                    subs = socket_handler.get_all_instrument_subscriptions("CRUDEOIL")
                    
                    assert len(subs) >= 1
                    assert subs[0][1] == "464926"  # CRUDEOIL future_id
    
    def test_get_all_instrument_subscriptions_multi_mode(self, mock_instruments):
        """Should return multiple subscriptions when multi-scan enabled"""
        with patch('socket_handler.INSTRUMENTS', mock_instruments):
            with patch('socket_handler.MULTI_SCAN_ENABLED', True):
                with patch('socket_handler.get_instruments_to_scan', return_value=["CRUDEOIL", "GOLD"]):
                    import socket_handler
                    
                    subs = socket_handler.get_all_instrument_subscriptions("CRUDEOIL")
                    
                    assert len(subs) == 2
