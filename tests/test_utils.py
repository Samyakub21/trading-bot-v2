"""
Unit tests for utils.py functions
"""

import pytest
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_dynamic_sl,
    is_market_open,
    can_place_new_trade,
    check_daily_limits,
    check_cooldown,
    check_signal_cooldown,
    get_performance_stats,
    save_state,
    load_state,
    MAX_DAILY_LOSS,
    MAX_TRADES_PER_DAY,
    COOLDOWN_AFTER_LOSS,
    SIGNAL_COOLDOWN
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_df_buy():
    """Create sample DataFrame for BUY signal testing"""
    data = {
        'open': [100, 102, 104, 105, 106],
        'high': [103, 105, 107, 108, 109],
        'low': [99, 101, 103, 104, 105],
        'close': [102, 104, 106, 107, 108],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2026-01-10 09:00', periods=5, freq='15min')
    return df


@pytest.fixture
def sample_df_sell():
    """Create sample DataFrame for SELL signal testing"""
    data = {
        'open': [110, 108, 106, 104, 102],
        'high': [112, 110, 108, 106, 104],
        'low': [108, 106, 104, 102, 100],
        'close': [108, 106, 104, 102, 100],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2026-01-10 09:00', periods=5, freq='15min')
    return df


@pytest.fixture
def temp_state_file(tmp_path):
    """Create temporary state file for testing"""
    state_file = tmp_path / "test_state.json"
    return str(state_file)


# =============================================================================
# STOP LOSS TESTS
# =============================================================================

class TestGetDynamicSL:
    """Tests for dynamic stop loss calculation"""
    
    def test_buy_sl_below_current_price(self, sample_df_buy):
        """SL for BUY should be below current price"""
        sl = get_dynamic_sl("BUY", sample_df_buy)
        current_price = sample_df_buy.iloc[-1]['close']
        assert sl < current_price, f"BUY SL ({sl}) should be below price ({current_price})"
    
    def test_sell_sl_above_current_price(self, sample_df_sell):
        """SL for SELL should be above current price"""
        sl = get_dynamic_sl("SELL", sample_df_sell)
        current_price = sample_df_sell.iloc[-1]['close']
        assert sl > current_price, f"SELL SL ({sl}) should be above price ({current_price})"
    
    def test_buy_sl_uses_swing_low(self, sample_df_buy):
        """BUY SL should be based on swing low of last 2 candles"""
        sl = get_dynamic_sl("BUY", sample_df_buy, buffer=2)
        swing_low = sample_df_buy.iloc[-3:-1]['low'].min()
        expected_sl = swing_low - 2
        assert sl == int(expected_sl), f"Expected SL {expected_sl}, got {sl}"
    
    def test_sell_sl_uses_swing_high(self, sample_df_sell):
        """SELL SL should be based on swing high of last 2 candles"""
        sl = get_dynamic_sl("SELL", sample_df_sell, buffer=2)
        swing_high = sample_df_sell.iloc[-3:-1]['high'].max()
        expected_sl = swing_high + 2
        assert sl == int(expected_sl), f"Expected SL {expected_sl}, got {sl}"
    
    def test_minimum_sl_distance_buy(self):
        """BUY SL should maintain minimum 5 point distance"""
        # Create df where swing low is very close to current price
        data = {
            'open': [100, 100, 100],
            'high': [101, 101, 101],
            'low': [99.5, 99.8, 99.9],
            'close': [100, 100, 100],
            'volume': [1000, 1000, 1000]
        }
        df = pd.DataFrame(data)
        df.index = pd.date_range(start='2026-01-10', periods=3, freq='15min')
        
        sl = get_dynamic_sl("BUY", df)
        current_price = df.iloc[-1]['close']
        assert (current_price - sl) >= 5, "Minimum SL distance should be 5 points"
    
    def test_empty_dataframe_returns_default(self):
        """Empty or small DataFrame should return default SL"""
        df = pd.DataFrame()
        sl = get_dynamic_sl("BUY", df)
        assert sl == 20, "Default SL should be 20"
    
    def test_insufficient_data_returns_default(self):
        """DataFrame with less than 3 rows should return default"""
        data = {'low': [100, 101], 'high': [105, 106], 'close': [103, 104]}
        df = pd.DataFrame(data)
        sl = get_dynamic_sl("BUY", df)
        assert sl == 20, "Default SL should be 20 for insufficient data"


# =============================================================================
# MARKET HOURS TESTS
# =============================================================================

class TestMarketHours:
    """Tests for market hours checking"""
    
    @patch('utils.datetime')
    def test_market_open_during_hours(self, mock_datetime):
        """Market should be open during trading hours on weekday"""
        # Create a mock now object
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0  # Monday
        mock_now.strftime.return_value = "10:30"
        mock_datetime.now.return_value = mock_now
        
        is_open, msg = is_market_open("09:00", "15:30")
        assert is_open == True
        assert "Open" in msg
    
    @patch('utils.datetime')
    def test_market_closed_weekend(self, mock_datetime):
        """Market should be closed on weekends"""
        with patch('utils.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 5  # Saturday
            mock_now.strftime.return_value = "10:30"
            mock_dt.now.return_value = mock_now
            
            is_open, msg = is_market_open("09:00", "15:30")
            assert is_open == False
            assert "Weekend" in msg
    
    @patch('utils.datetime')
    def test_market_closed_before_open(self, mock_datetime):
        """Market should be closed before opening time"""
        with patch('utils.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 0  # Monday
            mock_now.strftime.return_value = "08:30"
            mock_dt.now.return_value = mock_now
            
            is_open, msg = is_market_open("09:00", "15:30")
            assert is_open == False
            assert "not open yet" in msg
    
    @patch('utils.datetime')
    def test_market_closed_after_close(self, mock_datetime):
        """Market should be closed after closing time"""
        with patch('utils.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.weekday.return_value = 0  # Monday
            mock_now.strftime.return_value = "16:00"
            mock_dt.now.return_value = mock_now
            
            is_open, msg = is_market_open("09:00", "15:30")
            assert is_open == False
            assert "closed" in msg.lower()


class TestCanPlaceNewTrade:
    """Tests for new trade time restrictions"""
    
    @patch('utils.datetime')
    def test_can_trade_before_cutoff(self, mock_datetime):
        """Should allow trades before cutoff time"""
        with patch('utils.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "14:30"
            mock_dt.now.return_value = mock_now
            
            can_trade, msg = can_place_new_trade("15:00")
            assert can_trade == True
    
    @patch('utils.datetime')
    def test_cannot_trade_after_cutoff(self, mock_datetime):
        """Should not allow trades after cutoff time"""
        with patch('utils.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "15:30"
            mock_dt.now.return_value = mock_now
            
            can_trade, msg = can_place_new_trade("15:00")
            assert can_trade == False
            assert "No new trades" in msg


# =============================================================================
# RISK MANAGEMENT TESTS
# =============================================================================

class TestDailyLimits:
    """Tests for daily limit checking"""
    
    @patch('utils.load_daily_pnl')
    def test_within_limits(self, mock_load):
        """Should allow trading within limits"""
        mock_load.return_value = {"pnl": -1000, "trades": 2, "date": "2026-01-10"}
        
        within, msg = check_daily_limits()
        assert within == True
    
    @patch('utils.load_daily_pnl')
    def test_loss_limit_exceeded(self, mock_load):
        """Should block trading when loss limit exceeded"""
        mock_load.return_value = {"pnl": -MAX_DAILY_LOSS - 100, "trades": 2, "date": "2026-01-10"}
        
        within, msg = check_daily_limits()
        assert within == False
        assert "loss limit" in msg.lower()
    
    @patch('utils.load_daily_pnl')
    def test_trade_count_exceeded(self, mock_load):
        """Should block trading when max trades exceeded"""
        mock_load.return_value = {"pnl": 1000, "trades": MAX_TRADES_PER_DAY + 1, "date": "2026-01-10"}
        
        within, msg = check_daily_limits()
        assert within == False
        assert "Max trades" in msg


class TestCooldown:
    """Tests for loss cooldown"""
    
    def test_no_cooldown_without_loss(self):
        """No cooldown when no previous loss"""
        ok, msg = check_cooldown(None)
        assert ok == True
        assert "No cooldown" in msg
    
    def test_cooldown_active(self):
        """Cooldown should be active after recent loss"""
        recent_loss_time = datetime.now() - timedelta(seconds=60)
        ok, msg = check_cooldown(recent_loss_time)
        assert ok == False
        assert "remaining" in msg
    
    def test_cooldown_expired(self):
        """Cooldown should expire after COOLDOWN_AFTER_LOSS seconds"""
        old_loss_time = datetime.now() - timedelta(seconds=COOLDOWN_AFTER_LOSS + 60)
        ok, msg = check_cooldown(old_loss_time)
        assert ok == True
        assert "complete" in msg.lower()


class TestSignalCooldown:
    """Tests for signal/whipsaw cooldown"""
    
    def test_no_cooldown_first_signal(self):
        """No cooldown for first signal"""
        ok, msg = check_signal_cooldown("BUY", None, None)
        assert ok == True
    
    def test_no_cooldown_different_signal(self):
        """No cooldown for opposite direction signal"""
        recent_time = datetime.now() - timedelta(seconds=60)
        ok, msg = check_signal_cooldown("SELL", "BUY", recent_time)
        assert ok == True
    
    def test_cooldown_same_signal(self):
        """Cooldown should apply for same direction signal"""
        recent_time = datetime.now() - timedelta(seconds=60)
        ok, msg = check_signal_cooldown("BUY", "BUY", recent_time)
        assert ok == False
        assert "cooldown" in msg.lower()
    
    def test_cooldown_expired_same_signal(self):
        """Cooldown should expire for same direction after SIGNAL_COOLDOWN"""
        old_time = datetime.now() - timedelta(seconds=SIGNAL_COOLDOWN + 60)
        ok, msg = check_signal_cooldown("BUY", "BUY", old_time)
        assert ok == True


# =============================================================================
# PERFORMANCE STATS TESTS
# =============================================================================

class TestPerformanceStats:
    """Tests for performance statistics calculation"""
    
    @patch('utils.load_trade_history')
    def test_empty_history(self, mock_load):
        """Should return None for empty history"""
        mock_load.return_value = []
        stats = get_performance_stats()
        assert stats is None
    
    @patch('utils.load_trade_history')
    def test_win_rate_calculation(self, mock_load):
        """Should correctly calculate win rate"""
        mock_load.return_value = [
            {"pnl": 100, "exit_time": "2026-01-10 10:00:00", "r_multiple": 2.0},
            {"pnl": 150, "exit_time": "2026-01-10 11:00:00", "r_multiple": 3.0},
            {"pnl": -50, "exit_time": "2026-01-10 12:00:00", "r_multiple": -1.0},
            {"pnl": 200, "exit_time": "2026-01-10 13:00:00", "r_multiple": 4.0},
        ]
        
        stats = get_performance_stats()
        assert stats is not None
        assert stats['total_trades'] == 4
        assert stats['wins'] == 3
        assert stats['losses'] == 1
        assert stats['win_rate'] == 75.0
    
    @patch('utils.load_trade_history')
    def test_pnl_calculation(self, mock_load):
        """Should correctly calculate P&L metrics"""
        mock_load.return_value = [
            {"pnl": 100, "exit_time": "2026-01-10 10:00:00"},
            {"pnl": -50, "exit_time": "2026-01-10 11:00:00"},
        ]
        
        stats = get_performance_stats()
        assert stats['total_pnl'] == 50
        assert stats['gross_profit'] == 100
        assert stats['gross_loss'] == 50
        assert stats['profit_factor'] == 2.0
    
    @patch('utils.load_trade_history')
    def test_consecutive_tracking(self, mock_load):
        """Should track consecutive wins/losses"""
        mock_load.return_value = [
            {"pnl": 100, "exit_time": "2026-01-10 10:00:00"},
            {"pnl": 100, "exit_time": "2026-01-10 11:00:00"},
            {"pnl": 100, "exit_time": "2026-01-10 12:00:00"},
            {"pnl": -50, "exit_time": "2026-01-10 13:00:00"},
            {"pnl": -50, "exit_time": "2026-01-10 14:00:00"},
        ]
        
        stats = get_performance_stats()
        assert stats['max_consecutive_wins'] == 3
        assert stats['max_consecutive_losses'] == 2


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================

class TestStateManagement:
    """Tests for state save/load"""
    
    def test_load_default_state(self, tmp_path, monkeypatch):
        """Should return default state when no file exists"""
        monkeypatch.setattr('utils.STATE_FILE', str(tmp_path / "nonexistent.json"))
        monkeypatch.setattr('utils.USE_DATABASE', False)
        
        state = load_state()
        assert state['status'] == False
        assert state['type'] is None
        assert state['instrument'] is None
    
    def test_save_and_load_state(self, tmp_path, monkeypatch):
        """Should correctly save and load state"""
        state_file = str(tmp_path / "test_state.json")
        monkeypatch.setattr('utils.STATE_FILE', state_file)
        
        test_state = {
            "status": True,
            "type": "BUY",
            "instrument": "CRUDEOIL",
            "entry": 6000,
            "sl": 5980
        }
        
        save_state(test_state)
        loaded = load_state()
        
        assert loaded['status'] == True
        assert loaded['type'] == "BUY"
        assert loaded['instrument'] == "CRUDEOIL"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
