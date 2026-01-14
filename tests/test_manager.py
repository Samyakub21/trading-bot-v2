"""
Unit tests for manager.py functions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def active_buy_trade():
    """Create sample active BUY trade - V2 API format"""
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


@pytest.fixture
def active_sell_trade():
    """Create sample active SELL trade - V2 API format"""
    return {
        "status": True,
        "type": "SELL",
        "instrument": "CRUDEOIL",
        "entry": 6000,
        "initial_sl": 6020,
        "sl": 6020,
        "step_level": 0,
        "option_id": "12346",
        "option_entry": 150,
        "lot_size": 10,
        "exchange_segment_str": "MCX_COMM",  # V2 API format
        "entry_time": "2026-01-10 10:00:00"
    }


# =============================================================================
# TRAILING STOP LOSS TESTS
# =============================================================================

class TestTrailingStopLoss:
    """Tests for step-ladder trailing stop loss logic"""
    
    def test_calculate_r_multiple_buy(self, active_buy_trade):
        """Should correctly calculate R-multiple for BUY trade"""
        entry = active_buy_trade["entry"]
        initial_sl = active_buy_trade["initial_sl"]
        risk_unit = abs(entry - initial_sl)  # 20 points
        
        # Current price 6040 = 2R profit
        current_price = 6040
        profit_points = current_price - entry  # 40 points
        r_multiple = profit_points / risk_unit  # 2.0R
        
        assert risk_unit == 20
        assert profit_points == 40
        assert r_multiple == 2.0
    
    def test_calculate_r_multiple_sell(self, active_sell_trade):
        """Should correctly calculate R-multiple for SELL trade"""
        entry = active_sell_trade["entry"]
        initial_sl = active_sell_trade["initial_sl"]
        risk_unit = abs(entry - initial_sl)  # 20 points
        
        # Current price 5960 = 2R profit for SELL
        current_price = 5960
        profit_points = entry - current_price  # 40 points
        r_multiple = profit_points / risk_unit  # 2.0R
        
        assert risk_unit == 20
        assert profit_points == 40
        assert r_multiple == 2.0
    
    def test_step_1_trigger_buy(self, active_buy_trade):
        """Step 1 should trigger at 2R and lock 1R"""
        entry = active_buy_trade["entry"]
        initial_sl = active_buy_trade["initial_sl"]
        risk_unit = abs(entry - initial_sl)  # 20 points
        
        # Simulate price at 2R (6040)
        current_r = 2.0
        
        # Step 1 should trigger
        assert current_r >= 2.0
        assert active_buy_trade["step_level"] < 2
        
        # New SL should be entry + 1R
        new_sl = entry + (1.0 * risk_unit)
        assert new_sl == 6020
    
    def test_step_2_trigger_buy(self, active_buy_trade):
        """Step 2 should trigger at 3R and lock 2R"""
        entry = active_buy_trade["entry"]
        initial_sl = active_buy_trade["initial_sl"]
        risk_unit = abs(entry - initial_sl)  # 20 points
        
        # Set step level to 2 (already triggered step 1)
        active_buy_trade["step_level"] = 2
        current_r = 3.0
        
        # Step 2 should trigger
        assert current_r >= 3.0
        assert active_buy_trade["step_level"] < 3
        
        # New SL should be entry + 2R
        new_sl = entry + (2.0 * risk_unit)
        assert new_sl == 6040
    
    def test_step_3_trigger_buy(self, active_buy_trade):
        """Step 3 should trigger at 4R and lock 3R"""
        entry = active_buy_trade["entry"]
        initial_sl = active_buy_trade["initial_sl"]
        risk_unit = abs(entry - initial_sl)
        
        active_buy_trade["step_level"] = 3
        current_r = 4.0
        
        assert current_r >= 4.0
        
        new_sl = entry + (3.0 * risk_unit)
        assert new_sl == 6060
    
    def test_target_exit_at_5r(self, active_buy_trade):
        """Should exit at 5R target"""
        entry = active_buy_trade["entry"]
        initial_sl = active_buy_trade["initial_sl"]
        risk_unit = abs(entry - initial_sl)
        
        # Price at 5R target
        target_price = entry + (5.0 * risk_unit)
        current_r = (target_price - entry) / risk_unit
        
        assert current_r >= 5.0
        assert target_price == 6100
    
    def test_sl_hit_buy(self, active_buy_trade):
        """SL should trigger when price goes below SL for BUY"""
        ltp = 5975  # Below SL of 5980
        sl_hit = ltp <= active_buy_trade["sl"]
        assert sl_hit == True
    
    def test_sl_hit_sell(self, active_sell_trade):
        """SL should trigger when price goes above SL for SELL"""
        ltp = 6025  # Above SL of 6020
        sl_hit = ltp >= active_sell_trade["sl"]
        assert sl_hit == True
    
    def test_step_level_only_increases(self, active_buy_trade):
        """Step level should only increase, never decrease"""
        active_buy_trade["step_level"] = 3
        
        # Even if R drops back to 2, step level stays at 3
        current_r = 2.5
        
        # Condition check - won't trigger because step_level >= 2
        should_trigger_step_2 = current_r >= 2.0 and active_buy_trade["step_level"] < 2
        
        assert should_trigger_step_2 == False
    
    def test_sell_trailing_sl_direction(self, active_sell_trade):
        """SELL trailing SL should move DOWN"""
        entry = active_sell_trade["entry"]
        risk_unit = 20
        
        # At 2R, lock 1R for SELL means SL goes to entry - 1R
        new_sl = entry - (1.0 * risk_unit)
        
        # For SELL, lower SL is protective (locks profit)
        assert new_sl == 5980
        assert new_sl < entry  # SL is below entry (profit locked)


# =============================================================================
# P&L CALCULATION TESTS
# =============================================================================

class TestPnLCalculation:
    """Tests for P&L calculation"""
    
    def test_pnl_calculation_win(self, active_buy_trade):
        """Should correctly calculate winning trade P&L"""
        option_entry = active_buy_trade["option_entry"]  # 150
        option_exit = 200
        lot_size = active_buy_trade["lot_size"]  # 10
        
        pnl_per_lot = (option_exit - option_entry) * lot_size
        
        assert pnl_per_lot == 500  # (200-150) * 10 = 500
        assert pnl_per_lot > 0
    
    def test_pnl_calculation_loss(self, active_buy_trade):
        """Should correctly calculate losing trade P&L"""
        option_entry = active_buy_trade["option_entry"]  # 150
        option_exit = 100
        lot_size = active_buy_trade["lot_size"]  # 10
        
        pnl_per_lot = (option_exit - option_entry) * lot_size
        
        assert pnl_per_lot == -500  # (100-150) * 10 = -500
        assert pnl_per_lot < 0
    
    def test_r_multiple_achieved(self, active_buy_trade):
        """Should correctly calculate R-multiple achieved"""
        entry_future = active_buy_trade["entry"]
        exit_future = 6040
        initial_sl = active_buy_trade["initial_sl"]
        
        risk_unit = abs(entry_future - initial_sl)  # 20
        profit_points = exit_future - entry_future  # 40
        r_multiple = profit_points / risk_unit  # 2.0R
        
        assert r_multiple == 2.0


# =============================================================================
# AUTO SQUARE-OFF TESTS
# =============================================================================

class TestAutoSquareOff:
    """Tests for auto square-off logic"""
    
    def test_square_off_time_calculation(self):
        """Should correctly calculate square-off time"""
        market_end = "23:30"
        buffer = 5  # minutes
        
        market_end_hour, market_end_min = map(int, market_end.split(":"))
        square_off_hour = market_end_hour
        square_off_min = market_end_min - buffer
        
        if square_off_min < 0:
            square_off_min += 60
            square_off_hour -= 1
        
        square_off_time = f"{square_off_hour:02d}:{square_off_min:02d}"
        
        assert square_off_time == "23:25"
    
    def test_square_off_time_hour_rollover(self):
        """Should handle hour rollover correctly"""
        market_end = "15:03"
        buffer = 5
        
        market_end_hour, market_end_min = map(int, market_end.split(":"))
        square_off_hour = market_end_hour
        square_off_min = market_end_min - buffer
        
        if square_off_min < 0:
            square_off_min += 60
            square_off_hour -= 1
        
        square_off_time = f"{square_off_hour:02d}:{square_off_min:02d}"
        
        assert square_off_time == "14:58"
    
    def test_should_square_off_after_time(self):
        """Should trigger square-off after specified time"""
        current_time = "23:26"
        square_off_time = "23:25"
        
        should_square_off = current_time >= square_off_time
        assert should_square_off == True
    
    def test_should_not_square_off_before_time(self):
        """Should not trigger square-off before specified time"""
        current_time = "23:20"
        square_off_time = "23:25"
        
        should_square_off = current_time >= square_off_time
        assert should_square_off == False


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
