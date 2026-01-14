# =============================================================================
# BACKTEST ENGINE - Integrated Backtester for Trading Bot
# =============================================================================
"""
BacktestEngine that mocks the Dhan client and feeds historical data
into the scanner.py logic. This allows you to verify strategy performance
before deploying changes to live trading.

Usage:
    from backtest_engine import BacktestEngine
    
    engine = BacktestEngine(
        instrument="CRUDEOIL",
        start_date="2025-01-01",
        end_date="2025-12-31"
    )
    results = engine.run()
    engine.print_summary()
"""

import json
import logging
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# =============================================================================
# DATA CLASSES FOR BACKTEST RESULTS
# =============================================================================

@dataclass
class Trade:
    """Represents a single backtest trade."""
    instrument: str
    signal: str  # "BUY" or "SELL"
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    r_multiple: float = 0.0
    initial_sl: float = 0.0
    final_sl: float = 0.0
    lot_size: int = 1
    
    @property
    def is_win(self) -> bool:
        return self.pnl > 0
    
    @property
    def duration_minutes(self) -> int:
        if self.exit_time and self.entry_time:
            return int((self.exit_time - self.entry_time).total_seconds() / 60)
        return 0


@dataclass
class BacktestResults:
    """Container for backtest results and metrics."""
    instrument: str
    strategy: str
    start_date: datetime
    end_date: datetime
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.is_win)
    
    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if not t.is_win)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)
    
    @property
    def gross_profit(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl > 0)
    
    @property
    def gross_loss(self) -> float:
        return sum(t.pnl for t in self.trades if t.pnl < 0)
    
    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return abs(self.gross_profit / self.gross_loss)
    
    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.is_win]
        return sum(wins) / len(wins) if wins else 0.0
    
    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if not t.is_win]
        return sum(losses) / len(losses) if losses else 0.0
    
    @property
    def avg_r_multiple(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.r_multiple for t in self.trades) / len(self.trades)
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not self.trades:
            return 0.0
        
        equity = [0.0]
        for trade in self.trades:
            equity.append(equity[-1] + trade.pnl)
        
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate simplified Sharpe ratio (assuming 0% risk-free rate)."""
        if len(self.trades) < 2:
            return 0.0
        
        returns = [t.pnl for t in self.trades]
        avg_return = sum(returns) / len(returns)
        
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0.0
        
        return avg_return / std_dev
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "instrument": self.instrument,
            "strategy": self.strategy,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 2),
            "total_pnl": round(self.total_pnl, 2),
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "avg_r_multiple": round(self.avg_r_multiple, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


# =============================================================================
# MOCK DHAN CLIENT FOR BACKTESTING
# =============================================================================

class MockDhanClient:
    """
    Mock Dhan client that returns historical data for backtesting.
    Implements the same interface as dhanhq for seamless integration.
    """
    
    # Constants to match dhanhq
    BUY = "BUY"
    SELL = "SELL"
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    INTRADAY = "INTRADAY"
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize mock client with optional historical data.
        
        Args:
            historical_data: DataFrame with OHLCV data
        """
        self.historical_data = historical_data
        self._orders: List[Dict[str, Any]] = []
        self._order_id_counter = 10000
        self._current_time: Optional[datetime] = None
    
    def set_historical_data(self, data: pd.DataFrame) -> None:
        """Set historical data for backtesting."""
        self.historical_data = data
    
    def set_current_time(self, time: datetime) -> None:
        """Set the current simulation time."""
        self._current_time = time
    
    def intraday_minute_data(
        self,
        security_id: str,
        exchange_segment: str,
        instrument_type: str,
        from_date: str,
        to_date: str
    ) -> Dict[str, Any]:
        """
        Mock implementation of intraday_minute_data.
        Returns data from historical_data up to current simulation time.
        """
        if self.historical_data is None or self.historical_data.empty:
            return {"status": "failure", "remarks": "No historical data loaded"}
        
        # Filter data up to current time if set
        df = self.historical_data.copy()
        if self._current_time:
            df = df[df.index <= self._current_time]
        
        if df.empty:
            return {"status": "failure", "remarks": "No data for time range"}
        
        # Return in V2 API format
        return {
            "status": "success",
            "data": {
                "open": df['open'].tolist(),
                "high": df['high'].tolist(),
                "low": df['low'].tolist(),
                "close": df['close'].tolist(),
                "volume": df['volume'].tolist(),
                "timestamp": [int(t.timestamp()) for t in df.index]
            }
        }
    
    def place_order(
        self,
        security_id: str,
        exchange_segment: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        product_type: str,
        price: float = 0.0
    ) -> Dict[str, Any]:
        """
        Mock implementation of place_order.
        Immediately fills at the specified price.
        """
        self._order_id_counter += 1
        order_id = str(self._order_id_counter)
        
        order = {
            "orderId": order_id,
            "security_id": security_id,
            "exchange_segment": exchange_segment,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "price": price,
            "status": "TRADED",
            "tradedPrice": price,
            "averageTradedPrice": price,
            "timestamp": self._current_time or datetime.now()
        }
        self._orders.append(order)
        
        return {
            "status": "success",
            "data": {"orderId": order_id}
        }
    
    def get_order_by_id(self, order_id: str) -> Dict[str, Any]:
        """Mock implementation of get_order_by_id."""
        for order in self._orders:
            if order["orderId"] == order_id:
                return {
                    "status": "success",
                    "data": {
                        "orderId": order_id,
                        "orderStatus": "TRADED",
                        "averageTradedPrice": order["price"],
                        "tradedPrice": order["price"]
                    }
                }
        return {"status": "failure", "remarks": "Order not found"}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Mock implementation of cancel_order."""
        return {"status": "success"}
    
    def get_fund_limits(self) -> Dict[str, Any]:
        """Mock implementation - always return sufficient funds."""
        return {
            "status": "success",
            "data": {
                "availabelBalance": 1000000,
                "utilizedAmount": 0
            }
        }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    
    The engine simulates the scanner.py trading logic by:
    1. Loading historical OHLCV data
    2. Stepping through time and calling the strategy
    3. Simulating order execution and trade management
    4. Collecting performance metrics
    """
    
    def __init__(
        self,
        instrument: str,
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        initial_capital: float = 100000,
        data_file: Optional[str] = None
    ):
        """
        Initialize the backtest engine.
        
        Args:
            instrument: Instrument to backtest (e.g., "CRUDEOIL")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format
            strategy_name: Strategy to use (default: instrument's default)
            strategy_params: Optional strategy parameter overrides
            initial_capital: Starting capital
            data_file: Optional path to CSV file with historical data
        """
        self.instrument = instrument
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.initial_capital = initial_capital
        self.data_file = data_file
        
        # Initialize components
        self.mock_dhan = MockDhanClient()
        self.historical_data: Optional[pd.DataFrame] = None
        self.results: Optional[BacktestResults] = None
        
        # Trading state
        self.active_trade: Optional[Dict[str, Any]] = None
        self.trades: List[Trade] = []
        
        # Load instrument config
        from instruments import INSTRUMENTS
        self.inst_config = INSTRUMENTS.get(instrument, {})
        self.lot_size = self.inst_config.get("lot_size", 1)
    
    def load_data(self, data: Optional[pd.DataFrame] = None) -> bool:
        """
        Load historical data for backtesting.
        
        Args:
            data: Optional DataFrame to use directly
            
        Returns:
            True if data loaded successfully
        """
        if data is not None:
            self.historical_data = data
        elif self.data_file and Path(self.data_file).exists():
            # Load from CSV file
            self.historical_data = pd.read_csv(
                self.data_file,
                parse_dates=['time'],
                index_col='time'
            )
        else:
            # Try to load from Dhan API (for demo, create synthetic data)
            logging.warning("No historical data provided. Using synthetic data for demo.")
            self.historical_data = self._generate_synthetic_data()
        
        if self.historical_data is not None:
            # Ensure proper datetime index
            if not isinstance(self.historical_data.index, pd.DatetimeIndex):
                self.historical_data.index = pd.to_datetime(self.historical_data.index)
            
            # Filter to date range
            self.historical_data = self.historical_data[
                (self.historical_data.index >= self.start_date) &
                (self.historical_data.index <= self.end_date)
            ]
            
            self.mock_dhan.set_historical_data(self.historical_data)
            return True
        
        return False
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        import numpy as np
        
        # Generate minute data
        periods = int((self.end_date - self.start_date).total_seconds() / 60)
        periods = min(periods, 50000)  # Limit for performance
        
        dates = pd.date_range(start=self.start_date, periods=periods, freq='1min')
        
        # Random walk with trend
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.002, periods)
        
        base_price = 6000  # Base price for crude oil
        prices = base_price * np.cumprod(1 + returns)
        
        # Create OHLCV
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(base_price)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.002, periods))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.002, periods))
        df['volume'] = np.random.randint(1000, 10000, periods)
        
        return df
    
    def run(self) -> BacktestResults:
        """
        Run the backtest simulation.
        
        Returns:
            BacktestResults object with performance metrics
        """
        if self.historical_data is None or self.historical_data.empty:
            if not self.load_data():
                raise ValueError("No historical data available for backtesting")
        
        # Get strategy
        from strategies import get_strategy
        strategy = get_strategy(
            self.instrument,
            self.strategy_name,
            self.strategy_params
        )
        
        logging.info(f"🔄 Starting backtest: {self.instrument} with {strategy.name}")
        logging.info(f"   Period: {self.start_date.date()} to {self.end_date.date()}")
        
        # Resample to 15min and 60min for strategy
        df_15 = self.historical_data.resample('15min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        df_60 = self.historical_data.resample('60min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # Step through each 15min candle
        for i in range(50, len(df_15)):  # Start after warmup period
            current_time = df_15.index[i]
            self.mock_dhan.set_current_time(current_time)
            
            # Get data up to current time
            df_15_slice = df_15.iloc[:i+1].copy()
            df_60_slice = df_60[df_60.index <= current_time].copy()
            
            if len(df_60_slice) < 50:
                continue
            
            current_price = df_15_slice.iloc[-1]['close']
            
            # Check if we have an active trade
            if self.active_trade:
                self._manage_trade(current_price, current_time, df_15_slice)
            else:
                # Look for new signal
                signal_info = strategy.analyze(df_15_slice, df_60_slice)
                if signal_info:
                    self._enter_trade(signal_info, current_time)
        
        # Close any remaining trade
        if self.active_trade:
            final_price = df_15.iloc[-1]['close']
            self._exit_trade(final_price, df_15.index[-1], "END_OF_BACKTEST")
        
        # Compile results
        self.results = BacktestResults(
            instrument=self.instrument,
            strategy=strategy.name,
            start_date=self.start_date,
            end_date=self.end_date,
            trades=self.trades
        )
        
        return self.results
    
    def _enter_trade(self, signal_info: Dict[str, Any], entry_time: datetime) -> None:
        """Enter a new trade based on signal."""
        price = signal_info['price']
        signal = signal_info['signal']
        
        # Calculate dynamic SL (simplified)
        risk_pct = 0.01  # 1% risk
        if signal == "BUY":
            initial_sl = price * (1 - risk_pct)
        else:
            initial_sl = price * (1 + risk_pct)
        
        self.active_trade = {
            "instrument": self.instrument,
            "signal": signal,
            "entry_time": entry_time,
            "entry_price": price,
            "initial_sl": initial_sl,
            "current_sl": initial_sl,
            "step_level": 0,
        }
        
        logging.debug(f"📈 ENTRY: {signal} @ {price:.2f} | SL: {initial_sl:.2f} | Time: {entry_time}")
    
    def _manage_trade(
        self,
        current_price: float,
        current_time: datetime,
        df_15: pd.DataFrame
    ) -> None:
        """Manage active trade - check SL, trailing, targets."""
        if not self.active_trade:
            return
        
        trade = self.active_trade
        signal = trade["signal"]
        entry_price = trade["entry_price"]
        initial_sl = trade["initial_sl"]
        current_sl = trade["current_sl"]
        
        # Check SL hit
        sl_hit = False
        if signal == "BUY" and current_price <= current_sl:
            sl_hit = True
        elif signal == "SELL" and current_price >= current_sl:
            sl_hit = True
        
        if sl_hit:
            self._exit_trade(current_price, current_time, "SL_HIT")
            return
        
        # Calculate R-multiple
        risk_unit = abs(entry_price - initial_sl)
        if risk_unit == 0:
            risk_unit = entry_price * 0.01
        
        if signal == "BUY":
            profit_points = current_price - entry_price
        else:
            profit_points = entry_price - current_price
        
        r_multiple = profit_points / risk_unit
        
        # Target exit (5R)
        if r_multiple >= 5.0:
            self._exit_trade(current_price, current_time, "TARGET_5R")
            return
        
        # Trailing stop logic
        new_sl = current_sl
        step_level = trade["step_level"]
        
        if r_multiple >= 4.0 and step_level < 4:
            lock_r = 3.0
            trade["step_level"] = 4
        elif r_multiple >= 3.0 and step_level < 3:
            lock_r = 2.0
            trade["step_level"] = 3
        elif r_multiple >= 2.0 and step_level < 2:
            lock_r = 1.0
            trade["step_level"] = 2
        else:
            lock_r = 0
        
        if lock_r > 0:
            if signal == "BUY":
                new_sl = entry_price + (lock_r * risk_unit)
            else:
                new_sl = entry_price - (lock_r * risk_unit)
            
            if (signal == "BUY" and new_sl > current_sl) or \
               (signal == "SELL" and new_sl < current_sl):
                trade["current_sl"] = new_sl
                logging.debug(f"   🔒 Trailing SL to {new_sl:.2f} (Lock {lock_r}R)")
    
    def _exit_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> None:
        """Exit the current trade and record results."""
        if not self.active_trade:
            return
        
        trade = self.active_trade
        signal = trade["signal"]
        entry_price = trade["entry_price"]
        initial_sl = trade["initial_sl"]
        
        # Calculate P&L
        if signal == "BUY":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price
        
        pnl = pnl_points * self.lot_size
        
        # Calculate R-multiple
        risk_unit = abs(entry_price - initial_sl)
        if risk_unit > 0:
            r_multiple = pnl_points / risk_unit
        else:
            r_multiple = 0
        
        # Record trade
        completed_trade = Trade(
            instrument=self.instrument,
            signal=signal,
            entry_time=trade["entry_time"],
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=pnl,
            r_multiple=r_multiple,
            initial_sl=initial_sl,
            final_sl=trade["current_sl"],
            lot_size=self.lot_size
        )
        self.trades.append(completed_trade)
        
        result_emoji = "✅" if pnl > 0 else "❌"
        logging.debug(f"{result_emoji} EXIT: {exit_reason} @ {exit_price:.2f} | "
                     f"P&L: ₹{pnl:.2f} ({r_multiple:.1f}R)")
        
        self.active_trade = None
    
    def print_summary(self) -> None:
        """Print a summary of backtest results."""
        if not self.results:
            print("No results to display. Run the backtest first.")
            return
        
        r = self.results
        
        print("\n" + "=" * 60)
        print(f"📊 BACKTEST RESULTS: {r.instrument} ({r.strategy})")
        print("=" * 60)
        print(f"Period: {r.start_date.date()} to {r.end_date.date()}")
        print("-" * 60)
        print(f"Total Trades:     {r.total_trades}")
        print(f"Winning Trades:   {r.winning_trades}")
        print(f"Losing Trades:    {r.losing_trades}")
        print(f"Win Rate:         {r.win_rate:.1f}%")
        print("-" * 60)
        print(f"Total P&L:        ₹{r.total_pnl:,.2f}")
        print(f"Gross Profit:     ₹{r.gross_profit:,.2f}")
        print(f"Gross Loss:       ₹{r.gross_loss:,.2f}")
        print(f"Profit Factor:    {r.profit_factor:.2f}")
        print("-" * 60)
        print(f"Avg Win:          ₹{r.avg_win:,.2f}")
        print(f"Avg Loss:         ₹{r.avg_loss:,.2f}")
        print(f"Avg R-Multiple:   {r.avg_r_multiple:.2f}R")
        print("-" * 60)
        print(f"Max Drawdown:     ₹{r.max_drawdown:,.2f}")
        print(f"Sharpe Ratio:     {r.sharpe_ratio:.2f}")
        print("=" * 60 + "\n")
    
    def save_results(self, filepath: str) -> None:
        """Save backtest results to JSON file."""
        if not self.results:
            return
        
        output = {
            "summary": self.results.to_dict(),
            "trades": [
                {
                    "instrument": t.instrument,
                    "signal": t.signal,
                    "entry_time": t.entry_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl": round(t.pnl, 2),
                    "r_multiple": round(t.r_multiple, 2),
                }
                for t in self.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filepath}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for running backtests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trading Bot Backtest Engine")
    parser.add_argument("--instrument", "-i", default="CRUDEOIL",
                       help="Instrument to backtest (default: CRUDEOIL)")
    parser.add_argument("--start", "-s", default="2025-01-01",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", default="2025-12-31",
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", "-st", default=None,
                       help="Strategy name (TrendFollowing, MeanReversion, MomentumBreakout)")
    parser.add_argument("--data", "-d", default=None,
                       help="Path to CSV data file")
    parser.add_argument("--output", "-o", default=None,
                       help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(message)s')
    
    # Run backtest
    engine = BacktestEngine(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        strategy_name=args.strategy,
        data_file=args.data
    )
    
    results = engine.run()
    engine.print_summary()
    
    if args.output:
        engine.save_results(args.output)


if __name__ == "__main__":
    main()
