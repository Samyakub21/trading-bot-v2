# =============================================================================
# STRATEGIES - Modular Strategy Interface for Trading Bot
# =============================================================================
"""
This module provides a Strategy Pattern implementation allowing different
trading strategies per instrument. Each strategy can have its own:
- Signal generation logic (RSI, EMA, VWAP combinations)
- Parameter thresholds (RSI levels, volume multipliers)
- Entry/Exit rules

Usage:
    from strategies import get_strategy
    strategy = get_strategy("CRUDEOIL")
    signal = strategy.analyze(df_15, df_60)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pandas_ta as ta
import logging


# =============================================================================
# ABSTRACT BASE STRATEGY
# =============================================================================


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    All strategies must implement the analyze() method which takes OHLCV data
    and returns a signal dictionary or None.
    """

    def __init__(self, instrument: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize strategy with instrument and optional parameters.

        Args:
            instrument: The instrument key (e.g., "CRUDEOIL", "NIFTY")
            params: Optional dictionary of strategy parameters
        """
        self.instrument = instrument
        self.params = params or {}
        self._set_default_params()

    @abstractmethod
    def _set_default_params(self) -> None:
        """Set default parameters for the strategy. Override in subclasses."""
        pass

    @abstractmethod
    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze price data and generate trading signal.

        Args:
            df_15: 15-minute OHLCV DataFrame
            df_60: 60-minute OHLCV DataFrame

        Returns:
            Signal dictionary with keys: signal, price, rsi, etc.
            Returns None if no signal generated.
        """
        pass

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with fallback to default."""
        return self.params.get(key, default)

    def update_params(self, new_params: Dict[str, Any]) -> None:
        """Update strategy parameters dynamically."""
        self.params.update(new_params)

    @property
    def name(self) -> str:
        """Return strategy name for logging."""
        return self.__class__.__name__


# =============================================================================
# TREND FOLLOWING STRATEGY (Default - RSI + EMA + VWAP)
# =============================================================================


class TrendFollowingStrategy(Strategy):
    """
    Trend Following Strategy using RSI, EMA, and VWAP.

    Best suited for: Commodities (CRUDEOIL, GOLD, SILVER, NATURALGAS)

    Entry Rules:
    - BUY: Price > EMA50 (60min), Price > VWAP (15min), RSI > bullish_threshold
    - SELL: Price < EMA50 (60min), Price < VWAP (15min), RSI < bearish_threshold
    """

    def _set_default_params(self) -> None:
        """Set default parameters for trend following."""
        defaults = {
            "rsi_bullish_threshold": 60,
            "rsi_bearish_threshold": 40,
            "rsi_length": 14,
            "ema_length": 50,
            "volume_multiplier": 1.2,
            "volume_window": 20,
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Analyze using trend following logic."""
        try:
            # Get parameters
            rsi_bullish = self.get_param("rsi_bullish_threshold", 60)
            rsi_bearish = self.get_param("rsi_bearish_threshold", 40)
            rsi_length = self.get_param("rsi_length", 14)
            ema_length = self.get_param("ema_length", 50)
            volume_mult = self.get_param("volume_multiplier", 1.2)
            vol_window = self.get_param("volume_window", 20)

            # Calculate indicators
            df_60["EMA"] = ta.ema(df_60["close"], length=ema_length)
            df_15.ta.vwap(append=True)
            df_15["RSI"] = ta.rsi(df_15["close"], length=rsi_length)
            df_15["vol_avg"] = df_15["volume"].rolling(window=vol_window).mean()

            trend = df_60.iloc[-2]
            trigger = df_15.iloc[-2]

            price = trigger["close"]
            vwap_val = trigger.get("VWAP_D", 0)
            current_volume = trigger["volume"]
            avg_volume = trigger.get("vol_avg", current_volume)
            rsi_val = trigger["RSI"]
            ema_val = trend["EMA"]
            trend_close = trend["close"]

            # Volume confirmation
            volume_confirmed = (
                current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True
            )

            signal = None
            signal_strength = 0

            # BULLISH Signal
            if (
                (trend_close > ema_val)
                and (trigger["close"] > vwap_val)
                and (rsi_val > rsi_bullish)
                and volume_confirmed
            ):
                signal = "BUY"
                signal_strength = (rsi_val - rsi_bullish) + (
                    (trend_close - ema_val) / ema_val * 100
                )
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 10

            # BEARISH Signal
            elif (
                (trend_close < ema_val)
                and (trigger["close"] < vwap_val)
                and (rsi_val < rsi_bearish)
                and volume_confirmed
            ):
                signal = "SELL"
                signal_strength = (rsi_bearish - rsi_val) + (
                    (ema_val - trend_close) / ema_val * 100
                )
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 10

            if signal:
                return {
                    "instrument": self.instrument,
                    "signal": signal,
                    "price": price,
                    "rsi": rsi_val,
                    "volume": current_volume,
                    "avg_volume": avg_volume,
                    "vwap": vwap_val,
                    "ema": ema_val,
                    "signal_strength": signal_strength,
                    "strategy": self.name,
                    "df_15": df_15,
                }

            return None

        except Exception as e:
            logging.error(f"[{self.name}] Analysis error for {self.instrument}: {e}")
            return None


# =============================================================================
# MEAN REVERSION STRATEGY
# =============================================================================


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy using RSI oversold/overbought with Bollinger Bands.

    Best suited for: Index options (NIFTY, BANKNIFTY) in ranging markets

    Entry Rules:
    - BUY: RSI < oversold_level AND price near lower Bollinger Band
    - SELL: RSI > overbought_level AND price near upper Bollinger Band
    """

    def _set_default_params(self) -> None:
        """Set default parameters for mean reversion."""
        defaults = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rsi_length": 14,
            "bb_length": 20,
            "bb_std": 2.0,
            "volume_multiplier": 1.0,  # Less strict volume requirement
            "band_threshold": 0.02,  # 2% from band for entry
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Analyze using mean reversion logic."""
        try:
            # Get parameters
            rsi_oversold = self.get_param("rsi_oversold", 30)
            rsi_overbought = self.get_param("rsi_overbought", 70)
            rsi_length = self.get_param("rsi_length", 14)
            bb_length = self.get_param("bb_length", 20)
            bb_std = self.get_param("bb_std", 2.0)
            band_threshold = self.get_param("band_threshold", 0.02)
            volume_mult = self.get_param("volume_multiplier", 1.0)

            # Calculate indicators on 15min timeframe
            df_15["RSI"] = ta.rsi(df_15["close"], length=rsi_length)

            # Bollinger Bands
            bbands = ta.bbands(df_15["close"], length=bb_length, std=bb_std)
            if bbands is not None:
                df_15["BB_upper"] = bbands[f"BBU_{bb_length}_{bb_std}"]
                df_15["BB_lower"] = bbands[f"BBL_{bb_length}_{bb_std}"]
                df_15["BB_mid"] = bbands[f"BBM_{bb_length}_{bb_std}"]

            df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

            trigger = df_15.iloc[-2]

            price = trigger["close"]
            rsi_val = trigger["RSI"]
            bb_upper = trigger.get("BB_upper", price * 1.02)
            bb_lower = trigger.get("BB_lower", price * 0.98)
            bb_mid = trigger.get("BB_mid", price)
            current_volume = trigger["volume"]
            avg_volume = trigger.get("vol_avg", current_volume)

            # Volume check (less strict for mean reversion)
            volume_ok = (
                current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True
            )

            signal = None
            signal_strength = 0

            # Calculate distance from bands as percentage
            lower_distance = (price - bb_lower) / bb_lower if bb_lower > 0 else 0
            upper_distance = (bb_upper - price) / bb_upper if bb_upper > 0 else 0

            # BULLISH Signal (Oversold near lower band)
            if rsi_val < rsi_oversold and lower_distance < band_threshold and volume_ok:
                signal = "BUY"
                signal_strength = (rsi_oversold - rsi_val) + (
                    band_threshold - lower_distance
                ) * 100

            # BEARISH Signal (Overbought near upper band)
            elif (
                rsi_val > rsi_overbought
                and upper_distance < band_threshold
                and volume_ok
            ):
                signal = "SELL"
                signal_strength = (rsi_val - rsi_overbought) + (
                    band_threshold - upper_distance
                ) * 100

            if signal:
                return {
                    "instrument": self.instrument,
                    "signal": signal,
                    "price": price,
                    "rsi": rsi_val,
                    "volume": current_volume,
                    "avg_volume": avg_volume,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_mid": bb_mid,
                    "signal_strength": signal_strength,
                    "strategy": self.name,
                    "df_15": df_15,
                }

            return None

        except Exception as e:
            logging.error(f"[{self.name}] Analysis error for {self.instrument}: {e}")
            return None


# =============================================================================
# MOMENTUM BREAKOUT STRATEGY
# =============================================================================


class MomentumBreakoutStrategy(Strategy):
    """
    Momentum Breakout Strategy using price breakouts with volume confirmation.

    Best suited for: Volatile commodities (NATURALGAS, SILVER)

    Entry Rules:
    - BUY: Price breaks above recent high with strong volume and RSI momentum
    - SELL: Price breaks below recent low with strong volume and RSI momentum
    """

    def _set_default_params(self) -> None:
        """Set default parameters for momentum breakout."""
        defaults = {
            "lookback_period": 20,
            "breakout_threshold": 0.005,  # 0.5% above/below range
            "rsi_min_bullish": 55,  # RSI should show momentum
            "rsi_max_bearish": 45,
            "rsi_length": 14,
            "volume_multiplier": 1.5,  # Strong volume required
            "atr_multiplier": 1.0,
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Analyze using momentum breakout logic."""
        try:
            # Get parameters
            lookback = self.get_param("lookback_period", 20)
            breakout_pct = self.get_param("breakout_threshold", 0.005)
            rsi_min_bull = self.get_param("rsi_min_bullish", 55)
            rsi_max_bear = self.get_param("rsi_max_bearish", 45)
            rsi_length = self.get_param("rsi_length", 14)
            volume_mult = self.get_param("volume_multiplier", 1.5)

            # Calculate indicators
            df_15["RSI"] = ta.rsi(df_15["close"], length=rsi_length)
            df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

            # Calculate recent high/low
            df_15["recent_high"] = df_15["high"].rolling(window=lookback).max()
            df_15["recent_low"] = df_15["low"].rolling(window=lookback).min()

            trigger = df_15.iloc[-2]
            prev = df_15.iloc[-3]

            price = trigger["close"]
            rsi_val = trigger["RSI"]
            recent_high = prev["recent_high"]  # Use previous candle's level
            recent_low = prev["recent_low"]
            current_volume = trigger["volume"]
            avg_volume = trigger.get("vol_avg", current_volume)

            # Volume confirmation (strong volume required)
            volume_confirmed = (
                current_volume >= (avg_volume * volume_mult)
                if avg_volume > 0
                else False
            )

            signal = None
            signal_strength = 0

            # Calculate breakout levels
            upper_breakout = recent_high * (1 + breakout_pct)
            lower_breakout = recent_low * (1 - breakout_pct)

            # BULLISH Breakout
            if price > upper_breakout and rsi_val > rsi_min_bull and volume_confirmed:
                signal = "BUY"
                breakout_strength = (price - recent_high) / recent_high * 100
                signal_strength = breakout_strength + (rsi_val - rsi_min_bull)
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 20

            # BEARISH Breakout
            elif price < lower_breakout and rsi_val < rsi_max_bear and volume_confirmed:
                signal = "SELL"
                breakout_strength = (recent_low - price) / recent_low * 100
                signal_strength = breakout_strength + (rsi_max_bear - rsi_val)
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 20

            if signal:
                return {
                    "instrument": self.instrument,
                    "signal": signal,
                    "price": price,
                    "rsi": rsi_val,
                    "volume": current_volume,
                    "avg_volume": avg_volume,
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "signal_strength": signal_strength,
                    "strategy": self.name,
                    "df_15": df_15,
                }

            return None

        except Exception as e:
            logging.error(f"[{self.name}] Analysis error for {self.instrument}: {e}")
            return None


# =============================================================================
# STRATEGY REGISTRY & FACTORY
# =============================================================================

# Default strategy assignments per instrument
DEFAULT_STRATEGY_MAP = {
    "CRUDEOIL": "TrendFollowing",
    "NATURALGAS": "MomentumBreakout",
    "GOLD": "TrendFollowing",
    "SILVER": "MomentumBreakout",
    "NIFTY": "TrendFollowing",  # Can switch to MeanReversion for ranging markets
    "BANKNIFTY": "TrendFollowing",
}

# Strategy class registry
STRATEGY_CLASSES = {
    "TrendFollowing": TrendFollowingStrategy,
    "MeanReversion": MeanReversionStrategy,
    "MomentumBreakout": MomentumBreakoutStrategy,
}


def get_strategy(
    instrument: str,
    strategy_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Strategy:
    """
    Factory function to get a strategy instance for an instrument.

    Args:
        instrument: The instrument key (e.g., "CRUDEOIL")
        strategy_name: Optional strategy name override
        params: Optional parameters to pass to strategy

    Returns:
        Strategy instance configured for the instrument
    """
    # Use provided strategy name or look up default
    if strategy_name is None:
        strategy_name = DEFAULT_STRATEGY_MAP.get(instrument, "TrendFollowing")

    # Get strategy class
    strategy_class = STRATEGY_CLASSES.get(strategy_name, TrendFollowingStrategy)

    # Merge instrument-specific params from INSTRUMENTS config if available
    from instruments import INSTRUMENTS

    inst_config = INSTRUMENTS.get(instrument, {})

    # Get instrument-specific strategy params
    inst_params = inst_config.get("strategy_params", {})

    # Merge with provided params (provided params take precedence)
    merged_params = {**inst_params, **(params or {})}

    return strategy_class(instrument, merged_params)


def get_available_strategies() -> List[str]:
    """Return list of available strategy names."""
    return list(STRATEGY_CLASSES.keys())


def register_strategy(name: str, strategy_class: type) -> None:
    """Register a new strategy class."""
    if not issubclass(strategy_class, Strategy):
        raise ValueError(f"{strategy_class} must be a subclass of Strategy")
    STRATEGY_CLASSES[name] = strategy_class
