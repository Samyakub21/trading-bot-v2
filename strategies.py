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
from typing import Any, Dict, List, Optional, Tuple, cast
import pandas as pd
from datetime import datetime
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import logging
import numpy as np


def calculate_anchored_vwap(
    df: pd.DataFrame, reset_time: Optional[str] = None
) -> pd.Series:
    """Calculate Intraday VWAP anchored to the start of each day or session."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vp = typical_price * df["volume"]

    if reset_time:
        reset_dt = pd.Timestamp(reset_time).time()
        session = df.index.time >= reset_dt
        session_key = df.index.date.astype(str) + "_" + session.astype(str)
    else:
        session_key = df.index.date

    cumulative_vp = vp.groupby(session_key).cumsum()
    cumulative_vol = df["volume"].groupby(session_key).cumsum()
    return cumulative_vp / cumulative_vol.replace(0, np.nan)


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
        self.logger = logging.getLogger(f"Strategy.{instrument}")
        self._set_default_params()

    @abstractmethod
    def _set_default_params(self) -> None:
        """Set default parameters for the strategy. Override in subclasses."""
        pass

    @abstractmethod
    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
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

    Best suited for: Commodities (CRUDEOIL, NATURALGAS)

    Entry Rules:
    - BUY: Price > EMA50 (60min), Price > VWAP (15min), RSI > bullish_threshold, ADX > 25
    - SELL: Price < EMA50 (60min), Price < VWAP (15min), RSI < bearish_threshold, ADX > 25
    - 200 EMA alignment on 15min chart for long-term trend
    - ATR-based stop loss (2.0x for CRUDEOIL, 1.5x for others)
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
            "adx_threshold": 25,  # Minimum ADX for trend trades
            "atr_multiplier": 1.5,  # ATR multiplier for stop loss
            "atr_length": 14,
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Analyze using trend following logic."""
        try:
            # Universal time filter for NSE indices: No new entries between 11:30 AM and 1:30 PM IST
            if self.instrument in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                current_time = datetime.now().time()
                no_trade_start = datetime.strptime("11:30", "%H:%M").time()
                no_trade_end = datetime.strptime("13:30", "%H:%M").time()
                if no_trade_start <= current_time <= no_trade_end:
                    return None

            # Get parameters
            rsi_bullish = self.get_param("rsi_bullish_threshold", 60)
            rsi_bearish = self.get_param("rsi_bearish_threshold", 40)
            rsi_len = self.get_param("rsi_length", 14)
            ema_len = self.get_param("ema_length", 50)
            volume_mult = self.get_param("volume_multiplier", 1.2)
            vol_window = self.get_param("volume_window", 20)
            adx_threshold = self.get_param("adx_threshold", 25)
            atr_mult = self.get_param("atr_multiplier", 1.5)
            if self.instrument == "CRUDEOIL":
                atr_mult = 2.0  # Higher multiplier for Crude Oil noise
            atr_len = self.get_param("atr_length", 14)

            # Ensure we have enough historical bars to compute indicators
            required_15 = max(rsi_len, 200, atr_len, 14, vol_window)
            # Need at least a few extra bars to access -2 index safely
            if df_15.shape[0] < (required_15 + 3) or df_60.shape[0] < 2:
                self.logger.info(
                    f"SKIP [{self.instrument}]: Insufficient data for indicators (have {df_15.shape[0]}x15m, {df_60.shape[0]}x60m, need {required_15 + 3}x15m and 2x60m)"
                )
                return None

            # Calculate indicators
            # 1. EMA on 60min
            df_60["EMA"] = EMAIndicator(
                close=df_60["close"], window=ema_len
            ).ema_indicator()
            # 2. RSI on 15min
            df_15["RSI"] = RSIIndicator(close=df_15["close"], window=rsi_len).rsi()
            # 3. VWAP (Custom Anchored) - Session-based for MCX
            reset_time = (
                "15:30" if self.instrument in ["CRUDEOIL", "NATGASMINI"] else None
            )
            df_15["VWAP_D"] = calculate_anchored_vwap(df_15, reset_time)
            # 4. ADX on 15min
            df_15["ADX"] = ADXIndicator(
                high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
            ).adx()
            # 5. 200 EMA on 15min for alignment filter
            df_15["EMA200"] = EMAIndicator(
                close=df_15["close"], window=200
            ).ema_indicator()
            # 6. ATR for stop loss
            df_15["ATR"] = AverageTrueRange(
                high=df_15["high"],
                low=df_15["low"],
                close=df_15["close"],
                window=atr_len,
            ).average_true_range()
            df_15["vol_avg"] = df_15["volume"].rolling(window=vol_window).mean()

            trend = df_60.iloc[-2]
            trigger = df_15.iloc[-2]

            # Data robustness: Check for NaN in key indicators or zero volume
            if (
                pd.isna(trigger.get("RSI"))
                or pd.isna(trigger.get("ADX"))
                or pd.isna(trigger.get("VWAP_D"))
                or trigger.get("volume", 0) == 0
            ):
                self.logger.info(
                    f"SKIP [{self.instrument}]: Invalid data in trigger row (NaN or zero volume)"
                )
                return None

            price = trigger["close"]
            vwap_val = trigger.get("VWAP_D", 0)
            current_volume = trigger["volume"]
            avg_volume = trigger.get("vol_avg", current_volume)
            rsi_val = trigger["RSI"]
            ema_val = trend["EMA"]
            trend_close = trend["close"]
            adx_val = trigger["ADX"]
            ema200_val = trigger["EMA200"]
            atr_val = trigger["ATR"]

            # Adaptive ATR: Check for high volatility
            atr_rolling_mean = df_15["ATR"].rolling(window=20).mean().iloc[-1]
            if pd.isna(atr_rolling_mean):
                atr_rolling_mean = atr_val
            volatility_multiplier = 1.25 if atr_val > 1.5 * atr_rolling_mean else 1.0

            # Volume confirmation
            volume_confirmed = (
                current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True
            )

            # ADX filter: Only allow trend trades if ADX > threshold
            adx_confirmed = adx_val > adx_threshold

            # 200 EMA alignment filter for TrendFollowing
            alignment_confirmed = (
                (price > ema200_val) if self.instrument in ["BANKNIFTY"] else True
            )

            signal = None
            signal_strength = 0
            stop_loss = 0

            # BULLISH Signal
            if (
                (trend_close > ema_val)
                and (trigger["close"] > vwap_val)
                and (rsi_val > rsi_bullish)
                and volume_confirmed
                and adx_confirmed
                and alignment_confirmed
            ):
                signal = "BUY"
                signal_strength = (rsi_val - rsi_bullish) + (
                    (trend_close - ema_val) / ema_val * 100
                )
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 10
                # ATR-based stop loss
                stop_loss = price - (atr_val * atr_mult)

            # BEARISH Signal
            elif (
                (trend_close < ema_val)
                and (trigger["close"] < vwap_val)
                and (rsi_val < rsi_bearish)
                and volume_confirmed
                and adx_confirmed
            ):
                signal = "SELL"
                signal_strength = (rsi_bearish - rsi_val) + (
                    (ema_val - trend_close) / ema_val * 100
                )
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 10
                # ATR-based stop loss
                stop_loss = price + (atr_val * atr_mult)

            if signal:
                # Hybrid Dynamic Trailing Stop Loss Logic
                atr_mult = self.get_param("atr_multiplier", 1.5)

                if self.instrument == "CRUDEOIL":
                    atr_mult = 2.0  # Higher multiplier for Crude Oil noise
                atr_len = self.get_param("atr_length", 14)

                # Initial SL: atr_mult * ATR * volatility_multiplier
                initial_sl_distance = atr_val * atr_mult * volatility_multiplier

                # Activation Point: 1.5x ATR * volatility_multiplier favorable move (1:1 R:R)
                activation_distance = atr_val * 1.5 * volatility_multiplier

                # Dynamic Trailing: 1.5x ATR * volatility_multiplier from highest price after activation
                trail_distance = atr_val * 1.5 * volatility_multiplier

                # Maximum profit cap for indices (1:2.5 R:R)
                max_profit_mult = None
                if self.instrument in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                    max_profit_mult = 2.5  # 1:2.5 R:R cap for indices

                exit_logic = {
                    "initial_sl_distance": round(initial_sl_distance, 2),
                    "activation_atr_mult": 1.5
                    * volatility_multiplier,  # 1.5x ATR for activation
                    "trail_atr_mult": 1.5
                    * volatility_multiplier,  # 1.5x ATR for trailing
                    "max_profit_mult": max_profit_mult,  # None for commodities, 2.5 for indices
                    "atr_value": round(atr_val, 2),
                    "atr_multiplier": atr_mult * volatility_multiplier,
                }

                return {
                    "instrument": self.instrument,
                    "signal": signal,
                    "price": price,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "volume": current_volume,
                    "avg_volume": avg_volume,
                    "vwap": vwap_val,
                    "ema": ema_val,
                    "ema200": ema200_val,
                    "signal_strength": signal_strength,
                    "exit_logic": exit_logic,
                    "strategy": self.name,
                    "df_15": df_15,
                }

            # Log rejection reasons
            reasons = []
            if not volume_confirmed:
                reasons.append(
                    f"Volume too low ({current_volume:.0f} < {avg_volume * volume_mult:.0f})"
                )
            if not adx_confirmed:
                reasons.append(f"ADX too low ({adx_val:.1f} < {adx_threshold})")
            if not alignment_confirmed:
                reasons.append(
                    f"Price below 200EMA alignment ({price:.2f} < {ema200_val:.2f})"
                )

            # Check trend and price conditions
            bullish_trend = trend_close > ema_val
            bearish_trend = trend_close < ema_val
            bullish_price = trigger["close"] > vwap_val
            bearish_price = trigger["close"] < vwap_val
            bullish_rsi = rsi_val > rsi_bullish
            bearish_rsi = rsi_val < rsi_bearish

            if not (bullish_trend and bullish_price and bullish_rsi) and not (
                bearish_trend and bearish_price and bearish_rsi
            ):
                if not bullish_trend and not bearish_trend:
                    reasons.append("No clear trend direction")
                elif bullish_trend:
                    if not bullish_price:
                        reasons.append(
                            f"Price not above VWAP ({trigger['close']:.2f} <= {vwap_val:.2f})"
                        )
                    if not bullish_rsi:
                        reasons.append(f"RSI too low ({rsi_val:.1f} <= {rsi_bullish})")
                elif bearish_trend:
                    if not bearish_price:
                        reasons.append(
                            f"Price not below VWAP ({trigger['close']:.2f} >= {vwap_val:.2f})"
                        )
                    if not bearish_rsi:
                        reasons.append(f"RSI too high ({rsi_val:.1f} >= {rsi_bearish})")

            if reasons:
                self.logger.info(f"SKIP [{self.instrument}]: {', '.join(reasons)}")

            return None

        except Exception as e:
            self.logger.exception(f"Analysis error for {self.instrument}")
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
        self, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Analyze using mean reversion logic."""
        try:
            # Get parameters
            rsi_oversold = self.get_param("rsi_oversold", 30)
            rsi_overbought = self.get_param("rsi_overbought", 70)
            rsi_len = self.get_param("rsi_length", 14)
            bb_len = self.get_param("bb_length", 20)
            bb_std = self.get_param("bb_std", 2.0)
            band_threshold = self.get_param("band_threshold", 0.02)
            volume_mult = self.get_param("volume_multiplier", 1.0)

            # RSI
            df_15["RSI"] = RSIIndicator(close=df_15["close"], window=rsi_len).rsi()
            # Bollinger Bands
            bb_indicator = BollingerBands(
                close=df_15["close"], window=bb_len, window_dev=bb_std
            )
            df_15["BB_upper"] = bb_indicator.bollinger_hband()
            df_15["BB_lower"] = bb_indicator.bollinger_lband()
            df_15["BB_mid"] = bb_indicator.bollinger_mavg()

            df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

            # Use last 15m bar if it's recent (near real-time), otherwise use previous closed bar
            time_diff = datetime.now() - df_15.index.max().to_pydatetime()
            trigger_idx = -1 if time_diff.total_seconds() < 90 else -2
            trigger = df_15.iloc[trigger_idx]

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
            self.logger.exception(f"Analysis error for {self.instrument}")
            return None


# =============================================================================
# MOMENTUM BREAKOUT STRATEGY
# =============================================================================


class MomentumBreakoutStrategy(Strategy):
    """
    Momentum Breakout Strategy using price breakouts with volume confirmation.

    Best suited for: Volatile commodities (NATURALGAS)

    Entry Rules:
    - BUY: Price breaks above recent high with strong volume and RSI momentum
    - SELL: Price breaks below recent low with strong volume and RSI momentum
    - ADX > 25 for trend strength
    - ATR-based stop loss (1.5x default, 2.0x for Crude Oil)
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
            "adx_threshold": 25,  # Minimum ADX for trend trades
            "atr_multiplier": 1.5,  # ATR multiplier for stop loss
            "atr_length": 14,
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Analyze using momentum breakout logic."""
        try:
            # Universal time filter for NSE indices: No new entries between 11:30 AM and 1:30 PM IST
            if self.instrument in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                current_time = datetime.now().time()
                no_trade_start = datetime.strptime("11:30", "%H:%M").time()
                no_trade_end = datetime.strptime("13:30", "%H:%M").time()
                if no_trade_start <= current_time <= no_trade_end:
                    return None

            # Get parameters
            lookback = self.get_param("lookback_period", 20)
            breakout_pct = self.get_param("breakout_threshold", 0.005)
            rsi_min_bull = self.get_param("rsi_min_bullish", 55)
            rsi_max_bear = self.get_param("rsi_max_bearish", 45)
            rsi_len = self.get_param("rsi_length", 14)
            volume_mult = self.get_param("volume_multiplier", 1.5)
            adx_threshold = self.get_param("adx_threshold", 25)
            atr_mult = self.get_param("atr_multiplier", 1.5)
            atr_len = self.get_param("atr_length", 14)

            # Ensure sufficient data for indicators and lookback
            required_15 = max(rsi_len, atr_len, 14, lookback, 3)
            if df_15.shape[0] < (required_15 + 3):
                self.logger.info(
                    f"SKIP [{self.instrument}]: Insufficient 15m data (have {df_15.shape[0]}, need {required_15 + 3})"
                )
                return None

            # Instrument-specific overrides
            if self.instrument == "NATGASMINI":
                pass  # Use param value
            elif self.instrument == "MIDCPNIFTY":
                volume_mult = 2.0

            # Calculate indicators
            df_15["RSI"] = RSIIndicator(close=df_15["close"], window=rsi_len).rsi()
            df_15["ADX"] = ADXIndicator(
                high=df_15["high"], low=df_15["low"], close=df_15["close"], window=14
            ).adx()
            df_15["ATR"] = AverageTrueRange(
                high=df_15["high"],
                low=df_15["low"],
                close=df_15["close"],
                window=atr_len,
            ).average_true_range()
            df_15["vol_avg"] = df_15["volume"].rolling(window=20).mean()

            # Calculate recent high/low
            df_15["recent_high"] = df_15["high"].rolling(window=lookback).max()
            df_15["recent_low"] = df_15["low"].rolling(window=lookback).min()

            # Use last 15m bar if it's recent (near real-time), otherwise use previous closed bar
            trigger = df_15.iloc[-2]
            prev = df_15.iloc[-3]

            price = trigger["close"]
            rsi_val = trigger["RSI"]
            adx_val = trigger["ADX"]
            atr_val = trigger["ATR"]
            recent_high = prev["recent_high"]  # Use previous candle's level
            recent_low = prev["recent_low"]
            current_volume = trigger["volume"]
            avg_volume = trigger.get("vol_avg", current_volume)

            # Data robustness: Check for NaN in key indicators or zero volume
            if (
                pd.isna(trigger.get("RSI"))
                or pd.isna(trigger.get("ADX"))
                or trigger.get("volume", 0) == 0
            ):
                self.logger.info(
                    f"SKIP [{self.instrument}]: Invalid data in trigger row (NaN or zero volume)"
                )
                return None

            # Adaptive ATR: Check for high volatility
            atr_rolling_mean = df_15["ATR"].rolling(window=20).mean().iloc[-1]
            if pd.isna(atr_rolling_mean):
                atr_rolling_mean = atr_val
            volatility_multiplier = 1.25 if atr_val > 1.5 * atr_rolling_mean else 1.0

            # Volume confirmation (strong volume required)
            volume_confirmed = (
                current_volume >= (avg_volume * volume_mult)
                if avg_volume > 0
                else False
            )

            # ADX filter: Only allow signals if trend strength is sufficient (ADX > threshold)
            adx_confirmed = adx_val > adx_threshold

            signal = None
            signal_strength = 0
            stop_loss = 0

            # Calculate breakout levels
            upper_breakout = recent_high * (1 + breakout_pct)
            lower_breakout = recent_low * (1 - breakout_pct)

            # RSI slope check for NATGASMINI
            rsi_slope_ok = True
            if self.instrument == "NATGASMINI":
                prev_rsi = prev.get("RSI", rsi_val)
                rsi_slope_ok = (
                    (rsi_val > prev_rsi)
                    if rsi_val > rsi_min_bull
                    else (rsi_val < prev_rsi)
                )

            # BULLISH Breakout
            if (
                price > upper_breakout
                and rsi_val > rsi_min_bull
                and volume_confirmed
                and adx_confirmed
                and rsi_slope_ok
            ):
                signal = "BUY"
                breakout_strength = (price - recent_high) / recent_high * 100
                signal_strength = breakout_strength + (rsi_val - rsi_min_bull)
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 20
                # ATR-based stop loss for BUY
                stop_loss = price - (atr_val * atr_mult)

            # BEARISH Breakout
            elif (
                price < lower_breakout
                and rsi_val < rsi_max_bear
                and volume_confirmed
                and adx_confirmed
                and rsi_slope_ok
            ):
                signal = "SELL"
                breakout_strength = (recent_low - price) / recent_low * 100
                signal_strength = breakout_strength + (rsi_max_bear - rsi_val)
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 20
                # ATR-based stop loss for SELL
                stop_loss = price + (atr_val * atr_mult)

            if signal:
                # Hybrid Dynamic Trailing Stop Loss Logic
                atr_mult = self.get_param("atr_multiplier", 1.5)
                atr_len = self.get_param("atr_length", 14)

                # Initial SL: atr_mult * ATR * volatility_multiplier
                initial_sl_distance = atr_val * atr_mult * volatility_multiplier

                # Activation Point: 1.5x ATR * volatility_multiplier favorable move (1:1 R:R)
                activation_distance = atr_val * 1.5 * volatility_multiplier

                # Dynamic Trailing: 1.5x ATR * volatility_multiplier from highest price after activation
                trail_distance = atr_val * 1.5 * volatility_multiplier

                # Maximum profit cap for indices (1:2.5 R:R)
                max_profit_mult = None
                if self.instrument in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                    max_profit_mult = 2.5  # 1:2.5 R:R cap for indices

                exit_logic = {
                    "initial_sl_distance": round(initial_sl_distance, 2),
                    "activation_atr_mult": 1.5
                    * volatility_multiplier,  # 1.5x ATR for activation
                    "trail_atr_mult": 1.5
                    * volatility_multiplier,  # 1.5x ATR for trailing
                    "max_profit_mult": max_profit_mult,  # None for commodities, 2.5 for indices
                    "atr_value": round(atr_val, 2),
                    "atr_multiplier": atr_mult * volatility_multiplier,
                }

                result = {
                    "instrument": self.instrument,
                    "signal": signal,
                    "price": price,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "volume": current_volume,
                    "avg_volume": avg_volume,
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "signal_strength": signal_strength,
                    "exit_logic": exit_logic,
                    "strategy": self.name,
                    "df_15": df_15,
                }

                # Remove old trailing stop logic for MIDCPNIFTY (replaced by hybrid system)
                # The hybrid system handles all trailing logic now

                return result

            # If no signal, log rejection reasons to make skips visible in INFO logs
            reasons = []
            if not volume_confirmed:
                reasons.append(
                    f"Volume too low ({current_volume:.0f} < {avg_volume * volume_mult:.0f})"
                )
            if not adx_confirmed:
                reasons.append(f"ADX too low ({adx_val:.1f} < {adx_threshold})")
            # RSI checks
            if self.instrument == "NATGASMINI":
                if not rsi_slope_ok:
                    reasons.append("RSI slope not favorable")
                if rsi_val <= rsi_min_bull and rsi_val >= rsi_max_bear:
                    # If RSI is between thresholds, it's not showing momentum
                    reasons.append(f"RSI not in momentum range ({rsi_val:.1f})")
            else:
                if rsi_val <= rsi_min_bull and rsi_val >= rsi_max_bear:
                    reasons.append(f"RSI not in momentum range ({rsi_val:.1f})")

            # Breakout conditions
            if not (price > upper_breakout or price < lower_breakout):
                reasons.append(
                    f"No breakout ({price:.2f} not outside [{lower_breakout:.2f}, {upper_breakout:.2f}])"
                )

            if reasons:
                self.logger.info(f"SKIP [{self.instrument}]: {', '.join(reasons)}")

            return None

        except Exception as e:
            self.logger.exception(f"Analysis error for {self.instrument}")
            return None


# =============================================================================
# FINNIFTY SPECIFIC STRATEGY
# =============================================================================


class FinniftySpecificStrategy(Strategy):
    """
    Finnifty Specific Strategy with sector correlation and time filters.

    Best suited for: FINNIFTY (Nifty Financial Services)

    Features:
    - Sector correlation check with BANKNIFTY EMA50
    - Time filter preventing entries between 11:30 AM - 1:30 PM
    - 1:2 Risk-to-Reward ratio for quick scalps
    - ATR-based stop loss

    Entry Rules:
    - BUY: Price > EMA50 (60min), Price > VWAP (15min), RSI > 62, BANKNIFTY > EMA50, outside time filter
    - SELL: Price < EMA50 (60min), Price < VWAP (15min), RSI < 38, outside time filter
    """

    def _set_default_params(self) -> None:
        """Set default parameters for Finnifty specific strategy."""
        defaults = {
            "rsi_bullish_threshold": 62,
            "rsi_bearish_threshold": 38,
            "rsi_length": 14,
            "ema_length": 50,
            "volume_multiplier": 1.2,
            "volume_window": 20,
            "atr_multiplier": 1.5,  # ATR multiplier for stop loss
            "atr_length": 14,
        }
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    def analyze(
        self, df_15: pd.DataFrame, df_60: pd.DataFrame, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Analyze using Finnifty specific logic with correlation and time filters."""
        try:
            # Time filter: No new entries between 11:30 AM and 1:30 PM IST
            current_time = datetime.now().time()
            no_trade_start = datetime.strptime("11:30", "%H:%M").time()
            no_trade_end = datetime.strptime("13:30", "%H:%M").time()
            if no_trade_start <= current_time <= no_trade_end:
                return None

            # Get parameters
            rsi_bullish = self.get_param("rsi_bullish_threshold", 62)
            rsi_bearish = self.get_param("rsi_bearish_threshold", 38)
            rsi_len = self.get_param("rsi_length", 14)
            ema_len = self.get_param("ema_length", 50)
            volume_mult = self.get_param("volume_multiplier", 1.2)
            vol_window = self.get_param("volume_window", 20)
            atr_mult = self.get_param("atr_multiplier", 1.5)
            atr_len = self.get_param("atr_length", 14)

            # Calculate indicators
            # 1. EMA on 60min
            df_60["EMA"] = EMAIndicator(
                close=df_60["close"], window=ema_len
            ).ema_indicator()
            # 2. RSI on 15min
            df_15["RSI"] = RSIIndicator(close=df_15["close"], window=rsi_len).rsi()
            # 3. VWAP (Custom Anchored)
            df_15["VWAP_D"] = calculate_anchored_vwap(df_15)
            # 4. ATR for stop loss
            df_15["ATR"] = AverageTrueRange(
                high=df_15["high"],
                low=df_15["low"],
                close=df_15["close"],
                window=atr_len,
            ).average_true_range()
            df_15["vol_avg"] = df_15["volume"].rolling(window=vol_window).mean()

            trend = df_60.iloc[-2]
            # Use last 15m bar if it's recent (near real-time), otherwise use previous closed bar
            time_diff = datetime.now() - df_15.index.max().to_pydatetime()
            trigger_idx = -1 if time_diff.total_seconds() < 90 else -2
            trigger = df_15.iloc[trigger_idx]

            price = trigger["close"]
            vwap_val = trigger.get("VWAP_D", 0)
            current_volume = trigger["volume"]
            avg_volume = trigger.get("vol_avg", current_volume)
            rsi_val = trigger["RSI"]
            ema_val = trend["EMA"]
            trend_close = trend["close"]
            atr_val = trigger["ATR"]

            # Volume confirmation
            volume_confirmed = (
                current_volume >= (avg_volume * volume_mult) if avg_volume > 0 else True
            )

            # Sector correlation check for BANKNIFTY
            banknifty_df_60 = kwargs.get("banknifty_df_60")
            banknifty_correlation_ok = True
            if banknifty_df_60 is not None and not banknifty_df_60.empty:
                try:
                    banknifty_ema = EMAIndicator(
                        close=banknifty_df_60["close"], window=50
                    ).ema_indicator()
                    banknifty_close = banknifty_df_60.iloc[-2]["close"]
                    banknifty_correlation_ok = banknifty_close > banknifty_ema.iloc[-2]
                except Exception as e:
                    self.logger.warning(f"Could not check BANKNIFTY correlation: {e}")
                    banknifty_correlation_ok = True  # Default to allow if check fails

            signal = None
            signal_strength = 0
            stop_loss = 0

            # BULLISH Signal
            if (
                (trend_close > ema_val)
                and (trigger["close"] > vwap_val)
                and (rsi_val > rsi_bullish)
                and volume_confirmed
                and banknifty_correlation_ok
            ):
                signal = "BUY"
                signal_strength = (rsi_val - rsi_bullish) + (
                    (trend_close - ema_val) / ema_val * 100
                )
                if avg_volume > 0:
                    signal_strength += (current_volume / avg_volume - 1) * 10
                # ATR-based stop loss
                stop_loss = price - (atr_val * atr_mult)

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
                # ATR-based stop loss
                stop_loss = price + (atr_val * atr_mult)

            if signal:
                # Hybrid Dynamic Trailing Stop Loss Logic
                atr_mult = self.get_param("atr_multiplier", 1.5)

                # Initial SL: 1.5x ATR from entry
                initial_sl_distance = atr_val * 1.5

                # Activation Point: 1.5x ATR favorable move (1:1 R:R)
                activation_distance = atr_val * 1.5

                # Dynamic Trailing: 1.5x ATR from highest price after activation
                trail_distance = atr_val * 1.5

                # Maximum profit cap for indices (1:2.5 R:R)
                max_profit_mult = None
                if self.instrument in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]:
                    max_profit_mult = 2.5  # 1:2.5 R:R cap for indices

                exit_logic = {
                    "initial_sl_distance": round(initial_sl_distance, 2),
                    "activation_atr_mult": 1.5,  # 1.5x ATR for activation
                    "trail_atr_mult": 1.5,  # 1.5x ATR for trailing
                    "max_profit_mult": max_profit_mult,  # None for commodities, 2.5 for indices
                    "atr_value": round(atr_val, 2),
                    "atr_multiplier": atr_mult,
                }

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
                    "exit_logic": exit_logic,
                    "strategy": self.name,
                    "df_15": df_15,
                    "risk_reward_ratio": 2.0,  # 1:2 R:R for FINNIFTY scalps
                }

            return None

        except Exception as e:
            self.logger.exception(f"Analysis error for {self.instrument}")
            return None


# =============================================================================
# STRATEGY REGISTRY & FACTORY
# =============================================================================

# Default strategy assignments per instrument
DEFAULT_STRATEGY_MAP = {
    "CRUDEOIL": "TrendFollowing",
    # "NATURALGAS": "MomentumBreakout",
    "NATGASMINI": "MomentumBreakout",
    # "GOLD": "TrendFollowing",
    # "SILVER": "MomentumBreakout",
    "NIFTY": "TrendFollowing",  # Can switch to MeanReversion for ranging markets
    "BANKNIFTY": "TrendFollowing",
    "FINNIFTY": "FinniftySpecific",
}

# Strategy class registry
STRATEGY_CLASSES = {
    "TrendFollowing": TrendFollowingStrategy,
    "MeanReversion": MeanReversionStrategy,
    "MomentumBreakout": MomentumBreakoutStrategy,
    "FinniftySpecific": FinniftySpecificStrategy,
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
    inst_params = cast(Dict[str, Any], inst_config.get("strategy_params", {}))

    # Merge with provided params (provided params take precedence)
    merged_params = {**inst_params, **(params or {})}

    return strategy_class(instrument, merged_params)  # type: ignore


def get_available_strategies() -> List[str]:
    """Return list of available strategy names."""
    return list(STRATEGY_CLASSES.keys())


def register_strategy(name: str, strategy_class: type) -> None:
    """Register a new strategy class."""
    if not issubclass(strategy_class, Strategy):
        raise ValueError(f"{strategy_class} must be a subclass of Strategy")
    STRATEGY_CLASSES[name] = strategy_class
