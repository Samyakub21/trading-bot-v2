# =============================================================================
# CHART GENERATOR - Rich Trade Alert Images for Telegram
# =============================================================================
# Generates chart images with entry candle, stop loss, and indicators
# for visual verification of trade decisions
# =============================================================================

import io
import logging
from datetime import datetime
from typing import Any, Dict, Optional, cast

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf

from config import config

# =============================================================================
# CONFIGURATION
# =============================================================================
TELEGRAM_TOKEN = config.TELEGRAM_TOKEN
TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID

# Chart styling
CHART_STYLE = {
    "base_mpf_style": "charles",
    "figure_size": (12, 8),
    "candle_width": 0.6,
    "volume_alpha": 0.5,
    "entry_color": "#00FF00",  # Green for entry
    "sl_color": "#FF0000",  # Red for stop loss
    "target_color": "#FFA500",  # Orange for targets
    "ema_colors": ["#2196F3", "#FF9800", "#9C27B0"],  # Blue, Orange, Purple
}


def generate_trade_chart(
    df: pd.DataFrame,
    instrument: str,
    signal: str,
    entry_price: float,
    stop_loss: float,
    targets: Optional[list] = None,
    option_type: str = "",
    strike: int = 0,
    indicators: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    """
    Generate a chart image showing the trade setup.

    Args:
        df: DataFrame with OHLC data (must have datetime index)
        instrument: Instrument name (e.g., "CRUDEOIL")
        signal: Trade signal ("BUY" or "SELL")
        entry_price: Entry price level
        stop_loss: Stop loss price level
        targets: List of target prices
        option_type: Option type ("CE" or "PE")
        strike: Strike price
        indicators: Dict of indicator values to display

    Returns:
        PNG image bytes or None on failure
    """
    try:
        if df is None or df.empty or len(df) < 5:
            logging.warning("Chart generation: Insufficient data")
            return None

        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            logging.warning("Chart generation: Missing OHLC columns")
            return None

        # Take last 50 candles for better visibility
        df_chart = cast(pd.DataFrame, df.tail(50).copy())

        # Create figure with subplots
        fig = plt.figure(figsize=cast(tuple[int, int], CHART_STYLE["figure_size"]))

        # Main price chart (80% of height)
        ax_price = fig.add_axes((0.1, 0.25, 0.85, 0.65))

        # Volume chart (15% of height)
        ax_volume = fig.add_axes((0.1, 0.08, 0.85, 0.15), sharex=ax_price)

        # Plot candlesticks
        _plot_candlesticks(ax_price, df_chart)

        # Plot volume if available
        if "volume" in df_chart.columns:
            _plot_volume(ax_volume, df_chart)

        # Plot EMAs if available
        _plot_emas(ax_price, df_chart)

        # Draw entry line
        ax_price.axhline(
            y=entry_price,
            color=CHART_STYLE["entry_color"],
            linestyle="--",
            linewidth=2,
            label=f"Entry: {entry_price}",
        )

        # Draw stop loss line
        ax_price.axhline(
            y=stop_loss,
            color=CHART_STYLE["sl_color"],
            linestyle="--",
            linewidth=2,
            label=f"SL: {stop_loss}",
        )

        # Draw target lines if provided
        if targets:
            for i, target in enumerate(targets[:3], 1):  # Max 3 targets
                ax_price.axhline(
                    y=target,
                    color=CHART_STYLE["target_color"],
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"T{i}: {target}",
                )

        # Highlight entry candle (last candle)
        _highlight_entry_candle(ax_price, df_chart, signal)

        # Build title
        opt_info = f" {strike} {option_type}" if strike and option_type else ""
        title = f"{instrument}{opt_info} - {signal} Signal"
        ax_price.set_title(title, fontsize=14, fontweight="bold", pad=10)

        # Add indicator annotations
        if indicators:
            _add_indicator_annotations(ax_price, df_chart, indicators)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(
            0.99,
            0.01,
            f"Generated: {timestamp}",
            fontsize=8,
            ha="right",
            va="bottom",
            alpha=0.6,
        )

        # Configure legend
        ax_price.legend(loc="upper left", fontsize=9, framealpha=0.9)

        # Format axes
        ax_price.grid(True, alpha=0.3)
        ax_price.set_ylabel("Price", fontsize=10)
        ax_volume.set_ylabel("Volume", fontsize=10)

        # Format x-axis dates
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_price.xaxis.get_majorticklabels(), visible=False)

        ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.xticks(rotation=45)

        # Save to bytes buffer
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=100,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        buf.seek(0)

        plt.close(fig)

        return buf.getvalue()

    except Exception as e:
        logging.error(f"Chart generation error: {e}")
        plt.close("all")
        return None


def _plot_candlesticks(ax, df: pd.DataFrame) -> None:
    """Plot candlestick chart on the given axes."""
    df = cast(pd.DataFrame, df)
    try:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        # Convert to matplotlib dates
        dates = mdates.date2num(df.index.to_pydatetime())

        width = 0.0005  # Width of candlestick body

        for i in range(len(df)):
            open_price = df["open"].iloc[i]
            close_price = df["close"].iloc[i]
            high_price = df["high"].iloc[i]
            low_price = df["low"].iloc[i]

            # Determine candle color
            if close_price >= open_price:
                color = "#26A69A"  # Green
                body_bottom = open_price
                body_height = close_price - open_price
            else:
                color = "#EF5350"  # Red
                body_bottom = close_price
                body_height = open_price - close_price

            # Draw wick (high-low line)
            ax.plot(
                [dates[i], dates[i]], [low_price, high_price], color=color, linewidth=1
            )

            # Draw body
            if body_height > 0:
                rect = Rectangle(
                    (dates[i] - width / 2, body_bottom),
                    width,
                    body_height,
                    facecolor=color,
                    edgecolor=color,
                )
                ax.add_patch(rect)
            else:
                # Doji candle
                ax.plot(
                    [dates[i] - width / 2, dates[i] + width / 2],
                    [close_price, close_price],
                    color=color,
                    linewidth=2,
                )

    except Exception as e:
        logging.debug(f"Candlestick plot error: {e}")


def _plot_volume(ax, df: pd.DataFrame) -> None:
    """Plot volume bars on the given axes."""
    df = cast(pd.DataFrame, df)
    try:
        if "volume" not in df.columns:
            return

        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        dates = mdates.date2num(df.index.to_pydatetime())

        # Color volume bars based on price direction
        colors = [
            "#26A69A" if df["close"].iloc[i] >= df["open"].iloc[i] else "#EF5350"
            for i in range(len(df))
        ]

        ax.bar(
            dates,
            df["volume"],
            width=0.0005,
            color=colors,
            alpha=CHART_STYLE["volume_alpha"],
        )
        ax.set_ylim(bottom=0)

    except Exception as e:
        logging.debug(f"Volume plot error: {e}")


def _plot_emas(ax, df: pd.DataFrame) -> None:
    """Plot EMAs if they exist in the dataframe."""
    df = cast(pd.DataFrame, df)
    try:
        ema_cols = [col for col in df.columns if col.upper().startswith("EMA")]

        if not ema_cols:
            # Calculate common EMAs if not present
            for period, color in zip(
                [9, 21, 50], cast(list[str], CHART_STYLE["ema_colors"])
            ):
                if len(df) >= period:
                    ema = df["close"].ewm(span=period, adjust=False).mean()
                    ax.plot(
                        df.index,
                        ema,
                        color=color,
                        linewidth=1,
                        alpha=0.7,
                        label=f"EMA{period}",
                    )
        else:
            for i, col in enumerate(ema_cols[:3]):
                color = cast(list[str], CHART_STYLE["ema_colors"])[
                    i % len(cast(list[str], CHART_STYLE["ema_colors"]))
                ]
                ax.plot(  # type: ignore[index]
                    df.index,
                    df[col],
                    color=color,
                    linewidth=1,
                    alpha=0.7,
                    label=col.upper(),
                )

    except Exception as e:
        logging.debug(f"EMA plot error: {e}")


def _highlight_entry_candle(ax, df: pd.DataFrame, signal: str) -> None:
    """Highlight the entry candle with an arrow."""
    df = cast(pd.DataFrame, df)
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            return

        last_idx = df.index[-1]
        last_candle = df.iloc[-1]

        # Arrow properties based on signal direction
        if signal.upper() == "BUY":
            arrow_y = last_candle["low"] * 0.998
            arrow_color = "#00FF00"
            arrow_marker = "^"
        else:
            arrow_y = last_candle["high"] * 1.002
            arrow_color = "#FF0000"
            arrow_marker = "v"

        ax.scatter(
            [last_idx],
            [arrow_y],
            marker=arrow_marker,
            s=200,
            c=arrow_color,
            zorder=5,
            edgecolors="black",
        )

    except Exception as e:
        logging.debug(f"Entry highlight error: {e}")


def _add_indicator_annotations(
    ax, df: pd.DataFrame, indicators: Dict[str, Any]
) -> None:
    """Add indicator value annotations to the chart."""
    df = cast(pd.DataFrame, df)
    try:
        annotation_text = []

        if "rsi" in indicators:
            annotation_text.append(f"RSI: {indicators['rsi']:.1f}")
        if "macd" in indicators:
            annotation_text.append(f"MACD: {indicators['macd']:.2f}")
        if "volume_ratio" in indicators:
            annotation_text.append(f"Vol Ratio: {indicators['volume_ratio']:.2f}x")
        if "ema_alignment" in indicators:
            annotation_text.append(f"EMA: {indicators['ema_alignment']}")
        if "signal_strength" in indicators:
            annotation_text.append(f"Strength: {indicators['signal_strength']}")

        if annotation_text:
            text_str = " | ".join(annotation_text)

            # Position at top right of chart
            y_max = ax.get_ylim()[1]
            x_max = ax.get_xlim()[1]

            ax.text(
                x_max,
                y_max,
                text_str,
                fontsize=9,
                ha="right",
                va="top",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
                ),
            )

    except Exception as e:
        logging.debug(f"Indicator annotation error: {e}")


def send_chart_to_telegram(
    image_bytes: bytes,
    caption: str,
    token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> bool:
    """
    Send chart image to Telegram.

    Args:
        image_bytes: PNG image data
        caption: Caption text for the image
        token: Telegram bot token (uses default if None)
        chat_id: Telegram chat ID (uses default if None)

    Returns:
        True if sent successfully, False otherwise
    """
    import requests

    try:
        token = token or TELEGRAM_TOKEN
        chat_id = chat_id or TELEGRAM_CHAT_ID

        if not token or not chat_id:
            logging.error("Telegram credentials not configured")
            return False

        url = f"https://api.telegram.org/bot{token}/sendPhoto"

        files = {"photo": ("trade_chart.png", image_bytes, "image/png")}

        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}

        response = requests.post(url, files=files, data=data, timeout=30)

        if response.status_code == 200:
            logging.debug("Chart sent to Telegram successfully")
            return True
        else:
            logging.warning(f"Telegram photo send failed: {response.status_code}")
            return False

    except requests.exceptions.Timeout:
        logging.warning("Telegram photo send timeout")
        return False
    except Exception as e:
        logging.error(f"Telegram photo send error: {e}")
        return False


def send_rich_trade_alert(
    instrument: str,
    signal: str,
    entry_price: float,
    stop_loss: float,
    df: pd.DataFrame,
    strike: int = 0,
    option_type: str = "",
    targets: Optional[list] = None,
    indicators: Optional[Dict[str, Any]] = None,
    option_entry_price: float = 0,
) -> bool:
    """
    Send a rich trade alert with chart image to Telegram.

    This combines the text alert with a visual chart showing
    the entry candle, stop loss, and indicators.

    Args:
        instrument: Instrument name
        signal: Trade signal ("BUY" or "SELL")
        entry_price: Future entry price
        stop_loss: Stop loss price
        df: OHLC DataFrame
        strike: Option strike price
        option_type: Option type ("CE" or "PE")
        targets: List of target prices
        indicators: Dict of indicator values
        option_entry_price: Option premium entry price

    Returns:
        True if alert sent successfully
    """
    try:
        # Generate chart image
        chart_bytes = generate_trade_chart(
            df=df,
            instrument=instrument,
            signal=signal,
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=targets,
            option_type=option_type,
            strike=strike,
            indicators=indicators,
        )

        # Build caption
        opt_info = f" {strike} {option_type}" if strike and option_type else ""
        emoji = "📈" if signal.upper() == "BUY" else "📉"

        caption_parts = [
            f"{emoji} *{instrument}{opt_info} {signal}*",
            f"Entry: ₹{entry_price}",
            f"Stop Loss: ₹{stop_loss}",
        ]

        if option_entry_price > 0:
            caption_parts.append(f"Option Premium: ₹{option_entry_price}")

        if targets:
            targets_str = ", ".join([f"₹{t}" for t in targets[:3]])
            caption_parts.append(f"Targets: {targets_str}")

        if indicators:
            if "rsi" in indicators:
                caption_parts.append(f"RSI: {indicators['rsi']:.1f}")
            if "signal_strength" in indicators:
                caption_parts.append(f"Signal: {indicators['signal_strength']}")

        caption = "\n".join(caption_parts)

        # Send chart if generated, otherwise send text only
        if chart_bytes:
            success = send_chart_to_telegram(chart_bytes, caption)
            if success:
                return True

        # Fallback to text-only alert
        logging.info("Falling back to text-only alert")
        from utils import send_alert

        send_alert(caption)
        return True

    except Exception as e:
        logging.error(f"Rich trade alert error: {e}")
        return False


# =============================================================================
# STANDALONE TESTING
# =============================================================================
if __name__ == "__main__":
    import numpy as np

    # Create sample data for testing
    dates = pd.date_range(end=datetime.now(), periods=50, freq="15min")
    np.random.seed(42)

    # Generate realistic price data
    base_price = 5000
    returns = np.random.randn(50) * 0.002
    prices = base_price * np.cumprod(1 + returns)

    df_test = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(50) * 0.001),
            "high": prices * (1 + abs(np.random.randn(50)) * 0.003),
            "low": prices * (1 - abs(np.random.randn(50)) * 0.003),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 50),
        },
        index=dates,
    )

    # Test chart generation
    chart = generate_trade_chart(
        df=df_test,
        instrument="CRUDEOIL",
        signal="BUY",
        entry_price=df_test["close"].iloc[-1],
        stop_loss=df_test["low"].iloc[-1] - 10,
        targets=[df_test["close"].iloc[-1] + 20, df_test["close"].iloc[-1] + 40],
        option_type="CE",
        strike=5000,
        indicators={"rsi": 65.5, "volume_ratio": 1.5, "signal_strength": "STRONG"},
    )

    if chart:
        # Save locally for testing
        with open("test_chart.png", "wb") as f:
            f.write(chart)
        print("✅ Test chart saved to test_chart.png")
    else:
        print("❌ Chart generation failed")
