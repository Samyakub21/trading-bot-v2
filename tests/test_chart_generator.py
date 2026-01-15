"""
Unit tests for chart_generator.py
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
import io
import sys

# Mock matplotlib modules
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.dates"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()
sys.modules["mplfinance"] = MagicMock()

from chart_generator import generate_trade_chart


class TestChartGenerator:
    """Test cases for chart generation functionality."""

    def test_generate_chart_valid_data(self):
        """Test chart generation with valid OHLC data."""
        # Create sample OHLC data
        dates = pd.date_range(start="2023-01-01", periods=10, freq="5min")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                "close": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            },
            index=dates,
        )

        with patch("chart_generator.io.BytesIO") as mock_bytesio, patch(
            "chart_generator.plt"
        ) as mock_plt:

            # Mock the BytesIO to return a mock buffer
            mock_buf = MagicMock()
            mock_buf.getvalue.return_value = b"fake_png_data"
            mock_bytesio.return_value = mock_buf

            # Mock figure and its savefig method
            mock_fig = MagicMock()
            mock_plt.figure.return_value = mock_fig

            result = generate_trade_chart(
                df=df,
                instrument="CRUDEOIL",
                signal="BUY",
                entry_price=105.0,
                stop_loss=100.0,
                targets=[110.0, 115.0],
            )

            # Verify it returns bytes
            assert result == b"fake_png_data"
            # Verify BytesIO was called
            mock_bytesio.assert_called_once()
            # Verify savefig was called on the figure
            mock_fig.savefig.assert_called_once()

    def test_generate_chart_empty_data(self):
        """Test chart generation with empty DataFrame."""
        df = pd.DataFrame()

        result = generate_trade_chart(
            df=df,
            instrument="CRUDEOIL",
            signal="BUY",
            entry_price=105.0,
            stop_loss=100.0,
        )

        assert result is None

    def test_generate_chart_insufficient_data(self):
        """Test chart generation with insufficient data (less than 5 rows)."""
        dates = pd.date_range(start="2023-01-01", periods=3, freq="5min")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
            },
            index=dates,
        )

        result = generate_trade_chart(
            df=df,
            instrument="CRUDEOIL",
            signal="BUY",
            entry_price=105.0,
            stop_loss=100.0,
        )

        assert result is None

    def test_generate_chart_missing_columns(self):
        """Test chart generation with missing OHLC columns."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="5min")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                # Missing 'low' and 'close'
            },
            index=dates,
        )

        result = generate_trade_chart(
            df=df,
            instrument="CRUDEOIL",
            signal="BUY",
            entry_price=105.0,
            stop_loss=100.0,
        )

        assert result is None
