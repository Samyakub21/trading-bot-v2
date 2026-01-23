"""
Unit tests for economic_calendar.py
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

from economic_calendar import (
    EconomicCalendar,
    should_pause_trading,
    get_upcoming_events,
)


class TestEconomicCalendar:
    """Test cases for economic calendar functionality."""

    @patch("requests.get")
    def test_fetch_events_api_success(self, mock_get):
        """Test successful API fetch of economic events."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "date": "2026-01-15T14:30:00+00:00",
                "title": "Crude Oil Inventories",
                "impact": "High",
                "country": "US",
                "forecast": "2.5M",
                "previous": "1.8M",
            },
            {
                "date": "2026-01-15T15:00:00+00:00",
                "title": "Non-Farm Payrolls",
                "impact": "High",
                "country": "US",
            },
        ]
        mock_get.return_value = mock_response

        calendar = EconomicCalendar()
        events = calendar.fetch_events_from_api()

        assert len(events) == 2
        assert events[0].name == "Crude Oil Inventories"
        assert events[0].impact == "high"
        assert events[1].name == "Non-Farm Payrolls"
        assert events[1].impact == "high"

    @patch("requests.get")
    def test_fetch_events_api_failure(self, mock_get):
        """Test API failure fallback behavior."""
        # Mock API failure
        mock_get.side_effect = Exception("Connection timeout")

        calendar = EconomicCalendar()
        events = calendar.fetch_events_from_api()

        assert events == []

    @patch("json.load")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_load_cached_events(self, mock_exists, mock_file, mock_json_load):
        """Test loading events from cache file."""
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "events": [
                {
                    "name": "Test Event",
                    "timestamp": "2026-01-15T14:30:00",
                    "impact": "high",
                    "country": "US",
                }
            ],
            "last_update": datetime.now().isoformat(),
        }

        calendar = EconomicCalendar()
        result = calendar.load_cached_events()

        assert result == True
        assert len(calendar.events) == 1
        assert calendar.events[0].name == "Test Event"

    @patch("builtins.open", new_callable=mock_open)
    def test_save_events_to_cache(self, mock_file):
        """Test saving events to cache file."""
        calendar = EconomicCalendar()
        # Add a mock event
        from economic_calendar import EconomicEvent

        event = EconomicEvent(
            name="Test Event", timestamp=datetime.now(), impact="high", country="US"
        )
        calendar.events = [event]

        calendar.save_events_to_cache()

        # Verify file was written
        mock_file.assert_called_once()
        # Verify json.dump was called
        import json

        with patch("json.dump") as mock_json_dump:
            calendar.save_events_to_cache()
            mock_json_dump.assert_called_once()

    def test_is_high_impact(self):
        """Test filtering logic for high-impact events."""
        from economic_calendar import EconomicEvent

        # High impact event
        high_event = EconomicEvent(
            name="Crude Oil Inventories",
            timestamp=datetime.now() + timedelta(hours=1),
            impact="high",
            country="US",
        )

        # Medium impact event
        medium_event = EconomicEvent(
            name="Some Medium Event",
            timestamp=datetime.now() + timedelta(hours=1),
            impact="medium",
            country="US",
        )

        assert high_event.is_high_impact() == True
        assert medium_event.is_high_impact() == False

    def test_should_pause_trading(self):
        """Test trading pause logic."""
        # Mock event
        mock_event = MagicMock()
        mock_event.timestamp = datetime.now() + timedelta(minutes=5)
        mock_event.name = "Crude Oil Inventories"

        with patch("economic_calendar.get_calendar") as mock_get_cal:
            mock_calendar = MagicMock()
            mock_get_cal.return_value = mock_calendar
            mock_calendar.should_pause_trading.return_value = (True, mock_event)

            pause, reason = should_pause_trading("CRUDEOIL")

            assert pause == True
            assert "Crude Oil Inventories" in reason

    @patch("economic_calendar.EconomicCalendar.get_upcoming_events")
    def test_get_upcoming_events(self, mock_get_upcoming):
        """Test getting upcoming events."""
        mock_events = [MagicMock(), MagicMock()]
        mock_get_upcoming.return_value = mock_events

        with patch("economic_calendar.get_calendar") as mock_get_cal:
            mock_calendar = MagicMock()
            mock_get_cal.return_value = mock_calendar
            mock_calendar.get_upcoming_events.return_value = mock_events

            events = get_upcoming_events("CRUDEOIL", 24)

            assert len(events) == 2
