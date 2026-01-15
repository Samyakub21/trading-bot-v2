# =============================================================================
# ECONOMIC CALENDAR - News Filter for High-Impact Events
# =============================================================================
"""
Integrates economic calendar data to pause trading around high-impact events.
The bot will pause scanning 15 minutes before and after "Red Folder" events
that affect commodities (Crude Oil Inventories, US CPI, NFP, etc.)

Usage:
    from economic_calendar import should_pause_trading, get_upcoming_events

    # Check if trading should be paused
    pause, reason = should_pause_trading("CRUDEOIL")
    if pause:
        logging.info(f"Trading paused: {reason}")
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import threading
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cache file for economic events
DATA_DIR = Path(__file__).parent
EVENTS_CACHE_FILE = DATA_DIR / "economic_events_cache.json"

# Pause window around high-impact events (in minutes)
PAUSE_BEFORE_EVENT = 15  # Pause 15 minutes before event
PAUSE_AFTER_EVENT = 15  # Pause 15 minutes after event

# High-impact events that affect each instrument
# These are "Red Folder" events from economic calendars
HIGH_IMPACT_EVENTS = {
    "CRUDEOIL": [
        "Crude Oil Inventories",
        "EIA Crude Oil Stocks Change",
        "API Weekly Crude Oil Stock",
        "OPEC Meeting",
        "OPEC Monthly Report",
        "Baker Hughes Oil Rig Count",
    ],
    "NATURALGAS": [
        "Natural Gas Storage",
        "EIA Natural Gas Storage Change",
        "Baker Hughes Gas Rig Count",
    ],
    "GOLD": [
        "Fed Interest Rate Decision",
        "FOMC Statement",
        "FOMC Minutes",
        "Non-Farm Payrolls",
        "NFP",
        "Unemployment Rate",
        "CPI",
        "Core CPI",
        "PPI",
        "Core PPI",
        "Initial Jobless Claims",
        "GDP",
        "Retail Sales",
    ],
    "SILVER": [
        "Fed Interest Rate Decision",
        "FOMC Statement",
        "Non-Farm Payrolls",
        "CPI",
        "Core CPI",
        "Industrial Production",
    ],
    "NIFTY": [
        "RBI Interest Rate Decision",
        "RBI Monetary Policy",
        "India GDP",
        "India CPI",
        "India WPI",
        "India PMI",
        "Fed Interest Rate Decision",  # Global impact
    ],
    "BANKNIFTY": [
        "RBI Interest Rate Decision",
        "RBI Monetary Policy",
        "India GDP",
        "India CPI",
        "Fed Interest Rate Decision",
    ],
}

# Global events that affect all instruments
GLOBAL_HIGH_IMPACT = [
    "Fed Interest Rate Decision",
    "ECB Interest Rate Decision",
    "BOE Interest Rate Decision",
    "Non-Farm Payrolls",
    "US CPI",
    "US Core CPI",
    "US GDP",
]

# Update interval for fetching new events (in seconds)
EVENTS_UPDATE_INTERVAL = 3600  # 1 hour


# =============================================================================
# EVENT DATA STRUCTURES
# =============================================================================


class EconomicEvent:
    """Represents an economic calendar event."""

    def __init__(
        self,
        name: str,
        timestamp: datetime,
        impact: str,  # "low", "medium", "high"
        country: str,
        actual: Optional[str] = None,
        forecast: Optional[str] = None,
        previous: Optional[str] = None,
    ):
        self.name = name
        self.timestamp = timestamp
        self.impact = impact
        self.country = country
        self.actual = actual
        self.forecast = forecast
        self.previous = previous

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "impact": self.impact,
            "country": self.country,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EconomicEvent":
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            impact=data["impact"],
            country=data["country"],
            actual=data.get("actual"),
            forecast=data.get("forecast"),
            previous=data.get("previous"),
        )

    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"

    def affects_instrument(self, instrument: str) -> bool:
        """Check if this event affects the given instrument."""
        # Check instrument-specific events
        inst_events = HIGH_IMPACT_EVENTS.get(instrument, [])
        for event_pattern in inst_events:
            if event_pattern.lower() in self.name.lower():
                return True

        # Check global events
        for global_event in GLOBAL_HIGH_IMPACT:
            if global_event.lower() in self.name.lower():
                return True

        return False


# =============================================================================
# CALENDAR DATA FETCHING
# =============================================================================


class EconomicCalendar:
    """
    Fetches and caches economic calendar data.

    Supports multiple data sources with fallbacks:
    1. Investing.com (via scraping or API)
    2. ForexFactory (via scraping)
    3. TradingEconomics (if API key available)
    4. Local manual events file
    """

    def __init__(self):
        self.events: List[EconomicEvent] = []
        self.last_update: Optional[datetime] = None
        self._lock = threading.Lock()

    def load_cached_events(self) -> bool:
        """Load events from cache file."""
        try:
            if EVENTS_CACHE_FILE.exists():
                with open(EVENTS_CACHE_FILE, "r") as f:
                    data = json.load(f)

                self.events = [
                    EconomicEvent.from_dict(e) for e in data.get("events", [])
                ]
                self.last_update = (
                    datetime.fromisoformat(data["last_update"])
                    if data.get("last_update")
                    else None
                )

                # Check if cache is still valid (less than 1 hour old)
                if (
                    self.last_update
                    and (datetime.now() - self.last_update).total_seconds()
                    < EVENTS_UPDATE_INTERVAL
                ):
                    logging.debug(f"Loaded {len(self.events)} events from cache")
                    return True
            return False
        except Exception as e:
            logging.warning(f"Failed to load cached events: {e}")
            return False

    def save_events_to_cache(self) -> None:
        """Save current events to cache file."""
        try:
            data = {
                "events": [e.to_dict() for e in self.events],
                "last_update": datetime.now().isoformat(),
            }
            with open(EVENTS_CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save events cache: {e}")

    def fetch_events_from_api(self) -> List[EconomicEvent]:
        """
        Fetch economic events from external API.

        This is a placeholder that can be replaced with actual API integration.
        For production, consider:
        - Investing.com API
        - TradingEconomics API
        - Alpha Vantage Economic Calendar
        """
        events = []

        # Try to fetch from a free economic calendar API
        try:
            # Example: Using a hypothetical free API endpoint
            # Replace with actual API integration
            response = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10
            )

            if response.status_code == 200:
                data = response.json()

                for item in data:
                    try:
                        # Parse ForexFactory format
                        event_date = datetime.strptime(
                            item.get("date", ""), "%Y-%m-%dT%H:%M:%S%z"
                        )

                        # Convert impact to standard format
                        impact_map = {"High": "high", "Medium": "medium", "Low": "low"}
                        impact = impact_map.get(item.get("impact", ""), "low")

                        event = EconomicEvent(
                            name=item.get("title", "Unknown Event"),
                            timestamp=event_date.replace(
                                tzinfo=None
                            ),  # Convert to naive datetime
                            impact=impact,
                            country=item.get("country", "US"),
                            forecast=item.get("forecast"),
                            previous=item.get("previous"),
                        )
                        events.append(event)
                    except Exception as e:
                        logging.debug(f"Failed to parse event: {e}")
                        continue

                logging.info(f"Fetched {len(events)} events from economic calendar API")

        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to fetch events from API: {e}")
        except Exception as e:
            logging.warning(f"Error processing calendar data: {e}")

        return events

    def add_manual_events(self) -> None:
        """
        Add manually defined high-impact events.

        This can be used to add known events that might not be in the API,
        or for testing purposes.
        """
        manual_events_file = DATA_DIR / "manual_events.json"

        if manual_events_file.exists():
            try:
                with open(manual_events_file, "r") as f:
                    manual_data = json.load(f)

                for item in manual_data.get("events", []):
                    event = EconomicEvent(
                        name=item["name"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        impact=item.get("impact", "high"),
                        country=item.get("country", "US"),
                    )
                    self.events.append(event)

                logging.debug(
                    f"Added {len(manual_data.get('events', []))} manual events"
                )
            except Exception as e:
                logging.warning(f"Failed to load manual events: {e}")

    def update_events(self, force: bool = False) -> None:
        """
        Update economic events from all sources.

        Args:
            force: Force update even if cache is fresh
        """
        with self._lock:
            # Check if update is needed
            if not force and self.last_update:
                time_since_update = (datetime.now() - self.last_update).total_seconds()
                if time_since_update < EVENTS_UPDATE_INTERVAL:
                    return

            # Try to load from cache first
            if not force and self.load_cached_events():
                return

            # Fetch from API
            api_events = self.fetch_events_from_api()

            if api_events:
                self.events = api_events

            # Add manual events
            self.add_manual_events()

            # Update timestamp
            self.last_update = datetime.now()

            # Save to cache
            self.save_events_to_cache()

    def should_pause_trading(self) -> Tuple[bool, Optional[EconomicEvent]]:
        """
        Check if trading should be paused due to any high-impact event.
        Checks all configured instruments.
        """
        active_events = self.get_active_events(
            window_minutes=max(PAUSE_BEFORE_EVENT, PAUSE_AFTER_EVENT)
        )

        if active_events:
            return True, active_events[0]

        return False, None

    def get_upcoming_events(
        self,
        instrument: Optional[str] = None,
        hours_ahead: int = 24,
        high_impact_only: bool = True,
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events.

        Args:
            instrument: Filter events affecting this instrument
            hours_ahead: Look this many hours ahead
            high_impact_only: Only return high-impact events

        Returns:
            List of upcoming events
        """
        self.update_events()

        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = []
        for event in self.events:
            # Filter by time
            if event.timestamp < now or event.timestamp > cutoff:
                continue

            # Filter by impact
            if high_impact_only and not event.is_high_impact():
                continue

            # Filter by instrument
            if instrument and not event.affects_instrument(instrument):
                continue

            upcoming.append(event)

        # Sort by timestamp
        upcoming.sort(key=lambda e: e.timestamp)

        return upcoming

    def get_active_events(
        self, instrument: Optional[str] = None, window_minutes: int = 15
    ) -> List[EconomicEvent]:
        """
        Get events that are currently active (within the pause window).

        Args:
            instrument: Filter events affecting this instrument
            window_minutes: Minutes before/after event to consider it active

        Returns:
            List of active events
        """
        self.update_events()

        now = datetime.now()
        window_before = now + timedelta(minutes=window_minutes)
        window_after = now - timedelta(minutes=window_minutes)

        active = []
        for event in self.events:
            # Check if event is within the window
            if window_after <= event.timestamp <= window_before:
                # Filter by high impact
                if not event.is_high_impact():
                    continue

                # Filter by instrument
                if instrument and not event.affects_instrument(instrument):
                    continue

                active.append(event)

        return active


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_calendar: Optional[EconomicCalendar] = None
_calendar_lock = threading.Lock()


def get_calendar() -> EconomicCalendar:
    """Get the singleton EconomicCalendar instance."""
    global _calendar
    if _calendar is None:
        with _calendar_lock:
            if _calendar is None:
                _calendar = EconomicCalendar()
    return _calendar


# =============================================================================
# PUBLIC API
# =============================================================================


def should_pause_trading(instrument: str) -> Tuple[bool, str]:
    """
    Check if trading should be paused for the given instrument.

    Args:
        instrument: The instrument to check (e.g., "CRUDEOIL")

    Returns:
        Tuple of (should_pause, reason)
    """
    calendar = get_calendar()

    # Get active high-impact events
    active_events = calendar.get_active_events(
        instrument=instrument, window_minutes=max(PAUSE_BEFORE_EVENT, PAUSE_AFTER_EVENT)
    )

    if active_events:
        event = active_events[0]  # Get the most relevant event
        minutes_until = (event.timestamp - datetime.now()).total_seconds() / 60

        if minutes_until > 0:
            reason = f"High-impact event in {int(minutes_until)} min: {event.name}"
        else:
            minutes_ago = abs(minutes_until)
            reason = f"High-impact event {int(minutes_ago)} min ago: {event.name}"

        return True, reason

    return False, ""


def get_upcoming_events(
    instrument: Optional[str] = None, hours_ahead: int = 24
) -> List[Dict[str, Any]]:
    """
    Get upcoming high-impact events as a list of dictionaries.

    Args:
        instrument: Filter by instrument (optional)
        hours_ahead: Hours to look ahead

    Returns:
        List of event dictionaries
    """
    calendar = get_calendar()
    events = calendar.get_upcoming_events(
        instrument=instrument, hours_ahead=hours_ahead, high_impact_only=True
    )
    return [e.to_dict() for e in events]


def force_refresh_calendar() -> int:
    """
    Force refresh the economic calendar data.

    Returns:
        Number of events loaded
    """
    calendar = get_calendar()
    calendar.update_events(force=True)
    return len(calendar.events)


def add_custom_event(
    name: str, timestamp: str, impact: str = "high", country: str = "US"
) -> None:
    """
    Add a custom event to the calendar.

    Args:
        name: Event name
        timestamp: ISO format timestamp
        impact: "low", "medium", or "high"
        country: Country code
    """
    calendar = get_calendar()
    event = EconomicEvent(
        name=name,
        timestamp=datetime.fromisoformat(timestamp),
        impact=impact,
        country=country,
    )
    calendar.events.append(event)
    calendar.save_events_to_cache()


# =============================================================================
# BACKGROUND UPDATER
# =============================================================================


def start_calendar_updater(
    interval_seconds: int = EVENTS_UPDATE_INTERVAL,
) -> threading.Thread:
    """
    Start a background thread that periodically updates the calendar.

    Args:
        interval_seconds: Update interval in seconds

    Returns:
        The background thread
    """

    def updater():
        calendar = get_calendar()
        while True:
            try:
                calendar.update_events(force=True)
                logging.debug(
                    f"Economic calendar updated: {len(calendar.events)} events"
                )
            except Exception as e:
                logging.warning(f"Calendar update failed: {e}")

            time.sleep(interval_seconds)

    thread = threading.Thread(target=updater, daemon=True, name="CalendarUpdater")
    thread.start()
    return thread


# =============================================================================
# CLI / TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the calendar
    print("=" * 60)
    print("Economic Calendar Test")
    print("=" * 60)

    # Force refresh
    num_events = force_refresh_calendar()
    print(f"\nLoaded {num_events} events")

    # Check each instrument
    for instrument in ["CRUDEOIL", "GOLD", "NIFTY"]:
        pause, reason = should_pause_trading(instrument)
        status = "🔴 PAUSE" if pause else "🟢 OK"
        print(f"\n{instrument}: {status}")
        if reason:
            print(f"   Reason: {reason}")

        # Show upcoming events
        events = get_upcoming_events(instrument, hours_ahead=48)
        if events:
            print(f"   Upcoming events ({len(events)}):")
            for e in events[:3]:
                print(f"      - {e['name']} @ {e['timestamp']}")
