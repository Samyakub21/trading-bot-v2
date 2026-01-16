# =============================================================================
# DATABASE - SQLite Storage Layer for Trade Data
# =============================================================================
# Provides ACID-compliant storage for trade history, daily P&L, and state
# Replaces JSON file-based storage while maintaining backward compatibility

import sqlite3
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_DB_PATH = str(DATA_DIR / "trading_bot.db")


# =============================================================================
# DATABASE CONNECTION MANAGER
# =============================================================================
class DatabaseManager:
    """
    Thread-safe SQLite database manager with connection pooling.
    Provides ACID-compliant storage for all trading data.
    """

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, db_path: str = DEFAULT_DB_PATH):
        """Singleton pattern for database access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        if self._initialized:
            return

        self.db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._initialized = True

        # Initialize database schema
        self._init_schema()

        logging.info(f"📊 Database initialized: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA busy_timeout=30000")
        return self._local.connection

    @contextmanager
    def get_cursor(self, commit: bool = False):
        """Context manager for database cursor with optional auto-commit"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

    @contextmanager
    def transaction(self):
        """Context manager for write transactions with locking"""
        with self._write_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                logging.error(f"Database transaction failed: {e}")
                raise e

    def _init_schema(self):
        """Initialize database schema"""
        with self.transaction() as cursor:
            # Trade history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    option_type TEXT,
                    future_entry REAL NOT NULL,
                    future_exit REAL NOT NULL,
                    option_entry REAL,
                    option_exit REAL,
                    initial_sl REAL,
                    final_sl REAL,
                    max_step_level INTEGER DEFAULT 0,
                    pnl REAL NOT NULL,
                    r_multiple REAL,
                    exit_reason TEXT,
                    lot_size INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create index on entry_time for faster queries
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_history_entry_time 
                ON trade_history(entry_time)
            """
            )

            # Create index on instrument for filtering
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_trade_history_instrument 
                ON trade_history(instrument)
            """
            )

            # Daily P&L table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date TEXT PRIMARY KEY,
                    pnl REAL NOT NULL DEFAULT 0,
                    trades INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Active trade state table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    status INTEGER NOT NULL DEFAULT 0,
                    trade_type TEXT,
                    option_id TEXT,
                    entry REAL DEFAULT 0,
                    sl REAL DEFAULT 0,
                    initial_sl REAL DEFAULT 0,
                    step_level INTEGER DEFAULT 0,
                    instrument TEXT,
                    lot_size INTEGER,
                    exchange_segment_str TEXT,
                    option_entry REAL DEFAULT 0,
                    entry_time TEXT,
                    extra_data TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Insert default state row if not exists
            cursor.execute(
                """
                INSERT OR IGNORE INTO trade_state (id, status) VALUES (1, 0)
            """
            )

    # =========================================================================
    # TRADE HISTORY METHODS
    # =========================================================================
    def save_trade(self, trade_data: Dict[str, Any]) -> int:
        """
        Save a completed trade to history.

        Args:
            trade_data: Dictionary containing trade details

        Returns:
            The ID of the inserted trade record
        """
        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO trade_history (
                    instrument, entry_time, exit_time, trade_type, option_type,
                    future_entry, future_exit, option_entry, option_exit,
                    initial_sl, final_sl, max_step_level, pnl, r_multiple,
                    exit_reason, lot_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade_data.get("instrument", ""),
                    trade_data.get(
                        "entry_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    trade_data.get(
                        "exit_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ),
                    trade_data.get("trade_type", ""),
                    trade_data.get("option_type", ""),
                    trade_data.get("future_entry", 0),
                    trade_data.get("future_exit", 0),
                    trade_data.get("option_entry", 0),
                    trade_data.get("option_exit", 0),
                    trade_data.get("initial_sl", 0),
                    trade_data.get("final_sl", 0),
                    trade_data.get("max_step_level", 0),
                    trade_data.get("pnl", 0),
                    trade_data.get("r_multiple", 0),
                    trade_data.get("exit_reason", ""),
                    trade_data.get("lot_size", 0),
                ),
            )
            return cursor.lastrowid

    def get_trade_history(
        self,
        days: Optional[int] = None,
        instrument: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve trade history with optional filters.

        Args:
            days: Number of days to look back (None for all)
            instrument: Filter by instrument name
            limit: Maximum number of records to return

        Returns:
            List of trade dictionaries
        """
        query = "SELECT * FROM trade_history WHERE 1=1"
        params: List[Any] = []

        if days:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            query += " AND exit_time >= ?"
            params.append(cutoff_date)

        if instrument:
            query += " AND instrument = ?"
            params.append(instrument)

        query += " ORDER BY exit_time DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_trades_by_date_range(
        self, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Get trades within a specific date range"""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM trade_history 
                WHERE exit_time >= ? AND exit_time <= ?
                ORDER BY exit_time DESC
            """,
                (start_date, end_date),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # DAILY P&L METHODS
    # =========================================================================
    def get_daily_pnl(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily P&L data for a specific date.

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Dictionary with date, pnl, trades, wins, losses
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            else:
                return {"date": date, "pnl": 0, "trades": 0, "wins": 0, "losses": 0}

    # =========================================================================
    # BACKUP METHODS
    # =========================================================================
    def backup_database(self, backup_dir: Optional[str] = None) -> str:
        """
        Create a backup of the current database.

        Args:
            backup_dir: Directory to save backup to. Defaults to 'Extras/backup'

        Returns:
            Path to the backup file
        """
        if backup_dir is None:
            # Default to Extras/backup relative to project root
            backup_dir = str(Path(__file__).parent.parent / "Extras" / "backup")

        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_bot_backup_{timestamp}.db"
        target_file = backup_path / filename

        try:
            # Safely flush WAL file to disk before copying
            with self.get_cursor(commit=True) as cursor:
                cursor.execute("PRAGMA wal_checkpoint(FULL)")

            # Use SQLite Online Backup API if available, else file copy
            import shutil

            shutil.copy2(self.db_path, target_file)

            logging.info(f"✅ Database backup created: {target_file}")

            # Clean up old backups (keep last 7 days)
            self._cleanup_old_backups(backup_path)

            return str(target_file)
        except Exception as e:
            logging.error(f"❌ Database backup failed: {e}")
            return ""

    def _cleanup_old_backups(self, backup_path: Path, days: int = 7) -> None:
        """Remove backups older than N days"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            for item in backup_path.glob("trading_bot_backup_*.db"):
                if item.stat().st_mtime < cutoff.timestamp():
                    item.unlink()
                    logging.debug(f"Removed old backup: {item.name}")
        except Exception as e:
            logging.warning(f"Backup cleanup error: {e}")

    def update_daily_pnl(
        self, pnl_amount: float, is_win: bool, date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update daily P&L after a trade closes.

        Args:
            pnl_amount: P&L amount to add
            is_win: Whether the trade was profitable
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Updated daily P&L dictionary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        with self.transaction() as cursor:
            # Get current values
            cursor.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
            row = cursor.fetchone()

            if row:
                new_pnl = row["pnl"] + pnl_amount
                new_trades = row["trades"] + 1
                new_wins = row["wins"] + (1 if is_win else 0)
                new_losses = row["losses"] + (0 if is_win else 1)

                cursor.execute(
                    """
                    UPDATE daily_pnl 
                    SET pnl = ?, trades = ?, wins = ?, losses = ?, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE date = ?
                """,
                    (new_pnl, new_trades, new_wins, new_losses, date),
                )
            else:
                new_pnl = pnl_amount
                new_trades = 1
                new_wins = 1 if is_win else 0
                new_losses = 0 if is_win else 1

                cursor.execute(
                    """
                    INSERT INTO daily_pnl (date, pnl, trades, wins, losses)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (date, new_pnl, new_trades, new_wins, new_losses),
                )

            return {
                "date": date,
                "pnl": new_pnl,
                "trades": new_trades,
                "wins": new_wins,
                "losses": new_losses,
            }

    def get_pnl_history(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get P&L history for multiple days"""
        query = "SELECT * FROM daily_pnl"
        params = []

        if days:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            query += " WHERE date >= ?"
            params.append(cutoff_date)

        query += " ORDER BY date DESC"

        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # TRADE STATE METHODS
    # =========================================================================
    def save_state(self, state_data: Dict[str, Any]) -> None:
        """
        Save active trade state.

        Args:
            state_data: Dictionary containing trade state
        """
        # Extract known fields
        status = 1 if state_data.get("status", False) else 0
        trade_type = state_data.get("type")
        option_id = state_data.get("option_id")
        entry = state_data.get("entry", 0)
        sl = state_data.get("sl", 0)
        initial_sl = state_data.get("initial_sl", 0)
        step_level = state_data.get("step_level", 0)
        instrument = state_data.get("instrument")
        lot_size = state_data.get("lot_size")
        exchange_segment_str = state_data.get("exchange_segment_str")
        option_entry = state_data.get("option_entry", 0)
        entry_time = state_data.get("entry_time")

        # Store any additional fields as JSON
        known_keys = {
            "status",
            "type",
            "option_id",
            "entry",
            "sl",
            "initial_sl",
            "step_level",
            "instrument",
            "lot_size",
            "exchange_segment_str",
            "option_entry",
            "entry_time",
        }
        extra_data = {k: v for k, v in state_data.items() if k not in known_keys}
        extra_json = json.dumps(extra_data) if extra_data else None

        with self.transaction() as cursor:
            cursor.execute(
                """
                UPDATE trade_state SET
                    status = ?,
                    trade_type = ?,
                    option_id = ?,
                    entry = ?,
                    sl = ?,
                    initial_sl = ?,
                    step_level = ?,
                    instrument = ?,
                    lot_size = ?,
                    exchange_segment_str = ?,
                    option_entry = ?,
                    entry_time = ?,
                    extra_data = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """,
                (
                    status,
                    trade_type,
                    option_id,
                    entry,
                    sl,
                    initial_sl,
                    step_level,
                    instrument,
                    lot_size,
                    exchange_segment_str,
                    option_entry,
                    entry_time,
                    extra_json,
                ),
            )

    def load_state(self) -> Dict[str, Any]:
        """
        Load active trade state.

        Returns:
            Dictionary containing trade state
        """
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM trade_state WHERE id = 1")
            row = cursor.fetchone()

            if row:
                state = {
                    "status": bool(row["status"]),
                    "type": row["trade_type"],
                    "option_id": row["option_id"],
                    "entry": row["entry"] or 0,
                    "sl": row["sl"] or 0,
                    "initial_sl": row["initial_sl"] or 0,
                    "step_level": row["step_level"] or 0,
                    "instrument": row["instrument"],
                    "lot_size": row["lot_size"],
                    "exchange_segment_str": row["exchange_segment_str"],
                    "option_entry": row["option_entry"] or 0,
                    "entry_time": row["entry_time"],
                }

                # Merge extra data if present
                if row["extra_data"]:
                    try:
                        extra = json.loads(row["extra_data"])
                        state.update(extra)
                    except json.JSONDecodeError:
                        pass

                return state

            # Return default state
            return {
                "status": False,
                "type": None,
                "option_id": None,
                "entry": 0,
                "sl": 0,
                "initial_sl": 0,
                "step_level": 0,
                "instrument": None,
                "lot_size": None,
                "exchange_segment_str": None,
            }

    # =========================================================================
    # ANALYTICS & REPORTING
    # =========================================================================
    def get_performance_stats(
        self, days: Optional[int] = None, instrument: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive performance statistics.

        Args:
            days: Number of days to analyze (None for all time)
            instrument: Filter by specific instrument

        Returns:
            Dictionary with performance metrics or None if no data
        """
        history = self.get_trade_history(days=days, instrument=instrument)

        if not history:
            return None

        total_trades = len(history)
        wins = [t for t in history if t["pnl"] > 0]
        losses = [t for t in history if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in history)
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))

        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = (gross_profit / len(wins)) if wins else 0
        avg_loss = (gross_loss / len(losses)) if losses else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # R-multiple stats
        r_multiples = [t.get("r_multiple", 0) for t in history if t.get("r_multiple")]
        avg_r = (sum(r_multiples) / len(r_multiples)) if r_multiples else 0

        # Best and worst trades
        best_trade = max(history, key=lambda x: x["pnl"])["pnl"] if history else 0
        worst_trade = min(history, key=lambda x: x["pnl"])["pnl"] if history else 0

        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        # Sort by exit_time for streak calculation
        sorted_history = sorted(history, key=lambda x: x["exit_time"])

        for trade in sorted_history:
            if trade["pnl"] > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_r_multiple": avg_r,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "expectancy": (avg_win * win_rate / 100)
            - (avg_loss * (100 - win_rate) / 100),
        }

    def get_instrument_summary(self) -> List[Dict[str, Any]]:
        """Get performance summary grouped by instrument"""
        with self.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    instrument,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trade_history
                GROUP BY instrument
                ORDER BY total_pnl DESC
            """
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # =========================================================================
    # MIGRATION & UTILITIES
    # =========================================================================
    def migrate_from_json(
        self, trade_history_file: str, daily_pnl_file: str, state_file: str
    ) -> Tuple[int, int, bool]:
        """
        Migrate existing JSON data to SQLite database.

        Args:
            trade_history_file: Path to trade history JSON file
            daily_pnl_file: Path to daily P&L JSON file
            state_file: Path to state JSON file

        Returns:
            Tuple of (trades_migrated, daily_pnl_records_migrated, state_migrated)
        """
        trades_migrated = 0
        pnl_migrated = 0
        state_migrated = False

        # Migrate trade history
        if os.path.exists(trade_history_file):
            try:
                with open(trade_history_file, "r") as f:
                    history = json.load(f)

                for trade in history:
                    self.save_trade(trade)
                    trades_migrated += 1

                logging.info(f"Migrated {trades_migrated} trades from JSON")
            except Exception as e:
                logging.error(f"Error migrating trade history: {e}")

        # Migrate daily P&L
        if os.path.exists(daily_pnl_file):
            try:
                with open(daily_pnl_file, "r") as f:
                    pnl_data = json.load(f)

                # If it's a single day record
                if "date" in pnl_data:
                    with self.transaction() as cursor:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO daily_pnl (date, pnl, trades, wins, losses)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            (
                                pnl_data["date"],
                                pnl_data.get("pnl", 0),
                                pnl_data.get("trades", 0),
                                pnl_data.get("wins", 0),
                                pnl_data.get("losses", 0),
                            ),
                        )
                        pnl_migrated = 1

                logging.info(f"Migrated {pnl_migrated} daily P&L records from JSON")
            except Exception as e:
                logging.error(f"Error migrating daily P&L: {e}")

        # Migrate state
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)

                self.save_state(state_data)
                state_migrated = True
                logging.info("Migrated trade state from JSON")
            except Exception as e:
                logging.error(f"Error migrating state: {e}")

        return trades_migrated, pnl_migrated, state_migrated

    def close(self):
        """Close the database connection"""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================
_db_instance: Optional[DatabaseManager] = None


def get_database(db_path: str = DEFAULT_DB_PATH) -> DatabaseManager:
    """Get or create the database manager instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance


def init_database(db_path: str = DEFAULT_DB_PATH) -> DatabaseManager:
    """Initialize the database (called at startup)"""
    return get_database(db_path)
