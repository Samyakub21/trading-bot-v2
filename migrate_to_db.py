#!/usr/bin/env python
# =============================================================================
# MIGRATE TO DATABASE - One-time migration script
# =============================================================================
# Run this script once to migrate existing JSON data to SQLite database

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    """Migrate JSON files to SQLite database"""
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    from config import config
    from database import init_database
    
    # File paths
    trade_history_file = config.TRADE_HISTORY_FILE
    daily_pnl_file = config.DAILY_PNL_FILE
    state_file = config.STATE_FILE
    
    logging.info("=" * 60)
    logging.info("TRADING BOT - DATABASE MIGRATION")
    logging.info("=" * 60)
    
    # Check if files exist
    logging.info("\nChecking existing JSON files:")
    logging.info(f"  Trade History: {trade_history_file} - {'EXISTS' if os.path.exists(trade_history_file) else 'NOT FOUND'}")
    logging.info(f"  Daily P&L: {daily_pnl_file} - {'EXISTS' if os.path.exists(daily_pnl_file) else 'NOT FOUND'}")
    logging.info(f"  State: {state_file} - {'EXISTS' if os.path.exists(state_file) else 'NOT FOUND'}")
    
    # Confirm migration
    print("\n" + "=" * 60)
    response = input("Do you want to migrate data to SQLite database? (y/n): ")
    
    if response.lower() != 'y':
        logging.info("Migration cancelled.")
        return
    
    # Initialize database
    logging.info("\nInitializing SQLite database...")
    db = init_database(config.DATABASE_PATH)
    logging.info(f"Database created at: {config.DATABASE_PATH}")
    
    # Perform migration
    logging.info("\nMigrating data...")
    trades, pnl, state = db.migrate_from_json(
        trade_history_file,
        daily_pnl_file,
        state_file
    )
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("MIGRATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"  Trades migrated: {trades}")
    logging.info(f"  Daily P&L records: {pnl}")
    logging.info(f"  State migrated: {state}")
    
    # Backup recommendation
    logging.info("\n" + "-" * 60)
    logging.info("RECOMMENDATION:")
    logging.info("  1. Verify the migration by checking the database")
    logging.info("  2. Run the bot to ensure everything works")
    logging.info("  3. Once verified, you can archive the JSON files")
    logging.info("-" * 60)
    
    # Show sample data
    stats = db.get_performance_stats()
    if stats:
        logging.info("\nDatabase Statistics:")
        logging.info(f"  Total Trades: {stats['total_trades']}")
        logging.info(f"  Win Rate: {stats['win_rate']:.1f}%")
        logging.info(f"  Total P&L: ₹{stats['total_pnl']:.2f}")


if __name__ == '__main__':
    main()
