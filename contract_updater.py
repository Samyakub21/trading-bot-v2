# =============================================================================
# CONTRACT UPDATER - Auto-fetch Current Month Futures Contracts
# =============================================================================
# Automatically updates instrument configurations with current month contracts

import os
import csv
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from config import config
from dhanhq import dhanhq

# Initialize Dhan client (dhanhq v2.0)
dhan = dhanhq(config.CLIENT_ID, config.ACCESS_TOKEN)

# Data directory for runtime files
DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# Cache file for contract data
CONTRACT_CACHE_FILE = str(DATA_DIR / "contract_cache.json")
SCRIP_MASTER_CSV = "Extras/data/api-scrip-master-detailed.csv"

# Dhan API scrip master URL
SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"


# =============================================================================
# SCRIP MASTER MANAGEMENT
# =============================================================================

def download_scrip_master(output_path: str = SCRIP_MASTER_CSV) -> bool:
    """
    Download the latest scrip master CSV from Dhan.
    Returns True if successful.
    """
    try:
        logging.info("📥 Downloading latest scrip master...")
        
        response = requests.get(SCRIP_MASTER_URL, timeout=60)
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"✅ Scrip master downloaded to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to download scrip master: {e}")
        return False


def load_scrip_master(csv_path: str = SCRIP_MASTER_CSV) -> List[Dict]:
    """
    Load scrip master CSV into memory.
    Returns list of contract dictionaries.
    """
    contracts = []
    
    if not os.path.exists(csv_path):
        logging.warning(f"Scrip master not found at {csv_path}, attempting download...")
        if not download_scrip_master(csv_path):
            return []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                contracts.append(row)
        
        logging.info(f"📊 Loaded {len(contracts)} contracts from scrip master")
        return contracts
        
    except Exception as e:
        logging.error(f"Failed to load scrip master: {e}")
        return []


# =============================================================================
# CONTRACT LOOKUP
# =============================================================================

def find_current_month_future(
    underlying: str,
    exchange: str,
    contracts: List[Dict]
) -> Optional[Dict]:
    """
    Find the current month's futures contract for an underlying.
    
    Args:
        underlying: e.g., "CRUDEOIL", "GOLD", "NIFTY"
        exchange: e.g., "MCX_COMM", "NSE_FNO" (V2 API format)
        contracts: List of contract dictionaries from scrip master
    
    Returns:
        Contract dictionary or None
    """
    today = datetime.now().date()
    
    # Filter for matching underlying and exchange futures
    matching = []
    
    # V2 API uses MCX_COMM for commodities - normalize for matching
    exchange_normalized = exchange.replace("_COMM", "").upper()
    
    for contract in contracts:
        # Check exchange
        exch = contract.get('SEM_EXM_EXCH_ID', '').upper()
        if exchange_normalized == "MCX" and exch != "MCX":
            continue
        if exchange_normalized == "NSE_FNO" and exch not in ["NSE", "NFO"]:
            continue
        if exchange_normalized == "NSE" and exch not in ["NSE", "NFO"]:
            continue
        
        # Check instrument type (futures only - not options)
        inst_type = contract.get('SEM_INSTRUMENT_NAME', '').upper()
        # FUTCOM = commodity futures, FUTIDX = index futures, FUTSTK = stock futures
        # OPTFUT/OPTIDX = options (should be excluded)
        if inst_type not in ['FUTCOM', 'FUTIDX', 'FUTSTK']:
            continue
        
        # Check underlying name - must match at the start of trading symbol
        trading_symbol = contract.get('SEM_TRADING_SYMBOL', '').upper()
        custom_symbol = contract.get('SEM_CUSTOM_SYMBOL', '').upper()
        
        # More precise matching: symbol should start with underlying followed by separator
        # e.g., "NIFTY-Jan2026-FUT" for underlying "NIFTY"
        # Avoid matching "BANKNIFTY" when looking for "NIFTY"
        underlying_upper = underlying.upper()
        symbol_matches = (
            trading_symbol.startswith(underlying_upper + '-') or  # CRUDEOIL-16Jan2026-FUT
            trading_symbol.startswith(underlying_upper + 'M-') or  # CRUDEOILM-16Jan2026-FUT (mini)
            custom_symbol.startswith(underlying_upper + ' ')  # CRUDEOIL JAN FUT
        )
        if not symbol_matches:
            continue
        
        # Parse expiry date
        expiry_str = contract.get('SEM_EXPIRY_DATE', '')
        try:
            # Try different date formats (including datetime with time)
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']:
                try:
                    expiry_date = datetime.strptime(expiry_str, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                continue
            
            # Only consider contracts expiring in the future
            if expiry_date >= today:
                contract['_expiry_date'] = expiry_date
                matching.append(contract)
                
        except Exception:
            continue
    
    if not matching:
        logging.warning(f"No matching futures found for {underlying} on {exchange}")
        return None
    
    # Sort by expiry date (nearest first)
    matching.sort(key=lambda x: x['_expiry_date'])
    
    # Return the nearest expiry (current month or next available)
    return matching[0]


def find_current_month_option_chain(
    underlying: str,
    exchange: str,
    contracts: List[Dict],
    expiry_date: Optional[str] = None
) -> List[Dict]:
    """
    Find all options for an underlying with a specific expiry.
    
    Args:
        underlying: e.g., "CRUDEOIL", "GOLD", "NIFTY"
        exchange: e.g., "MCX_COMM", "NSE_FNO" (V2 API format)
        contracts: List of contract dictionaries
        expiry_date: Optional specific expiry date
    """
    today = datetime.now().date()
    
    # V2 API uses MCX_COMM for commodities - normalize for matching
    exchange_normalized = exchange.replace("_COMM", "").upper()
    
    if expiry_date:
        target_expiry = datetime.strptime(expiry_date, '%Y-%m-%d').date()
    else:
        # Use current month's expiry
        target_expiry = None
    
    options = []
    
    for contract in contracts:
        # Check exchange
        exch = contract.get('SEM_EXM_EXCH_ID', '').upper()
        if exchange_normalized == "MCX" and exch != "MCX":
            continue
        if exchange_normalized == "NSE_FNO" and exch not in ["NSE", "NFO"]:
            continue
        if exchange_normalized == "NSE" and exch not in ["NSE", "NFO"]:
            continue
        
        # Check instrument type (options)
        inst_type = contract.get('SEM_INSTRUMENT_NAME', '').upper()
        if 'OPT' not in inst_type:
            continue
        
        # Check underlying
        trading_symbol = contract.get('SEM_TRADING_SYMBOL', '').upper()
        if underlying.upper() not in trading_symbol:
            continue
        
        # Parse expiry
        expiry_str = contract.get('SEM_EXPIRY_DATE', '')
        try:
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']:
                try:
                    contract_expiry = datetime.strptime(expiry_str, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                continue
            
            if contract_expiry < today:
                continue
            
            if target_expiry and contract_expiry != target_expiry:
                continue
            
            options.append(contract)
            
        except Exception:
            continue
    
    return options


# =============================================================================
# INSTRUMENT CONFIGURATION UPDATE
# =============================================================================

def get_updated_instrument_config(
    instrument_key: str,
    current_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Get updated configuration for an instrument with current month contract.
    
    Args:
        instrument_key: e.g., "CRUDEOIL", "GOLD"
        current_config: Current instrument configuration
    
    Returns:
        Updated configuration dictionary or None
    """
    contracts = load_scrip_master()
    if not contracts:
        logging.error("Cannot update config without scrip master")
        return None
    
    # Determine exchange
    exchange = current_config.get('exchange_segment_str', 'MCX')
    
    # Find current month future
    future_contract = find_current_month_future(instrument_key, exchange, contracts)
    
    if not future_contract:
        logging.warning(f"Could not find current month future for {instrument_key}")
        return None
    
    # Extract details
    security_id = future_contract.get('SEM_SMST_SECURITY_ID', '')
    expiry_date = future_contract.get('_expiry_date')
    
    # For lot size: prefer the API value if > 1, otherwise keep original config
    # (Dhan API returns 1.0 for some MCX commodities which is incorrect)
    lot_size_raw = future_contract.get('SEM_LOT_UNITS', 1)
    api_lot_size = int(float(lot_size_raw))
    original_lot_size = current_config.get('lot_size', 1)
    lot_size = api_lot_size if api_lot_size > 1 else original_lot_size
    
    # Build updated config
    updated_config = current_config.copy()
    updated_config['future_id'] = security_id
    updated_config['expiry_date'] = expiry_date.strftime('%Y-%m-%d')
    updated_config['lot_size'] = lot_size
    
    logging.info(f"✅ Updated {instrument_key}: Future ID={security_id}, Expiry={expiry_date}, Lot={lot_size}")
    
    return updated_config


def update_all_instruments(instruments: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Update all instrument configurations with current month contracts.
    
    Args:
        instruments: Current INSTRUMENTS dictionary
    
    Returns:
        Updated instruments dictionary
    """
    logging.info("🔄 Updating all instrument configurations...")
    
    # Download fresh scrip master
    download_scrip_master()
    
    updated = {}
    
    for key, config in instruments.items():
        updated_config = get_updated_instrument_config(key, config)
        if updated_config:
            updated[key] = updated_config
        else:
            # Keep original if update fails
            updated[key] = config
            logging.warning(f"⚠️ Keeping original config for {key}")
    
    return updated


# =============================================================================
# CONTRACT CACHE
# =============================================================================

def save_contract_cache(contracts: Dict[str, Dict]):
    """Save contract configurations to cache file"""
    cache_data = {
        'updated_at': datetime.now().isoformat(),
        'contracts': contracts
    }
    
    with open(CONTRACT_CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    logging.info(f"💾 Contract cache saved to {CONTRACT_CACHE_FILE}")


def load_contract_cache() -> Optional[Dict[str, Dict]]:
    """Load contract configurations from cache if valid"""
    if not os.path.exists(CONTRACT_CACHE_FILE):
        return None
    
    try:
        with open(CONTRACT_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        # Check if cache is from today
        updated_at = datetime.fromisoformat(cache_data['updated_at'])
        if updated_at.date() == datetime.now().date():
            logging.info("📁 Using cached contract data")
            return cache_data['contracts']
        else:
            logging.info("📁 Contract cache is stale, will refresh")
            return None
            
    except Exception as e:
        logging.error(f"Failed to load contract cache: {e}")
        return None


def is_cache_valid() -> bool:
    """Check if contract cache is valid (from today)"""
    if not os.path.exists(CONTRACT_CACHE_FILE):
        return False
    
    try:
        with open(CONTRACT_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        updated_at = datetime.fromisoformat(cache_data['updated_at'])
        return updated_at.date() == datetime.now().date()
        
    except Exception:
        return False


# =============================================================================
# AUTO-UPDATE INTEGRATION
# =============================================================================

def auto_update_instruments_on_startup(instruments: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Auto-update instruments on bot startup.
    Uses cache if valid, otherwise fetches fresh data.
    
    Args:
        instruments: Current INSTRUMENTS dictionary
    
    Returns:
        Updated instruments dictionary
    """
    # Try cache first
    cached = load_contract_cache()
    if cached:
        # Merge cached data with base config
        for key in instruments:
            if key in cached:
                instruments[key]['future_id'] = cached[key].get('future_id', instruments[key]['future_id'])
                instruments[key]['expiry_date'] = cached[key].get('expiry_date', instruments[key]['expiry_date'])
                instruments[key]['lot_size'] = cached[key].get('lot_size', instruments[key]['lot_size'])
        return instruments
    
    # Update from scrip master
    updated = update_all_instruments(instruments)
    
    # Save to cache
    save_contract_cache(updated)
    
    return updated


def schedule_daily_update(instruments: Dict[str, Dict], update_time: str = "08:55"):
    """
    Schedule daily contract update before market opens.
    
    Args:
        instruments: INSTRUMENTS dictionary to update
        update_time: Time to run update (HH:MM format)
    """
    import schedule
    import threading
    
    def do_update():
        logging.info("⏰ Running scheduled contract update...")
        updated = update_all_instruments(instruments)
        
        # Update in place
        for key, config in updated.items():
            instruments[key] = config
        
        save_contract_cache(updated)
        logging.info("✅ Scheduled contract update complete")
    
    schedule.every().day.at(update_time).do(do_update)
    
    # Run scheduler in background thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            import time
            time.sleep(60)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    logging.info(f"📅 Contract update scheduled daily at {update_time}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_next_expiry_dates(underlying: str, exchange: str, count: int = 3) -> List[str]:
    """
    Get the next N expiry dates for an underlying.
    Useful for rolling contracts.
    """
    contracts = load_scrip_master()
    today = datetime.now().date()
    
    expiry_dates = set()
    
    for contract in contracts:
        # Filter by exchange
        exch = contract.get('SEM_EXM_EXCH_ID', '').upper()
        if exchange == "MCX" and exch != "MCX":
            continue
        if exchange == "NSE_FNO" and exch not in ["NSE", "NFO"]:
            continue
        
        # Filter by instrument type
        if 'FUT' not in contract.get('SEM_INSTRUMENT_NAME', '').upper():
            continue
        
        # Filter by underlying
        if underlying.upper() not in contract.get('SEM_TRADING_SYMBOL', '').upper():
            continue
        
        # Parse expiry
        expiry_str = contract.get('SEM_EXPIRY_DATE', '')
        try:
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']:
                try:
                    expiry_date = datetime.strptime(expiry_str, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                continue
            
            if expiry_date >= today:
                expiry_dates.add(expiry_date)
                
        except Exception:
            continue
    
    # Sort and return
    sorted_dates = sorted(expiry_dates)
    return [d.strftime('%Y-%m-%d') for d in sorted_dates[:count]]


def validate_contract(security_id: str, exchange: str = "MCX") -> bool:
    """
    Validate that a security ID is still active/valid.
    """
    try:
        # Try to get quote for the security
        quote = dhan.intraday_minute_data(
            security_id,
            exchange,
            "FUTURES",
            datetime.now().strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        
        return quote.get('status') != 'failure'
        
    except Exception as e:
        logging.error(f"Contract validation error: {e}")
        return False


def get_contract_details(security_id: str) -> Optional[Dict]:
    """
    Get full details for a specific contract.
    """
    contracts = load_scrip_master()
    
    for contract in contracts:
        if contract.get('SEM_SMST_SECURITY_ID', '') == security_id:
            return contract
    
    return None


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Contract Updater CLI")
    parser.add_argument("--download", action="store_true", help="Download fresh scrip master")
    parser.add_argument("--update", action="store_true", help="Update all instrument configs")
    parser.add_argument("--show", type=str, help="Show contract details for instrument (e.g., CRUDEOIL)")
    parser.add_argument("--expiries", type=str, help="Show next expiry dates for instrument")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if args.download:
        download_scrip_master()
    
    if args.update:
        from instruments import INSTRUMENTS
        updated = update_all_instruments(INSTRUMENTS)
        save_contract_cache(updated)
        
        print("\n📊 Updated Configurations:")
        for key, config in updated.items():
            print(f"  {key}:")
            print(f"    Future ID: {config['future_id']}")
            print(f"    Expiry: {config['expiry_date']}")
            print(f"    Lot Size: {config['lot_size']}")
    
    if args.show:
        from instruments import INSTRUMENTS
        inst = INSTRUMENTS.get(args.show.upper())
        if inst:
            config = get_updated_instrument_config(args.show.upper(), inst)
            if config:
                print(f"\n📊 {args.show.upper()} Configuration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
    
    if args.expiries:
        from instruments import INSTRUMENTS
        inst = INSTRUMENTS.get(args.expiries.upper(), {})
        exchange = inst.get('exchange_segment_str', 'MCX')
        dates = get_next_expiry_dates(args.expiries.upper(), exchange)
        print(f"\n📅 Next expiry dates for {args.expiries.upper()}:")
        for i, date in enumerate(dates, 1):
            print(f"  {i}. {date}")
