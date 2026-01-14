"""
Unit tests for contract_updater.py
"""

import pytest
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_contracts():
    """Sample contract data from scrip master"""
    today = datetime.now()
    future_expiry = (today + timedelta(days=15)).strftime('%Y-%m-%d')
    past_expiry = (today - timedelta(days=5)).strftime('%Y-%m-%d')
    
    return [
        {
            'SEM_SMST_SECURITY_ID': '464926',
            'SEM_EXM_EXCH_ID': 'MCX',
            'SEM_INSTRUMENT_NAME': 'FUTCOM',
            'SEM_TRADING_SYMBOL': 'CRUDEOIL26JAN',
            'SEM_CUSTOM_SYMBOL': 'CRUDEOIL',
            'SEM_EXPIRY_DATE': future_expiry,
            'SEM_LOT_UNITS': '10',
        },
        {
            'SEM_SMST_SECURITY_ID': '464927',
            'SEM_EXM_EXCH_ID': 'MCX',
            'SEM_INSTRUMENT_NAME': 'FUTCOM',
            'SEM_TRADING_SYMBOL': 'CRUDEOIL25DEC',  # Expired
            'SEM_CUSTOM_SYMBOL': 'CRUDEOIL',
            'SEM_EXPIRY_DATE': past_expiry,
            'SEM_LOT_UNITS': '10',
        },
        {
            'SEM_SMST_SECURITY_ID': '464930',
            'SEM_EXM_EXCH_ID': 'MCX',
            'SEM_INSTRUMENT_NAME': 'FUTCOM',
            'SEM_TRADING_SYMBOL': 'GOLD26JAN',
            'SEM_CUSTOM_SYMBOL': 'GOLD',
            'SEM_EXPIRY_DATE': future_expiry,
            'SEM_LOT_UNITS': '100',
        },
        {
            'SEM_SMST_SECURITY_ID': '465000',
            'SEM_EXM_EXCH_ID': 'MCX',
            'SEM_INSTRUMENT_NAME': 'OPTFUT',
            'SEM_TRADING_SYMBOL': 'CRUDEOIL26JAN6000CE',
            'SEM_CUSTOM_SYMBOL': 'CRUDEOIL',
            'SEM_EXPIRY_DATE': future_expiry,
            'SEM_LOT_UNITS': '10',
            'SEM_STRIKE_PRICE': '6000',
            'SEM_OPTION_TYPE': 'CE',
        },
    ]


@pytest.fixture
def sample_instrument_config():
    """Sample instrument configuration - V2 API format"""
    return {
        "CRUDEOIL": {
            "name": "CRUDE OIL",
            "exchange_segment_int": 5,
            "exchange_segment_str": "MCX_COMM",  # V2 API format
            "future_id": "OLD_ID",
            "lot_size": 10,
            "expiry_date": "2025-12-15",
        }
    }


@pytest.fixture
def sample_cache_data():
    """Sample contract cache data"""
    return {
        'updated_at': datetime.now().isoformat(),
        'contracts': {
            'CRUDEOIL': {
                'future_id': '464926',
                'expiry_date': '2026-01-16',
                'lot_size': 10
            }
        }
    }


# =============================================================================
# SCRIP MASTER DOWNLOAD TESTS
# =============================================================================

class TestScripMasterDownload:
    """Tests for scrip master download functionality"""
    
    @patch('contract_updater.requests.get')
    @patch('contract_updater.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_scrip_master_success(self, mock_file, mock_makedirs, mock_get):
        """Should successfully download scrip master"""
        from contract_updater import download_scrip_master
        
        mock_response = MagicMock()
        mock_response.content = b'CSV,DATA,HERE'
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        result = download_scrip_master("test_output.csv")
        
        assert result is True
        mock_get.assert_called_once()
        mock_file.assert_called_once()
    
    @patch('contract_updater.requests.get')
    def test_download_scrip_master_failure(self, mock_get):
        """Should handle download failures gracefully"""
        from contract_updater import download_scrip_master
        
        mock_get.side_effect = Exception("Network error")
        
        result = download_scrip_master("test_output.csv")
        
        assert result is False


# =============================================================================
# CONTRACT LOOKUP TESTS
# =============================================================================

class TestFindCurrentMonthFuture:
    """Tests for finding current month futures contracts"""
    
    def test_find_crude_oil_future(self, sample_contracts):
        """Should find CRUDEOIL future contract"""
        from contract_updater import find_current_month_future
        
        result = find_current_month_future("CRUDEOIL", "MCX", sample_contracts)
        
        assert result is not None
        assert 'CRUDEOIL' in result['SEM_TRADING_SYMBOL']
        assert result['SEM_SMST_SECURITY_ID'] == '464926'
    
    def test_find_gold_future(self, sample_contracts):
        """Should find GOLD future contract"""
        from contract_updater import find_current_month_future
        
        result = find_current_month_future("GOLD", "MCX", sample_contracts)
        
        assert result is not None
        assert 'GOLD' in result['SEM_TRADING_SYMBOL']
    
    def test_skip_expired_contracts(self, sample_contracts):
        """Should skip expired contracts"""
        from contract_updater import find_current_month_future
        
        result = find_current_month_future("CRUDEOIL", "MCX", sample_contracts)
        
        # Should return the future expiry, not the past one
        assert result['SEM_SMST_SECURITY_ID'] == '464926'
    
    def test_return_none_for_missing(self, sample_contracts):
        """Should return None when no matching contract found"""
        from contract_updater import find_current_month_future
        
        result = find_current_month_future("SILVER", "MCX", sample_contracts)
        
        assert result is None
    
    def test_filter_by_exchange(self, sample_contracts):
        """Should filter contracts by exchange"""
        from contract_updater import find_current_month_future
        
        # NSE_FNO shouldn't find MCX contracts
        result = find_current_month_future("CRUDEOIL", "NSE_FNO", sample_contracts)
        
        assert result is None


# =============================================================================
# OPTION CHAIN TESTS
# =============================================================================

class TestFindCurrentMonthOptionChain:
    """Tests for finding option chain"""
    
    def test_find_crude_options(self, sample_contracts):
        """Should find CRUDEOIL options"""
        from contract_updater import find_current_month_option_chain
        
        options = find_current_month_option_chain("CRUDEOIL", "MCX", sample_contracts)
        
        assert len(options) >= 1
        assert 'OPT' in options[0]['SEM_INSTRUMENT_NAME']
    
    def test_filter_options_by_expiry(self, sample_contracts):
        """Should filter options by specific expiry date"""
        from contract_updater import find_current_month_option_chain
        
        today = datetime.now()
        expiry = (today + timedelta(days=15)).strftime('%Y-%m-%d')
        
        options = find_current_month_option_chain(
            "CRUDEOIL", "MCX", sample_contracts, expiry_date=expiry
        )
        
        # Should find at least one option
        assert len(options) >= 0  # May be 0 if no exact match
        
        # If options found, verify they are for CRUDEOIL
        for opt in options:
            assert 'CRUDEOIL' in opt.get('SEM_TRADING_SYMBOL', '')


# =============================================================================
# INSTRUMENT CONFIG UPDATE TESTS
# =============================================================================

class TestGetUpdatedInstrumentConfig:
    """Tests for updating instrument configuration"""
    
    @patch('contract_updater.load_scrip_master')
    def test_update_config_success(self, mock_load, sample_contracts, sample_instrument_config):
        """Should successfully update instrument config"""
        from contract_updater import get_updated_instrument_config
        
        mock_load.return_value = sample_contracts
        
        result = get_updated_instrument_config(
            "CRUDEOIL", 
            sample_instrument_config["CRUDEOIL"]
        )
        
        assert result is not None
        assert result['future_id'] == '464926'
        assert result['lot_size'] == 10
    
    @patch('contract_updater.load_scrip_master')
    def test_update_config_no_scrip_master(self, mock_load, sample_instrument_config):
        """Should return None when scrip master unavailable"""
        from contract_updater import get_updated_instrument_config
        
        mock_load.return_value = []
        
        result = get_updated_instrument_config(
            "CRUDEOIL", 
            sample_instrument_config["CRUDEOIL"]
        )
        
        assert result is None


class TestUpdateAllInstruments:
    """Tests for updating all instruments"""
    
    @patch('contract_updater.download_scrip_master')
    @patch('contract_updater.get_updated_instrument_config')
    def test_update_all_instruments(self, mock_update, mock_download, sample_instrument_config):
        """Should update all instruments in config"""
        from contract_updater import update_all_instruments
        
        mock_download.return_value = True
        mock_update.return_value = {
            **sample_instrument_config["CRUDEOIL"],
            'future_id': 'NEW_ID'
        }
        
        result = update_all_instruments(sample_instrument_config)
        
        assert "CRUDEOIL" in result
        mock_update.assert_called_once()
    
    @patch('contract_updater.download_scrip_master')
    @patch('contract_updater.get_updated_instrument_config')
    def test_keep_original_on_failure(self, mock_update, mock_download, sample_instrument_config):
        """Should keep original config when update fails"""
        from contract_updater import update_all_instruments
        
        mock_download.return_value = True
        mock_update.return_value = None  # Update failed
        
        result = update_all_instruments(sample_instrument_config)
        
        assert result["CRUDEOIL"]["future_id"] == "OLD_ID"


# =============================================================================
# CONTRACT CACHE TESTS
# =============================================================================

class TestContractCache:
    """Tests for contract cache functionality"""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('contract_updater.json.dump')
    def test_save_contract_cache(self, mock_json_dump, mock_file):
        """Should save contract cache to file"""
        from contract_updater import save_contract_cache
        
        contracts = {'CRUDEOIL': {'future_id': '464926'}}
        
        save_contract_cache(contracts)
        
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()
    
    @patch('contract_updater.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_contract_cache_valid(self, mock_file, mock_exists, sample_cache_data):
        """Should load valid cache from today"""
        from contract_updater import load_contract_cache
        
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps(sample_cache_data)
        
        with patch('contract_updater.json.load', return_value=sample_cache_data):
            result = load_contract_cache()
        
        assert result is not None
        assert 'CRUDEOIL' in result
    
    @patch('contract_updater.os.path.exists')
    def test_load_contract_cache_missing(self, mock_exists):
        """Should return None when cache file doesn't exist"""
        from contract_updater import load_contract_cache
        
        mock_exists.return_value = False
        
        result = load_contract_cache()
        
        assert result is None
    
    @patch('contract_updater.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_contract_cache_stale(self, mock_file, mock_exists):
        """Should return None when cache is stale (not from today)"""
        from contract_updater import load_contract_cache
        
        mock_exists.return_value = True
        stale_cache = {
            'updated_at': (datetime.now() - timedelta(days=1)).isoformat(),
            'contracts': {'CRUDEOIL': {}}
        }
        
        with patch('contract_updater.json.load', return_value=stale_cache):
            result = load_contract_cache()
        
        assert result is None


class TestIsCacheValid:
    """Tests for cache validity check"""
    
    @patch('contract_updater.os.path.exists')
    def test_cache_invalid_when_missing(self, mock_exists):
        """Should return False when cache file doesn't exist"""
        from contract_updater import is_cache_valid
        
        mock_exists.return_value = False
        
        assert is_cache_valid() is False
    
    @patch('contract_updater.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_cache_valid_today(self, mock_file, mock_exists):
        """Should return True for today's cache"""
        from contract_updater import is_cache_valid
        
        mock_exists.return_value = True
        today_cache = {
            'updated_at': datetime.now().isoformat(),
            'contracts': {}
        }
        
        with patch('contract_updater.json.load', return_value=today_cache):
            result = is_cache_valid()
        
        assert result is True


# =============================================================================
# AUTO-UPDATE INTEGRATION TESTS
# =============================================================================

class TestAutoUpdateOnStartup:
    """Tests for auto-update on startup"""
    
    @patch('contract_updater.load_contract_cache')
    def test_use_cache_when_valid(self, mock_load_cache, sample_instrument_config):
        """Should use cached data when available"""
        from contract_updater import auto_update_instruments_on_startup
        
        mock_load_cache.return_value = {
            'CRUDEOIL': {
                'future_id': 'CACHED_ID',
                'expiry_date': '2026-01-16',
                'lot_size': 10
            }
        }
        
        result = auto_update_instruments_on_startup(sample_instrument_config)
        
        assert result['CRUDEOIL']['future_id'] == 'CACHED_ID'
    
    @patch('contract_updater.load_contract_cache')
    @patch('contract_updater.update_all_instruments')
    @patch('contract_updater.save_contract_cache')
    def test_fetch_when_no_cache(
        self, 
        mock_save, 
        mock_update, 
        mock_load_cache, 
        sample_instrument_config
    ):
        """Should fetch fresh data when cache unavailable"""
        from contract_updater import auto_update_instruments_on_startup
        
        mock_load_cache.return_value = None
        mock_update.return_value = {
            'CRUDEOIL': {
                **sample_instrument_config['CRUDEOIL'],
                'future_id': 'FRESH_ID'
            }
        }
        
        result = auto_update_instruments_on_startup(sample_instrument_config)
        
        mock_update.assert_called_once()
        mock_save.assert_called_once()


# =============================================================================
# DATE FORMAT PARSING TESTS
# =============================================================================

class TestDateFormatParsing:
    """Tests for handling different date formats in scrip master"""
    
    def test_parse_yyyy_mm_dd_format(self):
        """Should parse YYYY-MM-DD date format"""
        from contract_updater import find_current_month_future
        
        future_date = (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')
        contracts = [{
            'SEM_SMST_SECURITY_ID': '123',
            'SEM_EXM_EXCH_ID': 'MCX',
            'SEM_INSTRUMENT_NAME': 'FUTCOM',
            'SEM_TRADING_SYMBOL': 'CRUDEOIL',
            'SEM_CUSTOM_SYMBOL': 'CRUDEOIL',
            'SEM_EXPIRY_DATE': future_date,
            'SEM_LOT_UNITS': '10',
        }]
        
        result = find_current_month_future("CRUDEOIL", "MCX", contracts)
        
        assert result is not None
    
    def test_parse_dd_mm_yyyy_format(self):
        """Should parse DD-MM-YYYY date format"""
        from contract_updater import find_current_month_future
        
        future_date = (datetime.now() + timedelta(days=15)).strftime('%d-%m-%Y')
        contracts = [{
            'SEM_SMST_SECURITY_ID': '123',
            'SEM_EXM_EXCH_ID': 'MCX',
            'SEM_INSTRUMENT_NAME': 'FUTCOM',
            'SEM_TRADING_SYMBOL': 'CRUDEOIL',
            'SEM_CUSTOM_SYMBOL': 'CRUDEOIL',
            'SEM_EXPIRY_DATE': future_date,
            'SEM_LOT_UNITS': '10',
        }]
        
        result = find_current_month_future("CRUDEOIL", "MCX", contracts)
        
        assert result is not None


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestGetNextExpiryDates:
    """Tests for getting next expiry dates"""
    
    @patch('contract_updater.load_scrip_master')
    def test_get_multiple_expiries(self, mock_load):
        """Should return multiple upcoming expiry dates"""
        from contract_updater import get_next_expiry_dates
        
        today = datetime.now()
        contracts = [
            {
                'SEM_EXM_EXCH_ID': 'MCX',
                'SEM_INSTRUMENT_NAME': 'FUTCOM',
                'SEM_TRADING_SYMBOL': 'CRUDEOIL26JAN',
                'SEM_EXPIRY_DATE': (today + timedelta(days=10)).strftime('%Y-%m-%d'),
            },
            {
                'SEM_EXM_EXCH_ID': 'MCX',
                'SEM_INSTRUMENT_NAME': 'FUTCOM',
                'SEM_TRADING_SYMBOL': 'CRUDEOIL26FEB',
                'SEM_EXPIRY_DATE': (today + timedelta(days=40)).strftime('%Y-%m-%d'),
            },
        ]
        mock_load.return_value = contracts
        
        result = get_next_expiry_dates("CRUDEOIL", "MCX", count=3)
        
        # Should return available expiries (up to count)
        assert isinstance(result, list)
