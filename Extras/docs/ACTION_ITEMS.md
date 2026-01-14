# Action Items - Code Review Follow-up

This document contains prioritized action items based on the comprehensive code review.

---

## ✅ Completed Items

1. **Fixed unused global declarations** in `socket_handler.py`
   - Removed 3 unused global declarations (F824 warnings)
   - Status: ✅ Complete

2. **Fixed failing test** in `test_utils.py`
   - Fixed TestMarketHours::test_market_open_during_hours
   - Status: ✅ Complete

3. **All tests passing**
   - 146/146 tests now passing (100%)
   - Status: ✅ Complete

---

## 🔴 High Priority (Recommended for Next Sprint)

### 1. Complete State Management Migration
**Current Status:** Partially migrated  
**Issue:** Some global variables still exist in `socket_handler.py`  
**Action:**
```python
# Migrate these remaining globals to state_stores.py:
LATEST_LTP = 0
OPTION_LTP = 0
LAST_TICK_TIME = datetime.now()
LAST_OPTION_TICK_TIME = datetime.now()
INSTRUMENT_LTP = {}
```
**Benefit:** Improves thread safety and code maintainability  
**Estimated Effort:** 2-3 hours  
**Files to modify:** `socket_handler.py`, existing code already uses `state_stores.py`

### 2. Add Structured Logging
**Current Status:** Using basic logging  
**Issue:** No log levels or structured logging  
**Action:**
```python
# Replace basic logging with structured logging
import structlog
logger = structlog.get_logger()
logger.info("trade_entry", 
    instrument=instrument, 
    price=price, 
    quantity=quantity
)
```
**Benefit:** Better debugging, monitoring, and observability  
**Estimated Effort:** 4-6 hours  
**Files to modify:** All Python files with logging

### 3. Standardize Exception Handling
**Current Status:** Mix of broad and specific exception handling  
**Issue:** Some `except:` blocks catch all exceptions  
**Action:**
```python
# Replace bare except:
try:
    risky_operation()
except Exception as e:
    logger.error("operation_failed", error=str(e))
    # Handle or re-raise as appropriate
```
**Benefit:** Better error tracking and debugging  
**Estimated Effort:** 2-3 hours  
**Files to modify:** `contract_updater.py`, `dashboard.py`, `utils.py`

---

## 🟡 Medium Priority (Consider for Future Releases)

### 4. Refactor Large Modules
**Current Status:** Some files are >500 lines  
**Action:**
- Split `dashboard.py` (1031 lines) into:
  - `dashboard_auth.py` - Authentication logic
  - `dashboard_metrics.py` - Metrics and monitoring
  - `dashboard_ui.py` - UI components
- Split `scanner.py` (745 lines) into:
  - `signal_analyzer.py` - Signal analysis logic
  - `trade_executor.py` - Trade execution
  - `scanner.py` - Main coordination

**Benefit:** Improved maintainability and testability  
**Estimated Effort:** 6-8 hours

### 5. Add Type Hints to All Functions
**Current Status:** ~70% coverage  
**Action:**
- Add type hints to remaining functions in older modules
- Run mypy for type checking
```bash
pip install mypy
mypy --strict *.py
```
**Benefit:** Better IDE support, catch type errors early  
**Estimated Effort:** 4-6 hours

### 6. Add API Documentation
**Current Status:** In-code docstrings only  
**Action:**
```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Generate documentation
sphinx-quickstart docs/
sphinx-apidoc -o docs/source .
cd docs && make html
```
**Benefit:** Better onboarding for new developers  
**Estimated Effort:** 4-6 hours

### 7. Move Magic Numbers to Constants
**Current Status:** Some hardcoded values in code  
**Action:**
```python
# In config.py or constants.py
HEARTBEAT_TIMEOUT_SECONDS = 30
RECONNECT_DELAY_SECONDS = 5
MIN_TICK_INTERVAL_MS = 100

# Replace in code:
socket_ok = SOCKET_HEALTHY.wait(timeout=30)
# With:
socket_ok = SOCKET_HEALTHY.wait(timeout=HEARTBEAT_TIMEOUT_SECONDS)
```
**Benefit:** Easier to tune parameters, better code clarity  
**Estimated Effort:** 2-3 hours

---

## 🟢 Low Priority (Nice to Have)

### 8. Add Database Support
**Current Status:** JSON/CSV file storage  
**Action:**
- Add SQLite for local development
- Add PostgreSQL support for production
- Create migration scripts
```python
# Use SQLAlchemy for ORM
from sqlalchemy import create_engine
engine = create_engine('sqlite:///trading_bot.db')
```
**Benefit:** Better query performance, transaction support  
**Estimated Effort:** 12-16 hours

### 9. Add Architecture Diagrams
**Current Status:** No visual documentation  
**Action:**
- Create component diagram
- Create sequence diagrams for key flows
- Add to documentation
**Tools:** draw.io, Mermaid, or PlantUML  
**Benefit:** Better understanding for new developers  
**Estimated Effort:** 3-4 hours

### 10. Implement Retry Decorators
**Current Status:** Manual retry logic in some places  
**Action:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_api():
    return dhan.get_positions()
```
**Benefit:** Consistent retry logic, less code duplication  
**Estimated Effort:** 2-3 hours

### 11. Add Performance Benchmarks
**Current Status:** No performance testing  
**Action:**
- Create benchmark suite
- Measure key operations (signal analysis, order placement)
- Document acceptable performance thresholds
**Benefit:** Catch performance regressions  
**Estimated Effort:** 4-6 hours

### 12. Enhance Monitoring
**Current Status:** Basic Prometheus metrics in dashboard  
**Action:**
- Export metrics to Prometheus/Grafana
- Add alerting rules
- Create monitoring dashboard
**Benefit:** Better observability in production  
**Estimated Effort:** 6-8 hours

---

## 📋 Testing Improvements

### 13. Add Integration Tests for External APIs
**Current Status:** Mocked API calls in tests  
**Action:**
- Add optional integration tests with real API (test environment)
- Use pytest markers to separate unit and integration tests
```python
@pytest.mark.integration
def test_real_api_connection():
    # Test with real Dhan API
```
**Benefit:** Catch API changes early  
**Estimated Effort:** 4-6 hours

### 14. Add Load Testing
**Current Status:** No load testing  
**Action:**
- Use locust or pytest-benchmark
- Test concurrent order handling
- Test WebSocket performance under load
**Benefit:** Ensure system handles peak loads  
**Estimated Effort:** 4-6 hours

---

## 🔒 Security Enhancements

### 15. Implement Secret Rotation
**Current Status:** Static credentials  
**Action:**
- Add support for credential refresh
- Implement token rotation strategy
- Add expiration handling
**Benefit:** Better security for long-running deployments  
**Estimated Effort:** 6-8 hours

### 16. Add Rate Limiting
**Current Status:** No rate limiting on Telegram alerts  
**Action:**
```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=30, period=60)  # 30 calls per minute
def send_alert(msg: str):
    # Send Telegram alert
```
**Benefit:** Prevent API abuse  
**Estimated Effort:** 1-2 hours

---

## 📝 Documentation Improvements

### 17. Add Troubleshooting Guide
**Current Status:** Basic troubleshooting in README  
**Action:**
- Document common errors and solutions
- Add FAQ section
- Include debugging tips
**Benefit:** Faster issue resolution  
**Estimated Effort:** 2-3 hours

### 18. Add Trading Strategy Documentation
**Current Status:** Strategy embedded in code  
**Action:**
- Document signal generation logic
- Explain trailing stop-loss algorithm
- Document risk management rules
**Benefit:** Better understanding of bot behavior  
**Estimated Effort:** 3-4 hours

---

## 🎯 Summary of Priorities

**Sprint 1 (High Priority):**
1. Complete state management migration
2. Add structured logging
3. Standardize exception handling

**Sprint 2 (Medium Priority):**
4. Refactor large modules
5. Add type hints
6. Add API documentation
7. Move magic numbers to constants

**Sprint 3 (Low Priority):**
8. Add database support
9. Add architecture diagrams
10. Implement retry decorators
11-18. Other enhancements as needed

---

## 📊 Estimated Total Effort

- High Priority: ~8-12 hours
- Medium Priority: ~16-21 hours  
- Low Priority: ~40-50 hours

**Recommendation:** Focus on high-priority items first. They provide the most value with the least effort.

---

*Generated by GitHub Copilot Code Review Agent*  
*Last Updated: January 11, 2026*
