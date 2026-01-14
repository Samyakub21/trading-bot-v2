"""
Algo Trading Dashboard with Authentication, WebSocket Status, and Prometheus Metrics
✨ Modern Gen-Z UI with glassmorphism and professional design
Fixed issues:
- Removed st.rerun() infinite loop (using st.fragment for selective refresh)
- Updated file paths to new combined naming scheme
- Added basic authentication
- Improved error handling for missing files
- Upgraded to bcrypt password hashing with secure password policy
- Modern UI with dark theme, glassmorphism effects, and animations
"""

import streamlit as st
import pandas as pd
import json
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
import threading

# Import instruments for manual trade
from instruments import INSTRUMENTS, get_instruments_to_scan

# Trading config file path for live updates
TRADING_CONFIG_FILE = Path(__file__).parent / 'trading_config.json'

# =============================================================================
# 🎨 MODERN UI THEME & STYLES
# =============================================================================

MODERN_CSS = """
<style>
/* ============================================
   🎨 MODERN GEN-Z TRADING DASHBOARD THEME
   ============================================ */

/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Root variables */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    --danger-gradient: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    --dark-bg: #0f0f1a;
    --card-bg: rgba(255, 255, 255, 0.05);
    --glass-bg: rgba(255, 255, 255, 0.08);
    --text-primary: #ffffff;
    --text-secondary: rgba(255, 255, 255, 0.7);
    --border-glass: rgba(255, 255, 255, 0.1);
    --shadow-glow: 0 8px 32px rgba(102, 126, 234, 0.25);
    --neon-blue: #00d4ff;
    --neon-purple: #b24bf3;
    --neon-green: #00ff88;
    --neon-red: #ff3366;
}

/* Global styles */
.stApp {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 15, 26, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* Main content area */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 100%;
}

/* Metric cards - Modern glassmorphism */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    border-color: rgba(102, 126, 234, 0.5) !important;
}

[data-testid="stMetric"] label {
    color: rgba(255, 255, 255, 0.6) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ffffff 0%, rgba(255, 255, 255, 0.8) 100%);
    -webkit-background-clip: text;
    background-clip: text;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Positive delta styling */
[data-testid="stMetricDelta"] svg[data-testid="stArrowUp"] {
    color: #00ff88 !important;
}

[data-testid="stMetricDelta"]:has(svg[data-testid="stArrowUp"]) {
    color: #00ff88 !important;
}

/* Headers */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

h1 {
    font-size: 2.5rem !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.5rem !important;
}

/* Buttons - Modern style */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    text-transform: none !important;
    letter-spacing: 0.3px !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(56, 239, 125, 0.6) !important;
    background: linear-gradient(135deg, #38ef7d 0%, #11998e 100%) !important;
}

/* Secondary/Danger button */
.stButton > button[kind="secondary"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.1) 100%) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Emergency button in sidebar */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%) !important;
    box-shadow: 0 4px 15px rgba(235, 51, 73, 0.4) !important;
    animation: pulse-danger 2s infinite !important;
}

@keyframes pulse-danger {
    0%, 100% { box-shadow: 0 4px 15px rgba(235, 51, 73, 0.4); }
    50% { box-shadow: 0 4px 25px rgba(235, 51, 73, 0.7); }
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
}

/* Select boxes */
[data-testid="stSelectbox"] {
    background: transparent !important;
}

/* Sliders */
.stSlider > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

.stSlider > div > div > div > div {
    background: white !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

/* Checkboxes */
.stCheckbox > label > div[data-testid="stCheckbox"] {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    gap: 0.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.6) !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: white !important;
    background: rgba(255, 255, 255, 0.1) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding: 1.5rem 0 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 1rem !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(102, 126, 234, 0.5) !important;
}

/* Data frames */
.stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stDataFrame"] > div {
    background: rgba(255, 255, 255, 0.03) !important;
}

/* Alerts */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
}

[data-testid="stAlert"][data-baseweb="notification"] {
    background: rgba(255, 255, 255, 0.05) !important;
}

/* Success alert */
.element-container:has([data-testid="stAlert"]) .stAlert > div:first-child {
    border-radius: 12px !important;
}

/* Info box */
.stInfo {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
}

/* Warning box */
.stWarning {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.2) 0%, rgba(255, 152, 0, 0.2) 100%) !important;
    border: 1px solid rgba(255, 193, 7, 0.3) !important;
}

/* Error box */
.stError {
    background: linear-gradient(135deg, rgba(235, 51, 73, 0.2) 0%, rgba(244, 92, 67, 0.2) 100%) !important;
    border: 1px solid rgba(235, 51, 73, 0.3) !important;
}

/* Success box */
.stSuccess {
    background: linear-gradient(135deg, rgba(17, 153, 142, 0.2) 0%, rgba(56, 239, 125, 0.2) 100%) !important;
    border: 1px solid rgba(56, 239, 125, 0.3) !important;
}

/* Progress bars */
.stProgress > div > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}

.stProgress > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 10px !important;
}

/* Divider */
hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
    margin: 2rem 0 !important;
}

/* Radio buttons */
.stRadio > div {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stRadio > div > div > label {
    color: rgba(255, 255, 255, 0.8) !important;
    font-weight: 500 !important;
}

/* Plotly charts */
.js-plotly-plot {
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Custom dashboard header */
.dashboard-header {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

/* Status indicator animations */
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.4); }
    50% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.4); }
    50% { box-shadow: 0 0 0 10px rgba(255, 51, 102, 0); }
}

.status-active {
    animation: pulse-green 2s infinite;
}

.status-inactive {
    animation: pulse-red 2s infinite;
}

/* Instrument cards */
.instrument-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.instrument-card:hover {
    transform: translateX(5px);
    border-color: rgba(102, 126, 234, 0.5);
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Caption styling */
.stCaption {
    color: rgba(255, 255, 255, 0.5) !important;
}

/* Form styling */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* Login form special styling */
.login-box {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 24px;
    padding: 3rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

/* Toggle switch styling */
.toggle-container {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.toggle-container:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(102, 126, 234, 0.3);
}

/* Priority badge */
.priority-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.75rem;
}

.priority-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
.priority-2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); color: #000; }
.priority-3 { background: linear-gradient(135deg, #CD7F32, #8B4513); color: #fff; }
.priority-4 { background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; }
.priority-5 { background: linear-gradient(135deg, #11998e, #38ef7d); color: #fff; }
.priority-6 { background: linear-gradient(135deg, #f093fb, #f5576c); color: #fff; }

/* Floating action effect */
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}

.floating {
    animation: float 3s ease-in-out infinite;
}

/* Glow effect for important elements */
.glow-effect {
    box-shadow: 0 0 20px rgba(102, 126, 234, 0.4), 0 0 40px rgba(102, 126, 234, 0.2);
}

/* Live indicator */
.live-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(0, 255, 136, 0.15);
    border: 1px solid rgba(0, 255, 136, 0.3);
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #00ff88;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: #00ff88;
    border-radius: 50%;
    animation: pulse-green 1.5s infinite;
}

/* Number input arrows fix */
.stNumberInput button {
    background: rgba(255, 255, 255, 0.1) !important;
    border: none !important;
    color: white !important;
}

.stNumberInput button:hover {
    background: rgba(102, 126, 234, 0.3) !important;
}
</style>
"""

# Try to import bcrypt, fall back to hashlib if not available
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    import hashlib
    BCRYPT_AVAILABLE = False
    print("Warning: bcrypt not installed. Using SHA256 fallback (less secure). Run: pip install bcrypt")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Password Policy Configuration
PASSWORD_POLICY = {
    "min_length": 12,
    "require_uppercase": True,
    "require_lowercase": True,
    "require_digit": True,
    "require_special": True,
    "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?"
}

# =============================================================================
# SECURE PASSWORD HASHING
# =============================================================================

def hash_password_bcrypt(password: str) -> str:
    """Hash password using bcrypt (secure)"""
    if BCRYPT_AVAILABLE:
        salt = bcrypt.gensalt(rounds=12)  # Cost factor of 12
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    else:
        # Fallback to SHA256 with salt (less secure but better than plain SHA256)
        import hashlib
        import secrets
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${hashed.hex()}"


def verify_password_bcrypt(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash"""
    if BCRYPT_AVAILABLE:
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    else:
        # Fallback verification for PBKDF2
        try:
            import hashlib
            salt, stored_hash = hashed.split('$')
            new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return new_hash.hex() == stored_hash
        except Exception:
            return False


def validate_password_policy(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password against security policy.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: List[str] = []
    
    if len(password) < PASSWORD_POLICY["min_length"]:
        errors.append(f"Password must be at least {PASSWORD_POLICY['min_length']} characters long")
    
    if PASSWORD_POLICY["require_uppercase"] and not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    if PASSWORD_POLICY["require_lowercase"] and not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    if PASSWORD_POLICY["require_digit"] and not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    if PASSWORD_POLICY["require_special"]:
        special_pattern = f"[{re.escape(PASSWORD_POLICY['special_chars'])}]"
        if not re.search(special_pattern, password):
            errors.append(f"Password must contain at least one special character ({PASSWORD_POLICY['special_chars']})")
    
    return len(errors) == 0, errors


# Pre-hashed passwords using bcrypt (generate new ones with hash_password_bcrypt())
# IMPORTANT: Replace these with your own secure passwords!
# To generate a new hash, run in Python:
#   import bcrypt; print(bcrypt.hashpw(b"YourSecurePassword123!", bcrypt.gensalt(rounds=12)).decode())
#
# Default credentials (CHANGE THESE IN PRODUCTION!):
#   admin / Admin@Secure123!
#   trader / Trader@Secure456!

DASHBOARD_USERS = {
    # These are bcrypt hashes - regenerate for your own passwords using generate_password_hash.py
    # Default credentials (CHANGE IN PRODUCTION!):
    #   admin / Admin@Secure123!
    #   trader / Trader@Secure456!
    "admin": "$2b$12$gJLo70nFYFUahRt13a5xNexPtar5Y4/eHCO75qRscGxWP9uGbxPIm",
    "trader": "$2b$12$y47DcZCtOSpqKFFB8asZveb9BFrz49XuyVcorLY7bTJ49WtvDv61q",
}

# File Paths - Updated to new combined naming scheme (matching Tradebot.py)
DATA_DIR = Path(__file__).parent / 'data'
STATE_FILE = DATA_DIR / "trade_state_active.json"
DAILY_PNL_FILE = DATA_DIR / "daily_pnl_combined.json"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history_combined.json"
WEBSOCKET_STATUS_FILE = DATA_DIR / "websocket_status.json"

# Prometheus metrics port
PROMETHEUS_PORT = 8000

# Instruments list for quick reference (use INSTRUMENTS from instruments.py for full config)
INSTRUMENTS_LIST = ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "NIFTY", "BANKNIFTY"]

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not installed. Metrics disabled.")

# Initialize Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Use try/except to handle duplicate registration on streamlit reruns
    try:
        # Trading metrics
        TRADE_COUNTER = Counter(
            'trading_bot_trades_total',
            'Total number of trades executed',
            ['instrument', 'trade_type', 'outcome']
        )
        PNL_GAUGE = Gauge(
            'trading_bot_daily_pnl',
            'Current daily profit/loss in INR',
            ['instrument']
        )
        TOTAL_PNL_GAUGE = Gauge(
            'trading_bot_total_pnl',
            'Total cumulative P&L'
        )
        WIN_RATE_GAUGE = Gauge(
            'trading_bot_win_rate',
            'Current win rate percentage',
            ['instrument']
        )
        ACTIVE_TRADE_GAUGE = Gauge(
            'trading_bot_active_trade',
            'Whether there is an active trade (1=yes, 0=no)'
        )
        
        # WebSocket metrics
        WEBSOCKET_CONNECTED = Gauge(
            'trading_bot_websocket_connected',
            'WebSocket connection status (1=connected, 0=disconnected)'
        )
        WEBSOCKET_LATENCY = Histogram(
            'trading_bot_websocket_latency_ms',
            'WebSocket message latency in milliseconds',
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        )
        WEBSOCKET_MESSAGES = Counter(
            'trading_bot_websocket_messages_total',
            'Total WebSocket messages received',
            ['message_type']
        )
        WEBSOCKET_ERRORS = Counter(
            'trading_bot_websocket_errors_total',
            'Total WebSocket errors'
        )
        
        # Dashboard metrics
        DASHBOARD_LOGINS = Counter(
            'dashboard_logins_total',
            'Total dashboard login attempts',
            ['status']
        )
        
    except ValueError:
        # Metrics already registered (happens on streamlit rerun)
        pass

# Start Prometheus HTTP server (only once)
_prometheus_started = False

def start_prometheus_server():
    """Start Prometheus metrics server in background thread"""
    global _prometheus_started
    if PROMETHEUS_AVAILABLE and not _prometheus_started:
        try:
            start_http_server(PROMETHEUS_PORT)
            _prometheus_started = True
            print(f"Prometheus metrics available at http://localhost:{PROMETHEUS_PORT}/metrics")
        except OSError:
            # Port already in use (server already running)
            _prometheus_started = True

# =============================================================================
# AUTHENTICATION
# =============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt (or fallback)"""
    return hash_password_bcrypt(password)


def check_credentials(username: str, password: str) -> bool:
    """Verify user credentials using bcrypt"""
    if username in DASHBOARD_USERS:
        stored_hash = DASHBOARD_USERS[username]
        return verify_password_bcrypt(password, stored_hash)
    return False


def get_password_requirements() -> str:
    """Get formatted password requirements for display"""
    requirements = []
    requirements.append(f"• At least {PASSWORD_POLICY['min_length']} characters")
    if PASSWORD_POLICY["require_uppercase"]:
        requirements.append("• At least one uppercase letter (A-Z)")
    if PASSWORD_POLICY["require_lowercase"]:
        requirements.append("• At least one lowercase letter (a-z)")
    if PASSWORD_POLICY["require_digit"]:
        requirements.append("• At least one digit (0-9)")
    if PASSWORD_POLICY["require_special"]:
        requirements.append(f"• At least one special character")
    return "\n".join(requirements)


# Rate limiting for failed login attempts
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300  # 5 minutes


def check_login_rate_limit() -> Tuple[bool, int]:
    """
    Check if login is rate limited due to failed attempts.
    
    Returns:
        Tuple of (is_allowed, seconds_remaining)
    """
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
        st.session_state.last_failed_attempt = None
        st.session_state.lockout_until = None
    
    # Check if currently locked out
    if st.session_state.lockout_until:
        remaining = (st.session_state.lockout_until - datetime.now()).total_seconds()
        if remaining > 0:
            return False, int(remaining)
        else:
            # Lockout expired, reset
            st.session_state.login_attempts = 0
            st.session_state.lockout_until = None
    
    return True, 0


def record_failed_login() -> None:
    """Record a failed login attempt and potentially trigger lockout"""
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    
    st.session_state.login_attempts += 1
    st.session_state.last_failed_attempt = datetime.now()
    
    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.session_state.lockout_until = datetime.now() + timedelta(seconds=LOGIN_LOCKOUT_SECONDS)


def reset_login_attempts() -> None:
    """Reset login attempts after successful login"""
    st.session_state.login_attempts = 0
    st.session_state.lockout_until = None


def login_form():
    """Display login form and handle authentication with rate limiting"""
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .login-container {
            max-width: 450px;
            margin: 60px auto;
            padding: 0;
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-header h1 {
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .login-subtitle {
            color: rgba(255, 255, 255, 0.6);
            font-size: 1rem;
        }
        .security-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(56, 239, 125, 0.15);
            border: 1px solid rgba(56, 239, 125, 0.3);
            border-radius: 20px;
            font-size: 0.75rem;
            color: #38ef7d;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="login-header">
                <h1>🚀 TradingBot</h1>
                <p class="login-subtitle">Algo Trading Dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Check rate limit
        is_allowed, lockout_remaining = check_login_rate_limit()
        
        if not is_allowed:
            minutes = lockout_remaining // 60
            seconds = lockout_remaining % 60
            st.error(f"🔒 Too many failed attempts. Please try again in {minutes}m {seconds}s")
            st.info("💡 If you forgot your password, contact the administrator.")
            return
        
        with st.form("login_form"):
            st.markdown("#### 🔐 Sign In")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns([3, 1])
            with col_btn1:
                submit = st.form_submit_button("Sign In →", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("❌ Please enter both username and password")
                elif check_credentials(username, password):
                    reset_login_attempts()
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.now()
                    
                    if PROMETHEUS_AVAILABLE:
                        try:
                            DASHBOARD_LOGINS.labels(status='success').inc()
                        except:
                            pass
                    
                    st.rerun()
                else:
                    record_failed_login()
                    attempts_remaining = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                    
                    if attempts_remaining > 0:
                        st.error(f"❌ Invalid credentials. {attempts_remaining} attempt(s) remaining.")
                    else:
                        st.error(f"❌ Account locked for {LOGIN_LOCKOUT_SECONDS // 60} minutes.")
                    
                    if PROMETHEUS_AVAILABLE:
                        try:
                            DASHBOARD_LOGINS.labels(status='failed').inc()
                        except:
                            pass
        
        # Security notice
        if BCRYPT_AVAILABLE:
            st.markdown("""
                <div style="text-align: center; margin-top: 2rem;">
                    <div class="security-badge">
                        🔒 Secured with bcrypt encryption
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: center; margin-top: 2rem;">
                    <div class="security-badge" style="background: rgba(255, 193, 7, 0.15); border-color: rgba(255, 193, 7, 0.3); color: #ffc107;">
                        ⚠️ Using fallback security
                    </div>
                </div>
            """, unsafe_allow_html=True)

def require_auth(func):
    """Decorator to require authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            login_form()
            return None
        return func(*args, **kwargs)
    return wrapper

# =============================================================================
# EMERGENCY STOP FUNCTIONALITY
# =============================================================================

EMERGENCY_EXIT_SIGNAL_FILE = DATA_DIR / "emergency_exit_signal.json"
BOT_STOP_SIGNAL_FILE = DATA_DIR / "bot_control.json"


def _execute_emergency_stop():
    """
    Execute emergency stop: 
    1. Signal scanner to exit all positions
    2. Signal bot to stop running
    3. Write to Telegram alert
    """
    import logging
    
    try:
        # 1. Write emergency exit signal for scanner
        signal_data = {
            "timestamp": datetime.now().isoformat(),
            "action": "EMERGENCY_EXIT_ALL",
            "triggered_by": st.session_state.get('username', 'dashboard'),
            "reason": "PANIC BUTTON - Emergency Stop All"
        }
        
        with open(EMERGENCY_EXIT_SIGNAL_FILE, 'w') as f:
            json.dump(signal_data, f, indent=2)
        
        # 2. Write bot stop signal (for heartbeat monitor and bot loop)
        stop_signal = {
            "action": "STOP",
            "reason": "Emergency stop from dashboard",
            "timestamp": datetime.now().isoformat(),
            "triggered_by": st.session_state.get('username', 'dashboard')
        }
        
        with open(BOT_STOP_SIGNAL_FILE, 'w') as f:
            json.dump(stop_signal, f, indent=2)
        
        # 3. Try to send Telegram alert
        try:
            from utils import send_alert
            send_alert(
                "🚨 **EMERGENCY STOP TRIGGERED** 🚨\n\n"
                f"Triggered by: {st.session_state.get('username', 'Unknown')}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Actions taken:\n"
                "• Exit all positions signal sent\n"
                "• Bot stop signal sent\n\n"
                "⚠️ Please verify positions manually!"
            )
        except Exception as e:
            logging.warning(f"Could not send Telegram alert: {e}")
        
        st.sidebar.success("🛑 Emergency stop signal sent!")
        st.sidebar.warning("Check positions manually to confirm all exits.")
        
    except Exception as e:
        st.sidebar.error(f"❌ Emergency stop failed: {e}")
        logging.error(f"Emergency stop error: {e}")


def check_emergency_exit_pending() -> bool:
    """Check if an emergency exit signal is pending."""
    try:
        if EMERGENCY_EXIT_SIGNAL_FILE.exists():
            # Check if signal was recent (within last 5 minutes)
            with open(EMERGENCY_EXIT_SIGNAL_FILE, 'r') as f:
                data = json.load(f)
            
            signal_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
            if (datetime.now() - signal_time).total_seconds() < 300:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# DATA LOADING WITH ERROR HANDLING
# =============================================================================

def load_json_safe(filepath: Path, default: Any = None) -> Optional[Any]:
    """
    Safely load JSON file with comprehensive error handling
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    try:
        if not filepath.exists():
            return default
            
        # Check if file is empty
        if filepath.stat().st_size == 0:
            return default
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if data else default
            
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ Invalid JSON in {filepath.name}: {e}")
        return default
    except PermissionError:
        st.error(f"🔒 Permission denied reading {filepath.name}")
        return default
    except Exception as e:
        st.error(f"❌ Error loading {filepath.name}: {e}")
        return default

def load_trade_state() -> Dict[str, Any]:
    """Load current trade state with defaults"""
    default_state = {
        "status": False,
        "type": None,
        "instrument": None,
        "entry": 0,
        "current_sl_level": 0,
        "quantity": 0,
        "entry_time": None
    }
    return load_json_safe(STATE_FILE, default_state) or default_state

def load_daily_pnl() -> Dict[str, Any]:
    """Load daily P&L data with defaults"""
    default_pnl = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pnl": 0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "by_instrument": {}
    }
    data = load_json_safe(DAILY_PNL_FILE, default_pnl)
    
    # Reset if date changed
    if data and data.get("date") != datetime.now().strftime("%Y-%m-%d"):
        return default_pnl
        
    return data or default_pnl

def load_trade_history() -> List[Dict[str, Any]]:
    """Load trade history with defaults"""
    return load_json_safe(TRADE_HISTORY_FILE, []) or []

def load_websocket_status() -> Dict[str, Any]:
    """Load WebSocket status with defaults"""
    default_status = {
        "connected": False,
        "last_message_time": None,
        "latency_ms": 0,
        "messages_received": 0,
        "errors": 0,
        "reconnect_count": 0,
        "subscribed_symbols": [],
        "last_prices": {}
    }
    return load_json_safe(WEBSOCKET_STATUS_FILE, default_status) or default_status

# =============================================================================
# WEBSOCKET STATUS DASHBOARD
# =============================================================================

def render_websocket_status():
    """Render WebSocket connection status dashboard with modern styling"""
    st.markdown("""
        <h2 style="margin-bottom: 1.5rem;">🔌 WebSocket Status</h2>
    """, unsafe_allow_html=True)
    
    ws_status = load_websocket_status()
    
    # Connection status indicator
    col1, col2, col3, col4 = st.columns(4)
    
    is_connected = ws_status.get("connected", False)
    
    with col1:
        if is_connected:
            st.metric(
                label="🌐 Connection",
                value="🟢 Connected",
                delta="Active"
            )
        else:
            st.metric(
                label="🌐 Connection",
                value="🔴 Disconnected",
                delta="Offline",
                delta_color="inverse"
            )
        
    with col2:
        latency = ws_status.get("latency_ms", 0)
        if latency > 0:
            if latency < 50:
                latency_status = "⚡ Excellent"
            elif latency < 100:
                latency_status = "✅ Good"
            elif latency < 250:
                latency_status = "⚠️ Fair"
            else:
                latency_status = "🐢 Slow"
        else:
            latency_status = "—"
        st.metric(
            label="⚡ Latency",
            value=f"{latency:.0f} ms" if latency else "N/A",
            delta=latency_status
        )
        
    with col3:
        messages = ws_status.get("messages_received", 0)
        st.metric(
            label="📨 Messages",
            value=f"{messages:,}"
        )
        
    with col4:
        errors = ws_status.get("errors", 0)
        reconnects = ws_status.get("reconnect_count", 0)
        st.metric(
            label="⚠️ Errors / Reconnects",
            value=f"{errors} / {reconnects}",
            delta="Issues detected" if errors > 0 else "Stable",
            delta_color="inverse" if errors > 0 else "normal"
        )
    
    # Last message time
    last_msg_time = ws_status.get("last_message_time")
    if last_msg_time:
        try:
            last_msg_dt = datetime.fromisoformat(last_msg_time)
            time_ago = datetime.now() - last_msg_dt
            if time_ago.total_seconds() < 60:
                time_str = f"{int(time_ago.total_seconds())}s ago"
                time_color = "#38ef7d"
            elif time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                time_color = "#ffc107"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
                time_color = "#ff6b6b"
            
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                    <span style="color: rgba(255,255,255,0.6);">📡 Last Message:</span>
                    <span style="color: {time_color}; font-weight: 600; margin-left: 0.5rem;">{time_str}</span>
                </div>
            """, unsafe_allow_html=True)
        except:
            pass
    
    # Subscribed symbols and last prices
    with st.expander("📊 Live Prices", expanded=is_connected):
        subscribed = ws_status.get("subscribed_symbols", [])
        last_prices = ws_status.get("last_prices", {})
        
        if last_prices:
            # Create a styled table
            for symbol, price in last_prices.items():
                ltp = price.get('ltp', 0)
                change = price.get('change_pct', 0)
                change_color = "#38ef7d" if change >= 0 else "#ff6b6b"
                change_icon = "📈" if change >= 0 else "📉"
                
                st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;
                                background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                        <div>
                            <span style="font-weight: 600; color: white; font-size: 1.1rem;">{symbol}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-weight: 700; color: white; font-size: 1.2rem;">₹{ltp:,.2f}</span>
                            <span style="color: {change_color}; margin-left: 1rem; font-weight: 600;">
                                {change_icon} {change:+.2f}%
                            </span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 2rem; text-align: center;">
                    <span style="font-size: 2rem;">📡</span>
                    <p style="color: rgba(255,255,255,0.6); margin-top: 1rem;">
                        No live price data available. WebSocket may be disconnected.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # Update Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        try:
            WEBSOCKET_CONNECTED.set(1 if is_connected else 0)
        except:
            pass

# =============================================================================
# PROMETHEUS METRICS DISPLAY
# =============================================================================

def render_prometheus_info():
    """Show Prometheus metrics endpoint info"""
    with st.expander("📈 Prometheus Metrics", expanded=False):
        if PROMETHEUS_AVAILABLE:
            st.success(f"✅ Metrics server running on port {PROMETHEUS_PORT}")
            st.code(f"curl http://localhost:{PROMETHEUS_PORT}/metrics", language="bash")
            
            st.markdown("""
            **Available Metrics:**
            - `trading_bot_trades_total` - Total trades by instrument and outcome
            - `trading_bot_daily_pnl` - Current daily P&L
            - `trading_bot_total_pnl` - Cumulative P&L
            - `trading_bot_win_rate` - Win rate percentage
            - `trading_bot_active_trade` - Active trade indicator
            - `trading_bot_websocket_connected` - WebSocket status
            - `trading_bot_websocket_latency_ms` - Message latency histogram
            - `trading_bot_websocket_messages_total` - Total messages received
            - `trading_bot_websocket_errors_total` - Total errors
            - `dashboard_logins_total` - Dashboard login attempts
            """)
            
            # Grafana example
            st.markdown("**Grafana Dashboard Example:**")
            st.code("""
# Add to Prometheus scrape config (prometheus.yml):
scrape_configs:
  - job_name: 'trading_bot'
    static_configs:
      - targets: ['localhost:8000']
            """, language="yaml")
        else:
            st.warning("⚠️ Prometheus metrics disabled. Install prometheus_client:")
            st.code("pip install prometheus_client", language="bash")

# =============================================================================
# MAIN DASHBOARD COMPONENTS
# =============================================================================

def render_live_status():
    """Render live trading status section with modern UI"""
    
    state_data = load_trade_state()
    daily_data = load_daily_pnl()
    
    # Modern dashboard header
    is_active = state_data.get("status", False)
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <div>
                <h2 style="margin: 0; color: white;">📡 Live Trading Status</h2>
            </div>
            <div class="live-indicator">
                <span class="live-dot"></span>
                {"TRADING ACTIVE" if is_active else "SCANNING MARKETS"}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Bot Status
    with col1:
        instrument = state_data.get("instrument", "N/A")
        
        if is_active:
            st.metric(
                label="🤖 Bot Status",
                value="🟢 ACTIVE",
                delta=f"Trading {instrument}"
            )
        else:
            st.metric(
                label="🤖 Bot Status",
                value="💤 Scanning",
                delta="Awaiting signal"
            )
    
    # Current Trade Info
    with col2:
        if is_active:
            trade_type = state_data.get("type", "N/A")
            entry = state_data.get("entry", 0)
            emoji = "📈" if trade_type == "BUY" else "📉"
            st.metric(
                label=f"{emoji} Trade Type",
                value=trade_type,
                delta=f"Entry: ₹{entry:,.2f}" if entry else None
            )
        else:
            st.metric(label="📊 Trade Type", value="—")
    
    # Daily P&L
    with col3:
        pnl = daily_data.get("pnl", 0)
        pnl_emoji = "💰" if pnl >= 0 else "📉"
        st.metric(
            label=f"{pnl_emoji} Day's P&L",
            value=f"₹{pnl:,.2f}",
            delta=f"{'Profit' if pnl >= 0 else 'Loss'}",
            delta_color="normal" if pnl >= 0 else "inverse"
        )
        
        # Update Prometheus
        if PROMETHEUS_AVAILABLE:
            try:
                PNL_GAUGE.labels(instrument='combined').set(pnl)
            except:
                pass
    
    # Today's Stats
    with col4:
        trades = daily_data.get("trades", 0)
        wins = daily_data.get("wins", 0)
        losses = daily_data.get("losses", 0)
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        st.metric(
            label="📊 Today's Stats",
            value=f"{trades} Trades",
            delta=f"W:{wins} L:{losses} ({win_rate:.0f}%)"
        )
        
        # Update Prometheus
        if PROMETHEUS_AVAILABLE:
            try:
                ACTIVE_TRADE_GAUGE.set(1 if is_active else 0)
                WIN_RATE_GAUGE.labels(instrument='combined').set(win_rate)
            except:
                pass

def render_performance_analytics():
    """Render performance analytics section with modern styling"""
    st.markdown("""
        <h2 style="margin-bottom: 1.5rem;">📊 Performance Analytics</h2>
    """, unsafe_allow_html=True)
    
    history_data = load_trade_history()
    
    if not history_data:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 16px; padding: 3rem; text-align: center;">
                <span style="font-size: 3rem;">📭</span>
                <h3 style="margin: 1rem 0 0.5rem 0;">No Trade History Yet</h3>
                <p style="color: rgba(255,255,255,0.6);">Start the trading bot to generate data and see your performance here.</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    df = pd.DataFrame(history_data)
    
    # Convert timestamps safely
    for col in ['exit_time', 'entry_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = df.dropna(subset=['exit_time'])
    df.sort_values(by='exit_time', ascending=False, inplace=True)
    
    # Instrument filter
    available_instruments = df['instrument'].unique().tolist() if 'instrument' in df.columns else []
    
    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        filter_instrument = st.selectbox(
            "Filter by Instrument",
            ["All"] + available_instruments,
            index=0
        )
    
    # Apply filter
    if filter_instrument != "All" and 'instrument' in df.columns:
        df_filtered = df[df['instrument'] == filter_instrument]
    else:
        df_filtered = df
    
    if df_filtered.empty:
        st.warning("No trades found for selected filter.")
        return
    
    # Calculate metrics
    df_sorted = df_filtered.sort_values(by='exit_time', ascending=True)
    df_sorted['cumulative_pnl'] = df_sorted['pnl'].cumsum()
    
    total_pnl = df_sorted['pnl'].sum()
    win_count = len(df_sorted[df_sorted['pnl'] > 0])
    loss_count = len(df_sorted[df_sorted['pnl'] <= 0])
    win_rate = (win_count / len(df_sorted) * 100) if len(df_sorted) > 0 else 0
    
    # Update Prometheus
    if PROMETHEUS_AVAILABLE:
        try:
            TOTAL_PNL_GAUGE.set(total_pnl)
        except:
            pass
    
    # Summary metrics
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
    with metric_cols[1]:
        st.metric("Total Trades", len(df_sorted))
    with metric_cols[2]:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with metric_cols[3]:
        avg_win = df_sorted[df_sorted['pnl'] > 0]['pnl'].mean() if win_count > 0 else 0
        st.metric("Avg Win", f"₹{avg_win:,.2f}")
    with metric_cols[4]:
        avg_loss = df_sorted[df_sorted['pnl'] <= 0]['pnl'].mean() if loss_count > 0 else 0
        st.metric("Avg Loss", f"₹{avg_loss:,.2f}")
    
    # Charts with modern styling
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_equity = px.line(
            df_sorted,
            x='exit_time',
            y='cumulative_pnl',
            title='📈 Equity Curve',
            markers=True
        )
        fig_equity.update_traces(
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        )
        fig_equity.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)",
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.8)'),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.2)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.2)'
            ),
            title=dict(font=dict(size=16))
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with chart_col2:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Wins', 'Losses'],
            values=[win_count, loss_count],
            hole=0.6,
            marker=dict(
                colors=['#38ef7d', '#ff6b6b'],
                line=dict(color='rgba(255,255,255,0.2)', width=2)
            ),
            textinfo='percent+label',
            textfont=dict(size=14, color='white')
        )])
        fig_pie.update_layout(
            title='🎯 Win/Loss Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.8)'),
            showlegend=False,
            annotations=[dict(
                text=f'{win_rate:.0f}%<br>Win Rate',
                x=0.5, y=0.5,
                font_size=18,
                font_color='white',
                showarrow=False
            )],
            title_font_size=16
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # P&L by instrument chart (if multiple instruments)
    if 'instrument' in df.columns and len(df['instrument'].unique()) > 1:
        pnl_by_instrument = df.groupby('instrument')['pnl'].sum().reset_index()
        pnl_by_instrument['color'] = pnl_by_instrument['pnl'].apply(
            lambda x: '#38ef7d' if x >= 0 else '#ff6b6b'
        )
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=pnl_by_instrument['instrument'],
                y=pnl_by_instrument['pnl'],
                marker=dict(
                    color=pnl_by_instrument['color'],
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                text=[f'₹{v:,.0f}' for v in pnl_by_instrument['pnl']],
                textposition='outside',
                textfont=dict(color='white')
            )
        ])
        fig_bar.update_layout(
            title='💹 P&L by Instrument',
            xaxis_title='Instrument',
            yaxis_title='P&L (₹)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.8)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            title_font_size=16
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def render_trade_log():
    """Render detailed trade log"""
    st.subheader("📝 Trade Log")
    
    history_data = load_trade_history()
    
    if not history_data:
        st.info("No trades to display.")
        return
    
    df = pd.DataFrame(history_data)
    
    # Convert timestamps
    for col in ['exit_time', 'entry_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df = df.dropna(subset=['exit_time'])
    df.sort_values(by='exit_time', ascending=False, inplace=True)
    
    # Style function for P&L coloring
    def style_pnl(val):
        if pd.isna(val):
            return ''
        color = '#d4edda' if val > 0 else '#f8d7da'
        return f'background-color: {color}; color: black'
    
    # Select display columns (only existing ones)
    potential_cols = ['exit_time', 'instrument', 'trade_type', 'option_type', 
                      'future_entry', 'future_exit', 'pnl', 'r_multiple', 'exit_reason']
    display_cols = [col for col in potential_cols if col in df.columns]
    
    if not display_cols:
        st.warning("Trade history has unexpected format.")
        return
    
    # Format dict for numeric columns
    format_dict = {}
    if 'pnl' in display_cols:
        format_dict['pnl'] = '₹{:.2f}'
    if 'future_entry' in display_cols:
        format_dict['future_entry'] = '{:.2f}'
    if 'future_exit' in display_cols:
        format_dict['future_exit'] = '{:.2f}'
    if 'r_multiple' in display_cols:
        format_dict['r_multiple'] = '{:.2f}R'
    
    styled_df = df[display_cols].style.format(format_dict)
    
    if 'pnl' in display_cols:
        styled_df = styled_df.map(style_pnl, subset=['pnl'])
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# MANUAL TRADE FORM
# =============================================================================

def load_trading_config() -> Dict[str, Any]:
    """Load current trading configuration from file"""
    from config import DEFAULT_TRADING_CONFIG
    config = DEFAULT_TRADING_CONFIG.copy()
    
    if TRADING_CONFIG_FILE.exists():
        try:
            with open(TRADING_CONFIG_FILE, 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
        except Exception as e:
            st.warning(f"Could not load trading config: {e}")
    
    return config


def save_trading_config(config: Dict[str, Any]) -> bool:
    """Save trading configuration to file"""
    try:
        with open(TRADING_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save config: {e}")
        return False


def calculate_manual_sl_target(
    instrument: str,
    signal: str,
    entry_price: float
) -> Tuple[float, float, float]:
    """
    Calculate SL and targets for manual trade based on risk management rules.
    
    Returns:
        Tuple of (stop_loss, target_1, target_2)
    """
    inst_config = INSTRUMENTS.get(instrument, {})
    strike_step = inst_config.get("strike_step", 50)
    
    # Dynamic SL calculation based on 1R risk
    # Using a default risk of ~1% of entry price or minimum strike step
    risk_amount = max(entry_price * 0.01, strike_step)
    
    if signal == "BUY":
        stop_loss = round(entry_price - risk_amount, 2)
        target_1 = round(entry_price + (risk_amount * 2), 2)  # 1:2 RR
        target_2 = round(entry_price + (risk_amount * 3), 2)  # 1:3 RR
    else:  # SELL
        stop_loss = round(entry_price + risk_amount, 2)
        target_1 = round(entry_price - (risk_amount * 2), 2)  # 1:2 RR
        target_2 = round(entry_price - (risk_amount * 3), 2)  # 1:3 RR
    
    return stop_loss, target_1, target_2


def render_manual_trade():
    """Render manual trade entry form"""
    st.subheader("🎯 Manual Trade Entry")
    
    st.info("""
    **Manual Intervention Controls**  
    Use this form to manually enter trades when you see a signal the bot misses,
    or when you want to take a position based on news or manual analysis.
    The system will auto-calculate SL and Target based on the bot's risk management rules.
    """)
    
    # Check if there's already an active trade
    state_data = load_trade_state()
    if state_data.get("status", False):
        st.warning(f"⚠️ There is already an active trade in {state_data.get('instrument', 'Unknown')}. "
                   "Please close the existing trade before entering a new one.")
        
        # Show current trade info
        with st.expander("📊 Current Trade Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Instrument", state_data.get("instrument", "N/A"))
                st.metric("Trade Type", state_data.get("type", "N/A"))
            with col2:
                st.metric("Entry Price", f"₹{state_data.get('entry', 0):,.2f}")
                st.metric("Current SL", f"₹{state_data.get('sl', 0):,.2f}")
            with col3:
                st.metric("Option Entry", f"₹{state_data.get('option_entry', 0):,.2f}")
                st.metric("Step Level", state_data.get("step_level", 0))
        
        # Emergency exit button
        st.divider()
        st.markdown("### 🚨 Emergency Exit")
        st.warning("Use this only for emergency situations when you need to exit immediately.")
        
        if st.button("🛑 Emergency Close Position", type="primary", use_container_width=True):
            st.error("⚠️ Emergency exit requested. Please confirm in the Tradebot terminal.")
            # Create a signal file that the bot can pick up
            emergency_file = DATA_DIR / "emergency_exit_signal.json"
            with open(emergency_file, 'w') as f:
                json.dump({
                    "action": "EMERGENCY_EXIT",
                    "timestamp": datetime.now().isoformat(),
                    "requested_by": st.session_state.get('username', 'dashboard')
                }, f)
            st.success("Emergency exit signal sent to bot. Check bot terminal for confirmation.")
        
        return
    
    # Trade entry form
    st.markdown("### 📝 New Trade Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Instrument selection
        available_instruments = list(INSTRUMENTS.keys())
        selected_instrument = st.selectbox(
            "Select Instrument",
            available_instruments,
            index=0,
            help="Choose the instrument you want to trade"
        )
        
        # Get instrument details
        inst_config = INSTRUMENTS.get(selected_instrument, {})
        st.caption(f"📊 **{inst_config.get('name', selected_instrument)}** | "
                   f"Lot Size: {inst_config.get('lot_size', 'N/A')} | "
                   f"Exchange: {inst_config.get('exchange_segment_str', 'N/A')}")
    
    with col2:
        # Signal type
        signal_type = st.radio(
            "Trade Direction",
            ["BUY (Bullish)", "SELL (Bearish)"],
            horizontal=True,
            help="Select trade direction"
        )
        signal = "BUY" if "BUY" in signal_type else "SELL"
    
    st.divider()
    
    # Price inputs
    col3, col4, col5 = st.columns(3)
    
    with col3:
        current_price = st.number_input(
            "Current Future Price (₹)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Enter the current futures price from the market"
        )
    
    with col4:
        option_premium = st.number_input(
            "Option Premium (₹)",
            min_value=0.0,
            value=0.0,
            step=0.5,
            help="Enter the option premium you want to pay"
        )
    
    with col5:
        # Auto-calculate ATM strike
        strike_step = inst_config.get("strike_step", 50)
        if current_price > 0:
            atm_strike = round(current_price / strike_step) * strike_step
        else:
            atm_strike = 0
        
        atm_strike = st.number_input(
            "Strike Price",
            min_value=0,
            value=int(atm_strike),
            step=strike_step,
            help="ATM strike price (auto-calculated)"
        )
    
    # Auto-calculate SL and Targets
    if current_price > 0:
        stop_loss, target_1, target_2 = calculate_manual_sl_target(
            selected_instrument, signal, current_price
        )
        
        st.divider()
        st.markdown("### 📈 Auto-Calculated Risk Parameters")
        
        calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)
        
        with calc_col1:
            st.metric(
                "Stop Loss",
                f"₹{stop_loss:,.2f}",
                delta=f"{abs(current_price - stop_loss):.2f} pts risk",
                delta_color="inverse"
            )
        
        with calc_col2:
            st.metric(
                "Target 1 (1:2 RR)",
                f"₹{target_1:,.2f}",
                delta=f"+{abs(target_1 - current_price):.2f} pts"
            )
        
        with calc_col3:
            st.metric(
                "Target 2 (1:3 RR)",
                f"₹{target_2:,.2f}",
                delta=f"+{abs(target_2 - current_price):.2f} pts"
            )
        
        with calc_col4:
            risk_per_lot = abs(current_price - stop_loss) * inst_config.get("lot_size", 1)
            st.metric(
                "Max Risk/Lot",
                f"₹{risk_per_lot:,.2f}",
                delta="per lot"
            )
        
        # Allow manual override
        with st.expander("🔧 Manual Override (Advanced)", expanded=False):
            override_col1, override_col2, override_col3 = st.columns(3)
            
            with override_col1:
                manual_sl = st.number_input(
                    "Custom SL",
                    min_value=0.0,
                    value=float(stop_loss),
                    step=0.5
                )
            
            with override_col2:
                manual_target1 = st.number_input(
                    "Custom Target 1",
                    min_value=0.0,
                    value=float(target_1),
                    step=0.5
                )
            
            with override_col3:
                manual_target2 = st.number_input(
                    "Custom Target 2",
                    min_value=0.0,
                    value=float(target_2),
                    step=0.5
                )
            
            # Update values if manually changed
            if manual_sl != stop_loss:
                stop_loss = manual_sl
            if manual_target1 != target_1:
                target_1 = manual_target1
            if manual_target2 != target_2:
                target_2 = manual_target2
    
    st.divider()
    
    # Trade execution
    st.markdown("### 🚀 Execute Trade")
    
    # Confirmation checkbox
    confirm = st.checkbox(
        "I confirm this trade and understand the risks involved",
        value=False
    )
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button(
            f"📤 Place {signal} Order",
            type="primary",
            use_container_width=True,
            disabled=not (confirm and current_price > 0 and option_premium > 0)
        ):
            # Create manual trade signal file for the bot to pick up
            opt_type = "CE" if signal == "BUY" else "PE"
            manual_trade_signal = {
                "action": "MANUAL_ENTRY",
                "instrument": selected_instrument,
                "signal": signal,
                "option_type": opt_type,
                "atm_strike": atm_strike,
                "future_price": current_price,
                "option_premium": option_premium,
                "stop_loss": stop_loss,
                "target_1": target_1,
                "target_2": target_2,
                "timestamp": datetime.now().isoformat(),
                "requested_by": st.session_state.get('username', 'dashboard')
            }
            
            signal_file = DATA_DIR / "manual_trade_signal.json"
            with open(signal_file, 'w') as f:
                json.dump(manual_trade_signal, f, indent=2)
            
            st.success(f"""
            ✅ **Manual trade signal created!**  
            - **{inst_config.get('name')} {atm_strike} {opt_type}**
            - Entry: ₹{option_premium} | SL: ₹{stop_loss} | Target: ₹{target_1}
            
            ⚠️ The bot will pick up this signal on its next scan cycle.
            Please ensure the Tradebot is running.
            """)
            
            # Log the action
            if PROMETHEUS_AVAILABLE:
                try:
                    TRADE_COUNTER.labels(
                        instrument=selected_instrument,
                        trade_type=signal,
                        outcome='manual_signal'
                    ).inc()
                except:
                    pass
    
    with col_btn2:
        if st.button("🔄 Clear Form", use_container_width=True):
            st.rerun()


# =============================================================================
# SETTINGS PAGE - LIVE CONFIG UPDATES
# =============================================================================

def render_settings_page():
    """Render settings page for live configuration updates with modern UI"""
    st.markdown("""
        <h2 style="margin-bottom: 0.5rem;">⚙️ Bot Settings & Configuration</h2>
        <p style="color: rgba(255, 255, 255, 0.6); margin-bottom: 2rem;">
            Configure trading parameters in real-time. Changes take effect immediately.
        </p>
    """, unsafe_allow_html=True)
    
    # Load current config
    current_config = load_trading_config()
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Risk Management",
        "📊 Signal Settings", 
        "🎯 Instruments",
        "⏱️ Timing",
        "📁 Config File"
    ])
    
    # Track if any changes were made
    config_changed = False
    new_config = current_config.copy()
    
    # === TAB 1: Risk Management ===
    with tab1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(235, 51, 73, 0.1) 0%, rgba(244, 92, 67, 0.1) 100%); 
                        border: 1px solid rgba(235, 51, 73, 0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #ff6b6b;">💰 Risk Management Settings</h4>
                <p style="margin: 0; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                    Control your exposure and protect your capital
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_max_daily_loss = st.number_input(
                "🛡️ Maximum Daily Loss (₹)",
                min_value=1000,
                max_value=100000,
                value=int(current_config.get("MAX_DAILY_LOSS", 5000)),
                step=500,
                help="Trading stops when this loss limit is reached"
            )
            if new_max_daily_loss != current_config.get("MAX_DAILY_LOSS"):
                new_config["MAX_DAILY_LOSS"] = new_max_daily_loss
                config_changed = True
            
            new_max_trades = st.number_input(
                "📊 Maximum Trades Per Day",
                min_value=1,
                max_value=20,
                value=int(current_config.get("MAX_TRADES_PER_DAY", 5)),
                step=1,
                help="Maximum number of trades allowed per day"
            )
            if new_max_trades != current_config.get("MAX_TRADES_PER_DAY"):
                new_config["MAX_TRADES_PER_DAY"] = new_max_trades
                config_changed = True
        
        with col2:
            new_limit_buffer = st.number_input(
                "📈 Limit Order Buffer (%)",
                min_value=0.001,
                max_value=0.05,
                value=float(current_config.get("LIMIT_ORDER_BUFFER", 0.01)),
                step=0.005,
                format="%.3f",
                help="Buffer percentage for limit orders"
            )
            if new_limit_buffer != current_config.get("LIMIT_ORDER_BUFFER"):
                new_config["LIMIT_ORDER_BUFFER"] = new_limit_buffer
                config_changed = True
            
            new_squareoff_buffer = st.number_input(
                "⏰ Auto Square-Off Buffer (min)",
                min_value=1,
                max_value=30,
                value=int(current_config.get("AUTO_SQUARE_OFF_BUFFER", 5)),
                step=1,
                help="Minutes before market close to exit positions"
            )
            if new_squareoff_buffer != current_config.get("AUTO_SQUARE_OFF_BUFFER"):
                new_config["AUTO_SQUARE_OFF_BUFFER"] = new_squareoff_buffer
                config_changed = True
        
        # Risk Summary Cards
        st.markdown("<br>", unsafe_allow_html=True)
        daily_data = load_daily_pnl()
        remaining_loss = new_config.get('MAX_DAILY_LOSS', 5000) + daily_data.get('pnl', 0)
        used_pct = (1 - remaining_loss / new_config.get('MAX_DAILY_LOSS', 5000)) * 100 if new_config.get('MAX_DAILY_LOSS', 5000) > 0 else 0
        
        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        
        with risk_col1:
            st.metric("Max Daily Loss", f"₹{new_config.get('MAX_DAILY_LOSS', 5000):,}")
        with risk_col2:
            st.metric("Max Trades/Day", new_config.get('MAX_TRADES_PER_DAY', 5))
        with risk_col3:
            st.metric("Remaining Budget", f"₹{max(0, remaining_loss):,.0f}")
        with risk_col4:
            st.metric("Risk Used Today", f"{max(0, min(100, used_pct)):.1f}%")
    
    # === TAB 2: Signal Settings ===
    with tab2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                        border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #667eea;">📊 Signal Strength Settings</h4>
                <p style="margin: 0; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                    Configure technical indicators for signal generation
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📈 RSI Thresholds")
            new_rsi_bullish = st.slider(
                "RSI Bullish Threshold",
                min_value=50,
                max_value=80,
                value=int(current_config.get("RSI_BULLISH_THRESHOLD", 60)),
                help="RSI must be above this value for bullish signals"
            )
            if new_rsi_bullish != current_config.get("RSI_BULLISH_THRESHOLD"):
                new_config["RSI_BULLISH_THRESHOLD"] = new_rsi_bullish
                config_changed = True
            
            new_rsi_bearish = st.slider(
                "RSI Bearish Threshold",
                min_value=20,
                max_value=50,
                value=int(current_config.get("RSI_BEARISH_THRESHOLD", 40)),
                help="RSI must be below this value for bearish signals"
            )
            if new_rsi_bearish != current_config.get("RSI_BEARISH_THRESHOLD"):
                new_config["RSI_BEARISH_THRESHOLD"] = new_rsi_bearish
                config_changed = True
        
        with col2:
            st.markdown("##### 📊 Volume Settings")
            new_volume_mult = st.slider(
                "Volume Multiplier",
                min_value=1.0,
                max_value=3.0,
                value=float(current_config.get("VOLUME_MULTIPLIER", 1.2)),
                step=0.1,
                help="Volume must be this multiple of average"
            )
            if new_volume_mult != current_config.get("VOLUME_MULTIPLIER"):
                new_config["VOLUME_MULTIPLIER"] = new_volume_mult
                config_changed = True
            
            # Visual RSI Zone indicator
            st.markdown("##### 🎯 RSI Zones Visualization")
            
            # Create a visual representation
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, 
                    rgba(235, 51, 73, 0.3) 0%, 
                    rgba(235, 51, 73, 0.3) {new_rsi_bearish}%, 
                    rgba(255, 255, 255, 0.1) {new_rsi_bearish}%, 
                    rgba(255, 255, 255, 0.1) {new_rsi_bullish}%, 
                    rgba(56, 239, 125, 0.3) {new_rsi_bullish}%, 
                    rgba(56, 239, 125, 0.3) 100%);
                    border-radius: 10px; padding: 1rem; margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                        <span style="color: #ff6b6b;">📉 Bearish (0-{new_rsi_bearish})</span>
                        <span style="color: rgba(255,255,255,0.6);">Neutral</span>
                        <span style="color: #38ef7d;">📈 Bullish ({new_rsi_bullish}-100)</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # === TAB 3: Instruments Configuration ===
    with tab3:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(17, 153, 142, 0.1) 0%, rgba(56, 239, 125, 0.1) 100%); 
                        border: 1px solid rgba(56, 239, 125, 0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #38ef7d;">🎯 Instrument Configuration</h4>
                <p style="margin: 0; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                    Enable/disable instruments and set scanning priority
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Load current instrument settings
        enabled_instruments = current_config.get("ENABLED_INSTRUMENTS", list(INSTRUMENTS.keys()))
        instrument_priority = current_config.get("INSTRUMENT_PRIORITY", {
            "CRUDEOIL": 1, "GOLD": 2, "SILVER": 3, "NATURALGAS": 4, "NIFTY": 5, "BANKNIFTY": 6
        })
        
        st.markdown("##### 🔘 Enable/Disable Instruments")
        st.caption("Toggle which instruments the bot should scan and trade")
        
        # Create two columns for instruments
        inst_col1, inst_col2 = st.columns(2)
        instruments_list = list(INSTRUMENTS.keys())
        
        new_enabled = []
        
        for i, inst_key in enumerate(instruments_list):
            inst_data = INSTRUMENTS[inst_key]
            priority = instrument_priority.get(inst_key, i + 1)
            
            col = inst_col1 if i % 2 == 0 else inst_col2
            
            with col:
                # Styled instrument toggle card
                is_enabled = st.checkbox(
                    f"**{inst_data.get('name', inst_key)}**",
                    value=inst_key in enabled_instruments,
                    key=f"inst_toggle_{inst_key}",
                    help=f"Lot Size: {inst_data.get('lot_size')} | Exchange: {inst_data.get('exchange_segment_str')}"
                )
                
                if is_enabled:
                    new_enabled.append(inst_key)
                    
                    # Show priority selector when enabled
                    st.markdown(f"""
                        <div style="font-size: 0.8rem; color: rgba(255,255,255,0.5); margin-top: -10px; margin-bottom: 10px; padding-left: 2rem;">
                            📊 {inst_data.get('exchange_segment_str')} | Lot: {inst_data.get('lot_size')}
                        </div>
                    """, unsafe_allow_html=True)
        
        # Check if enabled instruments changed
        if set(new_enabled) != set(enabled_instruments):
            new_config["ENABLED_INSTRUMENTS"] = new_enabled
            config_changed = True
        
        st.divider()
        
        # Priority Settings
        st.markdown("##### 🏆 Instrument Priority")
        st.caption("Set priority for instrument selection when multiple signals occur (1 = highest priority)")
        
        new_priority = {}
        
        # Only show priority for enabled instruments
        if new_enabled:
            priority_cols = st.columns(min(len(new_enabled), 3))
            
            for i, inst_key in enumerate(new_enabled):
                col_idx = i % 3
                with priority_cols[col_idx]:
                    current_priority = instrument_priority.get(inst_key, i + 1)
                    
                    # Priority badge
                    badge_class = f"priority-{min(current_priority, 6)}"
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span class="priority-badge {badge_class}">{current_priority}</span>
                            <span style="font-weight: 600; color: white;">{INSTRUMENTS[inst_key].get('name', inst_key)}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    new_pri = st.number_input(
                        f"Priority",
                        min_value=1,
                        max_value=len(INSTRUMENTS),
                        value=current_priority,
                        key=f"priority_{inst_key}",
                        label_visibility="collapsed"
                    )
                    new_priority[inst_key] = new_pri
            
            # Check if priority changed
            if new_priority != instrument_priority:
                new_config["INSTRUMENT_PRIORITY"] = new_priority
                config_changed = True
        else:
            st.warning("⚠️ No instruments enabled. Enable at least one instrument above.")
        
        # Per-instrument signal settings
        st.divider()
        st.markdown("##### ⚙️ Per-Instrument Signal Settings")
        st.caption("Override global signal settings for specific instruments")
        
        # Get current per-instrument settings
        per_inst_settings = current_config.get("PER_INSTRUMENT_SETTINGS", {})
        
        # Expander for each enabled instrument
        for inst_key in new_enabled:
            inst_data = INSTRUMENTS[inst_key]
            strategy_params = inst_data.get("strategy_params", {})
            
            with st.expander(f"🔧 {inst_data.get('name', inst_key)} Settings", expanded=False):
                inst_settings = per_inst_settings.get(inst_key, {})
                
                inst_col1, inst_col2 = st.columns(2)
                
                with inst_col1:
                    use_custom = st.checkbox(
                        "Use custom settings for this instrument",
                        value=inst_settings.get("use_custom", False),
                        key=f"custom_{inst_key}"
                    )
                    
                    if use_custom:
                        custom_rsi_bull = st.slider(
                            "RSI Bullish",
                            50, 80,
                            value=inst_settings.get("rsi_bullish", strategy_params.get("rsi_bullish_threshold", 60)),
                            key=f"rsi_bull_{inst_key}"
                        )
                        custom_rsi_bear = st.slider(
                            "RSI Bearish",
                            20, 50,
                            value=inst_settings.get("rsi_bearish", strategy_params.get("rsi_bearish_threshold", 40)),
                            key=f"rsi_bear_{inst_key}"
                        )
                
                with inst_col2:
                    if use_custom:
                        custom_vol_mult = st.slider(
                            "Volume Multiplier",
                            1.0, 3.0,
                            value=float(inst_settings.get("volume_multiplier", strategy_params.get("volume_multiplier", 1.2))),
                            step=0.1,
                            key=f"vol_{inst_key}"
                        )
                        
                        # Save per-instrument settings
                        new_inst_settings = {
                            "use_custom": True,
                            "rsi_bullish": custom_rsi_bull,
                            "rsi_bearish": custom_rsi_bear,
                            "volume_multiplier": custom_vol_mult
                        }
                        
                        if new_inst_settings != inst_settings:
                            if "PER_INSTRUMENT_SETTINGS" not in new_config:
                                new_config["PER_INSTRUMENT_SETTINGS"] = {}
                            new_config["PER_INSTRUMENT_SETTINGS"][inst_key] = new_inst_settings
                            config_changed = True
                    else:
                        st.info(f"Using global settings\nRSI: {new_rsi_bearish}-{new_rsi_bullish}\nVolume: {new_volume_mult}x")
                        
                        # Remove custom settings if disabled
                        if inst_key in per_inst_settings:
                            if "PER_INSTRUMENT_SETTINGS" not in new_config:
                                new_config["PER_INSTRUMENT_SETTINGS"] = {}
                            new_config["PER_INSTRUMENT_SETTINGS"][inst_key] = {"use_custom": False}
                            config_changed = True
    
    # === TAB 4: Timing & Cooldowns ===
    with tab4:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%); 
                        border: 1px solid rgba(240, 147, 251, 0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #f093fb;">⏱️ Timing & Cooldown Settings</h4>
                <p style="margin: 0; color: rgba(255, 255, 255, 0.6); font-size: 0.9rem;">
                    Configure waiting periods between trades
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_loss_cooldown = st.number_input(
                "⏸️ Cooldown After Loss (seconds)",
                min_value=60,
                max_value=1800,
                value=int(current_config.get("COOLDOWN_AFTER_LOSS", 300)),
                step=60,
                help="Wait time after a losing trade"
            )
            if new_loss_cooldown != current_config.get("COOLDOWN_AFTER_LOSS"):
                new_config["COOLDOWN_AFTER_LOSS"] = new_loss_cooldown
                config_changed = True
            
            # Visual display
            mins = new_loss_cooldown // 60
            secs = new_loss_cooldown % 60
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 0.75rem; text-align: center;">
                    <span style="font-size: 1.5rem; font-weight: 700; color: #f093fb;">{mins}m {secs}s</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            new_signal_cooldown = st.number_input(
                "🔄 Signal Cooldown (seconds)",
                min_value=300,
                max_value=3600,
                value=int(current_config.get("SIGNAL_COOLDOWN", 900)),
                step=60,
                help="Minimum time between signals in same direction"
            )
            if new_signal_cooldown != current_config.get("SIGNAL_COOLDOWN"):
                new_config["SIGNAL_COOLDOWN"] = new_signal_cooldown
                config_changed = True
            
            mins2 = new_signal_cooldown // 60
            secs2 = new_signal_cooldown % 60
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); border-radius: 10px; padding: 0.75rem; text-align: center;">
                    <span style="font-size: 1.5rem; font-weight: 700; color: #667eea;">{mins2}m {secs2}s</span>
                </div>
            """, unsafe_allow_html=True)
    
    # === TAB 5: View Config File ===
    with tab5:
        st.markdown("##### 📁 Current Configuration File")
        
        if TRADING_CONFIG_FILE.exists():
            with open(TRADING_CONFIG_FILE, 'r') as f:
                config_content = f.read()
            st.code(config_content, language="json")
        else:
            st.warning("No trading_config.json file found. Using default values.")
            st.code(json.dumps(current_config, indent=2), language="json")
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download Config",
                data=json.dumps(new_config, indent=2),
                file_name="trading_config.json",
                mime="application/json",
                use_container_width=True
            )
    
    # === Save Button ===
    st.divider()
    
    col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
    
    with col_save2:
        if config_changed:
            st.warning("⚠️ You have unsaved changes!")
            
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                if save_trading_config(new_config):
                    st.success("""
                    ✅ **Configuration saved successfully!**  
                    Changes will take effect on the next scan cycle.
                    
                    Note: For some settings to fully apply, the bot may need to 
                    complete its current operation cycle.
                    """)
                    
                    # Log config change
                    config_log = DATA_DIR / "config_change_log.json"
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "user": st.session_state.get('username', 'dashboard'),
                        "changes": {k: new_config[k] for k in new_config if new_config[k] != current_config.get(k)}
                    }
                    
                    try:
                        existing_log = []
                        if config_log.exists():
                            with open(config_log, 'r') as f:
                                existing_log = json.load(f)
                        existing_log.append(log_entry)
                        # Keep last 100 entries
                        existing_log = existing_log[-100:]
                        with open(config_log, 'w') as f:
                            json.dump(existing_log, f, indent=2)
                    except Exception as e:
                        st.warning(f"Could not log config change: {e}")
                    
                    time.sleep(1)
                    st.rerun()
        else:
            st.success("✅ No changes to save")
    
    # Reset to defaults button
    with st.expander("🔄 Reset to Defaults", expanded=False):
        st.warning("This will reset all settings to their default values.")
        
        if st.button("Reset All Settings", type="secondary"):
            from config import DEFAULT_TRADING_CONFIG
            if save_trading_config(DEFAULT_TRADING_CONFIG):
                st.success("Settings reset to defaults!")
                time.sleep(1)
                st.rerun()


# =============================================================================
# AUTO-REFRESH FRAGMENT (Replaces infinite loop)
# =============================================================================

@st.fragment(run_every=5)  # Refresh every 5 seconds
def auto_refresh_data():
    """Auto-refreshing data fragment - avoids full page rerun"""
    render_live_status()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main dashboard application with modern UI"""
    
    # Page Config - Modern dark theme
    st.set_page_config(
        page_title="TradingBot Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply modern CSS styling
    st.markdown(MODERN_CSS, unsafe_allow_html=True)
    
    # Start Prometheus server
    start_prometheus_server()
    
    # Check authentication
    if not st.session_state.get('authenticated', False):
        login_form()
        return
    
    # Modern header with gradient title
    col_title, col_status, col_logout = st.columns([3, 1, 1])
    
    with col_title:
        st.markdown("""
            <h1 style="margin-bottom: 0;">🚀 TradingBot</h1>
            <p style="color: rgba(255, 255, 255, 0.6); margin-top: 0.25rem;">
                Algorithmic Trading Dashboard
            </p>
        """, unsafe_allow_html=True)
    
    with col_status:
        # Live status indicator
        state_data = load_trade_state()
        is_active = state_data.get("status", False)
        st.markdown(f"""
            <div style="text-align: right; padding-top: 1rem;">
                <div class="live-indicator" style="display: inline-flex; 
                    background: {'rgba(0, 255, 136, 0.15)' if is_active else 'rgba(255, 193, 7, 0.15)'}; 
                    border-color: {'rgba(0, 255, 136, 0.3)' if is_active else 'rgba(255, 193, 7, 0.3)'};
                    color: {'#00ff88' if is_active else '#ffc107'};">
                    <span class="live-dot" style="background: {'#00ff88' if is_active else '#ffc107'};"></span>
                    {"LIVE" if is_active else "SCANNING"}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_logout:
        st.markdown("""
            <div style="text-align: right; padding-top: 0.5rem;">
        """, unsafe_allow_html=True)
        st.caption(f"👤 {st.session_state.get('username', 'User')}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Modern Sidebar
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🤖</span>
            <h3 style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">Control Panel</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # ==========================================================================
    # EMERGENCY PANIC BUTTON - Always visible at top of sidebar
    # ==========================================================================
    st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, rgba(235, 51, 73, 0.15) 0%, rgba(244, 92, 67, 0.15) 100%);
                    border: 1px solid rgba(235, 51, 73, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <p style="margin: 0 0 0.5rem 0; font-size: 0.8rem; color: rgba(255,255,255,0.6);">EMERGENCY CONTROLS</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Panic button with confirmation
    if 'panic_confirm' not in st.session_state:
        st.session_state.panic_confirm = False
    
    if not st.session_state.panic_confirm:
        if st.sidebar.button("🛑 EMERGENCY STOP ALL", type="primary", use_container_width=True):
            st.session_state.panic_confirm = True
            st.rerun()
    else:
        st.sidebar.error("⚠️ CONFIRM EMERGENCY STOP?")
        st.sidebar.warning("This will exit ALL positions and stop the bot.")
        
        col_yes, col_no = st.sidebar.columns(2)
        with col_yes:
            if st.button("✅ YES", type="primary", use_container_width=True):
                _execute_emergency_stop()
                st.session_state.panic_confirm = False
                st.rerun()
        with col_no:
            if st.button("❌ No", use_container_width=True):
                st.session_state.panic_confirm = False
                st.rerun()
    
    st.sidebar.divider()
    
    # Quick Stats in Sidebar
    daily_data = load_daily_pnl()
    pnl = daily_data.get("pnl", 0)
    pnl_color = "#00ff88" if pnl >= 0 else "#ff6b6b"
    
    st.sidebar.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
            <p style="margin: 0; font-size: 0.75rem; color: rgba(255,255,255,0.5);">TODAY'S P&L</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: 700; color: {pnl_color};">
                ₹{pnl:,.2f}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Refresh controls
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5s)", value=True)
    
    if st.sidebar.button("⟳ Refresh Now", use_container_width=True):
        st.rerun()
    
    st.sidebar.divider()
    
    # Navigation with icons
    st.sidebar.markdown("##### Navigation")
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🎯 Manual Trade", "⚙️ Settings", "🔌 WebSocket", "📈 Metrics", "📝 Trade Log"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    
    # System Status
    st.sidebar.markdown("##### System Status")
    
    files_status = {
        "State File": STATE_FILE,
        "Daily P&L": DAILY_PNL_FILE,
        "Trade History": TRADE_HISTORY_FILE,
        "WebSocket": WEBSOCKET_STATUS_FILE
    }
    
    for name, path in files_status.items():
        icon = "🟢" if path.exists() else "🔴"
        st.sidebar.caption(f"{icon} {name}")
    
    # Main content based on navigation
    if page == "📊 Dashboard":
        # Use fragment for auto-refresh section
        if auto_refresh:
            auto_refresh_data()
        else:
            render_live_status()
        
        st.divider()
        render_performance_analytics()
    
    elif page == "🎯 Manual Trade":
        render_manual_trade()
    
    elif page == "⚙️ Settings":
        render_settings_page()
        
    elif page == "🔌 WebSocket":
        render_websocket_status()
        st.divider()
        
        # Manual WebSocket status update form removed (testing cleanup)
    
    elif page == "📈 Metrics":
        render_prometheus_info()
        st.divider()
        
        # Show current metrics values
        st.markdown("### 📊 Current Metric Values")
        
        daily = load_daily_pnl()
        history = load_trade_history()
        state = load_trade_state()
        ws = load_websocket_status()
        
        # Modern metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Daily P&L", f"₹{daily.get('pnl', 0):,.2f}")
            st.metric("📊 Today's Trades", daily.get('trades', 0))
        
        with col2:
            win_rate = (daily.get('wins', 0) / max(daily.get('trades', 1), 1) * 100)
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
            st.metric("🤖 Active Trade", "Yes" if state.get('status') else "No")
        
        with col3:
            st.metric("🔌 WebSocket", "Connected" if ws.get('connected') else "Disconnected")
            st.metric("⚡ Latency", f"{ws.get('latency_ms', 0)}ms")
        
        with col4:
            st.metric("📜 Total Trades", len(history))
            total_hist_pnl = sum(t.get('pnl', 0) for t in history)
            st.metric("💎 Total P&L", f"₹{total_hist_pnl:,.2f}")
    
    elif page == "📝 Trade Log":
        render_trade_log()

if __name__ == "__main__":
    main()