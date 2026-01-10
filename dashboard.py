"""
Algo Trading Dashboard with Authentication, WebSocket Status, and Prometheus Metrics
Fixed issues:
- Removed st.rerun() infinite loop (using st.fragment for selective refresh)
- Updated file paths to new combined naming scheme
- Added basic authentication
- Improved error handling for missing files
- Upgraded to bcrypt password hashing with secure password policy
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
DATA_DIR = Path(__file__).parent
STATE_FILE = DATA_DIR / "trade_state_active.json"
DAILY_PNL_FILE = DATA_DIR / "daily_pnl_combined.json"
TRADE_HISTORY_FILE = DATA_DIR / "trade_history_combined.json"
WEBSOCKET_STATUS_FILE = DATA_DIR / "websocket_status.json"

# Prometheus metrics port
PROMETHEUS_PORT = 8000

# Instruments list (should match Tradebot.py)
INSTRUMENTS = ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER", "NIFTY", "BANKNIFTY"]

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
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 40px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        .security-notice {
            font-size: 0.8em;
            color: #888;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🔐 Trading Dashboard Login")
        st.markdown("---")
        
        # Check rate limit
        is_allowed, lockout_remaining = check_login_rate_limit()
        
        if not is_allowed:
            minutes = lockout_remaining // 60
            seconds = lockout_remaining % 60
            st.error(f"🔒 Too many failed attempts. Please try again in {minutes}m {seconds}s")
            st.info("💡 If you forgot your password, contact the administrator.")
            return
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
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
            st.markdown('<p class="security-notice">🔒 Secured with bcrypt password hashing</p>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<p class="security-notice">⚠️ Using fallback security (install bcrypt for better protection)</p>', 
                       unsafe_allow_html=True)

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
    """Render WebSocket connection status dashboard"""
    st.subheader("🔌 WebSocket Status")
    
    ws_status = load_websocket_status()
    
    # Connection status indicator
    col1, col2, col3, col4 = st.columns(4)
    
    is_connected = ws_status.get("connected", False)
    
    with col1:
        status_icon = "🟢" if is_connected else "🔴"
        status_text = "Connected" if is_connected else "Disconnected"
        st.metric(
            label="Connection Status",
            value=f"{status_icon} {status_text}"
        )
        
    with col2:
        latency = ws_status.get("latency_ms", 0)
        latency_delta = None
        if latency > 0:
            if latency < 50:
                latency_delta = "Excellent"
            elif latency < 100:
                latency_delta = "Good"
            elif latency < 250:
                latency_delta = "Fair"
            else:
                latency_delta = "Poor"
        st.metric(
            label="Latency",
            value=f"{latency:.0f} ms" if latency else "N/A",
            delta=latency_delta
        )
        
    with col3:
        messages = ws_status.get("messages_received", 0)
        st.metric(
            label="Messages Received",
            value=f"{messages:,}"
        )
        
    with col4:
        errors = ws_status.get("errors", 0)
        reconnects = ws_status.get("reconnect_count", 0)
        st.metric(
            label="Errors / Reconnects",
            value=f"{errors} / {reconnects}",
            delta="⚠️" if errors > 0 else None,
            delta_color="inverse"
        )
    
    # Last message time
    last_msg_time = ws_status.get("last_message_time")
    if last_msg_time:
        try:
            last_msg_dt = datetime.fromisoformat(last_msg_time)
            time_ago = datetime.now() - last_msg_dt
            if time_ago.total_seconds() < 60:
                time_str = f"{int(time_ago.total_seconds())}s ago"
            elif time_ago.total_seconds() < 3600:
                time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
            st.caption(f"📡 Last message: {time_str}")
        except:
            pass
    
    # Subscribed symbols and last prices
    with st.expander("📊 Live Prices", expanded=is_connected):
        subscribed = ws_status.get("subscribed_symbols", [])
        last_prices = ws_status.get("last_prices", {})
        
        if last_prices:
            price_df = pd.DataFrame([
                {
                    "Symbol": symbol,
                    "LTP": f"₹{price.get('ltp', 0):,.2f}",
                    "Change": f"{price.get('change_pct', 0):+.2f}%",
                    "Volume": f"{price.get('volume', 0):,}",
                    "Updated": price.get('timestamp', 'N/A')
                }
                for symbol, price in last_prices.items()
            ])
            st.dataframe(price_df, use_container_width=True, hide_index=True)
        else:
            st.info("No live price data available. WebSocket may be disconnected.")
    
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
    """Render live trading status section"""
    st.subheader("📡 Live Trading Status")
    
    state_data = load_trade_state()
    daily_data = load_daily_pnl()
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Bot Status
    with col1:
        is_active = state_data.get("status", False)
        instrument = state_data.get("instrument", "N/A")
        
        if is_active:
            st.metric(
                label="Bot Status",
                value="🟢 ACTIVE TRADE",
                delta=instrument
            )
        else:
            st.metric(
                label="Bot Status",
                value="💤 Scanning...",
                delta="Waiting for signal"
            )
    
    # Current Trade Info
    with col2:
        if is_active:
            trade_type = state_data.get("type", "N/A")
            entry = state_data.get("entry", 0)
            st.metric(
                label="Trade Type",
                value=trade_type,
                delta=f"Entry: ₹{entry:,.2f}" if entry else None
            )
        else:
            st.metric(label="Trade Type", value="N/A")
    
    # Daily P&L
    with col3:
        pnl = daily_data.get("pnl", 0)
        st.metric(
            label="Day's P&L",
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
            label="Today's Stats",
            value=f"{trades} Trades",
            delta=f"{wins}W / {losses}L ({win_rate:.0f}%)"
        )
        
        # Update Prometheus
        if PROMETHEUS_AVAILABLE:
            try:
                ACTIVE_TRADE_GAUGE.set(1 if is_active else 0)
                WIN_RATE_GAUGE.labels(instrument='combined').set(win_rate)
            except:
                pass

def render_performance_analytics():
    """Render performance analytics section"""
    st.subheader("📊 Performance Analytics")
    
    history_data = load_trade_history()
    
    if not history_data:
        st.info("📭 No trade history found yet. Start the bot to generate data.")
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
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_equity = px.line(
            df_sorted,
            x='exit_time',
            y='cumulative_pnl',
            title='Equity Curve (Cumulative P&L)',
            markers=True
        )
        fig_equity.update_traces(line_color='#00CC96')
        fig_equity.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (₹)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_equity, use_container_width=True)
    
    with chart_col2:
        fig_pie = px.pie(
            names=['Wins', 'Losses'],
            values=[win_count, loss_count],
            title='Win/Loss Distribution',
            color_discrete_sequence=['#00CC96', '#EF553B']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # P&L by instrument chart (if multiple instruments)
    if 'instrument' in df.columns and len(df['instrument'].unique()) > 1:
        pnl_by_instrument = df.groupby('instrument')['pnl'].sum().reset_index()
        fig_bar = px.bar(
            pnl_by_instrument,
            x='instrument',
            y='pnl',
            title='P&L by Instrument',
            color='pnl',
            color_continuous_scale=['#EF553B', '#FFFF00', '#00CC96']
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
    """Main dashboard application"""
    
    # Page Config
    st.set_page_config(
        page_title="Algo Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Start Prometheus server
    start_prometheus_server()
    
    # Check authentication
    if not st.session_state.get('authenticated', False):
        login_form()
        return
    
    # Title with logout
    col_title, col_logout = st.columns([4, 1])
    with col_title:
        st.title("🤖 Algo Trading Live Dashboard")
    with col_logout:
        st.caption(f"👤 {st.session_state.get('username', 'User')}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Refresh rate selector (for manual refresh)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    st.sidebar.divider()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🔌 WebSocket", "📈 Metrics", "📝 Trade Log"],
        index=0
    )
    
    st.sidebar.divider()
    
    # File status indicators
    st.sidebar.markdown("**📁 Data Files:**")
    
    files_status = {
        "State": STATE_FILE,
        "Daily P&L": DAILY_PNL_FILE,
        "History": TRADE_HISTORY_FILE,
        "WebSocket": WEBSOCKET_STATUS_FILE
    }
    
    for name, path in files_status.items():
        icon = "✅" if path.exists() else "❌"
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
        
    elif page == "🔌 WebSocket":
        render_websocket_status()
        st.divider()
        
        # Manual WebSocket status update form
        with st.expander("🔧 Manual Status Update (Testing)", expanded=False):
            st.warning("This is for testing purposes only.")
            
            with st.form("ws_status_form"):
                ws_connected = st.checkbox("Connected", value=True)
                ws_latency = st.number_input("Latency (ms)", min_value=0, value=50)
                
                if st.form_submit_button("Update Status"):
                    test_status = {
                        "connected": ws_connected,
                        "last_message_time": datetime.now().isoformat(),
                        "latency_ms": ws_latency,
                        "messages_received": 1000,
                        "errors": 0,
                        "reconnect_count": 0,
                        "subscribed_symbols": ["CRUDEOIL", "GOLD"],
                        "last_prices": {
                            "CRUDEOIL": {"ltp": 6500.50, "change_pct": 1.25, "volume": 15000},
                            "GOLD": {"ltp": 72500.00, "change_pct": -0.35, "volume": 8000}
                        }
                    }
                    with open(WEBSOCKET_STATUS_FILE, 'w') as f:
                        json.dump(test_status, f, indent=2)
                    st.success("Status updated!")
                    st.rerun()
    
    elif page == "📈 Metrics":
        render_prometheus_info()
        st.divider()
        
        # Show current metrics values
        st.subheader("📊 Current Metric Values")
        
        daily = load_daily_pnl()
        history = load_trade_history()
        state = load_trade_state()
        ws = load_websocket_status()
        
        metrics_data = {
            "Daily P&L": f"₹{daily.get('pnl', 0):,.2f}",
            "Total Trades (Today)": daily.get('trades', 0),
            "Win Rate (Today)": f"{(daily.get('wins', 0) / max(daily.get('trades', 1), 1) * 100):.1f}%",
            "Active Trade": "Yes" if state.get('status') else "No",
            "WebSocket Connected": "Yes" if ws.get('connected') else "No",
            "WebSocket Latency": f"{ws.get('latency_ms', 0)}ms",
            "Historical Trades": len(history),
            "Total Historical P&L": f"₹{sum(t.get('pnl', 0) for t in history):,.2f}"
        }
        
        col1, col2 = st.columns(2)
        
        for i, (metric, value) in enumerate(metrics_data.items()):
            with col1 if i % 2 == 0 else col2:
                st.metric(metric, value)
    
    elif page == "📝 Trade Log":
        render_trade_log()

if __name__ == "__main__":
    main()