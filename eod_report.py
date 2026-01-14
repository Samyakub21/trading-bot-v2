# =============================================================================
# END OF DAY REPORT - Daily Trading Summary Generator
# =============================================================================
# Generates PDF/HTML reports with daily trades, P&L, and brokerage costs
# Schedules automatic report generation at 11:35 PM
# =============================================================================

import io
import json
import logging
import os
import smtplib
import schedule
import threading
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import config
from utils import (
    load_trade_history, load_daily_pnl, get_performance_stats,
    TRADE_HISTORY_FILE, DAILY_PNL_FILE, send_alert
)
from instruments import INSTRUMENTS

# =============================================================================
# CONFIGURATION
# =============================================================================
# Email configuration (from environment or config)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')  # App password for Gmail
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', EMAIL_ADDRESS)

# Report settings
REPORT_TIME = os.getenv('EOD_REPORT_TIME', '23:35')  # 11:35 PM
REPORT_DIR = Path(__file__).parent / 'reports'

# Brokerage costs (per lot)
BROKERAGE_COSTS = {
    'CRUDEOIL': {'brokerage': 20, 'stt': 0, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.03},
    'NATURALGAS': {'brokerage': 20, 'stt': 0, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.03},
    'GOLD': {'brokerage': 20, 'stt': 0, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.03},
    'SILVER': {'brokerage': 20, 'stt': 0, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.03},
    'NIFTY': {'brokerage': 20, 'stt': 0.05, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.015},
    'BANKNIFTY': {'brokerage': 20, 'stt': 0.05, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.015},
    'DEFAULT': {'brokerage': 20, 'stt': 0, 'exchange_txn': 0.05, 'gst': 3.6, 'sebi': 0.1, 'stamp': 0.03}
}


def calculate_brokerage(instrument: str, turnover: float) -> Dict[str, float]:
    """
    Calculate brokerage and other charges for a trade.
    
    Args:
        instrument: Instrument name
        turnover: Total turnover (entry_value + exit_value)
        
    Returns:
        Dict with breakdown of charges
    """
    costs = BROKERAGE_COSTS.get(instrument, BROKERAGE_COSTS['DEFAULT'])
    
    brokerage = costs['brokerage'] * 2  # Entry + Exit
    stt = turnover * costs['stt'] / 100
    exchange_txn = turnover * costs['exchange_txn'] / 100
    gst = (brokerage + exchange_txn) * 18 / 100  # GST on brokerage + txn charges
    sebi = turnover * costs['sebi'] / 100000  # SEBI charges per crore
    stamp = turnover * costs['stamp'] / 100
    
    total = brokerage + stt + exchange_txn + gst + sebi + stamp
    
    return {
        'brokerage': round(brokerage, 2),
        'stt': round(stt, 2),
        'exchange_txn': round(exchange_txn, 2),
        'gst': round(gst, 2),
        'sebi': round(sebi, 2),
        'stamp_duty': round(stamp, 2),
        'total': round(total, 2)
    }


def get_todays_trades() -> List[Dict[str, Any]]:
    """Get all trades from today."""
    history = load_trade_history()
    today = datetime.now().strftime('%Y-%m-%d')
    
    todays_trades = []
    for trade in history:
        exit_time = trade.get('exit_time', '')
        if exit_time.startswith(today):
            todays_trades.append(trade)
    
    return todays_trades


def generate_html_report(
    trades: List[Dict[str, Any]],
    daily_pnl: Dict[str, Any],
    performance_stats: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate an HTML report for the day's trading activity.
    
    Args:
        trades: List of trade records
        daily_pnl: Daily P&L data
        performance_stats: Overall performance statistics
        
    Returns:
        HTML string
    """
    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate totals
    total_pnl = sum(t.get('pnl', 0) for t in trades)
    total_brokerage = 0
    
    # Process trades for display
    trade_rows = []
    for trade in trades:
        instrument = trade.get('instrument', 'N/A')
        lot_size = trade.get('lot_size', INSTRUMENTS.get(instrument, {}).get('lot_size', 1))
        
        # Calculate turnover and brokerage
        option_entry = trade.get('option_entry', 0)
        option_exit = trade.get('option_exit', 0)
        turnover = (option_entry + option_exit) * lot_size
        
        brokerage = calculate_brokerage(instrument, turnover)
        total_brokerage += brokerage['total']
        
        pnl = trade.get('pnl', 0)
        net_pnl = pnl - brokerage['total']
        
        result_class = 'profit' if pnl > 0 else 'loss'
        result_emoji = '✅' if pnl > 0 else '❌'
        
        trade_rows.append(f"""
        <tr class="{result_class}">
            <td>{trade.get('entry_time', 'N/A')[-8:]}</td>
            <td>{trade.get('exit_time', 'N/A')[-8:]}</td>
            <td><strong>{instrument}</strong></td>
            <td>{trade.get('trade_type', 'N/A')}</td>
            <td>{trade.get('option_type', 'N/A')}</td>
            <td>₹{option_entry:.2f}</td>
            <td>₹{option_exit:.2f}</td>
            <td>₹{brokerage['total']:.2f}</td>
            <td class="{result_class}-text">{result_emoji} ₹{pnl:.2f}</td>
            <td class="{result_class}-text">₹{net_pnl:.2f}</td>
            <td>{trade.get('exit_reason', 'N/A')}</td>
        </tr>
        """)
    
    trade_rows_html = '\n'.join(trade_rows) if trade_rows else '<tr><td colspan="11">No trades today</td></tr>'
    
    net_pnl = total_pnl - total_brokerage
    
    # Performance summary
    wins = daily_pnl.get('wins', 0)
    losses = daily_pnl.get('losses', 0)
    total_trades = daily_pnl.get('trades', len(trades))
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Overall stats section
    overall_stats_html = ""
    if performance_stats:
        overall_stats_html = f"""
        <div class="stats-section">
            <h3>📊 Overall Performance (All Time)</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{performance_stats.get('total_trades', 0)}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{performance_stats.get('win_rate', 0):.1f}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">₹{performance_stats.get('total_pnl', 0):,.2f}</div>
                    <div class="stat-label">Total P&L</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{performance_stats.get('profit_factor', 0):.2f}</div>
                    <div class="stat-label">Profit Factor</div>
                </div>
            </div>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Trading Report - {today}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #e0e0e0;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                padding: 30px;
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                margin-bottom: 20px;
            }}
            .header h1 {{
                color: #00d4ff;
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            .header .date {{
                color: #888;
                font-size: 1.2em;
            }}
            .summary-cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: rgba(255,255,255,0.08);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .card.profit {{
                border-color: #00c853;
                background: rgba(0,200,83,0.1);
            }}
            .card.loss {{
                border-color: #ff5252;
                background: rgba(255,82,82,0.1);
            }}
            .card-value {{
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .card-value.profit {{ color: #00c853; }}
            .card-value.loss {{ color: #ff5252; }}
            .card-label {{
                color: #888;
                font-size: 0.9em;
                text-transform: uppercase;
            }}
            .trades-table {{
                width: 100%;
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                overflow: hidden;
                margin-bottom: 30px;
            }}
            .trades-table table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .trades-table th {{
                background: rgba(0,212,255,0.2);
                color: #00d4ff;
                padding: 15px 10px;
                text-align: left;
                font-size: 0.85em;
                text-transform: uppercase;
            }}
            .trades-table td {{
                padding: 12px 10px;
                border-bottom: 1px solid rgba(255,255,255,0.05);
                font-size: 0.9em;
            }}
            .trades-table tr:hover {{
                background: rgba(255,255,255,0.05);
            }}
            .profit-text {{ color: #00c853; font-weight: bold; }}
            .loss-text {{ color: #ff5252; font-weight: bold; }}
            .stats-section {{
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
            }}
            .stats-section h3 {{
                color: #00d4ff;
                margin-bottom: 20px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                padding: 15px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 1.5em;
                font-weight: bold;
                color: #00d4ff;
            }}
            .stat-label {{
                color: #888;
                font-size: 0.8em;
                margin-top: 5px;
            }}
            .footer {{
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.8em;
            }}
            .brokerage-breakdown {{
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 20px;
            }}
            .brokerage-breakdown h3 {{
                color: #ffa726;
                margin-bottom: 15px;
            }}
            .brokerage-item {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }}
            .brokerage-total {{
                font-weight: bold;
                color: #ffa726;
                font-size: 1.1em;
                border-top: 2px solid rgba(255,255,255,0.1);
                margin-top: 10px;
                padding-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Trading Bot Report</h1>
                <div class="date">{today}</div>
            </div>
            
            <div class="summary-cards">
                <div class="card {'profit' if net_pnl > 0 else 'loss'}">
                    <div class="card-value {'profit' if net_pnl > 0 else 'loss'}">₹{net_pnl:,.2f}</div>
                    <div class="card-label">Net P&L (After Charges)</div>
                </div>
                <div class="card">
                    <div class="card-value" style="color: #00d4ff;">{total_trades}</div>
                    <div class="card-label">Total Trades</div>
                </div>
                <div class="card profit">
                    <div class="card-value profit">{wins}</div>
                    <div class="card-label">Winning Trades</div>
                </div>
                <div class="card loss">
                    <div class="card-value loss">{losses}</div>
                    <div class="card-label">Losing Trades</div>
                </div>
                <div class="card">
                    <div class="card-value" style="color: #ffa726;">{win_rate:.1f}%</div>
                    <div class="card-label">Win Rate</div>
                </div>
            </div>
            
            <div class="trades-table">
                <table>
                    <thead>
                        <tr>
                            <th>Entry Time</th>
                            <th>Exit Time</th>
                            <th>Instrument</th>
                            <th>Type</th>
                            <th>Option</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>Charges</th>
                            <th>Gross P&L</th>
                            <th>Net P&L</th>
                            <th>Exit Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows_html}
                    </tbody>
                </table>
            </div>
            
            <div class="brokerage-breakdown">
                <h3>💰 Brokerage & Charges Summary</h3>
                <div class="brokerage-item">
                    <span>Gross P&L:</span>
                    <span class="{'profit-text' if total_pnl > 0 else 'loss-text'}">₹{total_pnl:,.2f}</span>
                </div>
                <div class="brokerage-item">
                    <span>Total Brokerage & Charges:</span>
                    <span style="color: #ffa726;">₹{total_brokerage:,.2f}</span>
                </div>
                <div class="brokerage-item brokerage-total">
                    <span>Net P&L:</span>
                    <span class="{'profit-text' if net_pnl > 0 else 'loss-text'}">₹{net_pnl:,.2f}</span>
                </div>
            </div>
            
            {overall_stats_html}
            
            <div class="footer">
                <p>Generated by Trading Bot • {now}</p>
                <p>This is an automated report. Do not reply to this email.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_pdf_report(
    trades: List[Dict[str, Any]],
    daily_pnl: Dict[str, Any],
    performance_stats: Optional[Dict[str, Any]] = None
) -> Optional[bytes]:
    """
    Generate a PDF report from the HTML content.
    
    Requires weasyprint or pdfkit to be installed.
    Falls back to HTML if PDF generation fails.
    
    Returns:
        PDF bytes or None if generation fails
    """
    try:
        # Try weasyprint first
        try:
            from weasyprint import HTML
            html_content = generate_html_report(trades, daily_pnl, performance_stats)
            pdf_bytes = HTML(string=html_content).write_pdf()
            return pdf_bytes
        except ImportError:
            pass
        
        # Try pdfkit as fallback
        try:
            import pdfkit
            html_content = generate_html_report(trades, daily_pnl, performance_stats)
            pdf_bytes = pdfkit.from_string(html_content, False)
            return pdf_bytes
        except ImportError:
            pass
        
        logging.warning("PDF generation libraries not available (weasyprint or pdfkit)")
        return None
        
    except Exception as e:
        logging.error(f"PDF generation error: {e}")
        return None


def save_report_locally(
    trades: List[Dict[str, Any]],
    daily_pnl: Dict[str, Any],
    performance_stats: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Save the report locally as HTML file.
    
    Returns:
        Path to saved file or None on failure
    """
    try:
        # Ensure reports directory exists
        REPORT_DIR.mkdir(exist_ok=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"trading_report_{today}.html"
        filepath = REPORT_DIR / filename
        
        html_content = generate_html_report(trades, daily_pnl, performance_stats)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"Report saved: {filepath}")
        return str(filepath)
        
    except Exception as e:
        logging.error(f"Failed to save report locally: {e}")
        return None


def send_report_email(
    trades: List[Dict[str, Any]],
    daily_pnl: Dict[str, Any],
    performance_stats: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Send the daily report via email.
    
    Returns:
        True if email sent successfully
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        logging.warning("Email credentials not configured. Skipping email report.")
        return False
    
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        net_pnl = sum(t.get('pnl', 0) for t in trades)
        
        # Create email message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"📊 Trading Report - {today} | Net P&L: ₹{net_pnl:,.2f}"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_RECIPIENT
        
        # Generate HTML content
        html_content = generate_html_report(trades, daily_pnl, performance_stats)
        msg.attach(MIMEText(html_content, 'html'))
        
        # Try to attach PDF
        pdf_bytes = generate_pdf_report(trades, daily_pnl, performance_stats)
        if pdf_bytes:
            pdf_attachment = MIMEBase('application', 'pdf')
            pdf_attachment.set_payload(pdf_bytes)
            encoders.encode_base64(pdf_attachment)
            pdf_attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="trading_report_{today}.pdf"'
            )
            msg.attach(pdf_attachment)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logging.info(f"📧 EOD report sent to {EMAIL_RECIPIENT}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        logging.error("Email authentication failed. Check EMAIL_ADDRESS and EMAIL_PASSWORD.")
        return False
    except smtplib.SMTPException as e:
        logging.error(f"SMTP error: {e}")
        return False
    except Exception as e:
        logging.error(f"Email send error: {e}")
        return False


def send_report_telegram() -> bool:
    """
    Send a summary of the daily report to Telegram.
    
    Returns:
        True if sent successfully
    """
    try:
        trades = get_todays_trades()
        daily_pnl = load_daily_pnl()
        
        if not trades:
            send_alert("📊 *End of Day Report*\n\nNo trades executed today.")
            return True
        
        # Calculate summary
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        total_brokerage = 0
        
        for trade in trades:
            instrument = trade.get('instrument', 'DEFAULT')
            lot_size = trade.get('lot_size', 1)
            option_entry = trade.get('option_entry', 0)
            option_exit = trade.get('option_exit', 0)
            turnover = (option_entry + option_exit) * lot_size
            brokerage = calculate_brokerage(instrument, turnover)
            total_brokerage += brokerage['total']
        
        net_pnl = total_pnl - total_brokerage
        wins = daily_pnl.get('wins', 0)
        losses = daily_pnl.get('losses', 0)
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Build message
        emoji = "📈" if net_pnl > 0 else "📉"
        result_emoji = "✅" if net_pnl > 0 else "❌"
        
        message = f"""
{emoji} *End of Day Report*
━━━━━━━━━━━━━━━━━━
📅 Date: {datetime.now().strftime('%Y-%m-%d')}

💰 *P&L Summary*
• Gross P&L: ₹{total_pnl:,.2f}
• Brokerage: ₹{total_brokerage:,.2f}
• *Net P&L: {result_emoji} ₹{net_pnl:,.2f}*

📊 *Statistics*
• Total Trades: {total_trades}
• Wins: {wins} | Losses: {losses}
• Win Rate: {win_rate:.1f}%

📝 *Trade Details*
"""
        
        # Add individual trade summaries
        for trade in trades[-5:]:  # Last 5 trades
            trade_emoji = "✅" if trade.get('pnl', 0) > 0 else "❌"
            message += f"{trade_emoji} {trade.get('instrument', 'N/A')} {trade.get('option_type', '')}: ₹{trade.get('pnl', 0):,.2f}\n"
        
        if len(trades) > 5:
            message += f"\n... and {len(trades) - 5} more trades"
        
        message += "\n━━━━━━━━━━━━━━━━━━"
        
        send_alert(message)
        return True
        
    except Exception as e:
        logging.error(f"Telegram report error: {e}")
        return False


def generate_and_send_eod_report() -> None:
    """
    Main function to generate and distribute the EOD report.
    
    This is called by the scheduler at 11:35 PM.
    """
    logging.info("🕐 Generating End of Day Report...")
    
    try:
        trades = get_todays_trades()
        daily_pnl = load_daily_pnl()
        performance_stats = get_performance_stats()
        
        # Save report locally
        save_report_locally(trades, daily_pnl, performance_stats)
        
        # Send via email
        send_report_email(trades, daily_pnl, performance_stats)
        
        # Send summary to Telegram
        send_report_telegram()
        
        logging.info("✅ End of Day Report completed")
        
    except Exception as e:
        logging.error(f"EOD Report generation failed: {e}")
        send_alert(f"⚠️ *EOD Report Failed*\nError: {str(e)[:100]}")


def schedule_eod_report() -> threading.Thread:
    """
    Schedule the EOD report job.
    
    Returns:
        Thread running the scheduler
    """
    schedule.every().day.at(REPORT_TIME).do(generate_and_send_eod_report)
    
    logging.info(f"📅 EOD Report scheduled for {REPORT_TIME} daily")
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    return scheduler_thread


def stop_eod_report() -> None:
    """Clear all scheduled jobs."""
    schedule.clear()
    logging.info("EOD Report scheduling stopped")


# =============================================================================
# STANDALONE TESTING
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Create sample trades for testing
    sample_trades = [
        {
            'instrument': 'CRUDEOIL',
            'entry_time': f"{datetime.now().strftime('%Y-%m-%d')} 10:15:00",
            'exit_time': f"{datetime.now().strftime('%Y-%m-%d')} 11:30:00",
            'trade_type': 'BUY',
            'option_type': 'CE',
            'future_entry': 5250,
            'future_exit': 5280,
            'option_entry': 120,
            'option_exit': 155,
            'initial_sl': 5230,
            'final_sl': 5250,
            'pnl': 3500,
            'r_multiple': 1.5,
            'exit_reason': 'Target Hit',
            'lot_size': 100
        },
        {
            'instrument': 'CRUDEOIL',
            'entry_time': f"{datetime.now().strftime('%Y-%m-%d')} 14:00:00",
            'exit_time': f"{datetime.now().strftime('%Y-%m-%d')} 15:15:00",
            'trade_type': 'SELL',
            'option_type': 'PE',
            'future_entry': 5300,
            'future_exit': 5320,
            'option_entry': 140,
            'option_exit': 105,
            'initial_sl': 5320,
            'final_sl': 5320,
            'pnl': -3500,
            'r_multiple': -1.0,
            'exit_reason': 'Stop Loss Hit',
            'lot_size': 100
        }
    ]
    
    sample_daily_pnl = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'pnl': 0,
        'trades': 2,
        'wins': 1,
        'losses': 1
    }
    
    # Generate and save test report
    html = generate_html_report(sample_trades, sample_daily_pnl)
    
    REPORT_DIR.mkdir(exist_ok=True)
    test_file = REPORT_DIR / 'test_report.html'
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Test report saved to: {test_file}")
    print("Open in browser to preview the report.")
