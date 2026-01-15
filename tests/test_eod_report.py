"""
Unit tests for eod_report.py
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
from email.mime.multipart import MIMEMultipart

from eod_report import (
    calculate_brokerage,
    generate_html_report,
    send_report_email,
    generate_and_send_eod_report
)


class TestEODReport:
    """Test cases for end-of-day report functionality."""

    def test_calculate_brokerage(self):
        """Test brokerage calculation for different instruments."""
        # Test CRUDEOIL brokerage
        result = calculate_brokerage("CRUDEOIL", 100000.0)

        expected_keys = ['brokerage', 'stt', 'exchange_txn', 'gst', 'sebi', 'stamp_duty', 'total']
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (int, float))

        # Test NATURALGAS brokerage
        result_ng = calculate_brokerage("NATURALGAS", 50000.0)

        for key in expected_keys:
            assert key in result_ng
            assert isinstance(result_ng[key], (int, float))

    @patch('eod_report.load_trade_history')
    @patch('eod_report.load_daily_pnl')
    @patch('eod_report.get_performance_stats')
    def test_generate_html_report_content(self, mock_perf_stats, mock_daily_pnl, mock_trade_history):
        """Test HTML report generation with P&L calculations."""
        # Mock data
        trades = [
            {
                'instrument': 'CRUDEOIL',
                'signal': 'BUY',
                'entry_price': 5000.0,
                'exit_price': 5100.0,
                'quantity': 10,
                'pnl': 1000.0,
                'entry_time': '2026-01-15 10:00:00',
                'exit_time': '2026-01-15 11:00:00'
            }
        ]

        daily_pnl = {
            'total_pnl': 1000.0,
            'total_trades': 1,
            'winning_trades': 1,
            'losing_trades': 0,
            'win_rate': 100.0
        }

        performance_stats = {
            'sharpe_ratio': 1.5,
            'max_drawdown': 500.0,
            'total_return': 1000.0
        }

        # Generate report
        html_content = generate_html_report(trades, daily_pnl, performance_stats)

        # Verify content contains expected elements
        assert isinstance(html_content, str)
        assert 'CRUDEOIL' in html_content
        assert '1000.0' in html_content  # P&L

    @patch('eod_report.EMAIL_ADDRESS', 'test@example.com')
    @patch('eod_report.EMAIL_PASSWORD', 'password')
    @patch('smtplib.SMTP')
    def test_send_email_success(self, mock_smtp):
        """Test successful email sending."""
        mock_smtp_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

        trades = [{'instrument': 'CRUDEOIL', 'pnl': 1000.0}]
        daily_pnl = {'total_pnl': 1000.0}

        result = send_report_email(trades, daily_pnl)

        assert result == True

        # Verify SMTP calls
        mock_smtp.assert_called_once_with('smtp.gmail.com', 587)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once()
        mock_smtp_instance.send_message.assert_called_once()

    @patch('smtplib.SMTP')
    def test_send_email_failure(self, mock_smtp):
        """Test email sending failure."""
        mock_smtp.side_effect = Exception("SMTP connection failed")

        trades = [{'instrument': 'CRUDEOIL', 'pnl': 1000.0}]
        daily_pnl = {'total_pnl': 1000.0}

        result = send_report_email(trades, daily_pnl)

        assert result == False

    @patch('eod_report.send_report_email')
    @patch('eod_report.generate_html_report')
    @patch('eod_report.get_todays_trades')
    @patch('eod_report.load_daily_pnl')
    @patch('eod_report.get_performance_stats')
    @patch('eod_report.save_report_locally')
    def test_generate_and_send_eod_report(self, mock_save, mock_perf, mock_daily_pnl, mock_get_trades, mock_generate_html, mock_send_email):
        """Test the complete EOD report generation and sending process."""
        mock_get_trades.return_value = [{'instrument': 'CRUDEOIL', 'pnl': 1000.0}]
        mock_daily_pnl.return_value = {'total_pnl': 1000.0}
        mock_perf.return_value = {'sharpe_ratio': 1.5}
        mock_generate_html.return_value = "<html>Test</html>"
        mock_send_email.return_value = True

        generate_and_send_eod_report()

        assert mock_get_trades.call_count >= 1
        assert mock_daily_pnl.call_count >= 1
        assert mock_perf.call_count >= 1
        mock_save.assert_called_once()
        mock_send_email.assert_called_once()