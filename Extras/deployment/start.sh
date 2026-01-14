#!/bin/bash
# Quick start script - just set environment variables and run

# Check if credentials are set
if [ -z "$DHAN_CLIENT_ID" ]; then
    echo "ERROR: Credentials not set!"
    echo ""
    echo "Please set environment variables first:"
    echo ""
    echo "export DHAN_CLIENT_ID=\"your_client_id\""
    echo "export DHAN_ACCESS_TOKEN=\"your_access_token\""
    echo "export TELEGRAM_TOKEN=\"your_telegram_token\""
    echo "export TELEGRAM_CHAT_ID=\"your_chat_id\""
    echo ""
    echo "OR create a credentials.json file (see credentials.example.json)"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found. Install dependencies first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install dhanhq pandas pandas-ta requests"
    exit 1
fi

# Run the bot
echo "Starting Trading Bot..."

# Perform simple file backup before start (safety first)
if [ -f "data/trading_bot.db" ]; then
    mkdir -p Extras/backup
    cp "data/trading_bot.db" "Extras/backup/trading_bot_prestart_$(date +%Y%m%d_%H%M%S).db"
fi

python Tradebot.py
