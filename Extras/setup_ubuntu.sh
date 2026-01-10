#!/bin/bash
# Quick setup script for Ubuntu server

echo "=== Trading Bot Setup for Ubuntu ==="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Warning: This script is designed for Ubuntu/Linux"
fi

# Function to prompt for credentials
setup_credentials() {
    echo "Choose credential setup method:"
    echo "1. Environment variables (Recommended for servers)"
    echo "2. credentials.json file (Quick local setup)"
    read -p "Enter choice [1-2]: " choice
    
    if [ "$choice" == "1" ]; then
        echo ""
        echo "Enter your credentials (they will be added to ~/.bashrc):"
        read -p "DHAN Client ID: " client_id
        read -p "DHAN Access Token: " access_token
        read -p "Telegram Bot Token: " telegram_token
        read -p "Telegram Chat ID: " chat_id
        
        echo "" >> ~/.bashrc
        echo "# Trading Bot Credentials" >> ~/.bashrc
        echo "export DHAN_CLIENT_ID=\"$client_id\"" >> ~/.bashrc
        echo "export DHAN_ACCESS_TOKEN=\"$access_token\"" >> ~/.bashrc
        echo "export TELEGRAM_TOKEN=\"$telegram_token\"" >> ~/.bashrc
        echo "export TELEGRAM_CHAT_ID=\"$chat_id\"" >> ~/.bashrc
        
        source ~/.bashrc
        
        echo ""
        echo "✓ Environment variables added to ~/.bashrc"
        echo "  Run 'source ~/.bashrc' in existing terminals"
        
    elif [ "$choice" == "2" ]; then
        if [ -f "credentials.example.json" ]; then
            cp credentials.example.json credentials.json
            echo ""
            echo "✓ Created credentials.json from example"
            echo "  Please edit credentials.json with your actual credentials"
            echo "  Run: nano credentials.json"
        else
            echo "Error: credentials.example.json not found"
            exit 1
        fi
    else
        echo "Invalid choice"
        exit 1
    fi
}

# Update system
echo "Step 1: Updating system packages..."
sudo apt update

# Install Python and pip
echo ""
echo "Step 2: Installing Python 3 and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
echo ""
echo "Step 3: Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Step 4: Installing Python packages..."
pip install --upgrade pip
pip install dhanhq pandas pandas-ta requests

# Setup credentials
echo ""
echo "Step 5: Setting up credentials..."
setup_credentials

# Create systemd service
echo ""
read -p "Do you want to create a systemd service to run bot automatically? [y/N]: " create_service

if [[ "$create_service" =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/tradingbot.service"
    CURRENT_USER=$(whoami)
    CURRENT_DIR=$(pwd)
    
    sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Trading Bot
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
Environment="PATH=$CURRENT_DIR/venv/bin"
ExecStart=$CURRENT_DIR/venv/bin/python Tradebot.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable tradingbot
    
    echo ""
    echo "✓ Systemd service created"
    echo "  Start: sudo systemctl start tradingbot"
    echo "  Stop:  sudo systemctl stop tradingbot"
    echo "  Status: sudo systemctl status tradingbot"
    echo "  Logs:  sudo journalctl -u tradingbot -f"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run the bot manually:"
echo "  source venv/bin/activate"
echo "  python Tradebot.py"
echo ""
echo "To run as service:"
echo "  sudo systemctl start tradingbot"
echo ""
