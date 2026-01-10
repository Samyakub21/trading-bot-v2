# Trading Bot Deployment Guide

## Quick Setup

### Option 1: Using Environment Variables (Recommended for Ubuntu Server)

1. Set environment variables before running the bot:

```bash
export DHAN_CLIENT_ID="your_client_id"
export DHAN_ACCESS_TOKEN="your_access_token"
export TELEGRAM_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Run the bot
python Tradebot.py
```

2. For persistent environment variables, add to `~/.bashrc` or `~/.profile`:

```bash
echo 'export DHAN_CLIENT_ID="your_client_id"' >> ~/.bashrc
echo 'export DHAN_ACCESS_TOKEN="your_access_token"' >> ~/.bashrc
echo 'export TELEGRAM_TOKEN="your_telegram_token"' >> ~/.bashrc
echo 'export TELEGRAM_CHAT_ID="your_chat_id"' >> ~/.bashrc
source ~/.bashrc
```

### Option 2: Using credentials.json (Quick Local Testing)

1. Copy the example file:
```bash
cp credentials.example.json credentials.json
```

2. Edit `credentials.json` with your actual credentials:
```json
{
    "CLIENT_ID": "1108779477",
    "ACCESS_TOKEN": "your_actual_token_here",
    "TELEGRAM_TOKEN": "8479812357:AAHCDeJ...",
    "TELEGRAM_CHAT_ID": "1036033920"
}
```

3. Run the bot:
```bash
python Tradebot.py
```

**Note:** `credentials.json` is in `.gitignore` and will NOT be committed to git.

## Ubuntu Server Deployment

### Step 1: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3 and pip
sudo apt install python3 python3-pip python3-venv -y

# Create project directory
mkdir -p ~/trading-bot
cd ~/trading-bot
```

### Step 2: Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Packages

```bash
pip install dhanhq pandas pandas-ta requests
```

### Step 4: Upload Your Code

```bash
# Upload files (use scp, rsync, or git clone)
scp Tradebot.py user@server:~/trading-bot/
scp config.py user@server:~/trading-bot/

# OR clone from git
git clone your_repo_url ~/trading-bot
```

### Step 5: Set Environment Variables

```bash
# Option A: Set in current session
export DHAN_CLIENT_ID="your_client_id"
export DHAN_ACCESS_TOKEN="your_access_token"
export TELEGRAM_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Option B: Create a .env file (more secure)
nano ~/.trading_bot_env

# Add these lines:
export DHAN_CLIENT_ID="your_client_id"
export DHAN_ACCESS_TOKEN="your_access_token"
export TELEGRAM_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Load it before running:
source ~/.trading_bot_env
```

### Step 6: Run the Bot

```bash
cd ~/trading-bot
source venv/bin/activate
python Tradebot.py
```

## Running as a System Service (Ubuntu)

Create a systemd service to run the bot automatically:

```bash
sudo nano /etc/systemd/system/tradingbot.service
```

Add this content:

```ini
[Unit]
Description=Trading Bot
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/trading-bot
Environment="DHAN_CLIENT_ID=your_client_id"
Environment="DHAN_ACCESS_TOKEN=your_access_token"
Environment="TELEGRAM_TOKEN=your_telegram_token"
Environment="TELEGRAM_CHAT_ID=your_chat_id"
ExecStart=/home/your_username/trading-bot/venv/bin/python Tradebot.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tradingbot
sudo systemctl start tradingbot

# Check status
sudo systemctl status tradingbot

# View logs
sudo journalctl -u tradingbot -f
```

## Security Best Practices

1. **Never commit credentials to git** - `.gitignore` prevents this
2. **Use environment variables on servers** - More secure than files
3. **Restrict file permissions** if using credentials.json:
   ```bash
   chmod 600 credentials.json
   ```
4. **Use separate read-only API keys** if possible
5. **Monitor logs** for unauthorized access attempts

## Troubleshooting

### "Missing credentials" error
- Check environment variables: `echo $DHAN_CLIENT_ID`
- Verify credentials.json exists and has valid JSON
- Ensure no extra spaces in credentials

### Module not found errors
- Activate virtual environment: `source venv/bin/activate`
- Install missing packages: `pip install package_name`

### Permission denied
- Check file permissions: `ls -la`
- Run with proper user, not root

## Updating Credentials

### Environment Variables:
```bash
export DHAN_ACCESS_TOKEN="new_token_here"
```

### credentials.json:
```bash
nano credentials.json
# Update the token
# Restart the bot
```

### Systemd Service:
```bash
sudo systemctl edit tradingbot
# Update environment variables
sudo systemctl restart tradingbot
```
