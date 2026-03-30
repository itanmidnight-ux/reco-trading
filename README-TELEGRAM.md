# Telegram Configuration Guide

This guide explains how to configure Telegram notifications for Reco-Trading bot with a robust connection.

## Table of Contents
1. [Create a Telegram Bot](#1-create-a-telegram-bot)
2. [Get Your Chat ID](#2-get-your-chat-id)
3. [Configure the Bot](#3-configure-the-bot)
4. [Notification Settings](#4-notification-settings)
5. [Commands Available](#5-commands-available)
6. [Testing the Connection](#6-testing-the-connection)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Create a Telegram Bot

### Step 1: Start BotFather
1. Open Telegram and search for `@BotFather`
2. Start a chat with BotFather
3. Send `/newbot` command

### Step 2: Choose a Name
1. BotFather will ask for a name for your bot
2. Enter: `RecoTrading` (or any name you prefer)

### Step 3: Choose a Username
1. BotFather will ask for a username (must end in `bot`)
2. Enter: `reco_trading_bot` (must be unique)

### Step 4: Save the Token
BotFather will give you a token like:
```
1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789
```

**⚠️ IMPORTANT**: Save this token securely. It gives full access to your bot!

---

## 2. Get Your Chat ID

### Method 1: Using @userinfobot
1. Search for `@userinfobot` in Telegram
2. Start the bot
3. It will display your Chat ID (numbers only)

### Method 2: Using @myidbot
1. Search for `@myidbot` in Telegram
2. Send `/getid` command
3. Bot will reply with your Chat ID

### Method 3: Direct Method
1. Open this URL in your browser (replace `YOUR_BOT_TOKEN`):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
2. If you get an empty response, send a message to your bot first
3. Refresh the page and look for `"chat":{"id":123456789}`

---

## 3. Configure the Bot

### Option A: Using .env file

Add these variables to your `.env` file:

```bash
# Telegram Configuration
TELEGRAM_ENABLED=true
TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789
TELEGRAM_CHAT_ID=123456789
```

### Option B: Using config.json

Add or update the `telegram` section in your config:

```json
{
  "telegram": {
    "enabled": true,
    "token": "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789",
    "chat_id": "123456789",
    "notification_settings": {
      "entries": true,
      "exits": true,
      "errors": true,
      "warnings": true,
      "status": true
    },
    "reload": true,
    "balance_dust_level": 0.01
  }
}
```

### Option C: Environment Variables in Docker

Add to your `docker-compose.yml` under `environment`:

```yaml
environment:
  - TELEGRAM_ENABLED=true
  - TELEGRAM_TOKEN=your_token_here
  - TELEGRAM_CHAT_ID=your_chat_id
```

---

## 4. Notification Settings

The bot sends the following notifications:

| Notification Type | Description | Default |
|-----------------|-------------|---------|
| **Entry** | Trade opened (pair, amount, entry price) | ✅ Enabled |
| **Exit** | Trade closed (pair, profit, exit reason) | ✅ Enabled |
| **Errors** | Critical errors | ✅ Enabled |
| **Warnings** | Warnings (e.g., cooldown activated) | ✅ Enabled |
| **Status** | Bot status changes | ✅ Enabled |

### Customizing Notifications

You can disable specific notifications by modifying the `notification_settings` in config:

```json
{
  "telegram": {
    "enabled": true,
    "token": "YOUR_TOKEN",
    "chat_id": "YOUR_CHAT_ID",
    "notification_settings": {
      "entries": true,
      "exits": true,
      "errors": true,
      "warnings": false,
      "status": true,
      "daily_summary": true,
      "emergency_stop": true,
      "pair_switch": true
    }
  }
}
```

---

## 5. Commands Available

Once connected, you can use these commands:

### Status & Profit
| Command | Description |
|---------|-------------|
| `/start` | Start bot interaction |
| `/status` | Show bot status |
| `/profit` | Total profit report |
| `/profit_long` | Long positions profit |
| `/profit_short` | Short positions profit |
| `/performance` | Performance by pair |
| `/daily` | Daily profit |
| `/weekly` | Weekly profit |
| `/monthly` | Monthly profit |

### Balance & Trades
| Command | Description |
|---------|-------------|
| `/balance` | Account balance |
| `/trades` | Recent trades (last 5) |
| `/count` | Open trades count |

### Control
| Command | Description |
|---------|-------------|
| `/pause` | Pause the bot |
| `/whitelist` | Show pair whitelist |
| `/blacklist` | Show pair blacklist |
| `/locks` | Show active locks |
| `/unlock` | Delete all locks |
| `/reload` | Reload configuration |
| `/show_config` | Show current config |

### Health & Info
| Command | Description |
|---------|-------------|
| `/health` | System health check |
| `/logs` | Recent logs |
| `/help` | Show all commands |

---

## 6. Testing the Connection

### Test 1: Bot Token
Open this URL in your browser:
```
https://api.telegram.org/botYOUR_TOKEN/getMe
```
Expected response: JSON with your bot info

### Test 2: Send Message
```
https://api.telegram.org/botYOUR_TOKEN/sendMessage?chat_id=YOUR_CHAT_ID&text=Hello
```

### Test 3: Bot Command
Start your bot and send `/start`

### Test 4: From the Bot
Send `/status` to your bot and verify you get a response

---

## 7. Troubleshooting

### Issue: "Chat not found"
- Make sure the Chat ID is correct (must be numbers, not @username)
- The bot must be started first: send `/start` to the bot

### Issue: "Bot blocked"
- You have blocked the bot
- Unblock: Go to Telegram Settings > Privacy > Block Users > Unblock bot

### Issue: "Not authorized"
- The bot token is invalid
- Get a new token from @BotFather

### Issue: Connection drops
- The bot uses polling by default
- For production, consider using webhooks for stability

### Issue: No notifications
- Check `TELEGRAM_ENABLED=true` in config
- Verify `notification_settings` are enabled
- Check bot logs for errors

### Issue: Rate limiting
- Telegram limits: 30 messages/second
- The bot automatically handles rate limiting

---

## Security Notes

1. **Never share your bot token**
2. **Use environment variables** instead of hardcoding in config files
3. **Restrict bot commands** via @BotFather if needed:
   - Send `/setcommands` to BotFather
   - Provide command list
4. **Enable 2FA** on your Telegram account

---

## Advanced: Webhook Configuration

For more robust connection in production, use webhooks instead of polling:

```python
# In your config
{
  "telegram": {
    "enabled": true,
    "token": "YOUR_TOKEN",
    "chat_id": "YOUR_CHAT_ID",
    "use_webhook": true,
    "webhook_url": "https://your-domain.com/telegram/webhook"
  }
}
```

Then set the webhook:
```
https://api.telegram.org/botYOUR_TOKEN/setWebhook?url=https://your-domain.com/telegram/webhook
```

---

## Quick Setup Checklist

- [ ] Created bot via @BotFather
- [ ] Saved bot token securely
- [ ] Got Chat ID via @userinfobot
- [ ] Added configuration to .env or config.json
- [ ] Set TELEGRAM_ENABLED=true
- [ ] Restarted the bot
- [ ] Sent /start to bot
- [ ] Tested /status command
- [ ] Verified notifications working

---

## Support

For issues with the Telegram integration, check:
1. Bot logs: `docker logs reco-trading-app`
2. API response: `https://api.telegram.org/botTOKEN/getWebhookInfo`
3. Network connectivity to Telegram servers
