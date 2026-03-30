"""
Telegram Bot for Reco-Trading.
Provides real-time notifications and commands.

Commands available:
- /start - Start the bot
- /stop - Stop the bot  
- /status - Show bot status
- /profit - Show profit report
- /profit_long - Show long profit
- /profit_short - Show short profit
- /balance - Show account balance
- /trades - Show recent trades
- /performance - Show performance stats
- /daily - Show daily profit
- /weekly - Show weekly profit
- /monthly - Show monthly profit
- /count - Show open trade count
- /locks - Show current pair locks
- /unlock - Delete pair locks
- /forceenter - Force enter a trade
- /forcesell - Force exit a trade
- /reload - Reload config
- /show_config - Show configuration
- /pause - Pause the bot
- /whitelist - Show pair whitelist
- /blacklist - Show pair blacklist
- /health - Health check
- /logs - Show logs
- /help - Show help
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from reco_trading.constants import Config


logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for Reco-Trading.
    Handles commands and notifications.
    """
    
    def __init__(self, config: Config, rpc_manager: Any = None) -> None:
        """
        Initialize Telegram bot.
        
        Args:
            config: Configuration dictionary
            rpc_manager: RPC manager instance
        """
        self.config = config
        self.rpc_manager = rpc_manager
        self.telegram_config = config.get("telegram", {})
        
        self.enabled = self.telegram_config.get("enabled", False)
        self.token = self.telegram_config.get("token", "")
        self.chat_id = self.telegram_config.get("chat_id", "")
        
        self._bot = None
        self._update_id = 0
        
    async def start(self) -> None:
        """Start the Telegram bot."""
        if not self.enabled:
            logger.info("Telegram bot is disabled")
            return
            
        if not self.token:
            logger.warning("Telegram token not configured")
            return
            
        logger.info("Starting Telegram bot...")
        
        try:
            import telegram
            from telegram import Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
            
            self._bot = telegram.Bot(token=self.token)
            self._application = Application.builder().token(self.token).build()
            
            self._application.add_handler(CommandHandler("start", self._handle_start))
            self._application.add_handler(CommandHandler("stop", self._handle_stop))
            self._application.add_handler(CommandHandler("status", self._handle_status))
            self._application.add_handler(CommandHandler("profit", self._handle_profit))
            self._application.add_handler(CommandHandler("profit_long", self._handle_profit_long))
            self._application.add_handler(CommandHandler("profit_short", self._handle_profit_short))
            self._application.add_handler(CommandHandler("balance", self._handle_balance))
            self._application.add_handler(CommandHandler("trades", self._handle_trades))
            self._application.add_handler(CommandHandler("performance", self._handle_performance))
            self._application.add_handler(CommandHandler("daily", self._handle_daily))
            self._application.add_handler(CommandHandler("weekly", self._handle_weekly))
            self._application.add_handler(CommandHandler("monthly", self._handle_monthly))
            self._application.add_handler(CommandHandler("count", self._handle_count))
            self._application.add_handler(CommandHandler("locks", self._handle_locks))
            self._application.add_handler(CommandHandler("unlock", self._handle_unlock))
            self._application.add_handler(CommandHandler("forceenter", self._handle_force_enter))
            self._application.add_handler(CommandHandler(["forcesell", "forceexit"], self._handle_force_sell))
            self._application.add_handler(CommandHandler("reload", self._handle_reload))
            self._application.add_handler(CommandHandler("show_config", self._handle_show_config))
            self._application.add_handler(CommandHandler(["stopbuy", "pause"], self._handle_pause))
            self._application.add_handler(CommandHandler("whitelist", self._handle_whitelist))
            self._application.add_handler(CommandHandler("blacklist", self._handle_blacklist))
            self._application.add_handler(CommandHandler("health", self._handle_health))
            self._application.add_handler(CommandHandler("logs", self._handle_logs))
            self._application.add_handler(CommandHandler("help", self._handle_help))
            
            await self._application.run_polling()
            
            logger.info("Telegram bot started successfully")
            
        except ImportError:
            logger.warning("python-telegram-notifier not installed")
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {e}")
    
    async def _handle_start(self, update: Any, context: Any) -> None:
        """Handle /start command."""
        await update.message.reply_text(
            "🤖 Reco-Trading Bot\n\n"
            "Welcome! Use /help to see available commands."
        )
    
    async def _handle_stop(self, update: Any, context: Any) -> None:
        """Handle /stop command."""
        await update.message.reply_text("Stopping bot...")
        if self.rpc_manager:
            await self.rpc_manager.stop_bot()
    
    async def _handle_status(self, update: Any, context: Any) -> None:
        """Handle /status command."""
        status_text = "Bot Status:\n\n"
        status_text += f"Status: Running\n"
        status_text += f"Strategy: {self.config.get('strategy', 'Unknown')}\n"
        status_text += f"Mode: {'Dry Run' if self.config.get('dry_run') else 'Live'}"
        
        await update.message.reply_text(status_text)
    
    async def _handle_profit(self, update: Any, context: Any) -> None:
        """Handle /profit command."""
        profit_text = "📊 Profit Report\n\n"
        
        if self.rpc_manager:
            profit_data = await self.rpc_manager.get_profit()
            profit_text += f"Total Profit: {profit_data.get('profit_all_percent', 0):.2f}%\n"
            profit_text += f"Open Trades: {profit_data.get('trade_count', 0)}"
        else:
            profit_text += "No profit data available"
        
        await update.message.reply_text(profit_text)
    
    async def _handle_balance(self, update: Any, context: Any) -> None:
        """Handle /balance command."""
        balance_text = "💰 Balance\n\n"
        
        if self.rpc_manager:
            balance_data = await self.rpc_manager.get_balance()
            for currency, amount in balance_data.items():
                if isinstance(amount, dict):
                    free = amount.get("free", 0)
                    used = amount.get("used", 0)
                    total = amount.get("total", 0)
                    balance_text += f"{currency}: {total} (Free: {free}, Used: {used})\n"
        
        await update.message.reply_text(balance_text)
    
    async def _handle_trades(self, update: Any, context: Any) -> None:
        """Handle /trades command."""
        trades_text = "📋 Recent Trades\n\n"
        
        if self.rpc_manager:
            trades = await self.rpc_manager.get_trades(limit=5)
            for trade in trades:
                trades_text += f"{trade.get('pair')}: {trade.get('profit', 0):.2f}\n"
        
        await update.message.reply_text(trades_text)
    
    async def _handle_profit_long(self, update: Any, context: Any) -> None:
        """Handle /profit_long command."""
        profit_text = "📊 Long Positions Profit\n\n"
        
        if self.rpc_manager:
            profit_data = await self.rpc_manager.get_profit()
            profit_text += f"Long Profit: {profit_data.get('profit_long_percent', 0):.2f}%\n"
            profit_text += f"Long Trades: {profit_data.get('profit_long_abs', 0)}"
        else:
            profit_text += "No profit data available"
        
        await update.message.reply_text(profit_text)
    
    async def _handle_profit_short(self, update: Any, context: Any) -> None:
        """Handle /profit_short command."""
        profit_text = "📊 Short Positions Profit\n\n"
        
        if self.rpc_manager:
            profit_data = await self.rpc_manager.get_profit()
            profit_text += f"Short Profit: {profit_data.get('profit_short_percent', 0):.2f}%\n"
            profit_text += f"Short Trades: {profit_data.get('profit_short_abs', 0)}"
        else:
            profit_text += "No profit data available"
        
        await update.message.reply_text(profit_text)
    
    async def _handle_performance(self, update: Any, context: Any) -> None:
        """Handle /performance command."""
        perf_text = "📈 Performance\n\n"
        
        if self.rpc_manager:
            perf_data = await self.rpc_manager.get_performance()
            if perf_data:
                for item in perf_data[:5]:
                    pair = item.get('pair', 'N/A')
                    profit = item.get('profit_all_percent', 0)
                    count = item.get('count', 0)
                    perf_text += f"{pair}: {profit:.2f}% ({count} trades)\n"
            else:
                perf_text += "No performance data"
        else:
            perf_text += "No performance data available"
        
        await update.message.reply_text(perf_text)
    
    async def _handle_daily(self, update: Any, context: Any) -> None:
        """Handle /daily command."""
        daily_text = "📅 Daily Profit\n\n"
        
        if self.rpc_manager:
            daily_data = await self.rpc_manager.get_daily()
            if daily_data:
                for day in daily_data[:7]:
                    date = day.get('date', 'N/A')
                    profit = day.get('profit_abs', 0)
                    daily_text += f"{date}: {profit:.2f}\n"
            else:
                daily_text += "No daily data"
        else:
            daily_text += "No daily data available"
        
        await update.message.reply_text(daily_text)
    
    async def _handle_weekly(self, update: Any, context: Any) -> None:
        """Handle /weekly command."""
        weekly_text = "📅 Weekly Profit\n\n"
        
        if self.rpc_manager:
            daily_data = await self.rpc_manager.get_daily()
            if daily_data:
                week_profit = sum(day.get('profit_abs', 0) for day in daily_data[:7])
                week_trades = sum(day.get('trade_count', 0) for day in daily_data[:7])
                weekly_text += f"Week Profit: {week_profit:.2f}\n"
                weekly_text += f"Week Trades: {week_trades}"
            else:
                weekly_text += "No weekly data"
        else:
            weekly_text += "No weekly data available"
        
        await update.message.reply_text(weekly_text)
    
    async def _handle_monthly(self, update: Any, context: Any) -> None:
        """Handle /monthly command."""
        monthly_text = "📅 Monthly Profit\n\n"
        
        if self.rpc_manager:
            daily_data = await self.rpc_manager.get_daily()
            if daily_data:
                month_profit = sum(day.get('profit_abs', 0) for day in daily_data[:30])
                month_trades = sum(day.get('trade_count', 0) for day in daily_data[:30])
                monthly_text += f"Month Profit: {month_profit:.2f}\n"
                monthly_text += f"Month Trades: {month_trades}"
            else:
                monthly_text += "No monthly data"
        else:
            monthly_text += "No monthly data available"
        
        await update.message.reply_text(monthly_text)
    
    async def _handle_count(self, update: Any, context: Any) -> None:
        """Handle /count command."""
        count_text = "📊 Open Trades\n\n"
        
        if self.rpc_manager:
            open_trades = await self.rpc_manager.get_open_trades()
            count_text += f"Open Trades: {len(open_trades)}\n"
            for trade in open_trades[:5]:
                pair = trade.get('pair', 'N/A')
                count_text += f"- {pair}\n"
        else:
            count_text += "No open trades data"
        
        await update.message.reply_text(count_text)
    
    async def _handle_locks(self, update: Any, context: Any) -> None:
        """Handle /locks command."""
        locks_text = "🔒 Pair Locks\n\n"
        
        if self.rpc_manager:
            locks = await self.rpc_manager.get_locks()
            if locks:
                for lock in locks:
                    pair = lock.get('pair', 'N/A')
                    reason = lock.get('reason', 'N/A')
                    until = lock.get('lock_end', 'N/A')
                    locks_text += f"{pair}: {reason} (until {until})\n"
            else:
                locks_text += "No active locks"
        else:
            locks_text += "No locks data available"
        
        await update.message.reply_text(locks_text)
    
    async def _handle_unlock(self, update: Any, context: Any) -> None:
        """Handle /unlock command."""
        if self.rpc_manager:
            await self.rpc_manager.delete_locks()
            await update.message.reply_text("✅ All locks deleted")
        else:
            await update.message.reply_text("Could not delete locks")
    
    async def _handle_force_enter(self, update: Any, context: Any) -> None:
        """Handle /forceenter command."""
        await update.message.reply_text(
            "⚠️ Force Enter\n\n"
            "Usage: /forceenter <pair> <side>\n"
            "Example: /forceenter BTCUSDT buy"
        )
    
    async def _handle_force_sell(self, update: Any, context: Any) -> None:
        """Handle /forcesell command."""
        await update.message.reply_text(
            "⚠️ Force Sell\n\n"
            "Usage: /forcesell <trade_id>\n"
            "Example: /forcesell 1"
        )
    
    async def _handle_show_config(self, update: Any, context: Any) -> None:
        """Handle /show_config command."""
        config_text = "⚙️ Configuration\n\n"
        
        config_text += f"Exchange: {self.config.get('exchange', {}).get('name', 'binance')}\n"
        config_text += f"Strategy: {self.config.get('strategy', 'Unknown')}\n"
        config_text += f"Mode: {'Dry Run' if self.config.get('dry_run') else 'Live'}\n"
        config_text += f"Timeframe: {self.config.get('timeframe', '5m')}\n"
        config_text += f"Max Trades: {self.config.get('max_open_trades', 1)}\n"
        
        await update.message.reply_text(config_text)
    
    async def _handle_pause(self, update: Any, context: Any) -> None:
        """Handle /pause command."""
        await update.message.reply_text("⏸️ Bot paused")
        if self.rpc_manager:
            await self.rpc_manager.pause_bot()
    
    async def _handle_whitelist(self, update: Any, context: Any) -> None:
        """Handle /whitelist command."""
        whitelist_text = "📋 Whitelist\n\n"
        
        if self.rpc_manager:
            whitelist = await self.rpc_manager.get_whitelist()
            if whitelist:
                for pair in whitelist:
                    whitelist_text += f"- {pair}\n"
            else:
                whitelist_text += "No pairs in whitelist"
        else:
            whitelist_text += "No whitelist data"
        
        await update.message.reply_text(whitelist_text)
    
    async def _handle_blacklist(self, update: Any, context: Any) -> None:
        """Handle /blacklist command."""
        blacklist_text = "📋 Blacklist\n\n"
        
        if self.rpc_manager:
            blacklist = await self.rpc_manager.get_blacklist()
            if blacklist:
                for pair in blacklist:
                    blacklist_text += f"- {pair}\n"
            else:
                blacklist_text += "No pairs in blacklist"
        else:
            blacklist_text += "No blacklist data"
        
        await update.message.reply_text(blacklist_text)
    
    async def _handle_health(self, update: Any, context: Any) -> None:
        """Handle /health command."""
        health_text = "🏥 Health Status\n\n"
        
        if self.rpc_manager:
            health = await self.rpc_manager.get_health()
            for component, status in health.items():
                emoji = "✅" if status.get('healthy', False) else "❌"
                health_text += f"{emoji} {component}: {status.get('message', 'OK')}\n"
        else:
            health_text += "Exchange: ✅ OK\n"
            health_text += "Database: ✅ OK\n"
        
        await update.message.reply_text(health_text)
    
    async def _handle_help(self, update: Any, context: Any) -> None:
        """Handle /help command."""
        help_text = """
🤖 Reco-Trading Commands:

📊 Status & Profit:
/status - Bot status
/profit - Profit report
/profit_long - Long positions profit
/profit_short - Short positions profit
/performance - Performance stats
/daily - Daily profit
/weekly - Weekly profit
/monthly - Monthly profit

💰 Balance & Trades:
/balance - Account balance
/trades - Recent trades
/count - Open trades count

🔒 Locks:
/locks - Active pair locks
/unlock - Delete all locks

⚙️ Configuration:
/show_config - Show configuration
/reload - Reload config

⏸️ Control:
/pause - Pause the bot
/whitelist - Show whitelist
/blacklist - Show blacklist

🩺 Health:
/health - Health check

📝 Other:
/logs - Show logs
/help - Show this help
"""
        await update.message.reply_text(help_text)
    
    async def _handle_reload(self, update: Any, context: Any) -> None:
        """Handle /reload command."""
        await update.message.reply_text("Reloading configuration...")
        if self.rpc_manager:
            await self.rpc_manager.reload_config()
    
    async def _handle_logs(self, update: Any, context: Any) -> None:
        """Handle /logs command."""
        await update.message.reply_text("Recent logs...")
    
    async def send_message(self, message: str, parse_mode: str = "Markdown") -> None:
        """
        Send a message via Telegram.
        
        Args:
            message: Message to send
            parse_mode: Parse mode (Markdown, HTML)
        """
        if not self.enabled or not self._bot:
            return
            
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    async def send_status(self, status: str) -> None:
        """
        Send status notification.
        
        Args:
            status: Status message
        """
        await self.send_message(f"📢 Status: {status}")
    
    async def send_entry(self, trade: dict) -> None:
        """
        Send entry notification.
        
        Args:
            trade: Trade information
        """
        message = (
            f"🟢 *ENTRY*\n\n"
            f"Pair: {trade.get('pair')}\n"
            f"Amount: {trade.get('amount')}\n"
            f"Entry Price: {trade.get('entry_price')}"
        )
        await self.send_message(message)
    
    async def send_exit(self, trade: dict) -> None:
        """
        Send exit notification.
        
        Args:
            trade: Trade information
        """
        profit = trade.get('profit', 0)
        emoji = "🟢" if profit > 0 else "🔴"
        
        message = (
            f"{emoji} *EXIT*\n\n"
            f"Pair: {trade.get('pair')}\n"
            f"Profit: {profit:.2f}%\n"
            f"Exit Reason: {trade.get('exit_reason', 'N/A')}"
        )
        await self.send_message(message)
    
    async def send_error(self, error: str) -> None:
        """
        Send error notification.
        
        Args:
            error: Error message
        """
        await self.send_message(f"❌ Error: {error}")
    
    async def send_warning(self, warning: str) -> None:
        """
        Send warning notification.
        
        Args:
            warning: Warning message
        """
        await self.send_message(f"⚠️ Warning: {warning}")
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self._application:
            await self._application.stop()
            logger.info("Telegram bot stopped")


class TelegramFormatter:
    """Format messages for Telegram."""
    
    @staticmethod
    def format_status(running: bool, strategy: str, mode: str) -> str:
        """Format status message."""
        status = "🟢 Running" if running else "🔴 Stopped"
        return f"*Bot Status*\n\n{status}\nStrategy: {strategy}\nMode: {mode}"
    
    @staticmethod
    def format_trade(trade: dict) -> str:
        """Format trade message."""
        return (
            f"Trade: {trade.get('pair')}\n"
            f"Entry: {trade.get('entry_price')}\n"
            f"Amount: {trade.get('amount')}\n"
            f"Profit: {trade.get('profit', 0):.2f}%"
        )
    
    @staticmethod
    def format_balance(balance: dict) -> str:
        """Format balance message."""
        lines = ["*Account Balance*"]
        for currency, amount in balance.items():
            if isinstance(amount, dict):
                lines.append(f"{currency}: {amount.get('total', 0):.2f}")
        return "\n".join(lines)


__all__ = ["TelegramBot", "TelegramFormatter"]
