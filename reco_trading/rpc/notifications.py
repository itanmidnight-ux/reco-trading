"""
Notifications System for Reco-Trading.
Handles sending notifications to various channels.
"""

import logging
from typing import Any
from enum import Enum


logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Notification types."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    ENTRY = "entry"
    EXIT = "exit"
    PROTECTION = "protection"


class NotificationManager:
    """
    Notification manager for sending alerts.
    Supports multiple notification channels.
    """
    
    def __init__(self, config: dict) -> None:
        """
        Initialize notification manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._telegram = None
        self._webhooks = []
        
        self._init_telegram()
        self._init_webhooks()
    
    def _init_telegram(self) -> None:
        """Initialize Telegram notifications."""
        telegram_config = self.config.get("telegram", {})
        
        if telegram_config.get("enabled", False):
            from reco_trading.rpc.telegram import TelegramBot
            self._telegram = TelegramBot(self.config)
            logger.info("Telegram notifications enabled")
    
    def _init_webhooks(self) -> None:
        """Initialize webhook notifications."""
        webhook_config = self.config.get("webhook", {})
        
        if webhook_config.get("enabled", False):
            self._webhooks.append(webhook_config)
            logger.info("Webhook notifications enabled")
    
    async def send(
        self,
        message: str,
        notification_type: NotificationType = NotificationType.INFO,
    ) -> None:
        """
        Send notification to all enabled channels.
        
        Args:
            message: Message to send
            notification_type: Type of notification
        """
        if self._telegram:
            await self._telegram.send_message(message)
        
        for webhook in self._webhooks:
            await self._send_webhook(webhook, message, notification_type)
    
    async def _send_webhook(
        self,
        webhook: dict,
        message: str,
        notification_type: NotificationType,
    ) -> None:
        """Send webhook notification."""
        import aiohttp
        
        url = webhook.get("url")
        if not url:
            return
        
        payload = {
            "message": message,
            "type": notification_type.value,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(url, json=payload)
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    async def notify_entry(self, trade: dict) -> None:
        """Notify about entry."""
        message = (
            f"🟢 *ENTRY*\n\n"
            f"Pair: {trade.get('pair')}\n"
            f"Amount: {trade.get('amount')}\n"
            f"Price: {trade.get('entry_price')}"
        )
        await self.send(message, NotificationType.ENTRY)
    
    async def notify_exit(self, trade: dict) -> None:
        """Notify about exit."""
        profit = trade.get('profit', 0)
        emoji = "🟢" if profit > 0 else "🔴"
        
        message = (
            f"{emoji} *EXIT*\n\n"
            f"Pair: {trade.get('pair')}\n"
            f"Profit: {profit:.2f}%\n"
            f"Reason: {trade.get('exit_reason', 'N/A')}"
        )
        await self.send(message, NotificationType.EXIT)
    
    async def notify_status(self, status: str) -> None:
        """Notify status change."""
        message = f"📢 Status: {status}"
        await self.send(message, NotificationType.INFO)
    
    async def notify_error(self, error: str) -> None:
        """Notify error."""
        message = f"❌ Error: {error}"
        await self.send(message, NotificationType.ERROR)
    
    async def notify_warning(self, warning: str) -> None:
        """Notify warning."""
        message = f"⚠️ Warning: {warning}"
        await self.send(message, NotificationType.WARNING)
    
    async def notify_protection(self, protection: str, reason: str) -> None:
        """Notify protection triggered."""
        message = f"🛡️ Protection: {protection}\nReason: {reason}"
        await self.send(message, NotificationType.PROTECTION)


__all__ = ["NotificationManager", "NotificationType"]
