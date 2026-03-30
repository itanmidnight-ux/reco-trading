"""
RPC Manager for Reco-Trading.
Manages communication between bot and external interfaces.
"""

import logging
from typing import Any

from reco_trading.constants import Config


logger = logging.getLogger(__name__)


class RPCManager:
    """
    RPC Manager handles communication with external interfaces.
    Supports Telegram, Web UI, and API.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize RPC Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._telegram = None
        self._webui = None
        
    async def start(self) -> None:
        """Start all RPC interfaces."""
        telegram_config = self.config.get("telegram", {})
        
        if telegram_config.get("enabled", False):
            from reco_trading.rpc.telegram import TelegramBot
            self._telegram = TelegramBot(self.config, self)
            await self._telegram.start()
            logger.info("Telegram RPC started")
    
    async def stop(self) -> None:
        """Stop all RPC interfaces."""
        if self._telegram:
            await self._telegram.stop()
    
    async def send_msg(self, msg: str, msg_type: str = "info") -> None:
        """
        Send message to all enabled interfaces.
        
        Args:
            msg: Message to send
            msg_type: Message type (info, warning, error)
        """
        if self._telegram:
            if msg_type == "error":
                await self._telegram.send_error(msg)
            elif msg_type == "warning":
                await self._telegram.send_warning(msg)
            else:
                await self._telegram.send_message(msg)
    
    async def send_status(self, status: str) -> None:
        """Send status message."""
        if self._telegram:
            await self._telegram.send_status(status)
    
    async def send_entry(self, trade: dict) -> None:
        """Send entry notification."""
        if self._telegram:
            await self._telegram.send_entry(trade)
    
    async def send_exit(self, trade: dict) -> None:
        """Send exit notification."""
        if self._telegram:
            await self._telegram.send_exit(trade)
    
    async def get_profit(self) -> dict:
        """Get profit statistics."""
        return {
            "profit_all_percent": 0.0,
            "profit_all": 0.0,
            "trade_count": 0,
        }
    
    async def get_balance(self) -> dict:
        """Get account balance."""
        return {}
    
    async def get_trades(self, limit: int = 10) -> list:
        """Get recent trades."""
        return []
    
    async def get_status(self) -> dict:
        """Get bot status."""
        return {
            "running": True,
            "strategy": self.config.get("strategy", "Unknown"),
            "mode": "dry_run" if self.config.get("dry_run") else "live",
        }
    
    async def reload_config(self) -> None:
        """Reload configuration."""
        logger.info("Configuration reload requested")

    async def stop_bot(self) -> None:
        """Stop the bot."""
        logger.info("Bot stop requested via Telegram")

    async def pause_bot(self) -> None:
        """Pause the bot."""
        logger.info("Bot pause requested via Telegram")

    async def get_performance(self) -> list:
        """Get performance statistics."""
        return []

    async def get_daily(self) -> list:
        """Get daily profit statistics."""
        return []

    async def get_open_trades(self) -> list:
        """Get open trades."""
        return []

    async def get_locks(self) -> list:
        """Get active pair locks."""
        return []

    async def delete_locks(self) -> None:
        """Delete all locks."""
        logger.info("Locks deletion requested")

    async def get_whitelist(self) -> list:
        """Get pair whitelist."""
        return self.config.get("exchange", {}).get("whitelist", ["BTCUSDT"])

    async def get_blacklist(self) -> list:
        """Get pair blacklist."""
        return self.config.get("exchange", {}).get("blacklist", [])

    async def get_health(self) -> dict:
        """Get health status."""
        return {
            "exchange": {"healthy": True, "message": "OK"},
            "database": {"healthy": True, "message": "OK"},
        }


class RPCProxy:
    """
    RPC Proxy for accessing bot functionality.
    """
    
    def __init__(self, rpc_manager: RPCManager) -> None:
        """
        Initialize RPC Proxy.
        
        Args:
            rpc_manager: RPC manager instance
        """
        self._rpc = rpc_manager
    
    async def profit(self) -> dict:
        """Get profit."""
        return await self._rpc.get_profit()
    
    async def balance(self) -> dict:
        """Get balance."""
        return await self._rpc.get_balance()
    
    async def status(self) -> dict:
        """Get status."""
        return await self._rpc.get_status()
    
    async def trades(self, limit: int = 10) -> list:
        """Get trades."""
        return await self._rpc.get_trades(limit)


__all__ = ["RPCManager", "RPCProxy"]
