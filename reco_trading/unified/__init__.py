"""Unified Trading System - Integration Module
Combines all trading systems: Copy Trading, Grid Bots, TradingView, Mobile, Cloud Sync, Marketplace
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

from reco_trading.copy_trading import (
    CopyTradingManager, SignalManager, get_copy_trading_manager, 
    TradingSignal, SignalSource
)
from reco_trading.grid_bot import (
    GridBotManager, GridConfig, GridMode, get_grid_bot_manager
)
from reco_trading.tradingview import (
    TradingViewWebhookHandler, WebhookConfig, get_webhook_handler
)
from reco_trading.marketplace import (
    TemplateMarketplace, get_template_marketplace
)
from reco_trading.cloud_sync import (
    CloudSyncManager, CloudConfig, get_cloud_sync_manager
)
from reco_trading.mobile import (
    MobileAPIManager, PushNotificationManager,
    get_mobile_api_manager, get_push_notification_manager
)

logger = logging.getLogger(__name__)


class UnifiedTradingSystem:
    """
    Unified Trading System that orchestrates all trading modules.
    This is the central hub that connects all systems together.
    """
    
    def __init__(self, settings: Any = None):
        self.settings = settings
        
        # Initialize all subsystems
        self.copy_trading = get_copy_trading_manager()
        self.grid_bot = get_grid_bot_manager()
        self.tradingview = get_webhook_handler()
        self.marketplace = get_template_marketplace()
        self.cloud_sync = get_cloud_sync_manager()
        self.mobile_api = get_mobile_api_manager()
        self.push_notifications = get_push_notification_manager()
        
        # Integration state
        self._is_running = False
        self._tasks: list[asyncio.Task] = []
        
        # Register callbacks
        self._setup_callbacks()
        
        logger.info("UnifiedTradingSystem initialized")
        
    def _setup_callbacks(self):
        """Setup integration callbacks between modules"""
        
        # TradingView webhook -> Copy Trading signals
        async def on_tradingview_signal(signal):
            await self._handle_tradingview_signal(signal)
            
        self.tradingview.register_callback(on_tradingview_signal)
        
    async def _handle_tradingview_signal(self, signal):
        """Handle incoming TradingView signal"""
        # Convert to internal signal and publish to copy trading
        internal_signal = TradingSignal(
            source=SignalSource.TRADINGVIEW,
            action=signal.signal_type.value,
            symbol=signal.symbol,
            price=signal.price,
            confidence=0.8,
            strategy_name=signal.strategy_name,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        await self.copy_trading.signal_manager.publish_signal(internal_signal)
        
        # Send push notification
        for device in self.mobile_api.devices.values():
            if device.is_active and device.signal_notifications:
                await self.push_notifications.send_signal_notification(
                    device.device_id,
                    {
                        "action": signal.signal_type.value,
                        "symbol": signal.symbol,
                        "confidence": 0.8,
                        "price": signal.price
                    }
                )
                
        logger.info(f"TradingView signal processed: {signal.symbol} {signal.signal_type.value}")
        
    async def start(self):
        """Start all trading systems"""
        if self._is_running:
            logger.warning("System already running")
            return
            
        logger.info("Starting Unified Trading System...")
        
        # Initialize cloud sync
        await self.cloud_sync.initialize()
        
        # Start mobile API server (in background)
        try:
            await self.mobile_api.start_server(port=8081)
        except Exception as e:
            logger.warning(f"Mobile API server failed to start: {e}")
            
        self._is_running = True
        logger.info("Unified Trading System started successfully")
        
    async def stop(self):
        """Stop all trading systems"""
        logger.info("Stopping Unified Trading System...")
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        self._tasks = []
        self._is_running = False
        
        logger.info("Unified Trading System stopped")
        
    # =======================
    # COPY TRADING API
    # =======================
    
    async def register_trader(self, **kwargs) -> str:
        """Register a new trader for copy trading"""
        from reco_trading.copy_trading import TraderProfile
        trader = TraderProfile(**kwargs)
        return await self.copy_trading.register_trader(trader)
    
    async def follow_trader(self, follower_id: str, trader_id: str, **config) -> str:
        """Follow a trader"""
        from reco_trading.copy_trading import FollowerConfig
        follower_config = FollowerConfig(
            follower_id=follower_id,
            trader_id=trader_id,
            **config
        )
        return await self.copy_trading.follow_trader(follower_config)
    
    async def get_top_traders(self, limit: int = 10):
        """Get top performing traders"""
        return self.copy_trading.get_top_traders(limit)
    
    def get_signals(self, source: SignalSource = None):
        """Get active trading signals"""
        return self.copy_trading.signal_manager.get_active_signals(source)
    
    # =======================
    # GRID BOT API
    # =======================
    
    async def create_grid_bot(self, **config) -> str:
        """Create a new grid trading bot"""
        grid_config = GridConfig(**config)
        return await self.grid_bot.create_bot(grid_config)
    
    async def start_grid_bot(self, config_id: str, current_price: float) -> bool:
        """Start a grid bot"""
        return await self.grid_bot.start_bot(config_id, current_price)
    
    async def stop_grid_bot(self, config_id: str):
        """Stop a grid bot"""
        return await self.grid_bot.stop_bot(config_id)
    
    def get_grid_status(self, config_id: str = None):
        """Get grid bot status"""
        if config_id:
            return self.grid_bot.get_bot_status(config_id)
        return self.grid_bot.get_all_bots_status()
    
    # =======================
    # TRADINGVIEW API
    # =======================
    
    def get_webhook_stats(self):
        """Get webhook processing statistics"""
        return self.tradingview.get_signal_stats()
    
    def get_webhook_signals(self):
        """Get recent webhook signals"""
        return self.tradingview.get_active_signals()
    
    # =======================
    # MARKETPLACE API
    # =======================
    
    async def create_template(self, **template_data):
        """Create a new strategy template"""
        from reco_trading.marketplace import StrategyTemplate
        template = StrategyTemplate(**template_data)
        return await self.marketplace.create_template(template)
    
    async def publish_template(self, template_id: str) -> bool:
        """Publish a template"""
        return await self.marketplace.publish_template(template_id)
    
    async def search_templates(self, **filters):
        """Search templates"""
        return await self.marketplace.search_templates(**filters)
    
    def get_featured_templates(self):
        """Get featured templates"""
        return asyncio.run(self.marketplace.get_featured_templates())
    
    # =======================
    # CLOUD SYNC API
    # =======================
    
    async def sync_now(self):
        """Perform manual sync"""
        return await self.cloud_sync.sync_now()
    
    async def restore_all(self):
        """Restore from backup"""
        return await self.cloud_sync.restore_all()
    
    def get_sync_status(self):
        """Get sync status"""
        return self.cloud_sync.get_status()
    
    async def add_to_sync(self, item_type: str, name: str, local_path: str):
        """Add item to sync queue"""
        from reco_trading.cloud_sync import SyncItem, SyncItemType
        item = SyncItem(
            item_type=SyncItemType(item_type),
            name=name,
            local_path=local_path
        )
        return await self.cloud_sync.add_sync_item(item)
    
    # =======================
    # MOBILE API
    # =======================
    
    async def send_trade_notification(self, device_id: str, trade_data: dict):
        """Send trade notification"""
        return await self.push_notifications.send_trade_notification(device_id, trade_data)
    
    async def send_alert(self, device_id: str, title: str, message: str):
        """Send alert notification"""
        return await self.push_notifications.send_alert_notification(
            device_id, {"title": title, "message": message}
        )
    
    def get_mobile_app(self):
        """Get mobile API manager"""
        return self.mobile_api
    
    # =======================
    # STATUS & MONITORING
    # =======================
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            "system_running": self._is_running,
            "copy_trading": {
                "active_traders": len(self.copy_trading.traders),
                "active_followers": len(self.copy_trading.followers),
                "copied_trades": len(self.copy_trading.copied_trades),
            },
            "grid_bots": {
                "total_bots": len(self.grid_bot.bots),
                "active_bots": self.grid_bot.get_active_bots_count(),
            },
            "tradingview": self.tradingview.get_signal_stats(),
            "marketplace": self.marketplace.get_marketplace_stats(),
            "cloud_sync": self.cloud_sync.get_status(),
            "mobile_devices": len(self.mobile_api.devices),
        }


# Global unified system instance
_unified_system: Optional[UnifiedTradingSystem] = None


def get_unified_trading_system(settings: Any = None) -> UnifiedTradingSystem:
    """Get or create the unified trading system"""
    global _unified_system
    if _unified_system is None:
        _unified_system = UnifiedTradingSystem(settings)
    return _unified_system


# Convenient access to subsystems
def get_copy_trading() -> CopyTradingManager:
    """Get copy trading manager"""
    return get_unified_trading_system().copy_trading


def get_grid_bots() -> GridBotManager:
    """Get grid bot manager"""
    return get_unified_trading_system().grid_bot


def get_tradingview_handler() -> TradingViewWebhookHandler:
    """Get TradingView handler"""
    return get_unified_trading_system().tradingview


def get_market() -> TemplateMarketplace:
    """Get marketplace"""
    return get_unified_trading_system().marketplace


def get_sync() -> CloudSyncManager:
    """Get cloud sync"""
    return get_unified_trading_system().cloud_sync


def get_mobile() -> MobileAPIManager:
    """Get mobile API"""
    return get_unified_trading_system().mobile_api


__all__ = [
    "UnifiedTradingSystem",
    "get_unified_trading_system",
    "get_copy_trading",
    "get_grid_bots",
    "get_tradingview_handler",
    "get_market",
    "get_sync",
    "get_mobile",
]