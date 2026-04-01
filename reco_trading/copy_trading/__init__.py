"""Copy Trading System for Reco-Trading
Provides signal marketplace, trader tracking, and automatic trade copying
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Source of trading signals"""
    INTERNAL = "internal"
    TRADINGVIEW = "tradingview"
    EXTERNAL_API = "external_api"
    COPY_TRADING = "copy_trading"
    COMMUNITY = "community"


class SignalAction(Enum):
    """Trading signal action"""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    BUY_TP = "buy_tp"
    SELL_TP = "sell_tp"


class TraderTier(Enum):
    """Trader performance tier"""
    NEWCOMER = "newcomer"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    LEGEND = "legend"


@dataclass
class TradingSignal:
    """Trading signal from any source"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: SignalSource = SignalSource.INTERNAL
    action: SignalAction = SignalAction.BUY
    symbol: str = ""
    price: float = 0.0
    confidence: float = 0.7
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)
    strategy_name: str = ""
    indicators: dict = field(default_factory=dict)
    risk_level: str = "medium"
    max_trades: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    

@dataclass
class TraderProfile:
    """Profile of a trader for copy trading"""
    trader_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    display_name: str = ""
    avatar_url: str = ""
    bio: str = ""
    tier: TraderTier = TraderTier.NEWCOMER
    
    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    avg_trade_duration: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Follower metrics
    follower_count: int = 0
    total_aum: float = 0.0
    success_rate: float = 0.0
    
    # Status
    is_public: bool = False
    is_verified: bool = False
    is_premium: bool = False
    subscription_required: bool = False
    subscription_price: float = 0.0
    
    # Settings
    allow_copy: bool = True
    min_follower_investment: float = 100.0
    max_followers: int = 1000
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    

@dataclass
class CopiedTrade:
    """Record of a copied trade"""
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_trader_id: str = ""
    follower_id: str = ""
    original_signal_id: str = ""
    
    symbol: str = ""
    action: str = ""
    amount: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    
    status: str = "open"
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    
    copy_ratio: float = 1.0


@dataclass
class FollowerConfig:
    """Configuration for a follower copying a trader"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    follower_id: str = ""
    trader_id: str = ""
    
    copy_ratio: float = 1.0
    max_investment: float = 10000.0
    stop_loss_percent: float = 5.0
    take_profit_percent: float = 10.0
    
    auto_close_on_sl: bool = True
    auto_close_on_tp: bool = True
    copy_stop_loss: bool = True
    copy_take_profit: bool = True
    
    is_active: bool = True
    
    created_at: datetime = field(default_factory=datetime.now)


class SignalManager:
    """Manages all trading signals"""
    
    def __init__(self):
        self.signals: dict[str, TradingSignal] = {}
        self.signal_history: list[TradingSignal] = []
        self.subscribers: list[asyncio.Queue] = []
        
    async def publish_signal(self, signal: TradingSignal) -> None:
        """Publish a new trading signal"""
        self.signals[signal.id] = signal
        self.signal_history.append(signal)
        
        # Notify subscribers
        for queue in self.subscribers:
            await queue.put(signal)
            
        logger.info(f"Signal published: {signal.action} {signal.symbol} @ {signal.price}")
        
    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to signals"""
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from signals"""
        if queue in self.subscribers:
            self.subscribers.remove(queue)
    
    def get_active_signals(self, source: Optional[SignalSource] = None) -> list[TradingSignal]:
        """Get currently active signals"""
        now = datetime.now()
        active = []
        
        for signal in self.signals.values():
            if signal.expires_at and signal.expires_at < now:
                continue
            if source and signal.source != source:
                continue
            active.append(signal)
            
        return active
    
    def get_signals_by_symbol(self, symbol: str) -> list[TradingSignal]:
        """Get signals for a specific symbol"""
        return [s for s in self.signals.values() if s.symbol == symbol]
    
    def get_signals_by_action(self, action: SignalAction) -> list[TradingSignal]:
        """Get signals by action type"""
        return [s for s in self.signals.values() if s.action == action]
    
    def cleanup_expired(self) -> int:
        """Remove expired signals"""
        now = datetime.now()
        expired_ids = []
        
        for sid, signal in self.signals.items():
            if signal.expires_at and signal.expires_at < now:
                expired_ids.append(sid)
                
        for sid in expired_ids:
            del self.signals[sid]
            
        return len(expired_ids)


class CopyTradingManager:
    """Manages copy trading operations"""
    
    def __init__(self, repository: Any = None):
        self.repository = repository
        self.traders: dict[str, TraderProfile] = {}
        self.followers: dict[str, list[FollowerConfig]] = {}
        self.copied_trades: dict[str, CopiedTrade] = {}
        self.signal_manager = SignalManager()
        
    async def register_trader(self, trader: TraderProfile) -> str:
        """Register a new trader"""
        self.traders[trader.trader_id] = trader
        logger.info(f"Trader registered: {trader.display_name}")
        return trader.trader_id
    
    async def update_trader_stats(self, trader_id: str, stats: dict) -> None:
        """Update trader statistics"""
        if trader_id in self.traders:
            trader = self.traders[trader_id]
            
            for key, value in stats.items():
                if hasattr(trader, key):
                    setattr(trader, key, value)
                    
            trader.updated_at = datetime.now()
            
            # Update tier based on performance
            await self._update_tier(trader)
            
    async def _update_tier(self, trader: TraderProfile) -> None:
        """Update trader tier based on performance"""
        if trader.success_rate >= 0.8 and trader.total_trades >= 100:
            trader.tier = TraderTier.LEGEND
        elif trader.success_rate >= 0.7 and trader.total_trades >= 50:
            trader.tier = TraderTier.PLATINUM
        elif trader.success_rate >= 0.6 and trader.total_trades >= 30:
            trader.tier = TraderTier.GOLD
        elif trader.success_rate >= 0.55 and trader.total_trades >= 20:
            trader.tier = TraderTier.SILVER
        elif trader.success_rate >= 0.5 and trader.total_trades >= 10:
            trader.tier = TraderTier.BRONZE
        else:
            trader.tier = TraderTier.NEWCOMER
    
    async def follow_trader(self, config: FollowerConfig) -> str:
        """Start following a trader"""
        follower_id = config.follower_id
        trader_id = config.trader_id
        
        if follower_id not in self.followers:
            self.followers[follower_id] = []
            
        # Check if already following
        for existing in self.followers[follower_id]:
            if existing.trader_id == trader_id:
                return existing.config_id
                
        self.followers[follower_id].append(config)
        logger.info(f"Follower {follower_id} started following trader {trader_id}")
        
        return config.config_id
    
    async def unfollow_trader(self, follower_id: str, trader_id: str) -> bool:
        """Stop following a trader"""
        if follower_id in self.followers:
            self.followers[follower_id] = [
                f for f in self.followers[follower_id]
                if f.trader_id != trader_id
            ]
            return True
        return False
    
    async def copy_signal(self, signal: TradingSignal) -> list[CopiedTrade]:
        """Copy a signal to all active followers"""
        copied_trades = []
        
        # Find all followers who are following traders that published this signal
        for follower_id, configs in self.followers.items():
            for config in configs:
                if not config.is_active:
                    continue
                    
                # Create copied trade
                trade = CopiedTrade(
                    original_trader_id=signal.metadata.get("trader_id", ""),
                    follower_id=follower_id,
                    original_signal_id=signal.id,
                    symbol=signal.symbol,
                    action=signal.action.value,
                    amount=signal.metadata.get("amount", 0),
                    entry_price=signal.price,
                    copy_ratio=config.copy_ratio
                )
                
                self.copied_trades[trade.trade_id] = trade
                copied_trades.append(trade)
                
        return copied_trades
    
    async def update_copied_trade(self, trade_id: str, current_price: float) -> Optional[CopiedTrade]:
        """Update a copied trade with current price"""
        if trade_id in self.copied_trades:
            trade = self.copied_trades[trade_id]
            trade.current_price = current_price
            
            if trade.action == "buy":
                trade.pnl = (current_price - trade.entry_price) * trade.amount
                trade.pnl_percent = ((current_price - trade.entry_price) / trade.entry_price) * 100
            else:
                trade.pnl = (trade.entry_price - current_price) * trade.amount
                trade.pnl_percent = ((trade.entry_price - current_price) / trade.entry_price) * 100
                
            return trade
        return None
    
    def get_top_traders(self, limit: int = 10, tier: Optional[TraderTier] = None) -> list[TraderProfile]:
        """Get top performing traders"""
        traders = list(self.traders.values())
        
        if tier:
            traders = [t for t in traders if t.tier == tier]
            
        # Sort by total_pnl_percent
        traders.sort(key=lambda x: x.total_pnl_percent, reverse=True)
        
        return traders[:limit]
    
    def get_trader_by_id(self, trader_id: str) -> Optional[TraderProfile]:
        """Get trader by ID"""
        return self.traders.get(trader_id)
    
    def get_following(self, follower_id: str) -> list[FollowerConfig]:
        """Get list of traders a follower is following"""
        return self.followers.get(follower_id, [])
    
    def get_copied_trades(self, follower_id: str) -> list[CopiedTrade]:
        """Get all copied trades for a follower"""
        return [t for t in self.copied_trades.values() if t.follower_id == follower_id]
    
    async def close_copied_trade(self, trade_id: str, exit_price: float) -> Optional[CopiedTrade]:
        """Close a copied trade"""
        if trade_id in self.copied_trades:
            trade = self.copied_trades[trade_id]
            trade.status = "closed"
            trade.closed_at = datetime.now()
            trade.current_price = exit_price
            
            if trade.action == "buy":
                trade.pnl = (exit_price - trade.entry_price) * trade.amount
                trade.pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
            else:
                trade.pnl = (trade.entry_price - exit_price) * trade.amount
                trade.pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100
                
            return trade
        return None


class SignalMarketplace:
    """Marketplace for trading signals"""
    
    def __init__(self, copy_trading: CopyTradingManager):
        self.copy_trading = copy_trading
        self.listings: dict[str, dict] = {}
        
    async def list_signal(self, signal: TradingSignal, price: float = 0.0, 
                          description: str = "") -> str:
        """List a signal in the marketplace"""
        listing_id = str(uuid.uuid4())
        
        self.listings[listing_id] = {
            "signal": signal,
            "price": price,
            "description": description,
            "purchases": 0,
            "rating": 0.0,
            "created_at": datetime.now()
        }
        
        return listing_id
    
    def get_listings(self, min_rating: float = 0.0, 
                     max_price: float = float('inf')) -> list[dict]:
        """Get marketplace listings"""
        return [
            l for l in self.listings.values()
            if l["rating"] >= min_rating and l["price"] <= max_price
        ]
    
    async def purchase_signal(self, listing_id: str, buyer_id: str) -> Optional[TradingSignal]:
        """Purchase a signal from the marketplace"""
        if listing_id in self.listings:
            listing = self.listings[listing_id]
            listing["purchases"] += 1
            return listing["signal"]
        return None
    
    async def rate_signal(self, listing_id: str, rating: float) -> bool:
        """Rate a signal"""
        if listing_id in self.listings:
            listing = self.listings[listing_id]
            current_rating = listing["rating"]
            purchases = listing["purchases"]
            listing["rating"] = (current_rating * purchases + rating) / (purchases + 1)
            return True
        return False


# Global instance
_copy_trading_manager: Optional[CopyTradingManager] = None
_signal_marketplace: Optional[SignalMarketplace] = None


def get_copy_trading_manager(repository: Any = None) -> CopyTradingManager:
    """Get or create the copy trading manager"""
    global _copy_trading_manager
    if _copy_trading_manager is None:
        _copy_trading_manager = CopyTradingManager(repository)
    return _copy_trading_manager


def get_signal_marketplace() -> SignalMarketplace:
    """Get or create the signal marketplace"""
    global _signal_marketplace
    if _signal_marketplace is None:
        _signal_marketplace = SignalMarketplace(get_copy_trading_manager())
    return _signal_marketplace


__all__ = [
    "SignalSource",
    "SignalAction", 
    "TraderTier",
    "TradingSignal",
    "TraderProfile",
    "CopiedTrade",
    "FollowerConfig",
    "SignalManager",
    "CopyTradingManager",
    "SignalMarketplace",
    "get_copy_trading_manager",
    "get_signal_marketplace",
]