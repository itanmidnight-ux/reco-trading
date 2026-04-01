#!/usr/bin/env python3
"""
Social Trading Marketplace
Premium social trading platform with copy trading, strategy sharing, and community features
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import hashlib

from pydantic import BaseModel, EmailStr
from ..config.settings import Settings

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy marketplace status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    VERIFIED = "verified"
    FEATURED = "featured"
    PRIVATE = "private"


class TradeSignal(BaseModel):
    """Trading signal for social trading"""
    symbol: str
    action: str  # buy, sell
    price: float
    amount: Optional[float] = None
    percentage: Optional[float] = None
    timestamp: datetime
    confidence: float = 0.5
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None


class TradingStrategy(BaseModel):
    """Shared trading strategy"""
    id: str
    name: str
    description: str
    author_id: str
    author_name: str
    status: StrategyStatus
    tags: List[str] = []
    
    # Performance metrics
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_hold_time: float = 0.0
    
    # Configuration
    risk_level: int = 1  # 1-10
    min_capital: float = 100.0
    exchanges: List[str] = []
    timeframes: List[str] = []
    
    # Marketplace
    price: float = 0.0
    subscription_type: str = "free"  # free, premium
    copy_count: int = 0
    rating: float = 0.0
    review_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class CopyTrader(BaseModel):
    """Copy trading configuration"""
    user_id: str
    strategy_id: str
    allocation_percentage: float = 10.0  # % of portfolio
    max_risk_per_trade: float = 2.0
    min_trade_size: float = 10.0
    max_leverage: int = 1
    copy_signals: bool = True
    copy_positions: bool = True
    enabled: bool = True
    started_at: datetime = field(default_factory=datetime.now)


class TradingSignal(BaseModel):
    """Real-time trading signal"""
    id: str
    strategy_id: str
    symbol: str
    action: str  # buy, sell, close
    timestamp: datetime = field(default_factory=datetime.now)
    price: float
    amount: Optional[float] = None
    percentage: Optional[float] = None
    confidence: float = 0.5
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SocialTradingMarketplace:
    """
    Advanced social trading marketplace with strategy marketplace,
    copy trading, and community features
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Data storage (in production, use database)
        self.strategies: Dict[str, TradingStrategy] = {}
        self.copy_traders: Dict[str, CopyTrader] = {}
        self.active_signals: Dict[str, TradingSignal] = {}
        self.user_portfolios: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.strategy_performance: Dict[str, List[Dict]] = {}
        self.copy_performance: Dict[str, List[Dict]] = {}
        
    async def publish_strategy(
        self,
        name: str,
        description: str,
        author_id: str,
        author_name: str,
        config: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> str:
        """Publish a new strategy to the marketplace"""
        
        # Generate strategy ID
        strategy_id = self._generate_strategy_id(name, author_id)
        
        # Extract performance from backtest results
        performance = backtest_results.get("performance", {})
        
        strategy = TradingStrategy(
            id=strategy_id,
            name=name,
            description=description,
            author_id=author_id,
            author_name=author_name,
            status=StrategyStatus.PUBLISHED,
            
            # Performance metrics
            win_rate=performance.get("win_rate", 0),
            total_trades=performance.get("total_trades", 0),
            profit_factor=performance.get("profit_factor", 0),
            max_drawdown=performance.get("max_drawdown", 0),
            sharpe_ratio=performance.get("sharpe_ratio", 0),
            avg_hold_time=performance.get("avg_hold_time", 0),
            
            # Configuration from strategy
            risk_level=config.get("risk_level", 5),
            min_capital=config.get("min_capital", 100),
            exchanges=config.get("exchanges", ["binance"]),
            timeframes=config.get("timeframes", ["5m"]),
            
            # Marketplace settings
            price=config.get("price", 0.0),
            subscription_type=config.get("subscription_type", "free"),
        )
        
        # Store strategy
        self.strategies[strategy_id] = strategy
        
        # Initialize performance tracking
        self.strategy_performance[strategy_id] = []
        
        self.logger.info(f"Strategy published: {name} by {author_name}")
        return strategy_id
        
    async def copy_trade_strategy(
        self,
        user_id: str,
        strategy_id: str,
        allocation_percentage: float = 10.0,
        max_risk_per_trade: float = 2.0
    ) -> bool:
        """Start copying a strategy"""
        
        # Validate strategy exists
        if strategy_id not in self.strategies:
            logger.error(f"Strategy {strategy_id} not found")
            return False
            
        strategy = self.strategies[strategy_id]
        
        # Check if user already copying this strategy
        copy_id = f"{user_id}_{strategy_id}"
        if copy_id in self.copy_traders:
            logger.warning(f"User {user_id} already copying strategy {strategy_id}")
            return False
            
        # Create copy trader configuration
        copy_trader = CopyTrader(
            user_id=user_id,
            strategy_id=strategy_id,
            allocation_percentage=allocation_percentage,
            max_risk_per_trade=max_risk_per_trade,
            min_trade_size=strategy.min_capital * 0.01,
        )
        
        # Store copy trader
        self.copy_traders[copy_id] = copy_trader
        
        # Update strategy copy count
        strategy.copy_count += 1
        
        # Initialize copy performance tracking
        self.copy_performance[copy_id] = []
        
        self.logger.info(f"User {user_id} started copying strategy {strategy.name}")
        return True
        
    async def emit_signal(
        self,
        strategy_id: str,
        symbol: str,
        action: str,
        price: float,
        amount: Optional[float] = None,
        percentage: Optional[float] = None,
        confidence: float = 0.5,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reasoning: Optional[str] = None
    ) -> str:
        """Emit a trading signal to followers"""
        
        # Validate strategy
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
            
        # Create signal
        signal = TradingSignal(
            id=self._generate_signal_id(),
            strategy_id=strategy_id,
            symbol=symbol,
            action=action,
            timestamp=datetime.now(),
            price=price,
            amount=amount,
            percentage=percentage,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )
        
        # Store signal
        self.active_signals[signal.id] = signal
        
        # Distribute to copy traders
        await self._distribute_to_copy_traders(signal)
        
        # Update strategy performance
        await self._update_strategy_performance(strategy_id, signal)
        
        self.logger.info(f"Signal emitted: {action} {symbol} @ {price} (strategy: {strategy_id})")
        return signal.id
        
    async def _distribute_to_copy_traders(self, signal: TradingSignal) -> None:
        """Distribute signal to active copy traders"""
        
        # Find all copy traders for this strategy
        target_copies = [
            copy for copy in self.copy_traders.values()
            if copy.strategy_id == signal.strategy_id and copy.enabled
        ]
        
        for copy_trader in target_copies:
            try:
                # Calculate position size for this user
                user_portfolio = self.user_portfolios.get(copy_trader.user_id, {})
                total_balance = sum(user_portfolio.values())
                
                if total_balance <= 0:
                    continue
                    
                # Calculate trade size based on allocation
                if signal.amount:
                    trade_size = signal.amount * (copy_trader.allocation_percentage / 100)
                elif signal.percentage:
                    trade_size = total_balance * signal.percentage * (copy_trader.allocation_percentage / 100)
                else:
                    continue  # Skip if neither amount nor percentage provided
                    
                # Apply risk limits
                max_allowed = total_balance * (copy_trader.max_risk_per_trade / 100)
                trade_size = min(trade_size, max_allowed)
                
                if trade_size < copy_trader.min_trade_size:
                    continue
                    
                # Execute trade for user (would integrate with exchange API)
                execution_result = await self._execute_copy_trade(
                    copy_trader.user_id,
                    signal.symbol,
                    signal.action,
                    trade_size,
                    signal.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )
                
                # Track performance
                self.copy_performance[f"{copy_trader.user_id}_{signal.strategy_id}"].append({
                    "signal_id": signal.id,
                    "timestamp": datetime.now(),
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "size": trade_size,
                    "execution": execution_result
                })
                
            except Exception as e:
                self.logger.error(f"Failed to copy signal for user {copy_trader.user_id}: {e}")
                
    async def _execute_copy_trade(
        self,
        user_id: str,
        symbol: str,
        action: str,
        amount: float,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute copy trade for user"""
        # This would integrate with the user's exchange account
        # For now, return a mock execution result
        return {
            "success": True,
            "order_id": f"copy_{user_id}_{symbol}_{int(datetime.now().timestamp())}",
            "executed_price": price,
            "executed_amount": amount,
            "fees": amount * price * 0.001,
            "status": "filled"
        }
        
    async def get_strategy_details(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed strategy information"""
        if strategy_id not in self.strategies:
            return None
            
        strategy = self.strategies[strategy_id]
        performance = self.strategy_performance.get(strategy_id, [])
        
        # Calculate recent performance
        recent_performance = performance[-10:] if len(performance) > 10 else performance
        recent_win_rate = sum(1 for p in recent_performance if p.get("win", False)) / len(recent_performance) if recent_performance else 0
        
        # Get top copiers
        top_copiers = sorted(
            [(f"{c.user_id}_{c.strategy_id}", c.allocation_percentage) 
             for c in self.copy_traders.values() if c.strategy_id == strategy_id],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            "strategy": strategy.dict(),
            "performance": {
                "recent_win_rate": recent_win_rate,
                "total_signals": len(performance),
                "copy_count": strategy.copy_count
            },
            "top_copiers": top_copiers,
            "latest_signals": [s.dict() for s in await self._get_strategy_signals(strategy_id, limit=5)]
        }
        
    async def _get_strategy_signals(self, strategy_id: str, limit: int = 10) -> List[TradingSignal]:
        """Get latest signals for a strategy"""
        signals = [s for s in self.active_signals.values() if s.strategy_id == strategy_id]
        signals.sort(key=lambda x: x.timestamp, reverse=True)
        return signals[:limit]
        
    async def get_trending_strategies(self, time_period: int = 7) -> List[Dict[str, Any]]:
        """Get trending strategies based on recent performance and adoption"""
        
        cutoff_date = datetime.now() - timedelta(days=time_period)
        
        trending = []
        for strategy in self.strategies.values():
            # Recent performance
            performance = self.strategy_performance.get(strategy.id, [])
            recent_perf = [p for p in performance if p.get("timestamp", datetime.now()) > cutoff_date]
            
            if not recent_perf:
                continue
                
            recent_win_rate = sum(1 for p in recent_perf if p.get("win", False)) / len(recent_perf)
            
            # Recent copy adoption
            recent_copies = sum(1 for c in self.copy_traders.values() 
                              if c.strategy_id == strategy.id and c.started_at > cutoff_date)
            
            # Calculate trending score
            trending_score = (
                recent_win_rate * 0.3 +
                min(recent_copies / 10, 1.0) * 0.4 +
                strategy.copy_count / 100 * 0.3
            )
            
            trending.append({
                "strategy": strategy.dict(),
                "trending_score": trending_score,
                "recent_win_rate": recent_win_rate,
                "recent_copies": recent_copies
            })
            
        # Sort by trending score
        trending.sort(key=lambda x: x["trending_score"], reverse=True)
        return trending[:10]
        
    async def verify_strategy(self, strategy_id: str, verifier_id: str) -> bool:
        """Verify a strategy (admin function)"""
        if strategy_id not in self.strategies:
            return False
            
        self.strategies[strategy_id].status = StrategyStatus.VERIFIED
        self.logger.info(f"Strategy {strategy_id} verified by {verifier_id}")
        return True
        
    async def _update_strategy_performance(self, strategy_id: str, signal: TradingSignal) -> None:
        """Update strategy performance tracking"""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = []
            
        # Add to performance history (mock result for now)
        self.strategy_performance[strategy_id].append({
            "signal_id": signal.id,
            "timestamp": datetime.now(),
            "win": True if signal.action == "buy" else False,  # Mock logic
            "pnl": 10.0 if signal.action == "buy" else -5.0,  # Mock PNL
        })
        
        # Update strategy metrics (periodic)
        if len(self.strategy_performance[strategy_id]) % 10 == 0:
            await self._recalculate_strategy_metrics(strategy_id)
            
    async def _recalculate_strategy_metrics(self, strategy_id: str) -> None:
        """Recalculate and update strategy performance metrics"""
        if strategy_id not in self.strategies or strategy_id not in self.strategy_performance:
            return
            
        strategy = self.strategies[strategy_id]
        performance = self.strategy_performance[strategy_id]
        
        if not performance:
            return
            
        # Calculate metrics
        wins = sum(1 for p in performance if p.get("win", False))
        total_trades = len(performance)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(p.get("pnl", 0) for p in performance)
        losing_trades = [p for p in performance if p.get("pnl", 0) < 0]
        max_drawdown = min(p.get("pnl", 0) for p in losing_trades) if losing_trades else 0
        
        # Update strategy
        strategy.win_rate = win_rate
        strategy.total_trades = total_trades
        strategy.max_drawdown = abs(max_drawdown)
        
        self.logger.info(f"Updated metrics for strategy {strategy.name}: win_rate={win_rate:.2%}")
        
    def _generate_strategy_id(self, name: str, author_id: str) -> str:
        """Generate unique strategy ID"""
        content = f"{name}_{author_id}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
        
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        content = f"signal_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# API endpoints for Social Trading
from fastapi import APIRouter, HTTPException, Depends
from typing import List

social_router = APIRouter(prefix="/api/social", tags=["social"])

# Global marketplace instance
marketplace = None

def get_marketplace() -> SocialTradingMarketplace:
    """Get marketplace instance"""
    global marketplace
    if marketplace is None:
        marketplace = SocialTradingMarketplace(None)  # Pass actual settings
    return marketplace

@social_router.post("/strategies")
async def create_strategy(
    name: str,
    description: str,
    author_id: str,
    author_name: str,
    config: Dict[str, Any],
    backtest_results: Dict[str, Any],
    mp: SocialTradingMarketplace = Depends(get_marketplace)
) -> Dict[str, str]:
    """Create and publish a new strategy"""
    strategy_id = await mp.publish_strategy(name, description, author_id, author_name, config, backtest_results)
    return {"strategy_id": strategy_id}

@social_router.get("/strategies")
async def list_strategies(
    status: Optional[str] = None,
    limit: int = 20,
    mp: SocialTradingMarketplace = Depends(get_marketplace)
) -> List[Dict[str, Any]]:
    """List available strategies"""
    strategies = []
    for strategy in mp.strategies.values():
        if status and strategy.status.value != status:
            continue
        strategies.append(strategy.dict())
    
    return strategies[:limit]

@social_router.get("/strategies/trending")
async def get_trending_strategies(
    time_period: int = 7,
    mp: SocialTradingMarketplace = Depends(get_marketplace)
) -> List[Dict[str, Any]]:
    """Get trending strategies"""
    return await mp.get_trending_strategies(time_period)

@social_router.post("/copy")
async def start_copying(
    user_id: str,
    strategy_id: str,
    allocation_percentage: float = 10.0,
    max_risk_per_trade: float = 2.0,
    mp: SocialTradingMarketplace = Depends(get_marketplace)
) -> Dict[str, str]:
    """Start copying a strategy"""
    success = await mp.copy_trade_strategy(user_id, strategy_id, allocation_percentage, max_risk_per_trade)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start copying strategy")
    return {"status": "success", "message": f"Now copying strategy {strategy_id}"}

@social_router.post("/signals")
async def emit_signal(
    strategy_id: str,
    symbol: str,
    action: str,
    price: float,
    amount: Optional[float] = None,
    percentage: Optional[float] = None,
    confidence: float = 0.5,
    reasoning: Optional[str] = None,
    mp: SocialTradingMarketplace = Depends(get_marketplace)
) -> Dict[str, str]:
    """Emit a trading signal"""
    try:
        signal_id = await mp.emit_signal(
            strategy_id, symbol, action, price, amount, percentage, confidence, reasoning=reasoning
        )
        return {"signal_id": signal_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@social_router.get("/strategies/{strategy_id}")
async def get_strategy_details(
    strategy_id: str,
    mp: SocialTradingMarketplace = Depends(get_marketplace)
) -> Dict[str, Any]:
    """Get detailed strategy information"""
    details = await mp.get_strategy_details(strategy_id)
    if not details:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return details
