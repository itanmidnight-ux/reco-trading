#!/usr/bin/env python3
"""
Advanced Futures and Short Selling Support
Add professional futures trading with leverage, short selling, and position management
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from ccxt import Exchange
from ..config.settings import Settings

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side enumeration for futures trading"""
    LONG = "long"
    SHORT = "short" 
    BOTH = "both"


class OrderType(Enum):
    """Enhanced order types for futures"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss_limit"
    TAKE_PROFIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class FuturesPosition:
    """Futures position tracking"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    leverage: int = 1
    margin_type: str = "cross"  # cross or isolated
    liquidation_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)
    
    @property
    def pnl_percentage(self) -> float:
        """PNL as percentage of entry price"""
        if self.entry_price and self.entry_price != 0:
            return (self.unrealized_pnl / abs(self.size * self.entry_price)) * 100
        return 0.0
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable"""
        return self.unrealized_pnl > 0


@dataclass
class LeverageConfig:
    """Leverage configuration by symbol category"""
    spot_max: int = 1
    futures_max: int = 125
    default_leverage: int = 10
    symbol_specific: Dict[str, int] = None
    
    def __post_init__(self):
        if self.symbol_specific is None:
            self.symbol_specific = {
                "BTCUSDT": 50,
                "ETHUSDT": 50,
                "SOLUSDT": 25,
                "BNBUSDT": 25,
            }


class FuturesManager:
    """
    Professional futures trading manager with short selling capabilities
    Advanced features: dynamic leverage, position sizing, risk management
    """
    
    def __init__(self, exchange: Exchange, settings: Settings):
        self.exchange = exchange
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Position tracking
        self.positions: Dict[str, FuturesPosition] = {}
        self.leverage_config = LeverageConfig()
        
        # Risk management
        self.max_positions = 5
        self.max_leverage = settings.futures_max_leverage or 50
        self.position_size_limit = 0.1  # 10% of equity per position
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit
        
    async def initialize(self) -> None:
        """Initialize futures manager"""
        await self.update_positions()
        await self.configure_leverage()
        
    async def configure_leverage(self, symbol: Optional[str] = None, leverage: Optional[int] = None) -> Dict[str, Any]:
        """Configure leverage for specific symbol or default"""
        if symbol and leverage:
            try:
                if hasattr(self.exchange, 'set_leverage'):
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self.exchange.set_leverage, leverage, symbol),
                        timeout=10.0
                    )
                    self.logger.info(f"Set leverage {leverage}x for {symbol}")
                    return result
            except Exception as e:
                self.logger.error(f"Failed to set leverage for {symbol}: {e}")
                
        # Set default leverage for all symbols
        default_leverage = self.leverage_config.default_leverage
        if hasattr(self.exchange, 'set_leverage'):
            try:
                for sym_lev in self.leverage_config.symbol_specific.items():
                    await asyncio.wait_for(
                        asyncio.to_thread(self.exchange.set_leverage, sym_lev[1], sym_lev[0]),
                        timeout=10.0
                    )
                self.logger.info(f"Configured symbol-specific leverage")
            except Exception as e:
                self.logger.warning(f"Failed to configure symbol leverage: {e}")
                
        return {"status": "configured", "default_leverage": default_leverage}
        
    async def open_long_position(
        self,
        symbol: str,
        amount: float,
        entry_price: Optional[float] = None,
        leverage: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Open long position with optional stop loss and take profit"""
        
        # Calculate position size based on equity
        balance = await self.get_available_balance()
        max_position_value = balance * self.position_size_limit
        
        # Apply leverage if specified
        if leverage:
            max_position_value *= leverage
            
        # Limit position size
        if (amount * (entry_price or 0)) > max_position_value:
            amount = max_position_value / (entry_price or 1)
            self.logger.info(f"Reduced position size to {amount} based on risk limits")
            
        # Create market order for long
        order = await self.create_futures_order(
            symbol=symbol,
            side="buy",
            amount=amount,
            price=entry_price,
            order_type="market" if not entry_price else "limit",
            position_side="long"
        )
        
        if order and order.get('status') == 'closed':
            # Track position
            entry_price_filled = order.get('price') or entry_price
            position = FuturesPosition(
                symbol=symbol,
                side=PositionSide.LONG,
                size=amount,
                entry_price=entry_price_filled,
                current_price=entry_price_filled,
                unrealized_pnl=0.0,
                leverage=leverage or self.leverage_config.default_leverage,
            )
            
            self.positions[symbol] = position
            
            # Place stop loss and take profit
            if stop_loss or self.stop_loss_percentage:
                sl_price = stop_loss or entry_price_filled * (1 - self.stop_loss_percentage)
                await self.place_stop_loss(symbol, "long", amount, sl_price)
                
            if take_profit or self.take_profit_percentage:
                tp_price = take_profit or entry_price_filled * (1 + self.take_profit_percentage)
                await self.place_take_profit(symbol, "long", amount, tp_price)
                
            self.logger.info(f"Opened LONG position: {symbol} @ {entry_price_filled}")
            
        return order
        
    async def open_short_position(
        self,
        symbol: str,
        amount: float,
        entry_price: Optional[float] = None,
        leverage: Optional[int] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Open short position (selling first)"""
        
        # For short selling, we sell first then buy back later
        order = await self.create_futures_order(
            symbol=symbol,
            side="sell",
            amount=amount,
            price=entry_price,
            order_type="market" if not entry_price else "limit",
            position_side="short"
        )
        
        if order and order.get('status') == 'closed':
            # Track short position (negative size indicates short)
            entry_price_filled = order.get('price') or entry_price
            position = FuturesPosition(
                symbol=symbol,
                side=PositionSide.SHORT,
                size=-amount,  # Negative for short
                entry_price=entry_price_filled,
                current_price=entry_price_filled,
                unrealized_pnl=0.0,
                leverage=leverage or self.leverage_config.default_leverage,
            )
            
            self.positions[symbol] = position
            
            # For shorts, stop loss is above entry, take profit is below entry
            if stop_loss or self.stop_loss_percentage:
                sl_price = stop_loss or entry_price_filled * (1 + self.stop_loss_percentage)
                await self.place_stop_loss(symbol, "short", amount, sl_price)
                
            if take_profit or self.take_profit_percentage:
                tp_price = take_profit or entry_price_filled * (1 - self.take_profit_percentage)
                await self.place_take_profit(symbol, "short", amount, tp_price)
                
            self.logger.info(f"Opened SHORT position: {symbol} @ {entry_price_filled}")
            
        return order
        
    async def close_position(self, symbol: str, reason: str = "manual") -> Dict[str, Any]:
        """Close position (long or short)"""
        position = self.positions.get(symbol)
        if not position:
            return {"error": f"No position found for {symbol}"}
            
        # Close position opposite to opening
        close_side = "sell" if position.side == PositionSide.LONG else "buy"
        close_amount = abs(position.size)
        
        order = await self.create_futures_order(
            symbol=symbol,
            side=close_side,
            amount=close_amount,
            order_type="market",
            position_side=position.side.value
        )
        
        if order and order.get('status') == 'closed':
            # Calculate realized PnL
            close_price = order.get('price')
            position.realized_pnl = self.calculate_realized_pnl(position, close_price)
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.info(f"Closed {position.side.value} position: {symbol} @ {close_price} ({reason})")
            
        return order
        
    async def create_futures_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        order_type: str = "market",
        position_side: Optional[str] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create futures order with hedge mode support"""
        order_params = params or {}
        
        # Set position side for hedge mode
        if position_side:
            order_params['positionSide'] = position_side
            
        # Handle different order types
        if order_type == "stop_loss" and price:
            order_params.update({
                'type': 'stop',
                'stopPrice': price,
            })
        
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.exchange.create_order,
                    symbol,
                    order_type,
                    side,
                    amount,
                    price,
                    order_params
                ),
                timeout=30.0
            )
        except Exception as e:
            self.logger.error(f"Futures order failed: {e}")
            raise
            
    async def place_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float) -> Dict[str, Any]:
        """Place stop loss order"""
        # For long positions, stop loss is sell order
        # For short positions, stop loss is buy order
        stop_side = "sell" if side == "long" else "buy"
        
        return await self.create_futures_order(
            symbol=symbol,
            side=stop_side,
            amount=amount,
            price=stop_price,
            order_type="stop_loss",
            params={'stopPrice': stop_price, 'type': 'stop_market'}
        )
        
    async def place_take_profit(self, symbol: str, side: str, amount: float, take_profit_price: float) -> Dict[str, Any]:
        """Place take profit order"""
        # For long positions, take profit is sell order  
        # For short positions, take profit is buy order
        tp_side = "sell" if side == "long" else "buy"
        
        return await self.create_futures_order(
            symbol=symbol,
            side=tp_side,
            amount=amount,
            price=take_profit_price,
            order_type="take_profit",
            params={'stopPrice': take_profit_price, 'type': 'limit'}
        )
        
    async def update_positions(self) -> None:
        """Update all open positions from exchange"""
        if not hasattr(self.exchange, 'fetch_positions'):
            return
            
        try:
            exchange_positions = await asyncio.wait_for(
                asyncio.to_thread(self.exchange.fetch_positions),
                timeout=30.0
            )
            
            # Clear and rebuild positions
            self.positions.clear()
            
            for pos_data in exchange_positions:
                if float(pos_data.get('contracts', 0)) == 0:
                    continue  # Skip empty positions
                    
                symbol = pos_data['symbol']
                size = float(pos_data['contracts'])
                side = PositionSide.SHORT if size < 0 else PositionSide.LONG
                
                position = FuturesPosition(
                    symbol=symbol,
                    side=side,
                    size=abs(size),
                    entry_price=float(pos_data.get('entryPrice', 0)),
                    current_price=float(pos_data.get('markPrice', 0)),
                    unrealized_pnl=float(pos_data.get('unrealizedPnl', 0)),
                    realized_pnl=float(pos_data.get('realizedPnl', 0)),
                    liquidation_price=float(pos_data.get('liquidationPrice', 0)) or None,
                )
                
                self.positions[symbol] = position
                
        except Exception as e:
            self.logger.warning(f"Failed to update positions: {e}")
            
    def calculate_realized_pnl(self, position: FuturesPosition, close_price: float) -> float:
        """Calculate realized PnL for closed position"""
        if position.side == PositionSide.LONG:
            # Long: (sell_price - buy_price) * size
            return (close_price - position.entry_price) * abs(position.size)
        else:
            # Short: (buy_price - sell_price) * size
            return (position.entry_price - close_price) * abs(position.size)
            
    async def get_available_balance(self) -> float:
        """Get available balance for futures trading"""
        try:
            balance = await asyncio.wait_for(
                asyncio.to_thread(self.exchange.fetch_balance),
                timeout=30.0
            )
            return balance.get('USDT', {}).get('free', 0)
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            return 0
            
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        
        return {
            "positions": {
                symbol: {
                    "side": pos.side.value,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "pnl_percentage": pos.pnl_percentage,
                    "leverage": pos.leverage,
                    "is_profitable": pos.is_profitable,
                    "liquidation_price": pos.liquidation_price,
                }
                for symbol, pos in self.positions.items()
            },
            "summary": {
                "total_positions": len(self.positions),
                "total_unrealized_pnl": total_pnl,
                "total_realized_pnl": total_realized,
                "profitable_positions": sum(1 for pos in self.positions.values() if pos.is_profitable),
                "losing_positions": sum(1 for pos in self.positions.values() if not pos.is_profitable),
            }
        }
        
    async def emergency_close_all(self) -> List[Dict[str, Any]]:
        """Emergency close all positions"""
        close_results = []
        
        for symbol in list(self.positions.keys()):
            try:
                result = await self.close_position(symbol, "emergency")
                close_results.append({"symbol": symbol, "result": result})
            except Exception as e:
                self.logger.error(f"Emergency close failed for {symbol}: {e}")
                close_results.append({"symbol": symbol, "error": str(e)})
                
        return close_results


class AdvancedShortSellingStrategy:
    """Enhanced short selling strategy with risk management"""
    
    def __init__(self, futures_manager: FuturesManager):
        self.futures_manager = futures_manager
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.overbought_rsi = 70
        self.oversold_rsi = 30
        self.trend_confirmation_periods = 10
        self.min_profit_target = 0.03  # 3%
        self.max_loss_allowed = 0.02   # 2%
        
    async def analyze_short_opportunity(self, symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
        """Analyze for short selling opportunities"""
        try:
            # Fetch OHLCV data
            ohlcv = await asyncio.wait_for(
                asyncio.to_thread(
                    self.futures_manager.exchange.fetch_ohlcv,
                    symbol, timeframe, None, 100
                ),
                timeout=30.0
            )
            
            if len(ohlcv) < 50:
                return {"opportunity": False, "reason": "Insufficient data"}
                
            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate indicators
            import talib
            rsi = talib.RSI(df['close'].values, 14)[-1]
            macd, macd_signal, _ = talib.MACD(df['close'].values)
            
            # Check overbought condition
            if rsi < self.overbought_rsi:
                return {"opportunity": False, "reason": f"RSI not overbought: {rsi:.1f}"}
                
            # Check trend direction (should be turning down)
            if macd[-1] > macd_signal[-1]:
                return {"opportunity": False, "reason": "MACD not bearish"}
                
            # Calculate potential entry and exit
            current_price = df['close'].iloc[-1]
            stop_loss_price = current_price * (1 + self.max_loss_allowed)
            take_profit_price = current_price * (1 - self.min_profit_target)
            
            return {
                "opportunity": True,
                "entry_price": current_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "rsi": rsi,
                "macd_signal": "bearish",
                "confidence": min(0.8, (rsi - 70) / 30),  # Higher RSI = higher confidence
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {symbol}: {e}")
            return {"opportunity": False, "reason": f"Analysis error: {e}"}
            
    async def execute_short_trade(
        self,
        symbol: str,
        amount: float,
        leverage: int = 10,
        analysis: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute short selling trade with risk management"""
        
        # Get analysis if not provided
        if not analysis:
            analysis = await self.analyze_short_opportunity(symbol)
            
        if not analysis.get('opportunity'):
            return {"error": "No short opportunity detected", "analysis": analysis}
            
        try:
            # Execute short position
            result = await self.futures_manager.open_short_position(
                symbol=symbol,
                amount=amount,
                entry_price=analysis.get('entry_price'),
                leverage=leverage,
                stop_loss=analysis.get('stop_loss'),
                take_profit=analysis.get('take_profit')
            )
            
            self.logger.info(f"Short trade executed: {symbol} @ {analysis.get('entry_price')}")
            
            return {
                "success": True,
                "symbol": symbol,
                "result": result,
                "analysis": analysis,
                "leverage": leverage,
            }
            
        except Exception as e:
            self.logger.error(f"Short trade failed: {e}")
            return {"success": False, "error": str(e), "analysis": analysis}