#!/usr/bin/env python3
"""
Mobile App Backend for Trading Bot Monitoring
Provides API endpoints for mobile application (iOS/Android)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class MobileNotification(BaseModel):
    """Mobile notification structure"""
    title: str
    message: str
    type: str  # info, warning, error, success
    timestamp: datetime
    action_url: Optional[str] = None


class MobileTradeAlert(BaseModel):
    """Trade alert for mobile notifications"""
    symbol: str
    side: str
    amount: float
    price: float
    type: str  # entry, exit, stop_loss, take_profit
    pnl: Optional[float] = None
    timestamp: datetime


class MobileDashboardData(BaseModel):
    """Complete dashboard data for mobile app"""
    portfolio_balance: Dict[str, float]
    active_positions: List[Dict[str, Any]]
    recent_trades: List[Dict[str, Any]]
    daily_pnl: float
    win_rate: float
    bot_status: str
    last_update: datetime
    notifications: List[MobileNotification]


class MobileAppBackend:
    """
    Mobile backend service providing optimized data for mobile apps
    Features: real-time updates, push notifications, optimized payloads
    """
    
    def __init__(self, bot_engine, settings):
        self.bot_engine = bot_engine
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        
        # Notification queue
        self.notification_queue: List[MobileNotification] = []
        self.max_notifications = 50
        
        # Optimize for mobile (small payloads, frequent updates)
        self.update_interval = 5  # seconds
        self.max_payload_size = 1024 * 10  # 10KB max payload
        
    async def initialize(self) -> None:
        """Initialize mobile backend"""
        self.logger.info("Initializing Mobile App Backend")
        
        # Start background tasks
        asyncio.create_task(self._background_updater())
        asyncio.create_task(self._notification_processor())
        
    async def register_connection(self, websocket: WebSocket) -> None:
        """Register WebSocket connection for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.logger.info(f"Mobile client connected. Total connections: {len(self.active_connections)}")
        
    async def disconnect_connection(self, websocket: WebSocket) -> None:
        """Disconnect WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"Mobile client disconnected. Total connections: {len(self.active_connections)}")
            
    async def broadcast_update(self, data: Dict[str, Any]) -> None:
        """Broadcast update to all connected mobile clients"""
        if not self.active_connections:
            return
            
        # Optimize payload for mobile
        mobile_data = self._optimize_for_mobile(data)
        
        # Check payload size
        payload_size = len(json.dumps(mobile_data).encode('utf-8'))
        if payload_size > self.max_payload_size:
            self.logger.warning(f"Payload too large: {payload_size} bytes (max: {self.max_payload_size})")
            mobile_data = self._compress_payload(mobile_data)
            
        # Send to all connections
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(mobile_data)
            except Exception as e:
                self.logger.warning(f"Failed to send update: {e}")
                disconnected.append(connection)
                
        # Remove disconnected connections
        for connection in disconnected:
            await self.disconnect_connection(connection)
            
    def _optimize_for_mobile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize data for mobile consumption"""
        mobile_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.bot_engine.get_status() if self.bot_engine else "offline",
        }
        
        # Add portfolio data (simplified)
        if "portfolio" in data:
            portfolio = data["portfolio"]
            mobile_data["portfolio"] = {
                "total_value": portfolio.get("total_balance", {}).get("USDT", 0),
                "daily_pnl": portfolio.get("daily_pnl", 0),
                "pnl_percentage": portfolio.get("pnl_percentage", 0),
            }
            
        # Add active positions (limited)
        if "positions" in data:
            positions = data["positions"]
            mobile_data["positions"] = positions[:5]  # Limit to 5 positions
            
        # Add recent trades (limited)
        if "trades" in data:
            trades = data["trades"]
            mobile_data["recent_trades"] = trades[:10]  # Limit to 10 trades
            
        # Add notifications
        if self.notification_queue:
            mobile_data["notifications"] = [
                {
                    "title": n.title,
                    "message": n.message,
                    "type": n.type,
                    "timestamp": n.timestamp.isoformat(),
                }
                for n in self.notification_queue[-5:]  # Show last 5 notifications
            ]
            
        return mobile_data
        
    def _compress_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress payload to fit size limits"""
        compressed = data.copy()
        
        # Remove non-essential fields
        if "positions" in compressed and len(compressed["positions"]) > 3:
            compressed["positions"] = compressed["positions"][:3]
            
        if "recent_trades" in compressed and len(compressed["recent_trades"]) > 5:
            compressed["recent_trades"] = compressed["recent_trades"][:5]
            
        # Simplify notification messages
        if "notifications" in compressed:
            for notif in compressed["notifications"]:
                notif["message"] = notif["message"][:100]  # Truncate messages
                
        return compressed
        
    async def add_notification(self, notification: MobileNotification) -> None:
        """Add notification to queue"""
        self.notification_queue.append(notification)
        
        # Limit queue size
        if len(self.notification_queue) > self.max_notifications:
            self.notification_queue = self.notification_queue[-self.max_notifications:]
            
        # Broadcast immediate update
        await self.broadcast_update({"notification": notification.dict()})
        
    async def add_trade_alert(self, alert: MobileTradeAlert) -> None:
        """Add trade alert notification"""
        side_emoji = "🟢" if alert.side == "buy" else "🔴"
        type_suffix = {
            "entry": "→",
            "exit": "←", 
            "stop_loss": "🛑",
            "take_profit": "🎯"
        }.get(alert.type, "")
        
        message = f"{side_emoji} {alert.symbol} {type_suffix} ${alert.price:.4f}"
        if alert.pnl is not None:
            message += f" PnL: ${alert.pnl:.2f}"
            
        notification = MobileNotification(
            title=f"Trade {alert.type.title()}",
            message=message,
            type="success" if alert.type == "take_profit" else "info",
            timestamp=alert.timestamp
        )
        
        await self.add_notification(notification)
        
    async def _background_updater(self) -> None:
        """Background task to send periodic updates"""
        while True:
            try:
                # Get current bot data
                update_data = await self._get_current_data()
                
                # Broadcast to mobile clients
                await self.broadcast_update(update_data)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Background updater error: {e}")
                await asyncio.sleep(10)  # Wait before retrying
                
    async def _notification_processor(self) -> None:
        """Background task to process notifications"""
        while True:
            try:
                # Process any queued notifications
                if self.notification_queue:
                    # Here you could implement push notification services
                    # like Firebase, APNS, or SMS/email
                    
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Notification processor error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _get_current_data(self) -> Dict[str, Any]:
        """Get current bot data"""
        try:
            if not self.bot_engine:
                return {"status": "offline"}
                
            # Get portfolio data
            portfolio = await self._get_portfolio_data()
            
            # Get active positions
            positions = await self._get_active_positions()
            
            # Get recent trades
            trades = await self._get_recent_trades()
            
            return {
                "portfolio": portfolio,
                "positions": positions,
                "trades": trades,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get current data: {e}")
            return {"status": "error", "error": str(e)}
            
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """Get portfolio data"""
        try:
            # Get balance from exchange
            if hasattr(self.bot_engine.exchange, 'fetch_balance'):
                balance = await asyncio.wait_for(
                    asyncio.to_thread(self.bot_engine.exchange.fetch_balance),
                    timeout=10.0
                )
                
                return {
                    "total_balance": balance.get("free", {}),
                    "used_balance": balance.get("used", {}),
                    "total": balance.get("total", {}),
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to get portfolio data: {e}")
            
        return {"total_balance": {}, "used_balance": {}, "total": {}}
        
    async def _get_active_positions(self) -> List[Dict[str, Any]]:
        """Get active positions"""
        try:
            if hasattr(self.bot_engine, 'futures_manager'):
                positions = await self.bot_engine.futures_manager.get_position_summary()
                return list(positions.get("positions", {}).values())
                
        except Exception as e:
            self.logger.warning(f"Failed to get positions: {e}")
            
        return []
        
    async def _get_recent_trades(self) -> List[Dict[str, Any]]:
        """Get recent trades"""
        try:
            if hasattr(self.bot_engine, 'repository'):
                trades = await self.bot_engine.repository.get_recent_trades(limit=10)
                return [
                    {
                        "symbol": trade.symbol,
                        "side": trade.side,
                        "amount": trade.amount,
                        "price": trade.price,
                        "timestamp": trade.timestamp.isoformat(),
                    }
                    for trade in trades
                ]
                
        except Exception as e:
            self.logger.warning(f"Failed to get trades: {e}")
            
        return []


def create_mobile_app(bot_engine, settings) -> FastAPI:
    """Create FastAPI app for mobile backend"""
    
    mobile_backend = MobileAppBackend(bot_engine, settings)
    app = FastAPI(title="Reco Trading Mobile API", version="1.0.0")
    
    # CORS for mobile app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify mobile app domains
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup():
        await mobile_backend.initialize()
        
    @app.get("/api/mobile/status")
    async def get_status():
        """Get bot status"""
        return {
            "status": mobile_backend.bot_engine.get_status() if mobile_backend.bot_engine else "offline",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "3.0.0",
        }
        
    @app.get("/api/mobile/dashboard")
    async def get_dashboard():
        """Get mobile dashboard data"""
        data = await mobile_backend._get_current_data()
        mobile_data = mobile_backend._optimize_for_mobile(data)
        return mobile_data
        
    @app.get("/api/mobile/positions")
    async def get_positions():
        """Get active positions"""
        positions = await mobile_backend._get_active_positions()
        return {"positions": positions}
        
    @app.get("/api/mobile/trades")
    async def get_trades(limit: int = 20):
        """Get recent trades"""
        trades = await mobile_backend._get_recent_trades()
        return {"trades": trades[:limit]}
        
    @app.get("/api/mobile/portfolio")
    async def get_portfolio():
        """Get portfolio summary"""
        portfolio = await mobile_backend._get_portfolio_data()
        return portfolio
        
    @app.post("/api/mobile/emergency-stop")
    async def emergency_stop():
        """Emergency stop trading"""
        try:
            if mobile_backend.bot_engine:
                await mobile_backend.bot_engine.emergency_stop()
                
            notification = MobileNotification(
                title="⚠️ Emergency Stop",
                message="Trading has been stopped via mobile app",
                type="error",
                timestamp=datetime.now(timezone.utc)
            )
            
            await mobile_backend.add_notification(notification)
            
            return {"status": "stopped"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    @app.post("/api/mobile/close-position/{symbol}")
    async def close_position(symbol: str):
        """Close specific position"""
        try:
            if hasattr(mobile_backend.bot_engine, 'futures_manager'):
                result = await mobile_backend.bot_engine.futures_manager.close_position(symbol)
                
                notification = MobileNotification(
                    title="Position Closed",
                    message=f"Position for {symbol} closed via mobile app",
                    type="success",
                    timestamp=datetime.now(timezone.utc)
                )
                
                await mobile_backend.add_notification(notification)
                
                return {"status": "closed", "result": result}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    @app.websocket("/ws/mobile")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates"""
        await mobile_backend.register_connection(websocket)
        
        try:
            # Send initial data
            initial_data = await mobile_backend._get_current_data()
            await websocket.send_json(mobile_backend._optimize_for_mobile(initial_data))
            
            # Keep connection alive
            while True:
                try:
                    # Wait for ping from client
                    await websocket.receive_text(timeout=30)
                    
                    # Send current data as ping response
                    update_data = await mobile_backend._get_current_data()
                    await websocket.send_json(mobile_backend._optimize_for_mobile(update_data))
                    
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await websocket.send_json({"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()})
                    
        except WebSocketDisconnect:
            await mobile_backend.disconnect_connection(websocket)
        except Exception as e:
            mobile_backend.logger.error(f"WebSocket error: {e}")
            await mobile_backend.disconnect_connection(websocket)
            
    return app