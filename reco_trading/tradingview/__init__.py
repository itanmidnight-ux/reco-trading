"""TradingView Webhook Integration for Reco-Trading
Receives and processes TradingView alerts via webhooks
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from aiohttp import web
import re

logger = logging.getLogger(__name__)


class WebhookSource(Enum):
    """Webhook source type"""
    TRADINGVIEW = "tradingview"
    CUSTOM = "custom"
    API = "api"
    ALERT = "alert"


class WebhookSignalType(Enum):
    """Type of signal from webhook"""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_ALL = "close_all"
    CANCEL = "cancel"
    MODIFY = "modify"


@dataclass
class WebhookSignal:
    """Signal received from webhook"""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: WebhookSource = WebhookSource.TRADINGVIEW
    
    # Signal data
    signal_type: WebhookSignalType = WebhookSignalType.BUY
    symbol: str = ""
    price: float = 0.0
    
    # Trading parameters
    quantity: Optional[float] = None
    order_type: str = "market"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Meta
    strategy_name: str = ""
    timeframe: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: dict = field(default_factory=dict)
    
    # Validation
    is_valid: bool = False
    validation_error: Optional[str] = None
    
    # Processing
    processed: bool = False
    processed_at: Optional[datetime] = None


@dataclass
class WebhookConfig:
    """Configuration for webhook endpoint"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Endpoint config
    endpoint_path: str = "/webhook"
    secret_key: str = ""
    allow_any_source: bool = True
    allowed_sources: list[str] = field(default_factory=list)
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_signals_per_minute: int = 30
    
    # Filtering
    allowed_symbols: list[str] = field(default_factory=list)
    blocked_symbols: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    
    # Processing
    auto_validate: bool = True
    default_quantity: float = 100.0
    default_order_type: str = "market"
    
    # Status
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class TradingViewWebhookHandler:
    """Handles TradingView webhook requests"""
    
    # Common TradingView webhook patterns
    TRADINGVIEW_PATTERNS = {
        "buy": r"(buy|long|entry|entrada|compra)",
        "sell": r"(sell|short|exit|salida|venta)",
        "close": r"(close|cierre|close_all)",
    }
    
    # Symbol patterns
    SYMBOL_PATTERN = r"([A-Z]{2,10})(USDT|TUSD|USD|BTC|ETH|BNB)?"
    
    def __init__(self, config: Optional[WebhookConfig] = None):
        self.config = config or WebhookConfig()
        self.webhook_history: list[WebhookSignal] = []
        self.rate_limiter: dict[str, list[float]] = {}
        self._signal_callbacks: list[callable] = []
        
    def register_callback(self, callback: callable) -> None:
        """Register a callback for processed signals"""
        self._signal_callbacks.append(callback)
        
    async def process_webhook(self, payload: bytes, 
                              headers: dict = None) -> WebhookSignal:
        """Process incoming webhook payload"""
        signal = WebhookSignal(raw_data={})
        
        try:
            # Parse payload
            data = self._parse_payload(payload)
            signal.raw_data = data
            
            # Validate signature if configured
            if self.config.secret_key and headers:
                if not self._validate_signature(payload, headers, self.config.secret_key):
                    signal.validation_error = "Invalid signature"
                    return signal
            
            # Parse signal from payload
            signal = self._parse_signal(data, signal)
            
            # Validate signal
            if self.config.auto_validate:
                validation = await self._validate_signal(signal)
                if not validation["valid"]:
                    signal.validation_error = validation["error"]
                    return signal
                    
            signal.is_valid = True
            
            # Check rate limit
            if not await self._check_rate_limit():
                signal.validation_error = "Rate limit exceeded"
                return signal
                
            # Process callbacks
            for callback in self._signal_callbacks:
                try:
                    await callback(signal)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
            signal.processed = True
            signal.processed_at = datetime.now()
            
            logger.info(f"Webhook processed: {signal.signal_type.value} {signal.symbol}")
            
        except Exception as e:
            signal.validation_error = f"Processing error: {str(e)}"
            logger.error(f"Webhook processing error: {e}")
            
        self.webhook_history.append(signal)
        return signal
    
    def _parse_payload(self, payload: bytes) -> dict:
        """Parse webhook payload"""
        try:
            # Try JSON first
            return json.loads(payload)
        except json.JSONDecodeError:
            # Try URL-encoded
            result = {}
            for item in payload.decode().split("&"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    result[key] = value
            return result
    
    def _validate_signature(self, payload: bytes, headers: dict, secret: str) -> bool:
        """Validate webhook signature"""
        # TradingView sends signature in different headers
        signature = headers.get("X-Treezor-Signature") or \
                   headers.get("X-Webhook-Signature") or \
                   headers.get("Signature", "")
        
        if not signature:
            return self.config.allow_any_source
            
        # Create expected signature
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    def _parse_signal(self, data: dict, signal: WebhookSignal) -> WebhookSignal:
        """Parse TradingView alert data into signal"""
        
        # Extract symbol - common TradingView field names
        symbol = data.get("symbol") or \
                data.get("ticker") or \
                data.get("SYMBOL") or \
                data.get("pair") or \
                data.get("coin") or ""
        
        # Clean symbol
        signal.symbol = self._normalize_symbol(symbol)
        
        # Extract action
        action = data.get("action") or \
                data.get("signal") or \
                data.get("ORDER_ACTION") or \
                data.get("direction") or \
                data.get("type") or "buy"
        
        signal.signal_type = self._parse_signal_type(action)
        
        # Extract price
        price = data.get("price") or \
               data.get("close") or \
               data.get("last") or \
               data.get("ENTRY_PRICE") or \
               data.get("0")
        
        try:
            signal.price = float(price) if price else 0.0
        except (ValueError, TypeError):
            signal.price = 0.0
            
        # Extract quantity
        qty = data.get("quantity") or \
             data.get("qty") or \
             data.get("amount") or \
             data.get("size") or \
             data.get("CONTRACT_SIZE")
        
        if qty:
            try:
                signal.quantity = float(qty)
            except (ValueError, TypeError):
                signal.quantity = self.config.default_quantity
        else:
            signal.quantity = self.config.default_quantity
            
        # Extract stop loss
        sl = data.get("stop_loss") or \
            data.get("sl") or \
            data.get("STOP_LOSS") or \
            data.get("stop")
        
        if sl:
            try:
                signal.stop_loss = float(sl)
            except (ValueError, TypeError):
                pass
                
        # Extract take profit
        tp = data.get("take_profit") or \
            data.get("tp") or \
            data.get("TAKE_PROFIT") or \
            data.get("target")
        
        if tp:
            try:
                signal.take_profit = float(tp)
            except (ValueError, TypeError):
                pass
                
        # Extract strategy name
        signal.strategy_name = data.get("strategy") or \
                               data.get("strategy_name") or \
                               data.get("STRATEGY") or \
                               "TradingView"
                               
        # Extract timeframe
        signal.timeframe = data.get("timeframe") or \
                          data.get("interval") or \
                          data.get("TF") or \
                          "Unknown"
                          
        # Extract strategy order ID
        if "strategy_order_id" in data:
            signal.raw_data["strategy_order_id"] = data["strategy_order_id"]
            
        return signal
    
    def _parse_signal_type(self, action: str) -> WebhookSignalType:
        """Parse action string into signal type"""
        action_lower = action.lower()
        
        for signal_type, pattern in self.TRADINGVIEW_PATTERNS.items():
            if re.search(pattern, action_lower):
                return WebhookSignalType(signal_type)
                
        # Check for explicit types
        if "close" in action_lower or "exit" in action_lower:
            return WebhookSignalType.CLOSE
        elif "cancel" in action_lower:
            return WebhookSignalType.CANCEL
            
        return WebhookSignalType.BUY
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format"""
        if not symbol:
            return ""
            
        # Remove common prefixes/suffixes
        symbol = symbol.replace("BINANCE:", "").replace("TV:", "")
        
        # Extract base and quote
        match = re.search(self.SYMBOL_PATTERN, symbol)
        if match:
            base = match.group(1)
            quote = match.group(2) or "USDT"
            return f"{base}{quote}"
            
        return symbol.upper()
    
    async def _validate_signal(self, signal: WebhookSignal) -> dict:
        """Validate signal"""
        
        # Check symbol
        if not signal.symbol:
            return {"valid": False, "error": "No symbol provided"}
            
        # Check if symbol is allowed
        if self.config.allowed_symbols:
            if signal.symbol not in self.config.allowed_symbols:
                return {"valid": False, "error": "Symbol not allowed"}
                
        # Check if symbol is blocked
        if signal.symbol in self.config.blocked_symbols:
            return {"valid": False, "error": "Symbol is blocked"}
            
        # Check price
        if signal.price <= 0:
            return {"valid": False, "error": "Invalid price"}
            
        # Check quantity
        if signal.quantity and signal.quantity <= 0:
            return {"valid": False, "error": "Invalid quantity"}
            
        return {"valid": True, "error": None}
    
    async def _check_rate_limit(self) -> bool:
        """Check rate limit"""
        now = time.time()
        client_id = "default"  # In production, extract from request
        
        if client_id not in self.rate_limiter:
            self.rate_limiter[client_id] = []
            
        # Clean old entries
        self.rate_limiter[client_id] = [
            t for t in self.rate_limiter[client_id]
            if now - t < 60
        ]
        
        if len(self.rate_limiter[client_id]) >= self.config.max_signals_per_minute:
            return False
            
        self.rate_limiter[client_id].append(now)
        return True
    
    def get_active_signals(self, since: Optional[datetime] = None) -> list[WebhookSignal]:
        """Get recently processed signals"""
        if since:
            return [s for s in self.webhook_history 
                    if s.timestamp >= since and s.is_valid]
        return [s for s in self.webhook_history if s.is_valid]
    
    def get_signal_stats(self) -> dict:
        """Get webhook processing statistics"""
        total = len(self.webhook_history)
        valid = len([s for s in self.webhook_history if s.is_valid])
        processed = len([s for s in self.webhook_history if s.processed])
        
        signal_types = {}
        for s in self.webhook_history:
            key = s.signal_type.value
            signal_types[key] = signal_types.get(key, 0) + 1
            
        return {
            "total_webhooks": total,
            "valid_signals": valid,
            "processed_signals": processed,
            "by_type": signal_types,
        }


class WebhookServer:
    """HTTP server for receiving webhooks"""
    
    def __init__(self, handler: TradingViewWebhookHandler, port: int = 8080):
        self.handler = handler
        self.port = port
        self.app = web.Application()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup webhook routes"""
        self.app.router.add_post(
            self.handler.config.endpoint_path,
            self._handle_webhook
        )
        self.app.router.add_get("/health", self._health_check)
        
    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming webhook"""
        try:
            payload = await request.read()
            headers = dict(request.headers)
            
            signal = await self.handler.process_webhook(payload, headers)
            
            if signal.is_valid:
                return web.json_response({
                    "status": "success",
                    "signal_id": signal.signal_id,
                    "type": signal.signal_type.value,
                    "symbol": signal.symbol,
                })
            else:
                return web.json_response({
                    "status": "error",
                    "error": signal.validation_error,
                }, status=400)
                
        except Exception as e:
            logger.error(f"Webhook handle error: {e}")
            return web.json_response({
                "status": "error",
                "error": str(e),
            }, status=500)
    
    async def _health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "stats": self.handler.get_signal_stats(),
        })
    
    async def start(self):
        """Start the webhook server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()
        logger.info(f"Webhook server started on port {self.port}")
        
    async def stop(self):
        """Stop the webhook server"""
        pass


# Global instances
_webhook_handler: Optional[TradingViewWebhookHandler] = None
_webhook_server: Optional[WebhookServer] = None


def get_webhook_handler(config: Optional[WebhookConfig] = None) -> TradingViewWebhookHandler:
    """Get or create webhook handler"""
    global _webhook_handler
    if _webhook_handler is None:
        _webhook_handler = TradingViewWebhookHandler(config)
    return _webhook_handler


async def start_webhook_server(port: int = 8080) -> WebhookServer:
    """Start webhook server"""
    global _webhook_server
    handler = get_webhook_handler()
    _webhook_server = WebhookServer(handler, port)
    await _webhook_server.start()
    return _webhook_server


__all__ = [
    "WebhookSource",
    "WebhookSignalType",
    "WebhookSignal",
    "WebhookConfig",
    "TradingViewWebhookHandler",
    "WebhookServer",
    "get_webhook_handler",
    "start_webhook_server",
]