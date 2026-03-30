from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairMetrics:
    symbol: str
    price: float = 0.0
    volume_24h: float = 0.0
    volatility: float = 0.0
    trend_score: float = 0.0
    volume_score: float = 0.0
    liquidity_score: float = 0.0
    opportunity_score: float = 0.0
    momentum_score: float = 0.0
    market_regime: str = "UNKNOWN"
    trend_direction: str = "NEUTRAL"
    rsi: float = 50.0
    adx: float = 0.0
    volume_ratio: float = 1.0
    spread: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TradingPair:
    symbol: str
    base_asset: str
    quote_asset: str
    min_quantity: float = 0.0
    min_notional: float = 0.0
    step_size: float = 0.0
    tick_size: float = 0.0
    is_tradeable: bool = True


class MultiPairManager:
    """
    Manages multiple trading pairs with auto-selection.
    Automatically finds the best pair to trade based on conditions.
    """

    def __init__(self, exchange_client: Any, base_pairs: list[str] | None = None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_client
        
        self.default_pairs = base_pairs or [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", 
            "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
            "DOT/USDT", "MATIC/USDT", "LINK/USDT", "ATOM/USDT"
        ]
        
        self.pairs_metrics: dict[str, PairMetrics] = {}
        self.active_pair: str = "BTC/USDT"
        self.pair_history: dict[str, list[dict]] = {}
        
        self.scan_interval_seconds = 30
        self.min_volume_24h = 1_000_000
        self.max_volatility = 0.15
        self.min_liquidity_score = 0.3
        
        self._scan_task: asyncio.Task | None = None
        self._is_running = False
        self._last_error: str | None = None
        self._consecutive_errors: int = 0
        self._max_consecutive_errors = 5
        
        self._switch_cooldown_until: datetime | None = None
        self._min_switch_interval_seconds = 300
        
        self._weight_opportunity = 0.30
        self._weight_momentum = 0.25
        self._weight_trend = 0.20
        self._weight_volume = 0.15
        self._weight_liquidity = 0.10

    async def start(self) -> None:
        self._is_running = True
        await self._initialize_pairs()
        
        self.logger.info(f"Multi-pair manager initialized with {len(self.default_pairs)} pairs")
        
        await self._scan_all_pairs()
        await self._select_best_pair()
        
        best = self.pairs_metrics.get(self.active_pair)
        self.logger.info(f"Initial best pair: {self.active_pair} (opportunity: {best.opportunity_score if best else 0:.3f})")
        
        self._scan_task = asyncio.create_task(self._scan_loop())
        self.logger.info(f"Multi-pair manager started with {len(self.default_pairs)} pairs")

    async def stop(self) -> None:
        self._is_running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Multi-pair manager stopped")

    async def _initialize_pairs(self) -> None:
        try:
            for symbol in self.default_pairs:
                self.pairs_metrics[symbol] = PairMetrics(symbol=symbol)
                self.pair_history[symbol] = []
        except Exception as exc:
            self.logger.error(f"Failed to initialize pairs: {exc}")

    async def _scan_loop(self) -> None:
        while self._is_running:
            try:
                await self._scan_all_pairs()
                await self._select_best_pair()
                
                sorted_pairs = sorted(self.pairs_metrics.items(), key=lambda x: x[1].opportunity_score, reverse=True)
                top_3 = sorted_pairs[:3]
                top_info = ", ".join([f"{s}:{m.opportunity_score:.2f}" for s, m in top_3])
                self.logger.info(f"[SCAN] Pairs: {top_info} | Active: {self.active_pair}")
                
                await asyncio.sleep(self.scan_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"Scan loop error: {exc}")
                await asyncio.sleep(10)

    async def _scan_all_pairs(self) -> None:
        try:
            ccxt_exchange = getattr(self.exchange, "exchange", self.exchange)
            
            async def fetch_all_tickers():
                return await asyncio.to_thread(ccxt_exchange.fetch_tickers)
            
            tickers = await fetch_all_tickers()
            
            for symbol in self.default_pairs:
                if symbol not in tickers:
                    continue
                    
                ticker = tickers[symbol]
                metrics = self.pairs_metrics.get(symbol, PairMetrics(symbol=symbol))
                
                metrics.price = float(ticker.get("last", 0))
                metrics.volume_24h = float(ticker.get("quoteVolume", 0))
                
                if metrics.symbol in self.pair_history:
                    self.pair_history[symbol].append({
                        "price": metrics.price,
                        "volume": metrics.volume_24h,
                        "timestamp": datetime.now(timezone.utc)
                    })
                    if len(self.pair_history[symbol]) > 100:
                        self.pair_history[symbol] = self.pair_history[symbol][-100:]
                
                await self._analyze_pair(metrics)
                self.pairs_metrics[symbol] = metrics
                
        except Exception as exc:
            self.logger.error(f"Failed to scan pairs: {exc}")

    async def _analyze_pair(self, metrics: PairMetrics) -> None:
        history = self.pair_history.get(metrics.symbol, [])
        if len(history) < 2:
            metrics.opportunity_score = 0.5
            metrics.momentum_score = 0.5
            return
        
        prices = [h["price"] for h in history[-20:]]
        if len(prices) < 5:
            return
        
        returns = np.diff(np.log(prices))
        metrics.volatility = float(np.std(returns) * np.sqrt(1440))
        
        price_change_1h = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        price_change_4h = (prices[-1] - prices[-24]) / prices[-24] if len(prices) >= 24 else 0
        
        if price_change_1h > 0.02:
            metrics.trend_direction = "BULLISH"
            metrics.trend_score = min(1.0, abs(price_change_1h) * 10)
        elif price_change_1h < -0.02:
            metrics.trend_direction = "BEARISH"
            metrics.trend_score = min(1.0, abs(price_change_1h) * 10)
        else:
            metrics.trend_direction = "NEUTRAL"
            metrics.trend_score = 0.3
        
        if metrics.volume_24h > 50_000_000:
            metrics.volume_score = 1.0
        elif metrics.volume_24h > 10_000_000:
            metrics.volume_score = 0.7
        elif metrics.volume_24h > 1_000_000:
            metrics.volume_score = 0.4
        else:
            metrics.volume_score = 0.1
        
        avg_volume = np.mean([h["volume"] for h in history[-24:]]) if len(history) >= 24 else metrics.volume_24h
        metrics.volume_ratio = metrics.volume_24h / max(avg_volume, 1)
        
        if metrics.volatility > 0.08:
            metrics.liquidity_score = 1.0
        elif metrics.volatility > 0.04:
            metrics.liquidity_score = 0.7
        else:
            metrics.liquidity_score = 0.4
        
        momentum = price_change_1h * 0.6 + price_change_4h * 0.4
        metrics.momentum_score = max(0, min(1.0, (momentum + 0.05) * 10))
        
        opportunity = (1.0 - min(metrics.volatility / self.max_volatility, 1.0)) * 0.4
        opportunity += metrics.momentum_score * 0.3
        opportunity += metrics.trend_score * 0.2
        opportunity += metrics.volume_score * 0.1
        metrics.opportunity_score = min(1.0, opportunity)
        
        if metrics.volatility < 0.02:
            metrics.market_regime = "LOW_VOLATILITY"
        elif metrics.volatility < 0.06:
            metrics.market_regime = "NORMAL"
        elif metrics.volatility < 0.10:
            metrics.market_regime = "HIGH_VOLATILITY"
        else:
            metrics.market_regime = "EXTREME"
        
        metrics.last_updated = datetime.now(timezone.utc)

    async def _select_best_pair(self) -> None:
        best_pair = None
        best_score = -1.0
        
        active_metrics = self.pairs_metrics.get(self.active_pair)
        
        for symbol, metrics in self.pairs_metrics.items():
            if metrics.volume_24h < self.min_volume_24h:
                continue
            if metrics.volatility > self.max_volatility and metrics.volatility > 0.20:
                continue
            if metrics.liquidity_score < self.min_liquidity_score:
                continue
            
            total_score = (
                metrics.opportunity_score * self._weight_opportunity +
                metrics.momentum_score * self._weight_momentum +
                metrics.trend_score * self._weight_trend +
                metrics.volume_score * self._weight_volume +
                metrics.liquidity_score * self._weight_liquidity
            )
            
            if metrics.market_regime == "LOW_VOLATILITY":
                total_score *= 0.7
            elif metrics.market_regime == "NORMAL":
                total_score *= 1.0
            elif metrics.market_regime == "HIGH_VOLATILITY":
                total_score *= 1.1
            
            if total_score > best_score:
                best_score = total_score
                best_pair = symbol
        
        active_total_score = 0.0
        if active_metrics:
            active_total_score = (
                active_metrics.opportunity_score * self._weight_opportunity +
                active_metrics.momentum_score * self._weight_momentum +
                active_metrics.trend_score * self._weight_trend +
                active_metrics.volume_score * self._weight_volume +
                active_metrics.liquidity_score * self._weight_liquidity
            )
            if active_metrics.market_regime == "LOW_VOLATILITY":
                active_total_score *= 0.7
            elif active_metrics.market_regime == "NORMAL":
                active_total_score *= 1.0
            elif active_metrics.market_regime == "HIGH_VOLATILITY":
                active_total_score *= 1.1
        
        should_switch = False
        switch_reason = ""
        
        if best_pair and best_pair != self.active_pair:
            score_diff = best_score - active_total_score
            self.logger.info(f"=== SCORE COMPARISON: best={best_pair}({best_score:.3f}) vs active={self.active_pair}({active_total_score:.3f}), diff={score_diff:.3f}")
            if score_diff > 0.08:
                should_switch = True
                switch_reason = f"better_score ({best_score:.2f} vs {active_total_score:.2f})"
        
        if active_metrics and active_metrics.market_regime == "LOW_VOLATILITY":
            for symbol, metrics in self.pairs_metrics.items():
                if symbol != self.active_pair and metrics.market_regime in ["NORMAL", "HIGH_VOLATILITY"]:
                    if metrics.opportunity_score > active_metrics.opportunity_score:
                        should_switch = True
                        switch_reason = f"low_volatility ({active_metrics.market_regime} vs {metrics.market_regime})"
                        best_pair = symbol
                        best_score = metrics.opportunity_score
                        break
        
        if should_switch and best_pair:
            old_pair = self.active_pair
            self.active_pair = best_pair
            self.logger.warning(f">>> AUTO-SWITCH PAIR: {old_pair} -> {best_pair} reason: {switch_reason}")

    def get_best_pair(self) -> str:
        return self.active_pair

    def get_pair_metrics(self, symbol: str | None = None) -> PairMetrics | None:
        return self.pairs_metrics.get(symbol or self.active_pair)

    def get_all_metrics(self) -> list[dict[str, Any]]:
        sorted_pairs = sorted(
            self.pairs_metrics.items(),
            key=lambda x: x[1].opportunity_score,
            reverse=True
        )
        
        return [
            {
                "symbol": m.symbol,
                "price": m.price,
                "volume_24h": m.volume_24h,
                "volatility": m.volatility,
                "trend_direction": m.trend_direction,
                "opportunity_score": m.opportunity_score,
                "momentum_score": m.momentum_score,
                "market_regime": m.market_regime,
                "is_active": m.symbol == self.active_pair,
            }
            for _, m in sorted_pairs
        ]

    def should_switch_pair(self, consecutive_losses: int) -> bool:
        if consecutive_losses >= 3:
            return True
        
        metrics = self.pairs_metrics.get(self.active_pair)
        if metrics and metrics.opportunity_score < 0.3:
            return True
            
        return False

    def get_alternative_pairs(self, count: int = 3) -> list[str]:
        sorted_pairs = sorted(
            self.pairs_metrics.items(),
            key=lambda x: x[1].opportunity_score,
            reverse=True
        )
        
        return [s[0] for s in sorted_pairs[:count] if s[0] != self.active_pair]
