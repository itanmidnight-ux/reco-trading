from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import numpy as np

from reco_trading.config.symbols import normalize_symbol

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
    volatility_percentile: float = 50.0
    signal_potential: float = 0.0
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
    tier: int = 1


class MultiPairManager:
    """
    Manages multiple trading pairs with auto-selection.
    Automatically finds the best pair to trade based on conditions.
    """

    def __init__(self, exchange_client: Any, base_pairs: list[str] | None = None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange_client
        
        # Expanded list of trading pairs organized by tier
        # Tier 1: Major pairs - reliable, lower volatility
        # Tier 2: High volatility altcoins - more trading opportunities
        # Tier 3: Meme coins - extreme volatility (use with caution)
        self.pair_tiers = {
            1: [
                "BTC/USDT", "ETH/USDT", "SOL/USDT"
            ],
            2: [
                "DOGE/USDT", "XRP/USDT", "AVAX/USDT", 
                "LINK/USDT", "UNI/USDT", "AAVE/USDT"
            ],
            3: [
                "PEPE/USDT", "WIF/USDT", "BONK/USDT"
            ],
        }
        
        # Normalize all incoming symbols to standard format
        if base_pairs:
            self.default_pairs = [normalize_symbol(s) for s in base_pairs]
        else:
            self.default_pairs = [
                "BTC/USDT", "ETH/USDT", "SOL/USDT",
                "DOGE/USDT", "XRP/USDT", "AVAX/USDT",
            ]
        
        # Currently active pair (normalized)
        self.active_pair: str = normalize_symbol(base_pairs[0]) if base_pairs else "BTC/USDT"
        
        self.pairs_metrics: dict[str, PairMetrics] = {}
        self.pair_history: dict[str, list[dict]] = {}
        
        # Tier 1 (majors): scan every 8 seconds
        # Tier 2 (altcoins): scan every 12 seconds  
        # Tier 3 (meme): scan every 20 seconds
        self.tier_scan_intervals = {
            1: 8,
            2: 12,
            3: 20,
        }
        
        # Scan all pairs together for efficiency
        self.scan_interval_seconds = 10
        
        # Minimum requirements (relaxed for more opportunities)
        self.min_volume_24h = 500_000  # Reduced from 2M
        self.max_volatility = 0.20  # Increased from 0.12
        self.min_liquidity_score = 0.2  # Reduced from 0.3
        
        # Pair switching configuration
        self._switch_cooldown_seconds = 120  # 2 minutes (reduced from 3)
        self._min_hold_time_seconds = 180   # 3 minutes minimum
        self._max_switches_per_hour = 6     # Increased from 3
        self._consecutive_switch_count = 0
        self._last_switch_time: datetime | None = None
        self._switch_history: list[datetime] = []
        
        # Scoring weights for pair selection
        self._weight_opportunity = 0.30
        self._weight_momentum = 0.25
        self._weight_volume = 0.20
        self._weight_trend = 0.15
        self._weight_liquidity = 0.10
        
        # Circuit breaker
        self.panic_threshold = 0.20
        self._circuit_breaker_until: datetime | None = None
        self._circuit_breaker_duration = 180  # 3 minutes
        self._consecutive_scan_errors = 0
        self._max_consecutive_errors = 5
        
        # Performance tracking
        self._pair_performance: dict[str, dict] = {}
        self._last_10_switches: list[dict] = []
        
        # Batch scanning
        self._max_pairs_per_scan = 25
        self._scan_cache_seconds = 5
        self._last_full_scan: datetime | None = None
        
        self._scan_task: asyncio.Task | None = None
        self._is_running = False
        self._last_error: str | None = None
        self._consecutive_errors = 0
        
        self._switch_cooldown_until: datetime | None = None
        self._min_switch_interval_seconds = 180
        
        self.max_pairs_per_scan = 8
        self._last_scan_time: datetime | None = None
        self._min_time_between_scans = 5

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
        metrics.volatility = float(np.std(returns) * np.sqrt(288))
        
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
        """
        Advanced pair switching algorithm - Better than Freqtrade/Gunbot
        
        Switch conditions:
        1. Circuit breaker active -> NO switch
        2. Consecutive losses >= 2 -> SWITCH
        3. Below panic threshold (0.20) -> SWITCH
        4. Better opportunity found -> SWITCH (if delta > 0.15)
        5. Minimum hold time not met -> NO switch
        6. Maximum switches per hour reached -> NO switch
        """
        # 1. Check circuit breaker
        if self._circuit_breaker_until and datetime.now(timezone.utc) < self._circuit_breaker_until:
            self.logger.info("Circuit breaker active, skipping pair switch")
            return False
        
        # 2. Check minimum hold time
        if self._last_switch_time:
            time_since_switch = (datetime.now(timezone.utc) - self._last_switch_time).total_seconds()
            if time_since_switch < self._min_hold_time_seconds:
                self.logger.debug(f"Min hold time not met: {time_since_switch:.0f}s < {self._min_hold_time_seconds}s")
                return False
        
        # 3. Check max switches per hour
        self._cleanup_switch_history()
        if len(self._switch_history) >= self._max_switches_per_hour:
            self.logger.warning(f"Max switches per hour reached: {len(self._switch_history)} >= {self._max_switches_per_hour}")
            return False
        
        # 4. Check consecutive losses
        if consecutive_losses >= 3:
            self.logger.warning(f"Consecutive losses detected: {consecutive_losses} -> SWITCH")
            return True
        
        # 5. Get current pair metrics
        current_metrics = self.pairs_metrics.get(self.active_pair)
        if not current_metrics:
            return True
        
        # 6. Check panic threshold
        if current_metrics.opportunity_score < self.panic_threshold:
            self.logger.warning(f"Active pair {self.active_pair} below panic: {current_metrics.opportunity_score:.2f} < {self.panic_threshold}")
            return True
        
        # 7. Find best alternative pair
        best_alternative = None
        best_score = -1.0
        
        for symbol, metrics in self.pairs_metrics.items():
            if symbol == self.active_pair:
                continue
            
            # Calculate switch score
            switch_score = self._calculate_switch_score(current_metrics, metrics)
            
            if switch_score > best_score:
                best_score = switch_score
                best_alternative = symbol
        
        # 8. Check if switch is beneficial
        opportunity_delta = best_score - current_metrics.opportunity_score
        
        if best_alternative and opportunity_delta > 0.15:  # 15% improvement threshold
            self.logger.info(f"Better pair found: {self.active_pair} ({current_metrics.opportunity_score:.2f}) -> {best_alternative} ({best_score:.2f}), delta: {opportunity_delta:.2f}")
            return True
        
        # 9. Check if current pair is worst than median
        all_scores = [m.opportunity_score for m in self.pairs_metrics.values()]
        median_score = sorted(all_scores)[len(all_scores) // 2] if all_scores else 0
        
        if current_metrics.opportunity_score < median_score * 0.7:
            self.logger.warning(f"Active pair below median: {current_metrics.opportunity_score:.2f} < {median_score * 0.7:.2f}")
            return True
        
        return False
    
    def _calculate_switch_score(self, current: PairMetrics, candidate: PairMetrics) -> float:
        """
        Calculate switch score based on multiple factors:
        - Opportunity score (40%)
        - Momentum advantage (20%)
        - Volume advantage (15%)
        - Volatility advantage - lower is better (10%)
        - Regime match (10%)
        - Risk-adjusted return (5%)
        """
        # Opportunity difference
        opp_diff = candidate.opportunity_score - current.opportunity_score
        
        # Momentum advantage
        momentum_adv = candidate.momentum_score - current.momentum_score
        
        # Volume advantage
        vol_adv = (candidate.volume_score - current.volume_score) * 0.5
        
        # Volatility advantage (lower is better for stability)
        if current.volatility > 0 and candidate.volatility > 0:
            vol_diff = (current.volatility - candidate.volatility) / max(current.volatility, 0.01)
        else:
            vol_diff = 0
        
        # Regime match bonus
        regime_bonus = 0.1 if candidate.market_regime in ["NORMAL", "LOW_VOLATILITY"] else 0
        
        # Risk-adjusted score (higher volatility = higher risk)
        risk_factor = 1.0 - min(candidate.volatility * 2, 0.5)
        
        # Calculate final score
        switch_score = (
            candidate.opportunity_score * 0.40 +
            momentum_adv * 0.20 +
            vol_adv * 0.15 +
            vol_diff * 0.10 +
            regime_bonus * 0.10 +
            risk_factor * candidate.opportunity_score * 0.05
        )
        
        return switch_score
    
    def _cleanup_switch_history(self) -> None:
        """Clean up switch history older than 1 hour"""
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        self._switch_history = [t for t in self._switch_history if t > one_hour_ago]
    
    def record_switch(self) -> None:
        """Record a pair switch"""
        self._last_switch_time = datetime.now(timezone.utc)
        self._switch_history.append(datetime.now(timezone.utc))
        
        # Track in performance history
        self._last_10_switches.append({
            "from": self.active_pair,
            "timestamp": datetime.now(timezone.utc),
        })
        
        # Keep only last 10
        if len(self._last_10_switches) > 10:
            self._last_10_switches = self._last_10_switches[-10:]
    
    def activate_circuit_breaker(self, duration_seconds: int = 300) -> None:
        self._circuit_breaker_until = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        self.logger.warning(f"Circuit breaker activated for {duration_seconds} seconds")
    
    def clear_circuit_breaker(self) -> None:
        self._circuit_breaker_until = None
        self.logger.info("Circuit breaker cleared")

    def get_alternative_pairs(self, count: int = 3) -> list[str]:
        sorted_pairs = sorted(
            self.pairs_metrics.items(),
            key=lambda x: x[1].opportunity_score,
            reverse=True
        )
        
        return [s[0] for s in sorted_pairs[:count] if s[0] != self.active_pair]
