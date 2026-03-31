from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class WhaleData:
    address: str
    balance: float
    change_24h: float
    transaction_count: int
    last_active: datetime
    classification: str = "unknown"
    metadata: dict = field(default_factory=dict)


@dataclass
class OnChainMetrics:
    whale_activity: float
    exchange_flow: float
    network_velocity: float
    stablecoin_flow: float
    gas_price: float
    timestamp: datetime = field(default_factory=datetime.now)


class WhaleTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._whale_addresses: dict[str, WhaleData] = {}
        self._activity_history: list[dict] = []
        
        self._initialize_known_whales()
        
        self.logger.info("WhaleTracker initialized")

    def _initialize_known_whales(self) -> None:
        self._whale_addresses = {
            "0x8894E0a0c962CB723c1976a18E3632B2C72C9E9B": WhaleData(
                address="0x8894E0a0c962CB723c1976a18E3632B2C72C9E9B",
                balance=0,
                change_24h=0,
                transaction_count=0,
                last_active=datetime.now(),
                classification="exchange"
            ),
            "0x28fDE8f8d7d4EAf1EbB4f5C8e1aC1b6F8d9C7e3A": WhaleData(
                address="0x28fDE8f8d7d4EAf1EbB4f5C8e1aC1b6F8d9C7e3A",
                balance=0,
                change_24h=0,
                transaction_count=0,
                last_active=datetime.now(),
                classification="major_holder"
            ),
            "0x47AC0Fb4D2D9C4f9D1a5c4E5a8b3C6d9E0f1A2B3": WhaleData(
                address="0x47AC0Fb4D2D9C4f9D1a5c4E5a8b3C6d9E0f1A2B3",
                balance=0,
                change_24h=0,
                transaction_count=0,
                last_active=datetime.now(),
                classification="defi_protocol"
            )
        }

    async def fetch_whale_data(self, symbol: str = "ETH") -> list[WhaleData]:
        await asyncio.sleep(0.1)
        
        for address, whale in self._whale_addresses.items():
            whale.balance = 10000 + (hash(address) % 50000)
            whale.change_24h = (hash(address + str(int(time.time()))) % 200 - 100) / 100.0
            whale.transaction_count = abs(hash(address)) % 50 + 10
            whale.last_active = datetime.now()
        
        return list(self._whale_addresses.values())

    def detect_accumulation(self, threshold_pct: float = 10.0) -> list[WhaleData]:
        accumulating = []
        
        for whale in self._whale_addresses.values():
            if whale.change_24h > threshold_pct:
                whale.classification = "accumulating"
                accumulating.append(whale)
        
        return accumulating

    def detect_distribution(self, threshold_pct: float = -10.0) -> list[WhaleData]:
        distributing = []
        
        for whale in self._whale_addresses.values():
            if whale.change_24h < threshold_pct:
                whale.classification = "distributing"
                distributing.append(whale)
        
        return distributing

    def get_whale_activity_score(self) -> float:
        if not self._whale_addresses:
            return 0.0
        
        avg_change = sum(w.change_24h for w in self._whale_addresses.values()) / len(self._whale_addresses)
        
        activity = max(-1, min(1, avg_change / 20))
        
        return activity

    def add_activity_log(self, activity: dict) -> None:
        activity["timestamp"] = datetime.now()
        self._activity_history.append(activity)
        
        if len(self._activity_history) > 100:
            self._activity_history = self._activity_history[-100:]

    def get_tracker_stats(self) -> dict:
        return {
            "tracked_whales": len(self._whale_addresses),
            "activity_logs": len(self._activity_history),
            "activity_score": self.get_whale_activity_score(),
            "accumulating_count": len(self.detect_accumulation()),
            "distributing_count": len(self.detect_distribution())
        }


class ExchangeFlowAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._inflow_history: list[float] = []
        self._outflow_history: list[float] = []
        
        self._max_history = 100

    async def fetch_flow_data(self, symbol: str = "ETH") -> dict:
        import random
        await asyncio.sleep(0.05)
        
        inflow = random.uniform(10000, 50000)
        outflow = random.uniform(10000, 50000)
        
        self._inflow_history.append(inflow)
        self._outflow_history.append(outflow)
        
        if len(self._inflow_history) > self._max_history:
            self._inflow_history = self._inflow_history[-self._max_history:]
            self._outflow_history = self._outflow_history[-self._max_history:]
        
        return {
            "inflow_24h": inflow,
            "outflow_24h": outflow,
            "net_flow": inflow - outflow,
            "inflow_ratio": inflow / (inflow + outflow)
        }

    def get_flow_score(self) -> float:
        if not self._inflow_history or not self._outflow_history:
            return 0.0
        
        avg_inflow = sum(self._inflow_history[-10:]) / 10
        avg_outflow = sum(self._outflow_history[-10:]) / 10
        
        if avg_outflow == 0:
            return 0.5
        
        ratio = avg_inflow / avg_outflow
        
        return max(0, min(1, ratio))

    def detect_anomaly(self) -> bool:
        if len(self._inflow_history) < 10:
            return False
        
        recent_in = self._inflow_history[-5:]
        older_in = self._inflow_history[-10:-5]
        
        avg_recent = sum(recent_in) / len(recent_in)
        avg_older = sum(older_in) / len(older_in)
        
        change_pct = abs(avg_recent - avg_older) / avg_older if avg_older > 0 else 0
        
        return change_pct > 0.5


class SmartMoneyDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._whale_tracker = WhaleTracker()
        self._flow_analyzer = ExchangeFlowAnalyzer()
        
        self._smart_money_score = 0.5

    async def analyze(self, symbol: str = "ETH") -> dict:
        whales = await self._whale_tracker.fetch_whale_data(symbol)
        
        flow = await self._flow_analyzer.fetch_flow_data(symbol)
        
        whale_activity = self._whale_tracker.get_whale_activity_score()
        flow_score = self._flow_analyzer.get_flow_score()
        
        smart_money = (whale_activity * 0.6) + (flow_score * 0.4)
        
        self._smart_money_score = smart_money
        
        accumulating = self._whale_tracker.detect_accumulation()
        distributing = self._whale_tracker.detect_distribution()
        
        return {
            "smart_money_score": smart_money,
            "whale_activity": whale_activity,
            "exchange_flow": flow_score,
            "accumulating_whales": len(accumulating),
            "distributing_whales": len(distributing),
            "net_flow": flow.get("net_flow", 0),
            "signal": "BULLISH" if smart_money > 0.6 else "BEARISH" if smart_money < 0.4 else "NEUTRAL"
        }

    def get_signal(self) -> str:
        if self._smart_money_score > 0.6:
            return "BUY"
        elif self._smart_money_score < 0.4:
            return "SELL"
        return "HOLD"


class OnChainAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._whale_tracker = WhaleTracker()
        self._flow_analyzer = ExchangeFlowAnalyzer()
        self._smart_money = SmartMoneyDetector()
        
        self._metrics_history: list[OnChainMetrics] = []
        
        self.logger.info("OnChainAnalyzer initialized")

    async def analyze(self, symbol: str = "ETH") -> OnChainMetrics:
        whale_activity = self._whale_tracker.get_whale_activity_score()
        
        flow = await self._flow_analyzer.fetch_flow_data(symbol)
        flow_score = self._flow_analyzer.get_flow_score()
        
        smart_money_result = await self._smart_money.analyze(symbol)
        
        import random
        network_velocity = random.uniform(0.3, 0.7)
        stablecoin_flow = (flow.get("inflow_24h", 0) - flow.get("outflow_24h", 0)) / 10000
        gas_price = random.uniform(10, 100)
        
        metrics = OnChainMetrics(
            whale_activity=whale_activity,
            exchange_flow=flow_score,
            network_velocity=network_velocity,
            stablecoin_flow=stablecoin_flow,
            gas_price=gas_price
        )
        
        self._metrics_history.append(metrics)
        
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]
        
        return metrics

    def get_composite_signal(self) -> dict:
        if not self._metrics_history:
            return {"signal": "NEUTRAL", "confidence": 0}
        
        recent = self._metrics_history[-10:]
        
        avg_whale = sum(m.whale_activity for m in recent) / len(recent)
        avg_flow = sum(m.exchange_flow for m in recent) / len(recent)
        
        composite = (avg_whale + avg_flow) / 2
        
        if composite > 0.6:
            signal = "BULLISH"
            confidence = min(1.0, (composite - 0.6) / 0.4)
        elif composite < 0.4:
            signal = "BEARISH"
            confidence = min(1.0, (0.4 - composite) / 0.4)
        else:
            signal = "NEUTRAL"
            confidence = 0.5
        
        return {
            "signal": signal,
            "confidence": confidence,
            "whale_activity": avg_whale,
            "exchange_flow": avg_flow,
            "composite_score": composite
        }

    def get_analyzer_stats(self) -> dict:
        return {
            "metrics_collected": len(self._metrics_history),
            "whale_tracker": self._whale_tracker.get_tracker_stats(),
            "composite_signal": self.get_composite_signal()
        }


def create_onchain_features(metrics: OnChainMetrics) -> list[float]:
    return [
        metrics.whale_activity,
        metrics.exchange_flow,
        metrics.network_velocity,
        metrics.stablecoin_flow,
        metrics.gas_price / 100.0
    ]