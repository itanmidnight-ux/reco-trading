from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgePairInfo:
    """Edge information for a trading pair."""
    pair: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    risk_reward_ratio: float = 0.0
    min_stoploss: float = -0.15
    max_stoploss: float = -0.01
    recommended_stoploss: float = -0.05
    required_stoploss: float = -0.10
    edge: float = 0.0
    confidence: float = 0.0


@dataclass
class EdgeConfig:
    """Configuration for Edge analysis."""
    min_trades: int = 20
    min_winrate: float = 0.40
    min_profit_factor: float = 1.2
    max_drawdown: float = 0.50
    calculation_candles: int = 300
    stoploss_range_min: float = -0.15
    stoploss_range_max: float = -0.01
    stoploss_range_step: float = 0.01


class EdgeAnalysis:
    """
    Edge Analysis - Calculates the edge of trading pairs based on historical data.
    Similar to FreqTrade's Edge module.
    """

    def __init__(self, config: EdgeConfig | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or EdgeConfig()
        self.pair_info: dict[str, EdgePairInfo] = {}
        self._cached_pairs: set[str] = set()

    def calculate_edge(
        self,
        pair: str,
        entries: list[dict],
        timeframe: str = "5m",
    ) -> EdgePairInfo | None:
        """Calculate edge metrics for a pair based on historical entries."""
        
        if len(entries) < self.config.min_trades:
            self.logger.info(f"Not enough trades for {pair}: {len(entries)} < {self.config.min_trades}")
            return None

        try:
            winning_trades = [e for e in entries if e.get("pnl", 0) > 0]
            losing_trades = [e for e in entries if e.get("pnl", 0) <= 0]
            
            win_rate = len(winning_trades) / len(entries) if entries else 0
            
            avg_profit = np.mean([e.get("pnl", 0) for e in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(e.get("pnl", 0)) for e in losing_trades]) if losing_trades else 0
            
            profit_factor = avg_profit / avg_loss if avg_loss > 0 else 0
            
            expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
            
            cummulative_returns = np.cumsum([e.get("pnl", 0) for e in entries])
            running_max = np.maximum.accumulate(cummulative_returns)
            drawdowns = (cummulative_returns - running_max)
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            
            risk_reward = avg_profit / avg_loss if avg_loss > 0 else 0
            
            recommended_stoploss = self._calculate_stoploss(entries)
            required_stoploss = self._calculate_required_stoploss(entries, win_rate, profit_factor)
            
            edge_score = self._calculate_edge_score(
                win_rate=win_rate,
                profit_factor=profit_factor,
                expectancy=expectancy,
                max_drawdown=max_drawdown,
            )
            
            confidence = self._calculate_confidence(len(entries))
            
            pair_info = EdgePairInfo(
                pair=pair,
                total_trades=len(entries),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                avg_profit=avg_profit,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                expectancy=expectancy,
                max_drawdown=max_drawdown,
                risk_reward_ratio=risk_reward,
                recommended_stoploss=recommended_stoploss,
                required_stoploss=required_stoploss,
                edge=edge_score,
                confidence=confidence,
            )
            
            self.pair_info[pair] = pair_info
            return pair_info
            
        except Exception as exc:
            self.logger.error(f"Error calculating edge for {pair}: {exc}")
            return None

    def _calculate_stoploss(self, entries: list[dict]) -> float:
        """Calculate optimal stoploss based on losing trade distribution."""
        
        losses = [abs(e.get("pnl", 0)) for e in entries if e.get("pnl", 0) < 0]
        
        if not losses:
            return self.config.stoploss_range_min
        
        percentiles = [5, 10, 15, 20, 25]
        
        for percentile in percentiles:
            stoploss = -np.percentile(losses, percentile)
            
            if self.config.stoploss_range_min <= stoploss <= self.config.stoploss_range_max:
                return stoploss
        
        return -0.05

    def _calculate_required_stoploss(
        self,
        entries: list[dict],
        win_rate: float,
        profit_factor: float,
    ) -> float:
        """Calculate required stoploss to break even."""
        
        if profit_factor <= 1.0:
            return self.config.stoploss_range_min
        
        min_profit = min([e.get("pnl", 0) for e in entries if e.get("pnl", 0) > 0]) if entries else 0
        max_loss = max([abs(e.get("pnl", 0)) for e in entries if e.get("pnl", 0) < 0]) if entries else 0
        
        if min_profit > 0 and max_loss > 0:
            required_sl = -((win_rate * min_profit) / ((1 - win_rate) * max_loss))
            return max(self.config.stoploss_range_min, min(required_sl, self.config.stoploss_range_max))
        
        return -0.05

    def _calculate_edge_score(
        self,
        win_rate: float,
        profit_factor: float,
        expectancy: float,
        max_drawdown: float,
    ) -> float:
        """Calculate overall edge score."""
        
        winrate_score = win_rate * 0.30
        pf_score = min(profit_factor / 3.0, 1.0) * 0.25
        expectancy_score = max(min(expectancy * 10, 1.0), 0) * 0.25
        drawdown_score = (1 - min(max_drawdown, 1.0)) * 0.20
        
        edge = winrate_score + pf_score + expectancy_score + drawdown_score
        
        return min(max(edge, 0.0), 1.0)

    def _calculate_confidence(self, num_trades: int) -> float:
        """Calculate confidence based on number of trades."""
        
        if num_trades >= 200:
            return 1.0
        elif num_trades >= 100:
            return 0.9
        elif num_trades >= 50:
            return 0.75
        elif num_trades >= 20:
            return 0.5
        else:
            return 0.25

    def filter_pairs(self, tickers: dict | None = None) -> list[str]:
        """Filter pairs based on edge criteria."""
        
        valid_pairs = []
        
        for pair, info in self.pair_info.items():
            if info.total_trades < self.config.min_trades:
                continue
            
            if info.win_rate < self.config.min_winrate:
                continue
            
            if info.profit_factor < self.config.min_profit_factor:
                continue
            
            if info.max_drawdown > self.config.max_drawdown:
                continue
            
            if info.edge <= 0:
                continue
            
            valid_pairs.append(pair)
        
        valid_pairs.sort(key=lambda p: self.pair_info[p].edge, reverse=True)
        
        return valid_pairs

    def get_pair_info(self, pair: str) -> EdgePairInfo | None:
        """Get edge info for a specific pair."""
        return self.pair_info.get(pair)

    def get_top_pairs(self, n: int = 10) -> list[EdgePairInfo]:
        """Get top N pairs by edge score."""
        
        sorted_pairs = sorted(
            self.pair_info.values(),
            key=lambda x: x.edge,
            reverse=True
        )
        
        return sorted_pairs[:n]

    def get_all_pairs_info(self) -> dict[str, EdgePairInfo]:
        """Get all pair info."""
        return self.pair_info.copy()

    def reset(self) -> None:
        """Reset all edge data."""
        self.pair_info.clear()
        self._cached_pairs.clear()
        self.logger.info("Edge analysis reset")

    def export_to_dataframe(self) -> list[dict]:
        """Export edge data to dataframe-compatible format."""
        
        data = []
        
        for pair, info in self.pair_info.items():
            data.append({
                "pair": info.pair,
                "total_trades": info.total_trades,
                "win_rate": f"{info.win_rate:.2%}",
                "profit_factor": f"{info.profit_factor:.2f}",
                "expectancy": f"{info.expectancy:.4f}",
                "max_drawdown": f"{info.max_drawdown:.2%}",
                "recommended_stoploss": f"{info.recommended_stoploss:.2%}",
                "required_stoploss": f"{info.required_stoploss:.2%}",
                "edge": f"{info.edge:.4f}",
                "confidence": f"{info.confidence:.2%}",
            })
        
        return data


def calculate_stoploss_from_risk(
    entry_price: float,
    risk_percent: float,
    side: str = "long",
) -> float:
    """Calculate stoploss price from risk percentage."""
    
    if side.lower() == "long":
        return entry_price * (1 - risk_percent)
    else:
        return entry_price * (1 + risk_percent)


def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stoploss_price: float,
    risk_percent: float = 1.0,
) -> float:
    """Calculate position size based on risk management."""
    
    risk_amount = account_balance * (risk_percent / 100)
    
    price_difference = abs(entry_price - stoploss_price)
    
    if price_difference > 0:
        position_size = risk_amount / price_difference
    else:
        position_size = 0
    
    return position_size
