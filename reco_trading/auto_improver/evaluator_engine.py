"""
Evaluator Engine Module for Auto-Improver.
Performs backtesting and calculates performance metrics.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from reco_trading.auto_improver.data_collector import DataSet, MarketDataPoint
from reco_trading.auto_improver.strategy_generator import StrategyVariant

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of strategy evaluation."""
    variant_id: str
    variant_name: str
    metrics: dict[str, float]
    trades: list[dict[str, Any]]
    duration_minutes: float
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "success"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "metrics": self.metrics,
            "trades_count": len(self.trades),
            "duration_minutes": self.duration_minutes,
            "evaluated_at": self.evaluated_at.isoformat(),
            "status": self.status,
            "error_message": self.error_message,
        }


class BacktestEngine:
    """Engine for backtesting strategies."""

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.fee = 0.001

    async def run_backtest(
        self,
        variant: StrategyVariant,
        dataset: DataSet,
    ) -> EvaluationResult:
        """Run backtest for a strategy variant."""
        logger.info(f"Running backtest for {variant.name}")
        
        start_time = datetime.now()
        
        try:
            params = variant.parameters
            
            equity = self.initial_balance
            position = None
            trades = []
            
            symbol_data: dict[str, list[MarketDataPoint]] = {}
            for point in dataset.data_points:
                if point.symbol not in symbol_data:
                    symbol_data[point.symbol] = []
                symbol_data[point.symbol].append(point)
            
            for symbol, points in symbol_data.items():
                points_sorted = sorted(points, key=lambda p: p.timestamp)
                
                for i, point in enumerate(points_sorted[20:], 20):
                    closes = [p.close for p in points_sorted[max(0, i-20):i]]
                    
                    if len(closes) < 20:
                        continue
                    
                    current_price = point.close
                    ma = sum(closes[-10:]) / 10
                    rsi = self._calculate_rsi(closes)
                    
                    stop_loss = params.get("stop_loss", 0.03)
                    take_profit = params.get("take_profit", 0.06)
                    
                    if position is None:
                        if rsi < params.get("rsi_oversold", 30) and current_price > ma:
                            position = {
                                "entry_price": current_price,
                                "entry_time": point.timestamp,
                                "side": "long",
                                "size": equity * params.get("position_size", 0.2),
                            }
                    
                    elif position["side"] == "long":
                        pnl_pct = (current_price - position["entry_price"]) / position["entry_price"]
                        
                        if pnl_pct >= take_profit or pnl_pct <= -stop_loss:
                            trade_pnl = position["size"] * pnl_pct - (position["size"] * self.fee)
                            equity += trade_pnl
                            
                            trades.append({
                                "entry_time": position["entry_time"].isoformat(),
                                "exit_time": point.timestamp.isoformat(),
                                "symbol": symbol,
                                "side": "long",
                                "entry_price": position["entry_price"],
                                "exit_price": current_price,
                                "pnl": trade_pnl,
                                "pnl_pct": pnl_pct * 100,
                            })
                            
                            position = None
            
            duration = (datetime.now() - start_time).total_seconds() / 60
            
            result = EvaluationResult(
                variant_id=variant.id,
                variant_name=variant.name,
                metrics={},
                trades=trades,
                duration_minutes=duration,
            )
            
            result.metrics = self._calculate_metrics(trades, equity)
            
            logger.info(f"Backtest complete: ROI={result.metrics.get('roi', 0):.2f}%, win_rate={result.metrics.get('win_rate', 0):.2f}")
            return result
            
        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            return EvaluationResult(
                variant_id=variant.id,
                variant_name=variant.name,
                metrics={},
                trades=[],
                duration_minutes=0,
                status="failed",
                error_message=str(e),
            )

    def _calculate_rsi(self, closes: list[float], period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_metrics(self, trades: list[dict[str, Any]], final_equity: float) -> dict[str, float]:
        """Calculate performance metrics."""
        if not trades:
            return {
                "roi": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "avg_trade": 0.0,
            }
        
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        
        total_pnl = sum(t["pnl"] for t in trades)
        roi = (final_equity - self.initial_balance) / self.initial_balance * 100
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        returns = [t["pnl_pct"] / 100 for t in trades]
        sharpe = self._calculate_sharpe(returns)
        
        max_dd = self._calculate_max_drawdown(trades)
        
        avg_trade = total_pnl / len(trades) if trades else 0
        
        return {
            "roi": roi,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": len(trades),
            "avg_trade": avg_trade,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_win": sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
            "avg_loss": sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        }

    def _calculate_sharpe(self, returns: list[float], risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        
        if std_dev == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free) / std_dev * math.sqrt(252)
        return sharpe

    def _calculate_max_drawdown(self, trades: list[dict[str, Any]]) -> float:
        """Calculate maximum drawdown."""
        if not trades:
            return 0.0
        
        equity = self.initial_balance
        peak = equity
        max_dd = 0.0
        
        for trade in trades:
            equity += trade["pnl"]
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd


class EvaluatorEngine:
    """Engine for evaluating strategy variants."""

    def __init__(self, initial_balance: float = 10000.0):
        self.backtest_engine = BacktestEngine(initial_balance)
        self._evaluation_history: list[EvaluationResult] = []

    async def evaluate_variant(
        self,
        variant: StrategyVariant,
        dataset: DataSet,
    ) -> EvaluationResult:
        """Evaluate a single strategy variant."""
        logger.info(f"Evaluating variant {variant.name}")
        
        result = await self.backtest_engine.run_backtest(variant, dataset)
        
        self._evaluation_history.append(result)
        
        return result

    async def evaluate_multiple(
        self,
        variants: list[StrategyVariant],
        dataset: DataSet,
    ) -> list[EvaluationResult]:
        """Evaluate multiple strategy variants."""
        logger.info(f"Evaluating {len(variants)} variants")
        
        results = []
        
        for variant in variants:
            result = await self.evaluate_variant(variant, dataset)
            results.append(result)
        
        return results

    def get_evaluation_history(
        self,
        variant_id: str | None = None,
    ) -> list[EvaluationResult]:
        """Get evaluation history."""
        if variant_id:
            return [r for r in self._evaluation_history if r.variant_id == variant_id]
        return self._evaluation_history

    def get_best_evaluation(self, variant_id: str) -> EvaluationResult | None:
        """Get best evaluation for a variant."""
        results = self.get_evaluation_history(variant_id)
        
        if not results:
            return None
        
        return max(results, key=lambda r: r.metrics.get("roi", float("-inf")))

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._evaluation_history.clear()
        logger.info("Evaluation history cleared")
