"""
Monte Carlo Simulator Module.
Simulates strategy performance under random scenarios.
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of a Monte Carlo simulation."""
    simulation_id: int
    initial_balance: float
    final_balance: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    consecutive_losses: int
    timestamps: list[datetime]
    equity_curve: list[float]


@dataclass
class MonteCarloAnalysis:
    """Aggregated Monte Carlo analysis."""
    num_simulations: int
    success_rate: float
    median_return: float
    percentile_5_return: float
    percentile_95_return: float
    median_max_drawdown: float
    probability_of_ruin: float
    expected_return: float
    risk_adjusted_return: float
    simulation_results: list[SimulationResult]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MonteCarloSimulator:
    """Simulates strategy performance using Monte Carlo methods."""

    def __init__(
        self,
        num_simulations: int = 1000,
        initial_balance: float = 10000,
    ):
        self.num_simulations = num_simulations
        self.initial_balance = initial_balance

    def simulate(
        self,
        historical_trades: list[dict[str, Any]],
        num_simulations: Optional[int] = None,
    ) -> MonteCarloAnalysis:
        """Run Monte Carlo simulation on historical trades."""
        num_sims = num_simulations or self.num_simulations
        
        logger.info(f"Running {num_sims} Monte Carlo simulations")
        
        if not historical_trades:
            return self._empty_analysis()
        
        trade_returns = [t.get("pnl_pct", 0) / 100 for t in historical_trades if "pnl_pct" in t]
        trade_results = [1 if t.get("pnl", 0) > 0 else 0 for t in historical_trades]
        
        if not trade_returns:
            return self._empty_analysis()
        
        win_rate = sum(trade_results) / len(trade_results) if trade_results else 0.5
        
        mean_return = np.mean(trade_returns)
        std_return = np.std(trade_returns)
        
        results: list[SimulationResult] = []
        
        for sim_id in range(num_sims):
            result = self._run_single_simulation(
                sim_id,
                trade_returns,
                mean_return,
                std_return,
                win_rate,
            )
            results.append(result)
        
        return self._analyze_results(results)

    def _run_single_simulation(
        self,
        sim_id: int,
        trade_returns: list[float],
        mean_return: float,
        std_return: float,
        win_rate: float,
    ) -> SimulationResult:
        """Run a single Monte Carlo simulation."""
        balance = self.initial_balance
        equity_curve = [balance]
        timestamps = [datetime.now(timezone.utc)]
        
        max_balance = balance
        max_drawdown = 0
        total_trades = len(trade_returns) * 2
        consecutive_losses = 0
        max_consecutive_losses = 0
        wins = 0
        
        for i in range(total_trades):
            if random.random() < win_rate:
                pnl_pct = max(random.gauss(mean_return, std_return), -0.1)
                wins += 1
                consecutive_losses = 0
            else:
                pnl_pct = min(random.gauss(mean_return, std_return), 0.05)
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            balance *= (1 + pnl_pct)
            
            if balance > max_balance:
                max_balance = balance
            
            dd = (max_balance - balance) / max_balance if max_balance > 0 else 0
            max_drawdown = max(max_drawdown, dd)
            
            equity_curve.append(balance)
            timestamps.append(datetime.now(timezone.utc))
        
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        returns_for_sharpe = [(equity_curve[i+1] - equity_curve[i]) / equity_curve[i] 
                              for i in range(len(equity_curve) - 1) if equity_curve[i] > 0]
        
        sharpe = self._calculate_sharpe(returns_for_sharpe) if returns_for_sharpe else 0
        
        return SimulationResult(
            simulation_id=sim_id,
            initial_balance=self.initial_balance,
            final_balance=balance,
            total_return=total_return,
            max_drawdown=max_drawdown * 100,
            sharpe_ratio=sharpe,
            win_rate=wins / total_trades * 100 if total_trades > 0 else 0,
            total_trades=total_trades,
            consecutive_losses=max_consecutive_losses,
            timestamps=timestamps,
            equity_curve=equity_curve,
        )

    def _calculate_sharpe(self, returns: list[float], risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free / 252) / std_return * np.sqrt(252)

    def _analyze_results(self, results: list[SimulationResult]) -> MonteCarloAnalysis:
        """Analyze simulation results."""
        returns = [r.total_return for r in results]
        drawdowns = [r.max_drawdown for r in results]
        
        returns_sorted = sorted(returns)
        
        success_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
        probability_of_ruin = sum(1 for r in returns if r < -50) / len(returns) * 100
        
        expected_return = np.mean(returns)
        
        risk_adjusted = expected_return / (np.std(returns) + 0.01) if np.std(returns) > 0 else 0
        
        return MonteCarloAnalysis(
            num_simulations=len(results),
            success_rate=success_rate,
            median_return=np.median(returns),
            percentile_5_return=returns_sorted[int(len(returns) * 0.05)],
            percentile_95_return=returns_sorted[int(len(returns) * 0.95)],
            median_max_drawdown=np.median(drawdowns),
            probability_of_ruin=probability_of_ruin,
            expected_return=expected_return,
            risk_adjusted_return=risk_adjusted,
            simulation_results=results,
        )

    def _empty_analysis(self) -> MonteCarloAnalysis:
        """Create empty analysis."""
        return MonteCarloAnalysis(
            num_simulations=0,
            success_rate=0,
            median_return=0,
            percentile_5_return=0,
            percentile_95_return=0,
            median_max_drawdown=0,
            probability_of_ruin=100,
            expected_return=0,
            risk_adjusted_return=0,
            simulation_results=[],
        )

    def simulate_with_parameters(
        self,
        params: dict[str, Any],
        market_scenarios: int = 100,
    ) -> MonteCarloAnalysis:
        """Simulate with specific strategy parameters."""
        base_win_rate = params.get("win_rate", 50)
        base_profit = params.get("avg_profit", 2)
        base_loss = params.get("avg_loss", 1)
        
        trade_returns = []
        for _ in range(100):
            if random.random() < base_win_rate / 100:
                trade_returns.append(random.uniform(0.5, base_profit * 2) / 100)
            else:
                trade_returns.append(-random.uniform(0.5, base_loss * 2) / 100)
        
        results = []
        
        for _ in range(market_scenarios):
            market_impact = random.gauss(1, 0.2)
            
            adjusted_returns = [r * market_impact for r in trade_returns]
            
            balance = self.initial_balance
            for r in adjusted_returns:
                balance *= (1 + r)
            
            results.append(SimulationResult(
                simulation_id=0,
                initial_balance=self.initial_balance,
                final_balance=balance,
                total_return=(balance - self.initial_balance) / self.initial_balance * 100,
                max_drawdown=random.uniform(5, 30),
                sharpe_ratio=random.uniform(0, 2),
                win_rate=base_win_rate,
                total_trades=len(trade_returns),
                consecutive_losses=random.randint(0, 5),
                timestamps=[],
                equity_curve=[],
            ))
        
        return self._analyze_results(results)

    def is_robust(self, analysis: MonteCarloAnalysis) -> tuple[bool, str]:
        """Check if strategy is robust based on Monte Carlo results."""
        if analysis.probability_of_ruin > 10:
            return False, f"High probability of ruin: {analysis.probability_of_ruin:.1f}%"
        
        if analysis.percentile_5_return < -30:
            return False, f"5th percentile return too low: {analysis.percentile_5_return:.1f}%"
        
        if analysis.median_max_drawdown > 50:
            return False, f"High median drawdown: {analysis.median_max_drawdown:.1f}%"
        
        if analysis.success_rate < 50:
            return False, f"Low success rate: {analysis.success_rate:.1f}%"
        
        if analysis.risk_adjusted_return < 0.5:
            return False, f"Low risk-adjusted return: {analysis.risk_adjusted_return:.2f}"
        
        return True, "Strategy appears robust"

    def to_dict(self, analysis: Optional[MonteCarloAnalysis] = None) -> dict[str, Any]:
        """Export to dictionary."""
        if analysis is None:
            return {"status": "no_analysis"}
        
        robust, reason = self.is_robust(analysis)
        
        return {
            "num_simulations": analysis.num_simulations,
            "success_rate": analysis.success_rate,
            "median_return": analysis.median_return,
            "percentile_5_return": analysis.percentile_5_return,
            "percentile_95_return": analysis.percentile_95_return,
            "median_max_drawdown": analysis.median_max_drawdown,
            "probability_of_ruin": analysis.probability_of_ruin,
            "expected_return": analysis.expected_return,
            "risk_adjusted_return": analysis.risk_adjusted_return,
            "is_robust": robust,
            "robustness_reason": reason,
        }
