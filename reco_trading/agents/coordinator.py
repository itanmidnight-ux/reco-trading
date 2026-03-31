from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from reco_trading.agents.base_agent import (
    AgentConfig, AgentRole, AgentCapability, AgentMessage, AgentRegistry, 
    AgentMessageBus, MarketContext, TradingDecision
)
from reco_trading.agents.analyst_agent import AnalystAgent, AnalysisResult
from reco_trading.agents.risk_agent import RiskAgent, RiskAssessment
from reco_trading.agents.executor_agent import ExecutorAgent, ExecutionResult


@dataclass
class CoordinatorConfig:
    enable_parallel_processing: bool = True
    decision_timeout: int = 30
    min_consensus: float = 0.6
    enable_caching: bool = True
    max_concurrent_agents: int = 3
    consensus_weight_analyst: float = 0.30
    consensus_weight_risk: float = 0.30
    consensus_weight_ensemble: float = 0.40


@dataclass
class ConsensusDecision:
    action: str
    confidence: float
    reasoning: str
    agent_votes: dict[str, str]
    consensus_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class AgentCoordinator:
    def __init__(self, config: CoordinatorConfig | None = None):
        self.config = config or CoordinatorConfig()
        self.logger = logging.getLogger(__name__)
        
        self._registry = AgentRegistry()
        self._message_bus = AgentMessageBus()
        
        self._analyst = AnalystAgent()
        self._risk_agent = RiskAgent()
        self._executor = ExecutorAgent()
        
        self._register_agents()
        self._setup_message_topics()
        
        self._decision_history: list[ConsensusDecision] = []
        self._is_running = False

    def _register_agents(self) -> None:
        self._registry.register(self._analyst)
        self._registry.register(self._risk_agent)
        self._registry.register(self._executor)

    def _setup_message_topics(self) -> None:
        self._message_bus.subscribe("analyst", ["market_data", "analysis_request"])
        self._message_bus.subscribe("risk", ["risk_assessment", "trade_decision"])
        self._message_bus.subscribe("executor", ["execution_request", "order_update"])
        self._message_bus.subscribe("coordinator", ["consensus", "final_decision"])

    async def start(self) -> None:
        await self._analyst.start()
        await self._risk_agent.start()
        await self._executor.start()
        
        self._is_running = True
        self.logger.info("Agent Coordinator started with all agents")

    async def stop(self) -> None:
        await self._analyst.stop()
        await self._risk_agent.stop()
        await self._executor.stop()
        
        self._is_running = False
        self.logger.info("Agent Coordinator stopped")

    async def make_decision(self, market_context: MarketContext, 
                           portfolio_state: dict,
                           ensemble_prediction: dict | None = None) -> ConsensusDecision:
        start_time = datetime.now()
        
        analyst_result = await self._get_analyst_decision(market_context)
        
        ensemble_action = ensemble_prediction.get("action", "HOLD") if ensemble_prediction else "HOLD"
        ensemble_confidence = ensemble_prediction.get("confidence", 0.5) if ensemble_prediction else 0.5
        
        initial_decision = TradingDecision(
            action=ensemble_action,
            confidence=ensemble_confidence,
            reasoning=f"Ensemble prediction: {ensemble_prediction.get('reasoning', 'N/A')}" if ensemble_prediction else "No ensemble data",
            risk_score=0.5,
            metadata={
                "symbol": market_context.symbol,
                "price": market_context.price,
                "position_pct": 10.0
            }
        )
        
        risk_result = await self._risk_agent.process({
            "decision": initial_decision,
            "market_context": market_context,
            "portfolio_state": portfolio_state
        })
        
        risk_assessment = risk_result.get("assessment")
        adjusted_decision = risk_result.get("adjusted_decision", initial_decision)
        
        final_action = adjusted_decision.action
        final_confidence = adjusted_decision.confidence
        
        if risk_assessment and risk_assessment.recommendation == "ABORT":
            final_action = "HOLD"
            final_confidence *= 0.5
        
        agent_votes = {
            "analyst": analyst_result.trend if analyst_result else "NEUTRAL",
            "risk": risk_assessment.risk_level if risk_assessment else "MEDIUM",
            "ensemble": ensemble_action
        }
        
        consensus_score = self._calculate_consensus(agent_votes, risk_assessment)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > self.config.decision_timeout:
            self.logger.warning(f"Decision took {elapsed}s, exceeds timeout of {self.config.decision_timeout}s")
        
        decision = ConsensusDecision(
            action=final_action,
            confidence=final_confidence,
            reasoning=f"Analyst: {analyst_result.conclusion if analyst_result else 'N/A'} | Risk: {risk_assessment.risk_level if risk_assessment else 'N/A'}",
            agent_votes=agent_votes,
            consensus_score=consensus_score
        )
        
        self._decision_history.append(decision)
        
        return decision

    async def _get_analyst_decision(self, market_context: MarketContext) -> AnalysisResult:
        try:
            result = await self._analyst.process({
                "market_context": market_context,
                "symbol": market_context.symbol
            })
            return result.get("analysis")
        except Exception as e:
            self.logger.error(f"Analyst failed: {e}")
            return None

    def _calculate_consensus(self, agent_votes: dict[str, str], 
                            risk_assessment: RiskAssessment | None) -> float:
        score = 0.0
        
        if agent_votes.get("analyst") == "BULLISH":
            score += 0.25
        elif agent_votes.get("analyst") == "BEARISH":
            score -= 0.25
        
        if agent_votes.get("risk") in ["LOW", "MEDIUM"]:
            score += 0.25
        elif agent_votes.get("risk") in ["HIGH", "EXTREME"]:
            score -= 0.25
        
        if agent_votes.get("ensemble") == "BUY":
            score += 0.25
        elif agent_votes.get("ensemble") == "SELL":
            score -= 0.25
        else:
            score += 0.1
        
        if risk_assessment and risk_assessment.risk_score > 50:
            score -= 0.1
        
        return max(0, min(1, (score + 1) / 2))

    async def execute_trade(self, decision: ConsensusDecision,
                            market_context: MarketContext) -> ExecutionResult:
        if decision.action == "HOLD":
            return ExecutionResult(
                success=True,
                error="No execution - HOLD signal"
            )
        
        trade_decision = TradingDecision(
            action=decision.action,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            risk_score=1.0 - decision.consensus_score,
            metadata={
                "symbol": market_context.symbol,
                "quantity": self._calculate_position_size(decision.confidence),
                "price": market_context.price
            }
        )
        
        result = await self._executor.process({"decision": trade_decision})
        
        return result.get("result")

    def _calculate_position_size(self, confidence: float) -> float:
        base_size = 0.001
        
        if confidence > 0.8:
            return base_size * 2
        elif confidence > 0.6:
            return base_size * 1.5
        
        return base_size

    def get_coordinator_stats(self) -> dict:
        return {
            "total_decisions": len(self._decision_history),
            "agents": self._registry.get_stats(),
            "analyst_stats": self._analyst.get_stats(),
            "risk_stats": self._risk_agent.get_risk_stats(),
            "executor_stats": self._executor.get_execution_stats()
        }

    def get_recent_decisions(self, limit: int = 20) -> list[ConsensusDecision]:
        return self._decision_history[-limit:]


class MultiAgentTradingSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._coordinator = AgentCoordinator()
        self._is_running = False
        
        self._performance_tracker = {
            "total_decisions": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "profit": 0.0,
            "loss": 0.0
        }

    async def start(self) -> None:
        await self._coordinator.start()
        self._is_running = True
        self.logger.info("Multi-Agent Trading System started")

    async def stop(self) -> None:
        await self._coordinator.stop()
        self._is_running = False
        self.logger.info("Multi-Agent Trading System stopped")

    async def trade(self, market_context: MarketContext,
                   portfolio_state: dict,
                   ensemble_prediction: dict | None = None) -> tuple[ConsensusDecision, ExecutionResult]:
        decision = await self._coordinator.make_decision(
            market_context, 
            portfolio_state,
            ensemble_prediction
        )
        
        self._performance_tracker["total_decisions"] += 1
        
        execution = await self._coordinator.execute_trade(decision, market_context)
        
        if execution.success:
            self._performance_tracker["successful_trades"] += 1
        else:
            self._performance_tracker["failed_trades"] += 1
        
        return decision, execution

    def get_performance(self) -> dict:
        total = self._performance_tracker["total_decisions"]
        if total == 0:
            return {"total_trades": 0}
        
        return {
            **self._performance_tracker,
            "win_rate": self._performance_tracker["successful_trades"] / total,
            "profit_ratio": (
                self._performance_tracker["profit"] / 
                max(1, self._performance_tracker["loss"])
            ) if self._performance_tracker["loss"] > 0 else 0
        }

    def get_system_stats(self) -> dict:
        return {
            "is_running": self._is_running,
            "coordinator_stats": self._coordinator.get_coordinator_stats(),
            "performance": self.get_performance()
        }