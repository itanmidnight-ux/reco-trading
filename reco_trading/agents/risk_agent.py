from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from reco_trading.agents.base_agent import (
    BaseLLMAgent, AgentConfig, AgentRole, AgentCapability, MarketContext, TradingDecision
)


SYSTEM_PROMPT = """You are a professional risk management specialist for crypto trading.
Your role is to evaluate trade risks and provide risk-adjusted recommendations.

Evaluate every trade against these criteria:
1. Position Size Risk - Is the position too large?
2. Market Risk - Is the market conditions favorable?
3. Liquidity Risk - Can we exit easily?
4. Volatility Risk - Is volatility too high?
5. Correlation Risk - Is the position correlated with existing trades?

Always respond with:
1. RISK_LEVEL: [LOW/MEDIUM/HIGH/EXTREME]
2. RISK_SCORE: [0-100 numeric score]
3. RECOMMENDATION: [PROCEED/ADJUST/ABORT]
4. ADJUSTMENTS: [Any position size or parameter adjustments needed]

Be conservative when uncertain."""


@dataclass
class RiskAssessment:
    risk_level: str
    risk_score: float
    recommendation: str
    adjustments: dict
    factors: dict
    timestamp: datetime = field(default_factory=datetime.now)


class RiskAgent(BaseLLMAgent):
    def __init__(self, config: AgentConfig | None = None):
        if config is None:
            config = AgentConfig(
                name="risk_manager",
                role=AgentRole.RISK_MANAGER,
                capabilities=[AgentCapability.RISK_ASSESSMENT],
                model="llama3",
                temperature=0.2,
                max_tokens=512
            )
        super().__init__(config)
        
        self._max_position_pct = 0.10
        self._max_daily_loss = 0.05
        self._max_correlation = 0.7
        
        self._risk_history: list[RiskAssessment] = []

    async def process(self, input_data: dict) -> dict:
        decision = input_data.get("decision")
        market_context = input_data.get("market_context")
        portfolio_state = input_data.get("portfolio_state", {})
        
        risk_assessment = await self._assess_risk(decision, market_context, portfolio_state)
        
        self._risk_history.append(risk_assessment)
        
        return {
            "agent": self.config.name,
            "assessment": risk_assessment,
            "adjusted_decision": self._apply_risk_adjustments(decision, risk_assessment)
        }

    async def _assess_risk(self, decision: TradingDecision, 
                          market_context: MarketContext | None,
                          portfolio_state: dict) -> RiskAssessment:
        
        risk_factors = {}
        
        position_size_risk = self._check_position_size(decision, portfolio_state)
        risk_factors["position_size"] = position_size_risk
        
        market_risk = self._check_market_conditions(market_context)
        risk_factors["market"] = market_risk
        
        liquidity_risk = self._check_liquidity(market_context)
        risk_factors["liquidity"] = liquidity_risk
        
        volatility_risk = self._check_volatility(market_context)
        risk_factors["volatility"] = volatility_risk
        
        correlation_risk = self._check_correlation(portfolio_state)
        risk_factors["correlation"] = correlation_risk
        
        weights = {
            "position_size": 0.25,
            "market": 0.20,
            "liquidity": 0.20,
            "volatility": 0.20,
            "correlation": 0.15
        }
        
        weighted_risk = sum(
            risk_factors[key] * weights[key] 
            for key in weights
        )
        
        risk_score = min(100, weighted_risk * 100)
        
        if risk_score < 25:
            risk_level = "LOW"
            recommendation = "PROCEED"
        elif risk_score < 50:
            risk_level = "MEDIUM"
            recommendation = "PROCEED" if risk_score < 40 else "ADJUST"
        elif risk_score < 75:
            risk_level = "HIGH"
            recommendation = "ADJUST"
        else:
            risk_level = "EXTREME"
            recommendation = "ABORT"
        
        adjustments = self._calculate_adjustments(risk_factors, risk_score)
        
        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            recommendation=recommendation,
            adjustments=adjustments,
            factors=risk_factors
        )

    def _check_position_size(self, decision: TradingDecision, portfolio_state: dict) -> float:
        position_pct = decision.metadata.get("position_pct", 0)
        
        if not portfolio_state:
            return 0.0
        
        capital = portfolio_state.get("capital", 1000)
        position_value = capital * (position_pct / 100)
        
        if position_value > capital * self._max_position_pct:
            return 1.0
        
        return position_pct / (self._max_position_pct * 100)

    def _check_market_conditions(self, market_context: MarketContext | None) -> float:
        if not market_context:
            return 0.5
        
        regime = market_context.regime.upper()
        
        if regime in ["HIGH_VOL", "BEAR"]:
            return 0.8
        elif regime == "SIDEWAYS":
            return 0.4
        elif regime == "BULL":
            return 0.2
        
        return 0.5

    def _check_liquidity(self, market_context: MarketContext | None) -> float:
        if not market_context:
            return 0.5
        
        volume = market_context.volume_24h
        
        if volume > 100_000_000:
            return 0.1
        elif volume > 10_000_000:
            return 0.3
        elif volume > 1_000_000:
            return 0.6
        else:
            return 0.9

    def _check_volatility(self, market_context: MarketContext | None) -> float:
        if not market_context:
            return 0.5
        
        volatility = market_context.volatility
        
        if volatility < 0.02:
            return 0.1
        elif volatility < 0.05:
            return 0.3
        elif volatility < 0.10:
            return 0.6
        else:
            return 0.9

    def _check_correlation(self, portfolio_state: dict) -> float:
        positions = portfolio_state.get("positions", [])
        
        if len(positions) < 2:
            return 0.0
        
        return min(1.0, len(positions) / 10)

    def _calculate_adjustments(self, risk_factors: dict, risk_score: float) -> dict:
        adjustments = {}
        
        if risk_factors.get("position_size", 0) > 0.8:
            adjustments["reduce_position"] = True
            adjustments["new_position_pct"] = min(
                risk_factors["position_size"],
                self._max_position_pct * 100
            )
        
        if risk_factors.get("volatility", 0) > 0.7:
            adjustments["widen_stop_loss"] = True
            adjustments["stop_loss_multiplier"] = 1.5
        
        if risk_factors.get("liquidity", 0) > 0.7:
            adjustments["use_limit_order"] = True
        
        if risk_score > 50:
            adjustments["reduce_confidence"] = True
            adjustments["confidence_multiplier"] = 1 - (risk_score - 50) / 100
        
        return adjustments

    def _apply_risk_adjustments(self, decision: TradingDecision, 
                                assessment: RiskAssessment) -> TradingDecision:
        if assessment.adjustments.get("reduce_position"):
            new_pct = assessment.adjustments.get("new_position_pct", decision.metadata.get("position_pct", 10))
            decision.metadata["position_pct"] = new_pct
        
        if assessment.adjustments.get("reduce_confidence"):
            multiplier = assessment.adjustments.get("confidence_multiplier", 1.0)
            decision.confidence *= multiplier
        
        decision.metadata["risk_assessment"] = {
            "risk_level": assessment.risk_level,
            "risk_score": assessment.risk_score,
            "adjustments": assessment.adjustments
        }
        
        return decision

    async def check_daily_loss(self, portfolio_state: dict) -> bool:
        daily_pnl = portfolio_state.get("daily_pnl", 0)
        capital = portfolio_state.get("capital", 1000)
        
        loss_pct = abs(daily_pnl) / capital if daily_pnl < 0 else 0
        
        return loss_pct >= self._max_daily_loss

    async def check_position_limits(self, portfolio_state: dict) -> bool:
        positions = portfolio_state.get("positions", [])
        
        capital = portfolio_state.get("capital", 1000)
        total_exposed = sum(p.get("value", 0) for p in positions)
        
        return (total_exposed / capital) >= self._max_position_pct

    def get_risk_history(self, limit: int = 20) -> list[RiskAssessment]:
        return self._risk_history[-limit:]

    def get_risk_stats(self) -> dict:
        if not self._risk_history:
            return {"total_assessments": 0}
        
        avg_risk = sum(r.risk_score for r in self._risk_history) / len(self._risk_history)
        
        high_risk_count = sum(1 for r in self._risk_history if r.risk_score > 50)
        
        abort_count = sum(1 for r in self._risk_history if r.recommendation == "ABORT")
        
        return {
            "total_assessments": len(self._risk_history),
            "average_risk_score": avg_risk,
            "high_risk_count": high_risk_count,
            "abort_count": abort_count,
            "abort_rate": abort_count / len(self._risk_history)
        }


class PositionRiskCalculator:
    @staticmethod
    def calculate_kelly(capital: float, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        
        kelly = (b * p - (1 - p)) / b
        
        return max(0, min(kelly, 0.25))

    @staticmethod
    def calculate_sharpe(returns: list[float], risk_free_rate: float = 0.02) -> float:
        if len(returns) < 2:
            return 0.0
        
        import statistics
        
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / std_return

    @staticmethod
    def calculate_var(returns: list[float], confidence: float = 0.95) -> float:
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0