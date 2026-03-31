from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from reco_trading.agents.base_agent import (
    BaseLLMAgent, AgentConfig, AgentRole, AgentCapability, MarketContext
)
from reco_trading.agents.base_agent import TradingDecision


SYSTEM_PROMPT = """You are a professional crypto market analyst with expertise in:
- Technical analysis (candlesticks, patterns, indicators)
- Market microstructure
- Order flow analysis
- Price action trading
- Market regime identification

Your role is to analyze market conditions and provide clear, actionable insights.
Always respond with a structured analysis following this format:

1. TREND: [BULLISH/BEARISH/NEUTRAL]
2. MOMENTUM: [STRONG/MODERATE/WEAK]
3. KEY LEVELS: [Support/Resistance]
4. PATTERNS: [Any notable chart patterns]
5. CONCLUSION: [Summary with confidence 0-100]

Be concise but thorough. Use data-driven analysis."""


@dataclass
class AnalysisResult:
    trend: str
    momentum: str
    key_levels: dict
    patterns: list[str]
    conclusion: str
    confidence: float
    timestamp: datetime


class AnalystAgent(BaseLLMAgent):
    def __init__(self, config: AgentConfig | None = None):
        if config is None:
            config = AgentConfig(
                name="analyst",
                role=AgentRole.ANALYST,
                capabilities=[AgentCapability.MARKET_ANALYSIS],
                model="llama3",
                temperature=0.3,
                max_tokens=1024
            )
        super().__init__(config)
        self._analysis_cache = {}

    async def process(self, input_data: dict) -> dict:
        context = input_data.get("market_context")
        symbol = input_data.get("symbol", "UNKNOWN")
        
        if not context:
            return self._default_analysis(symbol)
        
        prompt = self._build_analysis_prompt(context)
        llm_result = await self._call_llm(prompt, SYSTEM_PROMPT)
        
        analysis = self._parse_llm_response(llm_result.get("response", ""), context)
        
        self._analysis_cache[symbol] = {
            "analysis": analysis,
            "timestamp": datetime.now()
        }
        
        return {
            "agent": self.config.name,
            "analysis": analysis,
            "raw_response": llm_result.get("response", "")
        }

    def _build_analysis_prompt(self, context: MarketContext) -> str:
        return f"""Analyze {context.symbol} market:

Price: ${context.price:,.2f}
Volume 24h: ${context.volume_24h:,.0f}
Volatility: {context.volatility:.2%}
Trend: {context.trend}
Regime: {context.regime}

Additional Data:
{self._format_additional_data(context.additional_data)}

Provide your analysis in the specified format."""

    def _format_additional_data(self, data: dict) -> str:
        if not data:
            return "None"
        
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _parse_llm_response(self, response: str, context: MarketContext) -> AnalysisResult:
        response_lower = response.lower()
        
        trend = "NEUTRAL"
        if "bullish" in response_lower:
            trend = "BULLISH"
        elif "bearish" in response_lower:
            trend = "BEARISH"
        
        momentum = "MODERATE"
        if "strong" in response_lower:
            momentum = "STRONG"
        elif "weak" in response_lower:
            momentum = "WEAK"
        
        confidence = 50.0
        if "90" in response_lower:
            confidence = 90.0
        elif "80" in response_lower:
            confidence = 80.0
        elif "70" in response_lower:
            confidence = 70.0
        elif "60" in response_lower:
            confidence = 60.0
        
        key_levels = {
            "support": context.price * 0.98,
            "resistance": context.price * 1.02
        }
        
        patterns = []
        if "flag" in response_lower:
            patterns.append("Bull Flag")
        elif "wedge" in response_lower:
            patterns.append("Wedge Pattern")
        if "double" in response_lower:
            patterns.append("Double Top/Bottom")
        
        conclusion = response.split("CONCLUSION:")[-1].strip() if "CONCLUSION:" in response else response[:200]
        
        return AnalysisResult(
            trend=trend,
            momentum=momentum,
            key_levels=key_levels,
            patterns=patterns,
            conclusion=conclusion[:200],
            confidence=confidence,
            timestamp=datetime.now()
        )

    def _default_analysis(self, symbol: str) -> dict:
        return {
            "agent": self.config.name,
            "analysis": AnalysisResult(
                trend="NEUTRAL",
                momentum="MODERATE",
                key_levels={"support": 0, "resistance": 0},
                patterns=[],
                conclusion="Insufficient data for analysis",
                confidence=30.0,
                timestamp=datetime.now()
            ),
            "raw_response": "Default response due to missing context"
        }

    async def analyze_market(self, context: MarketContext) -> AnalysisResult:
        result = await self.process({"market_context": context, "symbol": context.symbol})
        return result.get("analysis")

    async def compare_markets(self, contexts: list[MarketContext]) -> list[AnalysisResult]:
        results = []
        for ctx in contexts:
            result = await self.analyze_market(ctx)
            results.append(result)
        return results

    def get_cached_analysis(self, symbol: str) -> AnalysisResult | None:
        cached = self._analysis_cache.get(symbol)
        if cached:
            return cached.get("analysis")
        return None

    def clear_cache(self) -> None:
        self._analysis_cache.clear()
        super().clear_cache()


class TechnicalAnalyzer:
    @staticmethod
    def calculate_rsi(prices: list[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    def calculate_macd(prices: list[float]) -> dict:
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        def ema(data: list[float], period: int) -> float:
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        ema_12 = ema(prices, 12)
        ema_26 = ema(prices, 26)
        macd = ema_12 - ema_26
        
        signal = macd * 0.9
        
        return {
            "macd": macd,
            "signal": signal,
            "histogram": macd - signal
        }

    @staticmethod
    def detect_trend(prices: list[float], short_period: int = 20, long_period: int = 50) -> str:
        if len(prices) < long_period:
            return "NEUTRAL"
        
        short_ma = sum(prices[-short_period:]) / short_period
        long_ma = sum(prices[-long_period:]) / long_period
        
        if short_ma > long_ma * 1.02:
            return "BULLISH"
        elif short_ma < long_ma * 0.98:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def calculate_volatility(prices: list[float]) -> float:
        if len(prices) < 2:
            return 0.0
        
        import statistics
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if len(returns) < 2:
            return 0.0
        
        return statistics.stdev(returns)


class PatternDetector:
    PATTERNS = [
        "double_top", "double_bottom", "head_shoulders",
        "bull_flag", "bear_flag", "wedge", "triangle",
        "support", "resistance"
    ]

    @staticmethod
    def detect(prices: list[float], volumes: list[float]) -> list[str]:
        if len(prices) < 20:
            return []
        
        patterns = []
        
        if PatternDetector._check_double_top(prices):
            patterns.append("double_top")
        
        if PatternDetector._check_double_bottom(prices):
            patterns.append("double_bottom")
        
        if PatternDetector._check_bull_flag(prices):
            patterns.append("bull_flag")
        
        return patterns

    @staticmethod
    def _check_double_top(prices: list[float]) -> bool:
        if len(prices) < 20:
            return False
        
        recent = prices[-20:]
        max_price = max(recent)
        
        peaks = [i for i, p in enumerate(recent) if p > max_price * 0.98]
        
        return len(peaks) >= 2

    @staticmethod
    def _check_double_bottom(prices: list[float]) -> bool:
        if len(prices) < 20:
            return False
        
        recent = prices[-20:]
        min_price = min(recent)
        
        troughs = [i for i, p in enumerate(recent) if p < min_price * 1.02]
        
        return len(troughs) >= 2

    @staticmethod
    def _check_bull_flag(prices: list[float]) -> bool:
        if len(prices) < 15:
            return False
        
        recent = prices[-15:]
        first_half = recent[:7]
        second_half = recent[7:]
        
        first_trend = sum(first_half[-1] - first_half[0]) / len(first_half)
        second_trend = sum(second_half[-1] - second_half[0]) / len(second_half)
        
        return first_trend > 0 and second_trend < first_trend * 0.3