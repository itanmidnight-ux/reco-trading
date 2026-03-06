from types import SimpleNamespace

from reco_trading.hft.adaptive_market_maker import AdaptiveMarketMaker, MarketMakingState
from reco_trading.kernel.quant_kernel import MarketQuality, QuantKernel


class _DecisionEngineStub:
    def __init__(self, confidence: float, reason: str = 'stub') -> None:
        self.last_confidence = confidence
        self.last_reason = reason
        self.last_scores = {'global_probability': 0.7}


def _build_kernel(enabled: tuple[str, ...]) -> QuantKernel:
    kernel = QuantKernel.__new__(QuantKernel)
    kernel.s = SimpleNamespace(enabled_strategies=enabled, max_gap_ratio=0.02, enable_multi_exchange_arbitrage='multi_exchange_arbitrage' in enabled)
    kernel.decision_engine = _DecisionEngineStub(confidence=0.65)
    kernel.market_making_engine = AdaptiveMarketMaker(MarketMakingState())
    kernel._last_market_quality = MarketQuality(True, 'ok', 4.0, 0.01, 1000.0, 0.0)
    kernel._context = {'expected_edge': 0.015, 'friction_cost': 0.001}
    kernel.arbitrage_engine_enabled = 'multi_exchange_arbitrage' in enabled
    return kernel


def test_route_strategies_directional_only() -> None:
    kernel = _build_kernel(('directional',))
    decision, reasons = kernel._route_strategies('BUY', {'volatility': 0.01, 'atr': 5.0, 'signal_vector': {'noise_penalty': 0.2}}, 100.0)
    assert decision == 'BUY'
    assert 'selected_strategy=directional' in reasons


def test_route_strategies_uses_enabled_non_directional_votes() -> None:
    kernel = _build_kernel(('adaptive_market_making', 'multi_exchange_arbitrage'))
    decision, reasons = kernel._route_strategies('HOLD', {'volatility': 0.02, 'atr': 3.0, 'signal_vector': {'noise_penalty': 0.1}}, 100.0)
    assert any(reason.startswith('adaptive_market_making:') for reason in reasons)
    assert any(reason.startswith('multi_exchange_arbitrage:') for reason in reasons)
    assert decision in {'BUY', 'SELL', 'HOLD'}


def test_route_strategies_disabled_strategy_never_runs() -> None:
    kernel = _build_kernel(('directional',))
    decision, reasons = kernel._route_strategies('HOLD', {'volatility': 0.02, 'atr': 3.0, 'signal_vector': {'noise_penalty': 0.1}}, 100.0)
    assert not any(reason.startswith('adaptive_market_making:') for reason in reasons)
    assert not any(reason.startswith('multi_exchange_arbitrage:') for reason in reasons)
    assert decision == 'HOLD'
