import numpy as np
import pandas as pd

from reco_trading.core.institutional_risk import InstitutionalRiskManager, RiskConfig
from reco_trading.core.market_regime import MarketRegimeDetector


def test_institutional_risk_kill_switch_by_drawdown():
    risk = InstitutionalRiskManager(RiskConfig(max_drawdown=0.1))
    risk.update_equity(1000)
    risk.check_kill_switch(890)
    assert risk.kill_switch is True


def test_institutional_risk_position_size_caps_exposure():
    risk = InstitutionalRiskManager(RiskConfig(max_exposure=0.2))
    size = risk.calculate_position_size(equity=1000, atr=0.1, win_rate=0.9, reward_risk=2.0)
    assert size <= 200


def test_market_regime_predict_returns_contract():
    detector = MarketRegimeDetector(n_states=3)
    returns = np.random.normal(0, 0.01, 200)
    prices = pd.Series(np.cumsum(np.random.normal(0, 1, 200)) + 100)
    out = detector.predict(returns, prices)
    assert {'regime', 'volatility_state', 'trend_state', 'confidence'}.issubset(out.keys())
