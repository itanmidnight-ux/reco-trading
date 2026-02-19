from trading_system.app.models.ensemble.service import EnsembleOutput
from trading_system.app.services.decision_engine.service import DecisionEngineService
from trading_system.app.services.feature_engineering.pipeline import FeatureVector
from trading_system.app.services.sentiment.service import SentimentSnapshot


def _features(stat_conf: float = 0.9, stat_p: float = 0.05, volatility: float = 0.8) -> FeatureVector:
    return FeatureVector(
        rsi=55,
        ema9=101,
        ema21=100,
        ema50=99,
        macd_hist=0.2,
        atr=1.0,
        bollinger_zscore=0.4,
        orderbook_imbalance=0.2,
        delta_volume=10,
        spread_proxy=0.0005,
        volatility=volatility,
        zscore_price=0.7,
        skewness=0.1,
        kurtosis=2.2,
        hh_hl_lh_ll=1.0,
        breakout_score=1.0,
        consolidation_score=0.0,
        stat_t=2.4,
        stat_pvalue=stat_p,
        hurst_proxy=0.62,
        stationarity_proxy=0.71,
        stat_confidence=stat_conf,
    )


def _ensemble_output(mode: str = 'ACTIVE') -> EnsembleOutput:
    return EnsembleOutput(
        p_up=0.8,
        p_down=0.2,
        score=0.82,
        mode=mode,
        model_versions={'rf': 'v1', 'xgb': 'v1', 'lstm': 'v1'},
        model_sources={'rf': 'a', 'xgb': 'b', 'lstm': 'c'},
        regime='Trend_Bull',
    )


def test_decision_signal_long_when_score_high_and_positive_ev():
    engine = DecisionEngineService()
    sentiment = SentimentSnapshot(score=0.4, attention_event=False)
    d = engine.decide(_ensemble_output(), sentiment, _features())
    assert d.signal == 'LONG'
    assert d.expected_value > 0


def test_decision_hold_with_weak_statistics():
    engine = DecisionEngineService()
    sentiment = SentimentSnapshot(score=0.2, attention_event=False)
    d = engine.decide(_ensemble_output(), sentiment, _features(stat_conf=0.3, stat_p=0.6))
    assert d.signal == 'HOLD'


def test_decision_hold_when_ensemble_in_safe_mode():
    engine = DecisionEngineService()
    sentiment = SentimentSnapshot(score=0.8, attention_event=False)
    d = engine.decide(_ensemble_output(mode='SAFE_HOLD'), sentiment, _features())
    assert d.signal == 'HOLD'
