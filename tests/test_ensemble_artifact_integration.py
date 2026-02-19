from __future__ import annotations

import joblib

from trading_system.app.models.ensemble.service import EnsembleService
from trading_system.app.ml.model_registry import ModelRegistry
from trading_system.app.services.feature_engineering.pipeline import FeatureVector
from trading_system.app.services.regime_detection.service import RegimeState


class ConstantProbaModel:
    def __init__(self, up: float) -> None:
        self.up = up

    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        assert rows and len(rows[0]) > 0
        return [[1 - self.up, self.up]]


def _fv() -> FeatureVector:
    return FeatureVector(
        rsi=50,
        ema9=101,
        ema21=100,
        ema50=99,
        macd_hist=0.2,
        atr=1.0,
        bollinger_zscore=0.1,
        orderbook_imbalance=0.2,
        delta_volume=10,
        spread_proxy=0.0001,
        volatility=0.8,
        zscore_price=0.3,
        skewness=0.0,
        kurtosis=2.0,
        hh_hl_lh_ll=1.0,
        breakout_score=1.0,
        consolidation_score=0.0,
        stat_t=1.8,
        stat_pvalue=0.04,
        hurst_proxy=0.6,
        stationarity_proxy=0.7,
        stat_confidence=0.8,
    )


def test_ensemble_uses_registered_artifacts(tmp_path):
    registry = ModelRegistry(str(tmp_path / 'registry.txt'))
    rf_path = tmp_path / 'rf.joblib'
    xgb_path = tmp_path / 'xgb.joblib'
    lstm_path = tmp_path / 'lstm.joblib'

    joblib.dump(ConstantProbaModel(0.7), rf_path)
    joblib.dump(ConstantProbaModel(0.6), xgb_path)
    joblib.dump(ConstantProbaModel(0.8), lstm_path)

    registry.register('rf_classifier', 'rf-v1', str(rf_path), regime='Trend_Bull')
    registry.register('xgb_classifier', 'xgb-v2', str(xgb_path), regime='Trend_Bull')
    registry.register('lstm_classifier', 'lstm-v3', str(lstm_path), regime='Trend_Bull')

    ensemble = EnsembleService(registry=registry)
    regime = RegimeState(name='Trend_Bull', weights={'rf': 0.25, 'xgb': 0.45, 'lstm': 0.3})

    out = ensemble.infer(_fv(), regime)

    assert out.mode == 'ACTIVE'
    assert out.model_versions == {'rf': 'rf-v1', 'xgb': 'xgb-v2', 'lstm': 'lstm-v3'}
    assert out.model_sources == {'rf': str(rf_path), 'xgb': str(xgb_path), 'lstm': str(lstm_path)}
    assert out.p_up == 0.25 * 0.7 + 0.45 * 0.6 + 0.3 * 0.8


def test_ensemble_degrades_to_safe_hold_without_artifacts(tmp_path):
    registry = ModelRegistry(str(tmp_path / 'registry.txt'))
    ensemble = EnsembleService(registry=registry)

    regime = RegimeState(name='Range', weights={'rf': 0.45, 'xgb': 0.35, 'lstm': 0.2})
    out = ensemble.infer(_fv(), regime)

    assert out.mode == 'SAFE_HOLD'
    assert out.p_up == 0.5
    assert out.model_versions == {'rf': 'safe-hold', 'xgb': 'safe-hold', 'lstm': 'safe-hold'}
