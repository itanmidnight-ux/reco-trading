from __future__ import annotations

import numpy as np
import pandas as pd

from reco_trading.ai.model_stacking import StackingEnsemble, StackingFeatureBuilder


def _synthetic_frame(rows: int = 1200) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(0, 12 * np.pi, rows)
    noise = np.random.default_rng(7).normal(0.0, 0.04, rows)

    frame = pd.DataFrame(index=idx)
    frame["return"] = np.sin(x) * 0.01 + noise
    frame["ema12"] = 100 + np.sin(x) * 2
    frame["ema26"] = 100 + np.cos(x) * 2
    frame["macd"] = frame["ema12"] - frame["ema26"]
    frame["breakout20"] = (frame["return"].rolling(20).max().fillna(0) > 0.01).astype(float)
    frame["volatility20"] = frame["return"].rolling(20).std().fillna(frame["return"].std())
    frame["volume_norm"] = 1.0 + np.abs(np.sin(x))

    frame["zscore20"] = (frame["return"] - frame["return"].rolling(20).mean().fillna(0)) / (
        frame["return"].rolling(20).std().fillna(1e-4) + 1e-9
    )
    frame["rsi14"] = 50 + np.sin(x) * 10
    frame["atr14"] = frame["volatility20"] * 100
    frame["bb_dev"] = frame["zscore20"]

    frame["obi"] = np.sin(x)
    frame["cvd"] = np.cumsum(frame["return"])
    frame["spread"] = 0.0005 + np.abs(noise) * 0.01
    frame["vpin"] = np.clip(np.abs(np.cos(x)), 0.0, 1.0)
    frame["liquidity_shock"] = (frame["spread"] > frame["spread"].quantile(0.9)).astype(float)

    frame["target"] = (frame["return"].shift(-1).fillna(0) > 0).astype(int)
    return frame


def test_stacking_feature_builder_combines_sources() -> None:
    frame = _synthetic_frame(200)
    hmm = pd.DataFrame({"hmm_state": 1.0, "hmm_confidence": 0.7}, index=frame.index)
    meta = {"meta_confidence": 0.8, "meta_weight_momentum": 1.2, "meta_weight_reversion": 0.9}
    rl = {"size_multiplier": 1.1, "threshold_shift": -0.02, "risk_shift": 0.001, "pause_trading": False}

    features = StackingFeatureBuilder().build(frame, hmm_output=hmm, meta_adjustments=meta, rl_adjustments=rl)

    assert "momentum_score" in features.columns
    assert "mean_reversion_score" in features.columns
    assert "microstructure_pressure" in features.columns
    assert "hmm_state" in features.columns
    assert "meta_confidence" in features.columns
    assert "rl_size_multiplier" in features.columns


def test_stacking_ensemble_walk_forward_calibration_and_artifacts(tmp_path) -> None:
    frame = _synthetic_frame(1200)
    features = StackingFeatureBuilder().build(frame)

    ensemble = StackingEnsemble(
        model_type="gradient_boosting",
        calibrator="platt",
        artifact_dir=tmp_path,
    )
    ensemble.fit(features, frame["target"], train=700, test=120, step=120, min_train=500)

    probs = ensemble.predict_proba(features.tail(20))
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    artifacts = ensemble.save_artifacts(version="stacking-test-v1")
    assert artifacts.model_path.exists()
    assert artifacts.metadata_path.exists()

    restored = StackingEnsemble(artifact_dir=tmp_path)
    restored.load_artifacts("stacking-test-v1")
    probs_restored = restored.predict_proba(features.tail(5))
    assert probs_restored.shape == (5, 2)
