from __future__ import annotations

import asyncio

from reco_trading.ai.gpu_inference import BoostingInferenceWrapper, DeviceManager, InferenceBatchEngine


class _Estimator:
    def predict_proba(self, features):
        out = []
        for row in features:
            score = float(sum(row))
            out.append([1.0 - min(max(score, 0.0), 1.0), min(max(score, 0.0), 1.0)])
        return out


def test_device_manager_falls_back_to_cpu_without_torch() -> None:
    manager = DeviceManager()
    assert manager.cuda_available() is False
    assert manager.assign_device('model_a') == 'cpu'


def test_boosting_wrapper_predicts_probability_column() -> None:
    wrapper = BoostingInferenceWrapper(_Estimator())
    out = wrapper.predict_batch([[0.2], [0.7]])
    assert out == [0.2, 0.7]


def test_inference_batch_engine_micro_batch_for_boosting() -> None:
    engine = InferenceBatchEngine(max_batch_size=4, max_wait_ms=10)
    engine.register_boosting('boosting', _Estimator(), input_adapter=lambda payload: payload)

    async def _run():
        results = await asyncio.gather(
            engine.infer('boosting', [0.1]),
            engine.infer('boosting', [0.9]),
        )
        await engine.shutdown()
        return results

    results = asyncio.run(_run())
    assert results == [0.1, 0.9]
