from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - fallback cuando torch no está instalado
    torch = None  # type: ignore[assignment]


InputAdapter = Callable[[Any], Any]


@dataclass(slots=True)
class DeviceManager:
    """Gestiona la asignación de dispositivos y saneamiento de memoria para inferencia."""

    enable_cuda: bool = True
    max_memory_fraction: float | None = None
    clear_cache_threshold_mb: float = 1024.0
    _device_map: dict[str, str] = field(default_factory=dict, init=False)
    _next_cuda_index: int = field(default=0, init=False)

    def cuda_available(self) -> bool:
        return bool(torch is not None and self.enable_cuda and torch.cuda.is_available())

    def assign_device(self, model_name: str) -> str:
        if model_name in self._device_map:
            return self._device_map[model_name]

        if not self.cuda_available():
            self._device_map[model_name] = 'cpu'
            return 'cpu'

        device_count = max(int(torch.cuda.device_count()), 1)
        cuda_index = self._next_cuda_index % device_count
        self._next_cuda_index += 1
        device_name = f'cuda:{cuda_index}'
        self._device_map[model_name] = device_name

        if self.max_memory_fraction is not None:
            fraction = float(np.clip(self.max_memory_fraction, 0.05, 1.0))
            with contextlib.suppress(Exception):
                torch.cuda.set_per_process_memory_fraction(fraction, cuda_index)

        return device_name

    def conditional_cache_clear(self, device_name: str | None = None) -> None:
        if not self.cuda_available() or torch is None:
            return

        if device_name is None:
            indices = range(torch.cuda.device_count())
        elif device_name.startswith('cuda:'):
            indices = [int(device_name.split(':', maxsplit=1)[1])]
        else:
            return

        threshold_bytes = int(self.clear_cache_threshold_mb * 1024 * 1024)
        for idx in indices:
            with contextlib.suppress(Exception):
                reserved = int(torch.cuda.memory_reserved(idx))
                if reserved >= threshold_bytes:
                    torch.cuda.empty_cache()

    def handle_cuda_failure(self, model_name: str, err: Exception) -> bool:
        message = str(err).lower()
        if 'cuda' not in message and 'cublas' not in message and 'out of memory' not in message:
            return False
        self._device_map[model_name] = 'cpu'
        self.conditional_cache_clear('cuda:0')
        return True


@dataclass(slots=True)
class _InferenceJob:
    payload: Any
    future: asyncio.Future[Any]


@dataclass(slots=True)
class _ModelRuntime:
    name: str
    model: Any
    input_adapter: InputAdapter
    mode: str
    device_name: str
    queue: asyncio.Queue[_InferenceJob] | None = None
    worker: asyncio.Task[None] | None = None


class BoostingInferenceWrapper:
    """Wrapper uniforme para inferencia de modelos boosting/sklearn-like."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def predict_batch(self, features: list[Any]) -> list[float]:
        if hasattr(self.estimator, 'predict_proba'):
            probs = self.estimator.predict_proba(features)
            out = np.asarray(probs)
            if out.ndim == 2 and out.shape[1] > 1:
                return [float(v) for v in out[:, 1]]
            return [float(v) for v in out.reshape(-1)]

        preds = self.estimator.predict(features)
        return [float(v) for v in np.asarray(preds).reshape(-1)]


class InferenceBatchEngine:
    def __init__(
        self,
        *,
        device_manager: DeviceManager | None = None,
        max_batch_size: int = 16,
        max_wait_ms: float = 3.0,
    ) -> None:
        self.device_manager = device_manager or DeviceManager()
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_wait_ms = max(0.1, float(max_wait_ms))
        self._runtimes: dict[str, _ModelRuntime] = {}
        self._closed = False

    def register_transformer(self, model_name: str, model: Any, *, input_adapter: InputAdapter) -> None:
        self.register_model(model_name, model, input_adapter=input_adapter, mode='transformer')

    def register_policy_network(self, model_name: str, model: Any, *, input_adapter: InputAdapter) -> None:
        self.register_model(model_name, model, input_adapter=input_adapter, mode='policy_network')

    def register_boosting(self, model_name: str, estimator: Any, *, input_adapter: InputAdapter) -> None:
        wrapper = BoostingInferenceWrapper(estimator)
        self.register_model(model_name, wrapper, input_adapter=input_adapter, mode='boosting')

    def register_model(self, model_name: str, model: Any, *, input_adapter: InputAdapter, mode: str = 'generic') -> None:
        device_name = self.device_manager.assign_device(model_name)
        runtime = _ModelRuntime(
            name=model_name,
            model=self._move_to_device(model, device_name),
            input_adapter=input_adapter,
            mode=mode,
            device_name=device_name,
        )
        self._runtimes[model_name] = runtime

    async def infer(self, model_name: str, payload: Any, timeout_s: float | None = None) -> Any:
        if self._closed:
            raise RuntimeError('InferenceBatchEngine ya está cerrado')
        runtime = self._runtimes.get(model_name)
        if runtime is None:
            raise KeyError(f'Modelo no registrado: {model_name}')

        loop = asyncio.get_running_loop()
        if runtime.queue is None:
            runtime.queue = asyncio.Queue()
        if runtime.worker is None or runtime.worker.done():
            runtime.worker = loop.create_task(self._worker_loop(runtime))
        fut: asyncio.Future[Any] = loop.create_future()
        await runtime.queue.put(_InferenceJob(payload=payload, future=fut))
        if timeout_s is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout_s)

    async def shutdown(self) -> None:
        self._closed = True
        for runtime in self._runtimes.values():
            if runtime.worker is not None:
                runtime.worker.cancel()
        await asyncio.gather(
            *(rt.worker for rt in self._runtimes.values() if rt.worker is not None),
            return_exceptions=True,
        )

    async def _worker_loop(self, runtime: _ModelRuntime) -> None:
        try:
            while True:
                if runtime.queue is None:
                    await asyncio.sleep(0)
                    continue
                first = await runtime.queue.get()
                batch = [first]
                deadline = asyncio.get_running_loop().time() + self.max_wait_ms / 1000.0

                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(runtime.queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    batch.append(nxt)

                try:
                    outputs = await asyncio.to_thread(self._execute_batch_sync, runtime, batch)
                    for job, output in zip(batch, outputs):
                        if not job.future.done():
                            job.future.set_result(output)
                except Exception as err:
                    for job in batch:
                        if not job.future.done():
                            job.future.set_exception(err)
                finally:
                    for _ in batch:
                        runtime.queue.task_done()
        except asyncio.CancelledError:
            return

    def _execute_batch_sync(self, runtime: _ModelRuntime, batch: list[_InferenceJob]) -> list[Any]:
        adapted = [runtime.input_adapter(job.payload) for job in batch]

        if runtime.mode == 'boosting' and isinstance(runtime.model, BoostingInferenceWrapper):
            return runtime.model.predict_batch(adapted)

        if torch is None:
            return [self._call_model(runtime.model, item) for item in adapted]

        try:
            return self._execute_torch_batch(runtime, adapted)
        except Exception as err:
            if self.device_manager.handle_cuda_failure(runtime.name, err):
                runtime.device_name = 'cpu'
                runtime.model = self._move_to_device(runtime.model, 'cpu')
                return self._execute_torch_batch(runtime, adapted)
            raise

    def _execute_torch_batch(self, runtime: _ModelRuntime, adapted: list[Any]) -> list[Any]:
        if torch is None:
            return [self._call_model(runtime.model, item) for item in adapted]

        if runtime.device_name.startswith('cuda'):
            device_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            device_ctx = contextlib.nullcontext()

        with torch.inference_mode():
            with device_ctx:
                if all(isinstance(item, torch.Tensor) for item in adapted):
                    batch_tensor = torch.stack([item.float() for item in adapted], dim=0)
                    batch_tensor = batch_tensor.to(runtime.device_name)
                    outputs = runtime.model(batch_tensor)
                    out_tensor = outputs.detach().to('cpu')
                    if out_tensor.ndim == 0:
                        return [float(out_tensor.item())]
                    if out_tensor.ndim == 1:
                        return [float(v) for v in out_tensor.numpy()]
                    return [row for row in out_tensor.numpy()]

                return [self._call_model(runtime.model, item) for item in adapted]

    @staticmethod
    def _call_model(model: Any, item: Any) -> Any:
        if callable(model):
            out = model(item)
        elif hasattr(model, 'predict'):
            out = model.predict([item])
        else:
            raise TypeError('Modelo no soportado para inferencia')

        if isinstance(out, np.ndarray):
            if out.size == 1:
                return float(out.reshape(-1)[0])
            return out
        if isinstance(out, (list, tuple)) and len(out) == 1:
            return out[0]
        return out

    @staticmethod
    def _move_to_device(model: Any, device_name: str) -> Any:
        if torch is None:
            return model
        if hasattr(model, 'to'):
            model = model.to(device_name)
        if hasattr(model, 'eval'):
            model.eval()
        return model


def _versioned_artifact_path(artifact_dir: str | Path, base_name: str, suffix: str) -> Path:
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    return artifact_path / f'{base_name}_v{version}{suffix}'


def export_torchscript(model: Any, sample_input: Any, *, artifact_dir: str | Path, base_name: str = 'model') -> Path:
    if torch is None:
        raise RuntimeError('torch no está disponible para exportar TorchScript')

    path = _versioned_artifact_path(artifact_dir, base_name, '.pt')
    traced = torch.jit.trace(model, sample_input)
    traced.save(str(path))
    return path


def export_onnx(
    model: Any,
    sample_input: Any,
    *,
    artifact_dir: str | Path,
    base_name: str = 'model',
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> Path:
    if torch is None:
        raise RuntimeError('torch no está disponible para exportar ONNX')

    path = _versioned_artifact_path(artifact_dir, base_name, '.onnx')
    torch.onnx.export(
        model,
        sample_input,
        str(path),
        input_names=input_names or ['input'],
        output_names=output_names or ['output'],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    return path
