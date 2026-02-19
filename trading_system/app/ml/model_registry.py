from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelRecord:
    name: str
    version: str
    path: str
    regime: str | None = None


class ModelRegistry:
    def __init__(self, registry_file: str = 'artifacts/models/registry.txt') -> None:
        self.path = Path(registry_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, version: str, path: str, regime: str | None = None) -> None:
        regime_value = regime or ''
        with self.path.open('a', encoding='utf-8') as f:
            f.write(f'{name},{version},{path},{regime_value}\n')

    def latest(self, name: str) -> ModelRecord | None:
        if not self.path.exists():
            return None
        lines = [x.strip() for x in self.path.read_text(encoding='utf-8').splitlines() if x.strip()]
        for line in reversed(lines):
            parts = line.split(',', 3)
            n, v, p = parts[:3]
            regime = parts[3] if len(parts) > 3 and parts[3] else None
            if n == name:
                return ModelRecord(n, v, p, regime=regime)
        return None
