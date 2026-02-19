from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopStage:
    name: str
    order: int


STAGES = [
    LoopStage('FASE 0 — Inicialización', 0),
    LoopStage('FASE 1 — Esperar cierre de vela', 1),
    LoopStage('FASE 2 — Descargar nueva vela', 2),
    LoopStage('FASE 3 — Calcular indicadores', 3),
    LoopStage('FASE 4 — Detectar patrones', 4),
    LoopStage('FASE 5 — Evaluar estructura', 5),
    LoopStage('FASE 6 — Calcular scoring determinístico', 6),
    LoopStage('FASE 7 — Integrar ajuste ML', 7),
    LoopStage('FASE 8 — Gestión de riesgo', 8),
    LoopStage('FASE 9 — Ejecutar trade', 9),
    LoopStage('FASE 10 — Monitorear posición', 10),
    LoopStage('FASE 11 — Emitir métricas a dashboard', 11),
]


class LoopEngine:
    def stages(self) -> list[LoopStage]:
        return STAGES
