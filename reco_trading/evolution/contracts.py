from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class StrategyContract(Protocol):
    """Contrato base para componentes de estrategia cuantitativa."""

    def fit(self, frame: pd.DataFrame) -> None:
        """Entrena/actualiza el estado interno de la estrategia."""


@runtime_checkable
class ProbabilisticModelContract(Protocol):
    """Contrato de integración para modelos de predicción probabilística."""

    def fit(self, frame: pd.DataFrame) -> None:
        """Entrena el modelo con un dataframe enriquecido."""

    def predict_proba_up(self, frame: pd.DataFrame) -> float:
        """Retorna la probabilidad de movimiento alcista."""


@runtime_checkable
class ReversionModelContract(Protocol):
    """Contrato de integración para modelos de reversión a la media."""

    def fit(self, frame: pd.DataFrame) -> None:
        """Entrena el modelo con features de reversión."""

    def predict_reversion(self, frame: pd.DataFrame) -> float:
        """Retorna probabilidad de reversión."""
