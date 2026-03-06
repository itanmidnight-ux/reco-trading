from __future__ import annotations

import pytest
from pydantic import ValidationError

from reco_trading.config.settings import Settings


@pytest.fixture(autouse=True)
def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('BINANCE_API_KEY', 'k')
    monkeypatch.setenv('BINANCE_API_SECRET', 's')
    monkeypatch.setenv('POSTGRES_DSN', 'postgresql://u:p@localhost:5432/db')


def test_settings_reject_invalid_partial_exit_sum() -> None:
    with pytest.raises(ValidationError, match='partial_exit_fractions'):
        Settings(partial_exit_fractions=[0.7, 0.4])


def test_settings_reject_invalid_session_window() -> None:
    with pytest.raises(ValidationError, match='inicio < fin'):
        Settings(allowed_sessions_utc=[(20, 12)])


def test_settings_reject_invalid_volatility_multiplier_bounds() -> None:
    with pytest.raises(ValidationError, match='vol_min_multiplier'):
        Settings(vol_min_multiplier=2.0, vol_max_multiplier=1.0)


def test_settings_exposes_legacy_max_gap_ratio_alias() -> None:
    settings = Settings()
    assert settings.max_gap_ratio == settings.market_max_gap_ratio
