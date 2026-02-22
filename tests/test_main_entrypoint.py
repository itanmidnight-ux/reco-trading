import os
import pytest

import main


def test_validate_required_env_raises_on_missing(monkeypatch):
    monkeypatch.delenv('BINANCE_API_KEY', raising=False)
    monkeypatch.delenv('BINANCE_API_SECRET', raising=False)
    monkeypatch.delenv('POSTGRES_DSN', raising=False)
    with pytest.raises(RuntimeError, match='Faltan variables obligatorias'):
        main._validate_required_env()


def test_validate_required_env_passes(monkeypatch):
    monkeypatch.setenv('BINANCE_API_KEY', 'k')
    monkeypatch.setenv('BINANCE_API_SECRET', 's')
    monkeypatch.setenv('POSTGRES_DSN', 'postgresql://u:p@localhost:5432/db')
    main._validate_required_env()
