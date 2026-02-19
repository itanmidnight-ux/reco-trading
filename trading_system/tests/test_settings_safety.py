import hashlib

import pytest
from pydantic import ValidationError

from trading_system.app.config.settings import Settings


def _live_base(**overrides):
    data = {
        'mode': 'live',
        'app_env': 'prod',
        'testnet': False,
        'api_key': 'k',
        'api_secret': 's',
        'enable_live_trading': True,
        'live_ack_token': 'ENABLE_LIVE_TRADING',
        'runtime_ip': '10.20.30.40',
    }
    data['allowed_ip_hash'] = hashlib.sha256(data['runtime_ip'].encode('utf-8')).hexdigest()
    data.update(overrides)
    return data


def test_live_mode_requires_security_gates() -> None:
    with pytest.raises(ValidationError, match='enable_live_trading=true'):
        Settings(**_live_base(enable_live_trading=False))


def test_live_mode_requires_matching_ip_hash() -> None:
    with pytest.raises(ValidationError, match='runtime_ip does not match allowed_ip_hash'):
        Settings(**_live_base(allowed_ip_hash='invalid'))


def test_live_mode_requires_prod_and_mainnet() -> None:
    with pytest.raises(ValidationError, match='mode=live requires app_env=prod'):
        Settings(**_live_base(app_env='dev'))

    with pytest.raises(ValidationError, match='mode=live requires testnet=False'):
        Settings(**_live_base(testnet=True))


def test_paper_mode_must_use_testnet() -> None:
    with pytest.raises(ValidationError, match='mode=paper requires testnet=True'):
        Settings(mode='paper', testnet=False)


def test_live_mode_valid_configuration_passes() -> None:
    settings = Settings(**_live_base())
    assert settings.is_live_mode is True
