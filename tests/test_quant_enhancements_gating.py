from __future__ import annotations

from datetime import datetime, timezone

from reco_trading.kernel.quant_kernel import QuantKernel


class _SettingsStub:
    allowed_sessions_utc = [(12, 20)]
    enable_volatility_sizing = True
    vol_target_risk = 0.20
    vol_min_multiplier = 0.25
    vol_max_multiplier = 1.50


def test_session_filter_uses_utc_windows() -> None:
    kernel = QuantKernel.__new__(QuantKernel)
    kernel.s = _SettingsStub()

    inside = datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc)
    outside = datetime(2024, 1, 1, 22, 0, tzinfo=timezone.utc)

    assert QuantKernel._is_within_allowed_session(kernel, inside) is True
    assert QuantKernel._is_within_allowed_session(kernel, outside) is False


def test_volatility_multiplier_is_clamped() -> None:
    kernel = QuantKernel.__new__(QuantKernel)
    kernel.s = _SettingsStub()

    high_vol = QuantKernel._volatility_size_multiplier(kernel, volatility=2.0)
    low_vol = QuantKernel._volatility_size_multiplier(kernel, volatility=0.01)

    assert high_vol == 0.25
    assert low_vol == 1.50
