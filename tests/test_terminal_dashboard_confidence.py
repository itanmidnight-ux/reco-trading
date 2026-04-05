from reco_trading.ui.dashboard import _confidence_bar


def test_confidence_bar_accepts_fraction_and_percent_values() -> None:
    fraction = _confidence_bar(0.65)
    percent = _confidence_bar(65)
    assert "65.0%" in fraction
    assert "65.0%" in percent


def test_confidence_bar_never_breaks_on_empty_or_large_values() -> None:
    assert "0.0%" in _confidence_bar(None)
    assert "100.0%" in _confidence_bar(150)
