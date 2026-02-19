from trading_system.app.services.feature_engineering.statistics import statistical_confirmation


def test_statistical_confirmation_range():
    prices = [100 + i * 0.1 for i in range(150)]
    result = statistical_confirmation(prices)
    assert 0 <= result.trend_pseudo_pvalue <= 1
    assert 0 <= result.confidence <= 1
