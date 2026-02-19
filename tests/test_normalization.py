from trading_system.app.services.feature_engineering.normalization import FeatureNormalizer


def test_feature_normalization_bounds():
    n = FeatureNormalizer()
    result = n.normalize({'rsi': 80, 'volatility': 2.5, 'orderbook_imbalance': 0.5})
    assert 0 <= result.values['rsi'] <= 1
    assert 0 <= result.values['volatility'] <= 1
    assert 0 <= result.values['orderbook_imbalance'] <= 1
