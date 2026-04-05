from reco_trading.core.bot_engine import BotEngine


def test_volatility_filter_config_has_valid_atr_bounds():
    engine = BotEngine.__new__(BotEngine)
    engine.symbol = "BTC/USDT"
    config = engine._get_default_filter_config()
    assert config["atr_low_threshold"] < config["atr_high_threshold"]
