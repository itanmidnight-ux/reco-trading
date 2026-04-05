from reco_trading.core.bot_engine import BotEngine


def test_rsi_filter_uses_runtime_thresholds():
    engine = BotEngine.__new__(BotEngine)
    engine.runtime_filter_config = {"rsi_buy_threshold": 49.0, "rsi_sell_threshold": 54.0}
    assert engine._effective_rsi_buy_threshold() == 49.0
    assert engine._effective_rsi_sell_threshold() == 54.0
