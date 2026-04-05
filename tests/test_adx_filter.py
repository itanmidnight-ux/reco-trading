from reco_trading.core.bot_engine import BotEngine


def test_adx_filter_uses_runtime_threshold():
    engine = BotEngine.__new__(BotEngine)
    engine.runtime_filter_config = {"adx_threshold": 21.5}
    assert engine._effective_adx_threshold() == 21.5
