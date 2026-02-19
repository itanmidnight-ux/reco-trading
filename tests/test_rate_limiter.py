import asyncio
import time

from trading_system.app.core.rate_limiter import BinanceRateLimitController


def test_rate_usage_increments():
    async def _run():
        limiter = BinanceRateLimitController(safe_weight_limit=10, order_rate_per_sec=5)
        await limiter.reserve_weight(3)
        await limiter.reserve_weight(2)
        assert await limiter.usage_1m() == 5

    asyncio.run(_run())


def test_418_sets_cooldown():
    limiter = BinanceRateLimitController(safe_weight_limit=10, order_rate_per_sec=5)
    limiter.trigger_418_cooldown(2)
    assert limiter.cooldown_until > time.time()
