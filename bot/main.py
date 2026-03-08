from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from bot.config import BotConfig
from bot.core.execution_engine import ExecutionEngine
from bot.core.market_data import MarketDataService
from bot.core.portfolio import Portfolio
from bot.core.risk_manager import RiskManager
from bot.core.strategy import DirectionalStrategy
from bot.exchange.binance_client import BinanceClient
from bot.services.state_manager import StateManager
from bot.utils.logger import configure_logger, logger


class TradingBot:
    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.client = BinanceClient(config)
        self.market_data = MarketDataService(self.client)
        self.strategy = DirectionalStrategy(config)
        self.risk = RiskManager(config)
        self.execution = ExecutionEngine(config, self.client, self.risk)
        self.state = StateManager()
        self.portfolio: Portfolio = self.state.load()
        self._last_day = datetime.now(tz=UTC).date()

    async def refresh_balance(self) -> None:
        balance = await self.client.fetch_balance()
        quote_asset = self.config.symbol.split('/')[1]
        base_asset = self.config.symbol.split('/')[0]
        self.portfolio.quote_balance = float(balance.get(quote_asset, {}).get('free', 0.0))
        self.portfolio.base_balance = float(balance.get(base_asset, {}).get('free', 0.0))

    def _rotate_daily_counters(self) -> None:
        current_day = datetime.now(tz=UTC).date()
        if current_day != self._last_day:
            self.portfolio.reset_daily_counters()
            self._last_day = current_day
            logger.info('Daily risk counters reset')

    async def run(self) -> None:
        await self.client.initialize()
        try:
            while True:
                self._rotate_daily_counters()
                await self.refresh_balance()
                candles = await self.market_data.get_candles(self.config.symbol, self.config.timeframe)
                signal = self.strategy.generate_signal(candles)
                ticker = await self.client.fetch_ticker(self.config.symbol)
                self.execution.update_position_mark_to_market(self.portfolio, float(ticker['last']))

                if signal.action in {'buy', 'sell'} and self.portfolio.open_position is None:
                    await self.execution.execute_signal(
                        symbol=self.config.symbol,
                        side=signal.action,
                        expected_move=signal.expected_move,
                        portfolio=self.portfolio,
                    )
                logger.info(
                    'Loop done signal={} open_position={} balance={:.2f}',
                    signal.action,
                    bool(self.portfolio.open_position),
                    self.portfolio.quote_balance,
                )
                self.state.save(self.portfolio)
                await asyncio.sleep(self.config.loop_interval_seconds)
        finally:
            self.state.save(self.portfolio)
            await self.client.close()


async def _main() -> None:
    configure_logger()
    config = BotConfig()
    if not config.api_key or not config.api_secret:
        raise RuntimeError('Set BINANCE_API_KEY and BINANCE_API_SECRET before running the bot')
    bot = TradingBot(config)
    await bot.run()


if __name__ == '__main__':
    asyncio.run(_main())
