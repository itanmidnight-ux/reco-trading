from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import ccxt.async_support as ccxt
from ccxt.base.errors import DDoSProtection, ExchangeError, NetworkError, RateLimitExceeded
from loguru import logger

from reco_trading.core.institutional_risk_manager import InstitutionalRiskManager
from reco_trading.core.rate_limit_controller import AdaptiveRateLimitController
from reco_trading.hft.capital_allocator import AllocationRequest, CapitalAllocator
from reco_trading.hft.safety import HFTSafetyMonitor
from reco_trading.kernel.capital_governor import CapitalGovernor


@dataclass(slots=True)
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    best_ask_buy: float
    best_bid_sell: float
    mid_price: float
    spread: float
    expected_edge_bps: float


@dataclass(slots=True)
class ExecutionReport:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_order_id: str | None
    sell_order_id: str | None
    status: str
    spread: float
    expected_edge_bps: float
    details: dict[str, Any]


class ExchangeAdapter(ABC):
    name: str

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def create_order(self, symbol: str, side: str, amount: float, order_type: str = 'market') -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def get_balance(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class CcxtAsyncExchangeAdapter(ExchangeAdapter):
    def __init__(self, name: str, exchange: Any, *, max_calls: int = 8, period_seconds: float = 1.0) -> None:
        self.name = name
        self.exchange = exchange
        self._rate_limiter = AdaptiveRateLimitController(max_calls=max_calls, period_seconds=period_seconds)

    async def _retry(self, fn: Callable[..., Awaitable[Any]], *args: Any, retries: int = 7, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                await self._rate_limiter.acquire()
                return await fn(*args, **kwargs)
            except (RateLimitExceeded, NetworkError, DDoSProtection) as exc:
                last_exc = exc
                await asyncio.sleep(min(2**attempt, 30))
            except ExchangeError as exc:
                last_exc = exc
                if attempt == retries:
                    raise
                await asyncio.sleep(min(attempt, 5))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError('Retry loop finalizó sin resultado')

    async def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return await self._retry(self.exchange.fetch_order_book, symbol=symbol, limit=limit)

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        return await self._retry(self.exchange.fetch_ticker, symbol=symbol)

    async def create_order(self, symbol: str, side: str, amount: float, order_type: str = 'market') -> dict[str, Any]:
        return await self._retry(self.exchange.create_order, symbol, order_type, side.lower(), amount)

    async def get_balance(self) -> dict[str, Any]:
        return await self._retry(self.exchange.fetch_balance)

    async def close(self) -> None:
        await self.exchange.close()


class BinanceAdapter(CcxtAsyncExchangeAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        exchange = ccxt.binance(
            {
                'apiKey': config.get('api_key', ''),
                'secret': config.get('api_secret', ''),
                'enableRateLimit': True,
                'options': {'defaultType': config.get('default_type', 'spot')},
            }
        )
        if config.get('testnet', False):
            exchange.set_sandbox_mode(True)
        super().__init__('binance', exchange)


class KrakenAdapter(CcxtAsyncExchangeAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        exchange = ccxt.kraken(
            {
                'apiKey': config.get('api_key', ''),
                'secret': config.get('api_secret', ''),
                'enableRateLimit': True,
            }
        )
        super().__init__('kraken', exchange)


class CoinbaseAdapter(CcxtAsyncExchangeAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        exchange = ccxt.coinbase(
            {
                'apiKey': config.get('api_key', ''),
                'secret': config.get('api_secret', ''),
                'password': config.get('password', ''),
                'enableRateLimit': True,
            }
        )
        super().__init__('coinbase', exchange)


class BybitAdapter(CcxtAsyncExchangeAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        exchange = ccxt.bybit(
            {
                'apiKey': config.get('api_key', ''),
                'secret': config.get('api_secret', ''),
                'enableRateLimit': True,
                'options': {'defaultType': config.get('default_type', 'spot')},
            }
        )
        if config.get('testnet', False):
            exchange.set_sandbox_mode(True)
        super().__init__('bybit', exchange)


class ExchangeAdapterFactory:
    _registry: dict[str, type[ExchangeAdapter]] = {
        'binance': BinanceAdapter,
        'kraken': KrakenAdapter,
        'coinbase': CoinbaseAdapter,
        'bybit': BybitAdapter,
    }

    @classmethod
    def register(cls, exchange_name: str, adapter_cls: type[ExchangeAdapter]) -> None:
        cls._registry[exchange_name.lower()] = adapter_cls

    @classmethod
    def create(cls, exchange_name: str, config: dict[str, Any]) -> ExchangeAdapter:
        normalized_name = exchange_name.lower()
        adapter_cls = cls._registry.get(normalized_name)
        if adapter_cls is None:
            raise ValueError(f'Exchange no soportado: {exchange_name}')
        return adapter_cls(config)

    @classmethod
    def create_from_config(cls, config: dict[str, dict[str, Any]]) -> dict[str, ExchangeAdapter]:
        adapters: dict[str, ExchangeAdapter] = {}
        for exchange_name, exchange_config in config.items():
            enabled = exchange_config.get('enabled', True)
            if not enabled:
                continue
            adapters[exchange_name.lower()] = cls.create(exchange_name, exchange_config)
        return adapters




@dataclass(slots=True)
class OpportunityContext:
    equity: float
    daily_pnl: float
    atr: float
    annualized_volatility: float
    volatility_multiplier: float
    expected_win_rate: float
    avg_win: float
    avg_loss: float
    returns_matrix: Any
    notionals_by_exchange: dict[str, float] = field(default_factory=dict)
    notionals_by_asset: dict[str, float] = field(default_factory=dict)
    inventory_by_exchange: dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0

class MultiExchangeArbitrageEngine:
    def __init__(
        self,
        adapters: dict[str, ExchangeAdapter],
        min_edge_bps: float = 5.0,
        *,
        safety_monitor: HFTSafetyMonitor | None = None,
        capital_governor: CapitalGovernor | None = None,
    ) -> None:
        self.adapters = adapters
        self.min_edge_bps = min_edge_bps
        self.safety_monitor = safety_monitor
        self.capital_governor = capital_governor

    async def _fetch_books_and_tickers(
        self,
        symbol: str,
        exchange_names: list[str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        book_tasks = [self.adapters[name].get_order_book(symbol) for name in exchange_names]
        ticker_tasks = [self.adapters[name].get_ticker(symbol) for name in exchange_names]
        started_at = time.perf_counter()
        books_result, tickers_result = await asyncio.gather(asyncio.gather(*book_tasks), asyncio.gather(*ticker_tasks))
        elapsed_ms = (time.perf_counter() - started_at) * 1_000

        books = {name: result for name, result in zip(exchange_names, books_result, strict=True)}
        tickers = {name: result for name, result in zip(exchange_names, tickers_result, strict=True)}

        if self.safety_monitor is not None:
            per_exchange_latency_ms = elapsed_ms / max(len(exchange_names), 1)
            for exchange in exchange_names:
                self.safety_monitor.update_heartbeat(exchange)
                self.safety_monitor.detect_latency_spike(exchange, per_exchange_latency_ms)
                self.safety_monitor.detect_book_ticker_desync(exchange, books[exchange], tickers[exchange])
        return books, tickers

    @staticmethod
    def _best_prices(order_book: dict[str, Any]) -> tuple[float | None, float | None]:
        bids = order_book.get('bids') or []
        asks = order_book.get('asks') or []
        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None
        return best_bid, best_ask

    def _build_opportunity(
        self,
        symbol: str,
        sell_exchange: str,
        buy_exchange: str,
        sell_bid: float,
        buy_ask: float,
    ) -> ArbitrageOpportunity | None:
        mid_price = (sell_bid + buy_ask) / 2
        if mid_price <= 0:
            return None
        spread = (sell_bid - buy_ask) / mid_price
        edge_bps = spread * 10_000
        if edge_bps < self.min_edge_bps:
            return None
        return ArbitrageOpportunity(
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            best_ask_buy=buy_ask,
            best_bid_sell=sell_bid,
            mid_price=mid_price,
            spread=spread,
            expected_edge_bps=edge_bps,
        )

    async def scan_symbol(self, symbol: str) -> list[ArbitrageOpportunity]:
        exchange_names = list(self.adapters.keys())
        if self.safety_monitor is not None:
            exchange_names = [name for name in exchange_names if name not in self.safety_monitor.state.blocked_exchanges]
            if self.safety_monitor.state.auto_disable_arbitrage:
                logger.warning('safety_auto_disable_arbitrage_enabled', symbol=symbol)
                return []
        books, _ = await self._fetch_books_and_tickers(symbol, exchange_names)
        opportunities: list[ArbitrageOpportunity] = []

        for exchange_a, exchange_b in combinations(exchange_names, 2):
            bid_a, ask_a = self._best_prices(books[exchange_a])
            bid_b, ask_b = self._best_prices(books[exchange_b])
            if None in {bid_a, ask_a, bid_b, ask_b}:
                continue
            assert bid_a is not None and ask_a is not None and bid_b is not None and ask_b is not None

            for sell_exchange, buy_exchange, sell_bid, buy_ask in (
                (exchange_a, exchange_b, bid_a, ask_b),
                (exchange_b, exchange_a, bid_b, ask_a),
            ):
                opportunity = self._build_opportunity(symbol, sell_exchange, buy_exchange, sell_bid, buy_ask)
                if opportunity is None:
                    continue
                logger.info(
                    'arbitrage_opportunity_detected',
                    symbol=opportunity.symbol,
                    exchange_pair=f'{opportunity.sell_exchange}->{opportunity.buy_exchange}',
                    spread=opportunity.spread,
                    expected_edge_bps=opportunity.expected_edge_bps,
                )
                opportunities.append(opportunity)

        opportunities.sort(key=lambda item: item.expected_edge_bps, reverse=True)
        return opportunities

    async def scan(self, symbols: list[str]) -> list[ArbitrageOpportunity]:
        all_opportunities: list[ArbitrageOpportunity] = []
        for symbol in symbols:
            all_opportunities.extend(await self.scan_symbol(symbol))
        return sorted(all_opportunities, key=lambda item: item.expected_edge_bps, reverse=True)

    async def execute_opportunity(self, opportunity: ArbitrageOpportunity, amount: float) -> ExecutionReport:
        buy_adapter = self.adapters[opportunity.buy_exchange]
        sell_adapter = self.adapters[opportunity.sell_exchange]

        buy_order, sell_order = await asyncio.gather(
            buy_adapter.create_order(opportunity.symbol, side='buy', amount=amount),
            sell_adapter.create_order(opportunity.symbol, side='sell', amount=amount),
        )

        return ExecutionReport(
            symbol=opportunity.symbol,
            buy_exchange=opportunity.buy_exchange,
            sell_exchange=opportunity.sell_exchange,
            buy_order_id=str(buy_order.get('id')) if buy_order else None,
            sell_order_id=str(sell_order.get('id')) if sell_order else None,
            status='submitted',
            spread=opportunity.spread,
            expected_edge_bps=opportunity.expected_edge_bps,
            details={'buy_order': buy_order, 'sell_order': sell_order},
        )

    async def execute_with_risk_controls(
        self,
        opportunity: ArbitrageOpportunity,
        *,
        risk_manager: InstitutionalRiskManager,
        allocator: CapitalAllocator,
        context: OpportunityContext,
    ) -> ExecutionReport:
        capital_ticket = None
        if self.capital_governor is not None:
            capital_ticket = self.capital_governor.issue_ticket(
                strategy='arbitrage',
                exchange=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                requested_notional=max(opportunity.mid_price, 0.0),
                pnl_or_returns=context.returns_matrix.get(opportunity.symbol, []),
                spread_bps=max(opportunity.expected_edge_bps, 0.0),
                available_liquidity=max(opportunity.mid_price * 2.0, 1e-9),
                price_gap_pct=max(context.annualized_volatility / 252.0, 0.0),
            )
            valid_ticket, ticket_reason = self.capital_governor.validate_ticket(capital_ticket)
            if not valid_ticket:
                return ExecutionReport(
                    symbol=opportunity.symbol,
                    buy_exchange=opportunity.buy_exchange,
                    sell_exchange=opportunity.sell_exchange,
                    buy_order_id=None,
                    sell_order_id=None,
                    status='rejected_governor',
                    spread=opportunity.spread,
                    expected_edge_bps=opportunity.expected_edge_bps,
                    details={'reason': ticket_reason},
                )

        risk_assessment = risk_manager.assess(
            symbol=opportunity.symbol,
            side='BUY',
            equity=context.equity,
            daily_pnl=context.daily_pnl,
            current_price=opportunity.mid_price,
            atr=context.atr,
            annualized_volatility=context.annualized_volatility,
            volatility_multiplier=context.volatility_multiplier,
            expected_win_rate=context.expected_win_rate,
            avg_win=context.avg_win,
            avg_loss=context.avg_loss,
            returns_matrix=context.returns_matrix,
            exchange=opportunity.buy_exchange,
            notional_by_exchange=context.notionals_by_exchange,
            total_exposure=context.total_exposure,
            capital_ticket=capital_ticket,
        )
        if not risk_assessment.allowed:
            return ExecutionReport(
                symbol=opportunity.symbol,
                buy_exchange=opportunity.buy_exchange,
                sell_exchange=opportunity.sell_exchange,
                buy_order_id=None,
                sell_order_id=None,
                status='rejected_risk',
                spread=opportunity.spread,
                expected_edge_bps=opportunity.expected_edge_bps,
                details={'reason': risk_assessment.reason},
            )

        allocation = allocator.allocate(
            AllocationRequest(
                symbol=opportunity.symbol,
                buy_exchange=opportunity.buy_exchange,
                sell_exchange=opportunity.sell_exchange,
                mid_price=opportunity.mid_price,
                expected_edge_bps=opportunity.expected_edge_bps,
            ),
            equity=context.equity,
            inventory_by_exchange=context.inventory_by_exchange,
            notionals_by_exchange=context.notionals_by_exchange,
            notionals_by_asset=context.notionals_by_asset,
        )

        if not allocation.allowed:
            return ExecutionReport(
                symbol=opportunity.symbol,
                buy_exchange=opportunity.buy_exchange,
                sell_exchange=opportunity.sell_exchange,
                buy_order_id=None,
                sell_order_id=None,
                status='rejected_allocator',
                spread=opportunity.spread,
                expected_edge_bps=opportunity.expected_edge_bps,
                details={'reason': allocation.reason, 'allocator_debug': allocation.debug},
            )

        if self.safety_monitor is not None:
            if self.safety_monitor.state.auto_disable_arbitrage:
                return ExecutionReport(
                    symbol=opportunity.symbol,
                    buy_exchange=opportunity.buy_exchange,
                    sell_exchange=opportunity.sell_exchange,
                    buy_order_id=None,
                    sell_order_id=None,
                    status='rejected_safety',
                    spread=opportunity.spread,
                    expected_edge_bps=opportunity.expected_edge_bps,
                    details={'reason': 'auto_disable_arbitrage'},
                )

            capital_fraction = min(
                self.safety_monitor.allowed_capital_fraction(opportunity.buy_exchange),
                self.safety_monitor.allowed_capital_fraction(opportunity.sell_exchange),
            )
            if capital_fraction <= 0:
                return ExecutionReport(
                    symbol=opportunity.symbol,
                    buy_exchange=opportunity.buy_exchange,
                    sell_exchange=opportunity.sell_exchange,
                    buy_order_id=None,
                    sell_order_id=None,
                    status='rejected_safety',
                    spread=opportunity.spread,
                    expected_edge_bps=opportunity.expected_edge_bps,
                    details={'reason': 'exchange_blocked_by_safety'},
                )
        else:
            capital_fraction = 1.0

        amount = min(risk_assessment.position_size, allocation.units) * capital_fraction
        if amount <= 0:
            return ExecutionReport(
                symbol=opportunity.symbol,
                buy_exchange=opportunity.buy_exchange,
                sell_exchange=opportunity.sell_exchange,
                buy_order_id=None,
                sell_order_id=None,
                status='rejected_execution',
                spread=opportunity.spread,
                expected_edge_bps=opportunity.expected_edge_bps,
                details={'reason': 'amount_capped_to_zero'},
            )

        return await self.execute_opportunity(opportunity, amount=amount)

    async def close(self) -> None:
        await asyncio.gather(*(adapter.close() for adapter in self.adapters.values()))
