from __future__ import annotations

import os
from dataclasses import asdict

import asyncpg
from sqlalchemy import JSON, BigInteger, Column, DateTime, MetaData, Numeric, String, Table, func, insert
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from loguru import logger

metadata = MetaData()

orders = Table(
    'orders',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('exchange_order_id', String, unique=True, nullable=False),
    Column('symbol', String, nullable=False),
    Column('side', String, nullable=False),
    Column('price', Numeric),
    Column('amount', Numeric, nullable=False),
    Column('status', String, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

fills = Table(
    'fills',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('exchange_order_id', String, nullable=False),
    Column('symbol', String, nullable=False),
    Column('side', String, nullable=False),
    Column('fill_price', Numeric),
    Column('fill_amount', Numeric),
    Column('fee', Numeric),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

portfolio_state = Table(
    'portfolio_state',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('snapshot', JSON, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)


class Database:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.engine = create_async_engine(dsn, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self) -> None:
        await self._ensure_database_exists()
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def _ensure_database_exists(self) -> None:
        url = make_url(self.dsn)
        if not url.drivername.startswith('postgresql') or not url.database:
            return

        admin_url = self._to_asyncpg_url(url.set(database='postgres'))
        db_name = url.database

        conn = await asyncpg.connect(admin_url)
        try:
            exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', db_name)
            if exists:
                return
            escaped_db_name = db_name.replace('"', '""')
            await conn.execute(f'CREATE DATABASE "{escaped_db_name}"')
            logger.info(f'Base de datos "{db_name}" creada automáticamente.')
        finally:
            await conn.close()

    @staticmethod
    def _to_asyncpg_url(url: URL) -> str:
        return str(url.set(drivername='postgresql'))

    async def record_order(self, order: dict) -> None:
        payload = {
            'exchange_order_id': str(order.get('id')),
            'symbol': order.get('symbol', ''),
            'side': str(order.get('side', '')).upper(),
            'price': order.get('price') or order.get('average') or 0,
            'amount': order.get('amount') or order.get('filled') or 0,
            'status': order.get('status', 'unknown'),
        }
        async with self.session_factory() as session:
            await session.execute(insert(orders).values(**payload))
            await session.commit()

    async def record_fill(self, order: dict) -> None:
        fee = order.get('fee') if isinstance(order.get('fee'), dict) else {}
        payload = {
            'exchange_order_id': str(order.get('id')),
            'symbol': order.get('symbol', ''),
            'side': str(order.get('side', '')).upper(),
            'fill_price': order.get('average') or order.get('price') or 0,
            'fill_amount': order.get('filled') or order.get('amount') or 0,
            'fee': float(fee.get('cost') or 0.0),
        }
        async with self.session_factory() as session:
            await session.execute(insert(fills).values(**payload))
            await session.commit()

    async def snapshot_portfolio(self, state) -> None:
        async with self.session_factory() as session:
            await session.execute(insert(portfolio_state).values(snapshot=asdict(state)))
            await session.commit()

    async def fill_stats(self) -> dict:
        async with self.session_factory() as session:
            buy_spent = await session.scalar(
                fills.select()
                .with_only_columns(func.coalesce(func.sum(fills.c.fill_price * fills.c.fill_amount), 0))
                .where(fills.c.side == 'BUY')
            )
            sell_earned = await session.scalar(
                fills.select()
                .with_only_columns(func.coalesce(func.sum(fills.c.fill_price * fills.c.fill_amount), 0))
                .where(fills.c.side == 'SELL')
            )
            fees_paid = await session.scalar(fills.select().with_only_columns(func.coalesce(func.sum(fills.c.fee), 0)))
            total_fills = await session.scalar(fills.select().with_only_columns(func.count(fills.c.id)))

        spent = float(buy_spent or 0.0)
        earned = float(sell_earned or 0.0)
        fees = float(fees_paid or 0.0)
        return {
            'spent_usdt': spent,
            'earned_usdt': earned,
            'fees_usdt': fees,
            'realized_pnl_usdt': earned - spent - fees,
            'total_fills': int(total_fills or 0),
        }

    async def close(self) -> None:
        await self.engine.dispose()
