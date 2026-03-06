from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import JSON, BigInteger, Column, DateTime, ForeignKey, MetaData, Numeric, String, Table, Text, func, insert, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
    Column('decision_id', String),
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
    Column('order_id', BigInteger),
    Column('decision_id', String),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

portfolio_state = Table(
    'portfolio_state',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('snapshot', JSON, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

validation_history = Table(
    'validation_history',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('snapshot', JSON, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

candles = Table(
    'candles',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('symbol', String, nullable=False),
    Column('interval', String, nullable=False),
    Column('ts', BigInteger, nullable=False),
    Column('open', Numeric, nullable=False),
    Column('high', Numeric, nullable=False),
    Column('low', Numeric, nullable=False),
    Column('close', Numeric, nullable=False),
    Column('volume', Numeric, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

trade_signals = Table(
    'trade_signals',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('ts', BigInteger, nullable=False),
    Column('symbol', String, nullable=False),
    Column('signal', String, nullable=False),
    Column('score', Numeric, nullable=False),
    Column('expected_value', Numeric, nullable=False),
    Column('reason', Text, nullable=False, server_default=''),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

order_executions = Table(
    'order_executions',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('ts', BigInteger, nullable=False),
    Column('symbol', String, nullable=False),
    Column('side', String, nullable=False),
    Column('qty', Numeric, nullable=False),
    Column('price', Numeric, nullable=False),
    Column('status', String, nullable=False),
    Column('exchange_order_id', String),
    Column('pnl', Numeric, nullable=False, server_default='0'),
    Column('order_id', BigInteger),
    Column('decision_id', String),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

execution_idempotency_ledger = Table(
    'execution_idempotency_ledger',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('client_order_id', String, unique=True, nullable=False),
    Column('symbol', String, nullable=False),
    Column('side', String, nullable=False),
    Column('qty', Numeric, nullable=False),
    Column('status', String, nullable=False),
    Column('exchange_order_id', String, nullable=False, server_default=''),
    Column('decision_id', String),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column('updated_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

capital_reservations = Table(
    'capital_reservations',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('reservation_id', String, unique=True, nullable=False),
    Column('client_order_id', String, unique=True, nullable=False),
    Column('symbol', String, nullable=False),
    Column('side', String, nullable=False),
    Column('reserved_amount', Numeric, nullable=False),
    Column('used_amount', Numeric, nullable=False, server_default='0'),
    Column('status', String, nullable=False, server_default='active'),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column('updated_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

decision_audit = Table(
    'decision_audit',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('decision_id', String, unique=True, nullable=False),
    Column('snapshot', JSON, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

system_config_versions = Table(
    'system_config_versions',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('version', String, unique=True, nullable=False),
    Column('config_hash', String, nullable=False),
    Column('signature', Text, nullable=False),
    Column('status', String, nullable=False, server_default='pending'),
    Column('reason', Text),
    Column('metadata', JSON, nullable=False, server_default='{}'),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column('activated_at', DateTime(timezone=True)),
    Column('failed_at', DateTime(timezone=True)),
    Column('rolled_back_at', DateTime(timezone=True)),
    Column('updated_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

system_config_snapshots = Table(
    'system_config_snapshots',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('version_id', BigInteger, ForeignKey('system_config_versions.id', ondelete='CASCADE'), nullable=False),
    Column('snapshot_hash', String, nullable=False),
    Column('snapshot_signature', Text, nullable=False),
    Column('snapshot_payload', JSON, nullable=False),
    Column('reason', Text),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

system_deployments = Table(
    'system_deployments',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('version_id', BigInteger, ForeignKey('system_config_versions.id', ondelete='CASCADE'), nullable=False),
    Column('status', String, nullable=False),
    Column('reason', Text),
    Column('deployed_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column('completed_at', DateTime(timezone=True)),
)

system_rollbacks = Table(
    'system_rollbacks',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('deployment_id', BigInteger, ForeignKey('system_deployments.id', ondelete='SET NULL')),
    Column('from_version_id', BigInteger, ForeignKey('system_config_versions.id', ondelete='CASCADE'), nullable=False),
    Column('to_version_id', BigInteger, ForeignKey('system_config_versions.id', ondelete='CASCADE'), nullable=False),
    Column('status', String, nullable=False),
    Column('reason', Text, nullable=False),
    Column('triggered_by', String, nullable=False, server_default='automatic'),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column('completed_at', DateTime(timezone=True)),
)

security_audit_log = Table(
    'security_audit_log',
    metadata,
    Column('id', BigInteger, primary_key=True),
    Column('event_type', String, nullable=False),
    Column('actor', String, nullable=False),
    Column('target', String),
    Column('payload', JSON, nullable=False, server_default='{}'),
    Column('prev_event_hash', String),
    Column('event_hash', String, nullable=False),
    Column('created_at', DateTime(timezone=True), server_default=func.now(), nullable=False),
)

ConfigStatus = Literal['pending', 'active', 'failed', 'rolled_back']


class Database:
    def __init__(self, dsn: str, admin_dsn: str | None = None) -> None:
        self.dsn = dsn
        self.admin_dsn = admin_dsn
        self.engine = create_async_engine(dsn, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS decision_audit (
                    id BIGSERIAL PRIMARY KEY,
                    decision_id VARCHAR(64) NOT NULL UNIQUE,
                    snapshot JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS orders (
                    id BIGSERIAL PRIMARY KEY,
                    exchange_order_id VARCHAR(128) NOT NULL UNIQUE,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    price NUMERIC,
                    amount NUMERIC NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    decision_id VARCHAR(64),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fills (
                    id BIGSERIAL PRIMARY KEY,
                    exchange_order_id VARCHAR(128) NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    fill_price NUMERIC,
                    fill_amount NUMERIC,
                    fee NUMERIC,
                    order_id BIGINT,
                    decision_id VARCHAR(64),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS order_executions (
                    id BIGSERIAL PRIMARY KEY,
                    ts BIGINT NOT NULL,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    qty NUMERIC NOT NULL,
                    price NUMERIC NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    exchange_order_id VARCHAR(128),
                    pnl NUMERIC NOT NULL DEFAULT 0,
                    order_id BIGINT,
                    decision_id VARCHAR(64),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS execution_idempotency_ledger (
                    id BIGSERIAL PRIMARY KEY,
                    client_order_id VARCHAR(64) NOT NULL UNIQUE,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    qty NUMERIC NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    exchange_order_id VARCHAR(128) NOT NULL DEFAULT '',
                    decision_id VARCHAR(64),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS capital_reservations (
                    id BIGSERIAL PRIMARY KEY,
                    reservation_id VARCHAR(128) NOT NULL UNIQUE,
                    client_order_id VARCHAR(64) NOT NULL UNIQUE,
                    symbol VARCHAR(32) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    reserved_amount NUMERIC NOT NULL,
                    used_amount NUMERIC NOT NULL DEFAULT 0,
                    status VARCHAR(32) NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS validation_history (
                    id BIGSERIAL PRIMARY KEY,
                    snapshot JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """))
            await conn.execute(text('ALTER TABLE capital_reservations ADD COLUMN IF NOT EXISTS client_order_id VARCHAR(64)'))
            await conn.execute(
                text(
                    """
                    UPDATE capital_reservations
                    SET client_order_id = COALESCE(NULLIF(client_order_id, ''), reservation_id)
                    WHERE client_order_id IS NULL OR client_order_id = ''
                    """
                )
            )
            await conn.execute(text('CREATE UNIQUE INDEX IF NOT EXISTS ux_capital_reservations_client_order_id ON capital_reservations(client_order_id)'))
            await conn.execute(
                pg_insert(decision_audit)
                .values(decision_id='__bootstrap__', snapshot={'bootstrap': True})
                .on_conflict_do_nothing(index_elements=[decision_audit.c.decision_id])
            )
            await conn.execute(text('ALTER TABLE fills ADD COLUMN IF NOT EXISTS order_id BIGINT'))
            await conn.execute(text('ALTER TABLE order_executions ADD COLUMN IF NOT EXISTS order_id BIGINT'))
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'orders'
                          AND column_name = 'decision_id'
                    ) THEN
                        ALTER TABLE orders ADD COLUMN decision_id VARCHAR(64);
                    END IF;

                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_orders_decision_id') THEN
                        ALTER TABLE orders
                        ADD CONSTRAINT fk_orders_decision_id
                        FOREIGN KEY (decision_id) REFERENCES decision_audit(decision_id) ON DELETE SET NULL;
                    END IF;
                END
                $$;
            """))
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'fills'
                          AND column_name = 'order_id'
                    ) THEN
                        ALTER TABLE fills ADD COLUMN order_id BIGINT;
                    END IF;

                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_fills_order_id') THEN
                        ALTER TABLE fills
                        ADD CONSTRAINT fk_fills_order_id
                        FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE SET NULL;
                    END IF;
                END
                $$;
            """))
            await conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'order_executions'
                          AND column_name = 'order_id'
                    ) THEN
                        ALTER TABLE order_executions ADD COLUMN order_id BIGINT;
                    END IF;

                    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_order_executions_order_id') THEN
                        ALTER TABLE order_executions
                        ADD CONSTRAINT fk_order_executions_order_id
                        FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE SET NULL;
                    END IF;
                END
                $$;
            """))

    async def health_check(self) -> None:
        try:
            async with self.session_factory() as session:
                await session.execute(select(1))
        except Exception as exc:
            logger.exception('database_health_check_failed', error=str(exc))
            raise RuntimeError('Database health check failed: SELECT 1 did not return successfully.') from exc

    async def _execute_in_transaction(self, action: str, operation) -> Any:
        async with self.session_factory() as session:
            try:
                async with session.begin():
                    return await operation(session)
            except Exception:
                await session.rollback()
                logger.exception('database_transaction_error', action=action)
                raise

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _valid_signature(payload_hash: str, signature: str) -> bool:
        expected = hmac.new(b'reco_trading', payload_hash.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    @staticmethod
    def _audit_hash(event_type: str, actor: str, target: str | None, payload: dict[str, Any], prev_event_hash: str | None) -> str:
        body = json.dumps(
            {
                'event_type': event_type,
                'actor': actor,
                'target': target,
                'payload': payload,
                'prev_event_hash': prev_event_hash,
            },
            sort_keys=True,
            separators=(',', ':'),
        )
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    async def append_audit_event(
        self,
        *,
        event_type: str,
        actor: str,
        target: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        payload_data = payload or {}
        async with self.session_factory() as session:
            async with session.begin():
                prev_hash = await session.scalar(
                    select(security_audit_log.c.event_hash).order_by(security_audit_log.c.id.desc()).limit(1)
                )
                event_hash = self._audit_hash(event_type, actor, target, payload_data, prev_hash)
                result = await session.execute(
                    insert(security_audit_log)
                    .values(
                        event_type=event_type,
                        actor=actor,
                        target=target,
                        payload=payload_data,
                        prev_event_hash=prev_hash,
                        event_hash=event_hash,
                    )
                    .returning(security_audit_log.c.id)
                )
                return int(result.scalar_one())

    async def create_config_version(
        self,
        version: str,
        payload: dict[str, Any],
        signature: str,
        *,
        reason: str | None = None,
        metadata_payload: dict[str, Any] | None = None,
        actor: str = 'system',
    ) -> int:
        config_hash = self._payload_hash(payload)
        if not self._valid_signature(config_hash, signature):
            raise ValueError('Firma de configuración inválida.')

        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    insert(system_config_versions)
                    .values(
                        version=version,
                        config_hash=config_hash,
                        signature=signature,
                        status='pending',
                        reason=reason,
                        metadata=metadata_payload or {},
                    )
                    .returning(system_config_versions.c.id)
                )
                version_id = int(result.scalar_one())
                await session.execute(
                    insert(system_config_snapshots).values(
                        version_id=version_id,
                        snapshot_hash=config_hash,
                        snapshot_signature=signature,
                        snapshot_payload=payload,
                        reason=reason,
                    )
                )
                await self.append_audit_event(
                    event_type='config_change',
                    actor=actor,
                    target=version,
                    payload={'action': 'create_version', 'config_hash': config_hash, 'reason': reason},
                )
        return version_id

    async def activate_config_version(self, version: str, *, reason: str | None = None, actor: str = 'system') -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(system_config_versions)
                    .where(system_config_versions.c.status == 'active')
                    .values(status='rolled_back', reason='Replaced by newer active version', rolled_back_at=func.now(), updated_at=func.now())
                )
                result = await session.execute(
                    update(system_config_versions)
                    .where(system_config_versions.c.version == version)
                    .values(status='active', reason=reason, activated_at=func.now(), updated_at=func.now())
                    .returning(system_config_versions.c.id)
                )
                if result.scalar_one_or_none() is None:
                    raise ValueError(f'No existe la versión {version}.')
        await self.append_audit_event(
            event_type='config_change',
            actor=actor,
            target=version,
            payload={'action': 'activate', 'reason': reason},
        )

    async def register_config_failure(self, version: str, reason: str, *, actor: str = 'system') -> None:
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    update(system_config_versions)
                    .where(system_config_versions.c.version == version)
                    .values(status='failed', reason=reason, failed_at=func.now(), updated_at=func.now())
                    .returning(system_config_versions.c.id)
                )
                if result.scalar_one_or_none() is None:
                    raise ValueError(f'No existe la versión {version}.')
        await self.append_audit_event(
            event_type='security_event',
            actor=actor,
            target=version,
            payload={'action': 'config_failure', 'reason': reason},
        )

    async def register_deployment(
        self,
        version_id: int,
        status: ConfigStatus = 'pending',
        reason: str | None = None,
        *,
        actor: str = 'system',
        signature: str | None = None,
        deployment_hash: str | None = None,
    ) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    insert(system_deployments)
                    .values(version_id=version_id, status=status, reason=reason)
                    .returning(system_deployments.c.id)
                )
                deployment_id = int(result.scalar_one())
        await self.append_audit_event(
            event_type='security_event',
            actor=actor,
            target=str(version_id),
            payload={
                'action': 'register_deployment',
                'deployment_id': deployment_id,
                'status': status,
                'reason': reason,
                'signature': signature,
                'deployment_hash': deployment_hash,
            },
        )
        return deployment_id

    async def complete_deployment(
        self,
        deployment_id: int,
        status: ConfigStatus,
        reason: str | None = None,
        *,
        actor: str = 'system',
    ) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(system_deployments)
                    .where(system_deployments.c.id == deployment_id)
                    .values(status=status, reason=reason, completed_at=func.now())
                )
        await self.append_audit_event(
            event_type='security_event',
            actor=actor,
            target=str(deployment_id),
            payload={'action': 'complete_deployment', 'status': status, 'reason': reason},
        )

    async def execute_rollback(
        self,
        from_version: str,
        reason: str,
        *,
        deployment_id: int | None = None,
        actor: str = 'system',
    ) -> str:
        async with self.session_factory() as session:
            async with session.begin():
                from_row = (
                    await session.execute(
                        select(system_config_versions.c.id)
                        .where(system_config_versions.c.version == from_version)
                    )
                ).first()
                if from_row is None:
                    raise ValueError(f'No existe la versión {from_version}.')

                target_row = (
                    await session.execute(
                        select(system_config_versions.c.id, system_config_versions.c.version)
                        .where(
                            system_config_versions.c.status.in_(['active', 'rolled_back']),
                            system_config_versions.c.version != from_version,
                        )
                        .order_by(system_config_versions.c.activated_at.desc().nullslast(), system_config_versions.c.id.desc())
                        .limit(1)
                    )
                ).first()
                if target_row is None:
                    raise ValueError('No existe una versión previa para rollback.')

                from_id = int(from_row.id)
                to_id = int(target_row.id)
                to_version = str(target_row.version)

                await session.execute(
                    update(system_config_versions)
                    .where(system_config_versions.c.id == from_id)
                    .values(status='rolled_back', reason=reason, rolled_back_at=func.now(), updated_at=func.now())
                )
                await session.execute(
                    update(system_config_versions)
                    .where(system_config_versions.c.id == to_id)
                    .values(status='active', reason='Rollback target activated', activated_at=func.now(), updated_at=func.now())
                )
                await session.execute(
                    insert(system_rollbacks).values(
                        deployment_id=deployment_id,
                        from_version_id=from_id,
                        to_version_id=to_id,
                        status='active',
                        reason=reason,
                        triggered_by='automatic',
                        completed_at=func.now(),
                    )
                )
        await self.append_audit_event(
            event_type='recovery_action',
            actor=actor,
            target=from_version,
            payload={'action': 'rollback', 'reason': reason, 'deployment_id': deployment_id, 'rolled_back_to': to_version},
        )
        return to_version

    async def _resolve_order_id(self, session: AsyncSession, exchange_order_id: str) -> int | None:
        if not exchange_order_id:
            return None
        return await session.scalar(select(orders.c.id).where(orders.c.exchange_order_id == exchange_order_id).limit(1))

    async def record_order(self, order: dict) -> None:
        payload = {
            'exchange_order_id': str(order.get('id')),
            'symbol': order.get('symbol', ''),
            'side': str(order.get('side', '')).upper(),
            'price': order.get('price') or order.get('average') or 0,
            'amount': order.get('amount') or order.get('filled') or 0,
            'status': order.get('status', 'unknown'),
            'decision_id': order.get('decision_id'),
        }
        async def _op(session: AsyncSession) -> None:
            await session.execute(insert(orders).values(**payload))

        await self._execute_in_transaction('record_order', _op)

    async def record_fill(self, order: dict) -> None:
        fee = order.get('fee') if isinstance(order.get('fee'), dict) else {}
        exchange_order_id = str(order.get('id') or '')
        payload = {
            'exchange_order_id': exchange_order_id,
            'symbol': order.get('symbol', ''),
            'side': str(order.get('side', '')).upper(),
            'fill_price': order.get('average') or order.get('price') or 0,
            'fill_amount': order.get('filled') or order.get('amount') or 0,
            'fee': float(fee.get('cost') or 0.0),
            'decision_id': order.get('decision_id'),
        }
        async def _op(session: AsyncSession) -> None:
            payload['order_id'] = await self._resolve_order_id(session, exchange_order_id)
            await session.execute(insert(fills).values(**payload))

        await self._execute_in_transaction('record_fill', _op)

    async def snapshot_portfolio(self, state) -> None:
        async def _op(session: AsyncSession) -> None:
            await session.execute(insert(portfolio_state).values(snapshot=asdict(state)))

        await self._execute_in_transaction('snapshot_portfolio', _op)

    async def persist_validation_event(self, payload: dict[str, Any]) -> None:
        async def _op(session: AsyncSession) -> None:
            await session.execute(insert(validation_history).values(snapshot=payload))

        await self._execute_in_transaction('persist_validation_event', _op)

    async def persist_candle(self, payload: dict[str, Any]) -> None:
        async def _op(session: AsyncSession) -> None:
            await session.execute(insert(candles).values(**payload))

        await self._execute_in_transaction('persist_candle', _op)

    async def persist_trade_signal(self, payload: dict[str, Any]) -> None:
        async def _op(session: AsyncSession) -> None:
            await session.execute(insert(trade_signals).values(**payload))

        await self._execute_in_transaction('persist_trade_signal', _op)

    async def persist_order_execution(self, payload: dict[str, Any]) -> None:
        async def _op(session: AsyncSession) -> None:
            exchange_order_id = str(payload.get('exchange_order_id') or '')
            row = dict(payload)
            row['order_id'] = await self._resolve_order_id(session, exchange_order_id)
            await session.execute(insert(order_executions).values(**row))

        await self._execute_in_transaction('persist_order_execution', _op)

    async def persist_decision_snapshot(self, decision_id: str, payload: dict[str, Any]) -> None:
        async def _op(session: AsyncSession) -> None:
            stmt = (
                pg_insert(decision_audit)
                .values(decision_id=decision_id, snapshot=payload)
                .on_conflict_do_update(index_elements=[decision_audit.c.decision_id], set_={'snapshot': payload})
            )
            await session.execute(stmt)

        await self._execute_in_transaction('persist_decision_snapshot', _op)

    async def upsert_idempotency_ledger(self, payload: dict[str, Any]) -> None:
        data = {
            'client_order_id': str(payload.get('client_order_id') or ''),
            'symbol': str(payload.get('symbol') or ''),
            'side': str(payload.get('side') or '').upper(),
            'qty': float(payload.get('qty') or 0.0),
            'status': str(payload.get('status') or 'PENDING_SUBMIT'),
            'exchange_order_id': str(payload.get('exchange_order_id') or ''),
            'decision_id': payload.get('decision_id'),
            'updated_at': datetime.now(timezone.utc),
        }

        async def _op(session: AsyncSession) -> None:
            stmt = (
                pg_insert(execution_idempotency_ledger)
                .values(**data)
                .on_conflict_do_update(
                    index_elements=[execution_idempotency_ledger.c.client_order_id],
                    set_={
                        'status': data['status'],
                        'exchange_order_id': data['exchange_order_id'],
                        'qty': data['qty'],
                        'decision_id': data['decision_id'],
                        'updated_at': data['updated_at'],
                    },
                )
            )
            await session.execute(stmt)

        await self._execute_in_transaction('upsert_idempotency_ledger', _op)

    async def update_idempotency_status(self, client_order_id: str, status: str, exchange_order_id: str = '') -> None:
        async def _op(session: AsyncSession) -> None:
            await session.execute(
                update(execution_idempotency_ledger)
                .where(execution_idempotency_ledger.c.client_order_id == client_order_id)
                .values(status=status, exchange_order_id=exchange_order_id, updated_at=datetime.now(timezone.utc))
            )

        await self._execute_in_transaction('update_idempotency_status', _op)

    async def load_active_idempotency_entries(self) -> list[dict[str, Any]]:
        active = ('PENDING_SUBMIT', 'SUBMITTED', 'PARTIALLY_FILLED')
        async with self.session_factory() as session:
            rows = await session.execute(
                select(execution_idempotency_ledger).where(execution_idempotency_ledger.c.status.in_(active))
            )
        return [dict(r._mapping) for r in rows]

    async def reserve_capital(self, client_order_id: str, symbol: str, side: str, reserved_amount: float) -> None:
        payload = {
            'reservation_id': client_order_id,
            'client_order_id': client_order_id,
            'symbol': symbol,
            'side': side,
            'reserved_amount': float(max(reserved_amount, 0.0)),
            'status': 'active',
            'updated_at': datetime.now(timezone.utc),
        }

        async def _op(session: AsyncSession) -> None:
            stmt = (
                pg_insert(capital_reservations)
                .values(**payload)
                .on_conflict_do_nothing(index_elements=[capital_reservations.c.client_order_id])
            )
            await session.execute(stmt)

        await self._execute_in_transaction('reserve_capital', _op)

    async def finalize_capital_reservation(self, client_order_id: str, used_amount: float, status: str) -> None:
        async def _op(session: AsyncSession) -> None:
            await session.execute(
                update(capital_reservations)
                .where(capital_reservations.c.client_order_id == client_order_id)
                .values(used_amount=float(max(used_amount, 0.0)), status=status, updated_at=datetime.now(timezone.utc))
            )

        await self._execute_in_transaction('finalize_capital_reservation', _op)

    async def cleanup_stale_reservations(
        self,
        open_orders: list[dict[str, Any]] | None = None,
        active_client_order_ids: set[str] | None = None,
    ) -> int:
        open_orders = open_orders or []
        open_client_ids: set[str] = {
            str(item.get('clientOrderId') or item.get('client_order_id') or '')
            for item in open_orders
            if str(item.get('clientOrderId') or item.get('client_order_id') or '')
        }
        active_client_order_ids = active_client_order_ids or set()

        async with self.session_factory() as session:
            rows = await session.execute(
                select(
                    capital_reservations.c.reservation_id,
                    capital_reservations.c.client_order_id,
                ).where(func.lower(capital_reservations.c.status) == 'active')
            )
            stale_ids: list[str] = []
            for row in rows:
                client_order_id = str(row.client_order_id or row.reservation_id or '')
                if not client_order_id:
                    continue
                if client_order_id in open_client_ids:
                    continue
                if client_order_id in active_client_order_ids:
                    continue
                stale_ids.append(client_order_id)

        for client_order_id in stale_ids:
            await self.finalize_capital_reservation(client_order_id, 0.0, 'released')
        return len(stale_ids)

    @asynccontextmanager
    async def execution_advisory_lock(self, lock_key: int = 741852) -> Any:
        async with self.session_factory() as session:
            acquired = bool(await session.scalar(select(func.pg_try_advisory_lock(lock_key))))
            try:
                yield acquired
            finally:
                if acquired:
                    await session.scalar(select(func.pg_advisory_unlock(lock_key)))

    async def restore_position_snapshot(self, symbol: str) -> dict[str, float]:
        async with self.session_factory() as session:
            rows = await session.execute(
                select(fills.c.side, fills.c.fill_amount, fills.c.fill_price, fills.c.created_at)
                .where(fills.c.symbol == symbol)
                .order_by(fills.c.created_at.asc(), fills.c.id.asc())
            )
        position_qty = 0.0
        avg_entry = 0.0
        for row in rows:
            side = str(row.side or '').upper()
            qty = float(row.fill_amount or 0.0)
            px = float(row.fill_price or 0.0)
            if qty <= 0.0:
                continue
            if side == 'BUY':
                new_qty = position_qty + qty
                if new_qty <= 0.0:
                    continue
                avg_entry = ((avg_entry * position_qty) + (px * qty)) / new_qty
                position_qty = new_qty
            elif side == 'SELL':
                position_qty = max(position_qty - qty, 0.0)
                if position_qty <= 0.0:
                    avg_entry = 0.0
        return {'net_qty': position_qty, 'avg_entry': avg_entry if position_qty > 0.0 else 0.0}

    async def get_realized_pnl_from_fills(self, symbol: str | None = None) -> dict[str, float]:
        sell_stmt = select(func.coalesce(func.sum(fills.c.fill_price * fills.c.fill_amount), 0)).where(fills.c.side == 'SELL')
        buy_stmt = select(func.coalesce(func.sum(fills.c.fill_price * fills.c.fill_amount), 0)).where(fills.c.side == 'BUY')
        fees_stmt = select(func.coalesce(func.sum(fills.c.fee), 0))
        if symbol:
            sell_stmt = sell_stmt.where(fills.c.symbol == symbol)
            buy_stmt = buy_stmt.where(fills.c.symbol == symbol)
            fees_stmt = fees_stmt.where(fills.c.symbol == symbol)

        async with self.session_factory() as session:
            sell_notional = await session.scalar(sell_stmt)
            buy_notional = await session.scalar(buy_stmt)
            fees_paid = await session.scalar(fees_stmt)

        sells = float(sell_notional or 0.0)
        buys = float(buy_notional or 0.0)
        fees = float(fees_paid or 0.0)
        return {
            'sell_notional': sells,
            'buy_notional': buys,
            'fees_paid': fees,
            'realized_pnl': sells - buys - fees,
        }

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
