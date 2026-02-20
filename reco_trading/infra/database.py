from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict
from typing import Any, Literal

from sqlalchemy import JSON, BigInteger, Column, DateTime, ForeignKey, MetaData, Numeric, String, Table, Text, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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

ConfigStatus = Literal['pending', 'active', 'failed', 'rolled_back']


class Database:
    def __init__(self, dsn: str, admin_dsn: str | None = None) -> None:
        self.dsn = dsn
        self.admin_dsn = admin_dsn
        self.engine = create_async_engine(dsn, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _valid_signature(payload_hash: str, signature: str) -> bool:
        expected = hmac.new(b'reco_trading', payload_hash.encode('utf-8'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    async def create_config_version(
        self,
        version: str,
        payload: dict[str, Any],
        signature: str,
        *,
        reason: str | None = None,
        metadata_payload: dict[str, Any] | None = None,
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
        return version_id

    async def activate_config_version(self, version: str, *, reason: str | None = None) -> None:
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

    async def register_config_failure(self, version: str, reason: str) -> None:
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

    async def register_deployment(self, version_id: int, status: ConfigStatus = 'pending', reason: str | None = None) -> int:
        async with self.session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    insert(system_deployments)
                    .values(version_id=version_id, status=status, reason=reason)
                    .returning(system_deployments.c.id)
                )
                return int(result.scalar_one())

    async def complete_deployment(self, deployment_id: int, status: ConfigStatus, reason: str | None = None) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(system_deployments)
                    .where(system_deployments.c.id == deployment_id)
                    .values(status=status, reason=reason, completed_at=func.now())
                )

    async def execute_rollback(self, from_version: str, reason: str, *, deployment_id: int | None = None) -> str:
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
        return to_version

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
