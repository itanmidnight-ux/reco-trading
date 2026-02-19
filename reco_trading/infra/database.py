from __future__ import annotations

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


class Database:
    def __init__(self, dsn: str) -> None:
        self.engine = create_async_engine(dsn, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    def session(self) -> AsyncSession:
        return self.session_factory()
