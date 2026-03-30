"""
Test utilities for Reco-Trading
"""

import asyncio
import logging
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Base test case with common utilities."""
    name: str
    passed: bool = True
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class TestSuite:
    """Test suite container."""
    name: str
    cases: list[TestCase] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.cases if not c.passed)

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0


class TestRunner:
    """Test runner for executing test suites."""

    def __init__(self) -> None:
        self.suites: list[TestSuite] = []
        self.logger = logging.getLogger(__name__)

    def add_suite(self, suite: TestSuite) -> None:
        self.suites.append(suite)

    async def run_suite(self, suite: TestSuite) -> TestSuite:
        """Run a test suite."""
        suite.start_time = datetime.now(timezone.utc)
        self.logger.info(f"Running test suite: {suite.name}")

        for case in suite.cases:
            start = datetime.now(timezone.utc)
            try:
                if case.error is None:
                    case.passed = True
                else:
                    case.passed = False
            except Exception as e:
                case.passed = False
                case.error = str(e)
            case.duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        suite.end_time = datetime.now(timezone.utc)
        return suite

    def get_summary(self) -> dict[str, Any]:
        """Get test summary."""
        total_passed = sum(s.passed for s in self.suites)
        total_failed = sum(s.failed for s in self.suites)
        total_tests = total_passed + total_failed

        return {
            "suites": len(self.suites),
            "tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "duration_ms": sum(s.duration_ms for s in self.suites),
        }


def create_mock_dataframe(rows: int = 100) -> pd.DataFrame:
    """Create a mock OHLCV dataframe for testing."""
    dates = pd.date_range(start="2024-01-01", periods=rows, freq="5min", tz="UTC")
    base_price = 50000.0

    data = {
        "timestamp": dates,
        "open": base_price + np.cumsum(np.random.randn(rows) * 10),
        "high": base_price + np.cumsum(np.random.randn(rows) * 10) + abs(np.random.rand(rows) * 50),
        "low": base_price + np.cumsum(np.random.randn(rows) * 10) - abs(np.random.rand(rows) * 50),
        "close": base_price + np.cumsum(np.random.randn(rows) * 10),
        "volume": np.random.rand(rows) * 1000 + 500,
    }

    df = pd.DataFrame(data)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def create_mock_ticker(price: float = 50000.0) -> dict[str, Any]:
    """Create a mock ticker for testing."""
    return {
        "symbol": "BTC/USDT",
        "last": price,
        "bid": price - 1.0,
        "ask": price + 1.0,
        "volume": 1000.0,
        "percentage": 2.5,
        "timestamp": datetime.now(timezone.utc).timestamp() * 1000,
    }


def create_mock_orderbook(price: float = 50000.0, depth: int = 10) -> dict[str, Any]:
    """Create a mock orderbook for testing."""
    bids = [[price - i * 0.5, 10 - i] for i in range(depth)]
    asks = [[price + i * 0.5, 10 - i] for i in range(depth)]

    return {
        "bids": bids,
        "asks": asks,
        "timestamp": datetime.now(timezone.utc).timestamp() * 1000,
    }


def create_mock_trade(
    side: str = "buy",
    price: float = 50000.0,
    qty: float = 0.01,
) -> dict[str, Any]:
    """Create a mock trade for testing."""
    return {
        "id": "test-trade-123",
        "symbol": "BTC/USDT",
        "type": "market",
        "side": side,
        "price": price,
        "amount": qty,
        "cost": price * qty,
        "fee": {"cost": price * qty * 0.001, "currency": "USDT"},
        "timestamp": datetime.now(timezone.utc).timestamp() * 1000,
    }


class AsyncTestCase:
    """Base class for async test cases."""

    async def setup(self) -> None:
        """Setup before each test."""
        pass

    async def teardown(self) -> None:
        """Teardown after each test."""
        pass

    async def run(self) -> tuple[bool, str | None]:
        """Run the test. Returns (passed, error_message)."""
        try:
            await self.setup()
            await self.execute()
            await self.teardown()
            return True, None
        except Exception as e:
            return False, str(e)

    async def execute(self) -> None:
        """Override this method to implement the test."""
        raise NotImplementedError


class HealthCheckResult:
    """Result of a health check."""

    def __init__(
        self,
        name: str,
        healthy: bool,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.healthy = healthy
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "healthy": self.healthy,
            "message": self.message,
            "details": self.details,
        }


class HealthChecker:
    """Health checker for system components."""

    def __init__(self) -> None:
        self.checks: list[Callable[[], HealthCheckResult]] = []

    def register_check(self, name: str, check_fn: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        self.checks.append(check_fn)

    async def check_all(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        results = []
        for check in self.checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    name=getattr(check, "__name__", "unknown"),
                    healthy=False,
                    message=f"Check failed: {e}",
                ))
        return results

    def is_healthy(self) -> bool:
        """Check if all components are healthy."""
        results = asyncio.get_event_loop().run_until_complete(self.check_all())
        return all(r.healthy for r in results)


def mock_async_function(return_value: Any, raises: Exception | None = None):
    """Create a mock async function."""
    async def mock(*args, **kwargs):
        if raises:
            raise raises
        return return_value
    return mock


__all__ = [
    "TestCase",
    "TestSuite", 
    "TestRunner",
    "create_mock_dataframe",
    "create_mock_ticker",
    "create_mock_orderbook",
    "create_mock_trade",
    "AsyncTestCase",
    "HealthCheckResult",
    "HealthChecker",
    "mock_async_function",
]
