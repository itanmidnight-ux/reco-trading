from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    passed: bool
    error: str | None = None
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class TestRunner:
    def __init__(self) -> None:
        self.results: list[TestResult] = []
        self._test_functions: dict[str, Any] = {}

    def register(self, name: str):
        def decorator(func):
            self._test_functions[name] = func
            return func

        return decorator

    async def run_all(self, verbose: bool = True) -> bool:
        logger.info(f"Running {len(self._test_functions)} tests...")
        self.results.clear()

        for name, func in self._test_functions.items():
            start = datetime.now(timezone.utc)
            result = TestResult(name=name, passed=False)

            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()

                result.passed = True
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                result.duration_ms = duration

                if verbose:
                    logger.info(f"✓ {name} ({duration:.1f}ms)")

            except Exception as exc:  # noqa: BLE001
                result.passed = False
                result.error = str(exc)
                duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000
                result.duration_ms = duration

                if verbose:
                    logger.error(f"✗ {name}: {exc}")

            self.results.append(result)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        logger.info(f"Results: {passed} passed, {failed} failed")

        return failed == 0

    def get_summary(self) -> dict[str, Any]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "total_duration_ms": sum(r.duration_ms for r in self.results),
            "failures": [{"name": r.name, "error": r.error} for r in self.results if not r.passed],
        }


class ModuleValidator:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    async def validate_imports(self, module_path: str) -> bool:
        try:
            __import__(module_path)
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.error(f"Import validation failed for {module_path}: {exc}")
            return False

    async def validate_core_modules(self) -> dict[str, bool]:
        modules = [
            "reco_trading.core.bot_engine",
            "reco_trading.core.error_handler",
            "reco_trading.core.debugging",
            "reco_trading.risk.risk_manager",
            "reco_trading.strategy.signal_engine",
            "reco_trading.exchange.binance_client",
            "reco_trading.database.repository",
        ]

        results = {}
        for module in modules:
            results[module] = await self.validate_imports(module)

        return results


async def run_validation() -> bool:
    logger.info("Starting auto-validation...")

    validator = ModuleValidator()
    results = await validator.validate_core_modules()

    all_valid = all(results.values())

    for module, valid in results.items():
        status = "✓" if valid else "✗"
        logger.info(f"{status} {module}")

    if all_valid:
        logger.info("All core modules validated successfully")
    else:
        logger.error("Some modules failed validation")

    return all_valid


async def run_tests() -> bool:
    runner = TestRunner()

    @runner.register("test_imports")
    def test_imports():
        import reco_trading
        import reco_trading.core
        import reco_trading.risk

    @runner.register("test_config")
    def test_config():
        from reco_trading.config.settings import Settings

        settings = Settings()
        assert settings is not None

    @runner.register("test_logger")
    def test_logger():
        log = logging.getLogger("test")
        log.info("Test message")

    await runner.run_all()

    summary = runner.get_summary()
    logger.info(f"Test summary: {summary}")

    return summary["failed"] == 0


if __name__ == "__main__":
    async def main():
        valid = await run_validation()
        if not valid:
            sys.exit(1)

        tests_passed = await run_tests()
        if not tests_passed:
            sys.exit(1)

    asyncio.run(main())
