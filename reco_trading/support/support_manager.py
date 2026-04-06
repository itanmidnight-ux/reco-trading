from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    hostname: str = ""
    platform_arch: str = ""


@dataclass
class DiagnosticResult:
    name: str
    status: str  # "ok", "warning", "error"
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SupportTicket:
    ticket_id: str
    title: str
    description: str
    priority: str  # "low", "medium", "high", "critical"
    category: str  # "bug", "feature", "question", "performance"
    system_info: SystemInfo
    diagnostics: list[DiagnosticResult]
    logs: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "open"


class SupportManager:
    def __init__(self, bot_instance: Any = None):
        self.bot = bot_instance
        self._diagnostics_cache: dict[str, DiagnosticResult] = {}
        self._last_diagnostic_run: float = 0
        self._diagnostic_interval: float = 300.0  # 5 minutes

    def get_system_info(self) -> SystemInfo:
        info = SystemInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            hostname=socket.gethostname(),
            platform_arch=platform.machine(),
        )
        
        try:
            import psutil
            mem = psutil.virtual_memory()
            info.memory_total_gb = round(mem.total / (1024 ** 3), 2)
            info.memory_available_gb = round(mem.available / (1024 ** 3), 2)
            
            disk = psutil.disk_usage("/")
            info.disk_total_gb = round(disk.total / (1024 ** 3), 2)
            info.disk_free_gb = round(disk.free / (1024 ** 3), 2)
        except Exception:
            pass
        
        return info

    async def run_diagnostics(self, force: bool = False) -> list[DiagnosticResult]:
        now = time.time()
        
        if not force and (now - self._last_diagnostic_run) < self._diagnostic_interval:
            return list(self._diagnostics_cache.values())
        
        results: list[DiagnosticResult] = []
        
        results.append(await self._check_database())
        results.append(await self._check_exchange_connection())
        results.append(await self._check_market_data())
        results.append(await self._check_positions())
        results.append(await self._check_risk_limits())
        results.append(await self._check_system_resources())
        results.append(await self._check_api_latency())
        results.append(await self._check_error_rate())
        
        for result in results:
            self._diagnostics_cache[result.name] = result
        
        self._last_diagnostic_run = now
        
        return results

    async def _check_database(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "repository"):
                repo = self.bot.repository
                if hasattr(repo, "get_trades"):
                    trades = await repo.get_trades(limit=1)
                    return DiagnosticResult(
                        name="database",
                        status="ok",
                        message="Database connection healthy",
                        details={"trades_count": len(trades)},
                    )
            return DiagnosticResult(
                name="database",
                status="warning",
                message="Database connection not available or no trades found",
            )
        except Exception as e:
            return DiagnosticResult(
                name="database",
                status="error",
                message=f"Database error: {str(e)}",
                details={"exception": str(e), "traceback": traceback.format_exc()},
            )

    async def _check_exchange_connection(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "client"):
                client = self.bot.client
                if hasattr(client, "fetch_balance"):
                    balance = await client.fetch_balance()
                    if balance:
                        return DiagnosticResult(
                            name="exchange",
                            status="ok",
                            message="Exchange connection healthy",
                            details={"has_balance": True},
                        )
            return DiagnosticResult(
                name="exchange",
                status="warning",
                message="Exchange connection status unknown",
            )
        except Exception as e:
            return DiagnosticResult(
                name="exchange",
                status="error",
                message=f"Exchange connection error: {str(e)}",
                details={"exception": str(e)},
            )

    async def _check_market_data(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "snapshot"):
                snapshot = self.bot.snapshot
                last_update = snapshot.get("last_market_update", 0)
                now = time.time()
                age_seconds = now - last_update if last_update else 999
                
                if age_seconds < 30:
                    return DiagnosticResult(
                        name="market_data",
                        status="ok",
                        message=f"Market data fresh ({age_seconds:.1f}s old)",
                        details={"age_seconds": age_seconds},
                    )
                elif age_seconds < 120:
                    return DiagnosticResult(
                        name="market_data",
                        status="warning",
                        message=f"Market data stale ({age_seconds:.1f}s old)",
                        details={"age_seconds": age_seconds},
                    )
                else:
                    return DiagnosticResult(
                        name="market_data",
                        status="error",
                        message=f"Market data very stale ({age_seconds:.1f}s old)",
                        details={"age_seconds": age_seconds},
                    )
            return DiagnosticResult(
                name="market_data",
                status="warning",
                message="Market data status unknown",
            )
        except Exception as e:
            return DiagnosticResult(
                name="market_data",
                status="error",
                message=f"Market data check error: {str(e)}",
                details={"exception": str(e)},
            )

    async def _check_positions(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "position_manager"):
                pm = self.bot.position_manager
                positions = getattr(pm, "positions", [])
                num_positions = len(positions) if positions else 0
                
                if num_positions == 0:
                    return DiagnosticResult(
                        name="positions",
                        status="ok",
                        message="No open positions",
                        details={"positions": 0},
                    )
                elif num_positions <= 2:
                    return DiagnosticResult(
                        name="positions",
                        status="ok",
                        message=f"{num_positions} open position(s)",
                        details={"positions": num_positions},
                    )
                else:
                    return DiagnosticResult(
                        name="positions",
                        status="warning",
                        message=f"Many open positions ({num_positions})",
                        details={"positions": num_positions},
                    )
            return DiagnosticResult(
                name="positions",
                status="ok",
                message="Position manager not available",
            )
        except Exception as e:
            return DiagnosticResult(
                name="positions",
                status="error",
                message=f"Position check error: {str(e)}",
                details={"exception": str(e)},
            )

    async def _check_risk_limits(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "snapshot"):
                snapshot = self.bot.snapshot
                
                issues: list[str] = []
                
                if snapshot.get("trading_paused_by_drawdown"):
                    issues.append("drawdown_pause_active")
                
                if snapshot.get("emergency_stop_active"):
                    issues.append("emergency_stop_active")
                
                if snapshot.get("user_paused"):
                    issues.append("user_paused")
                
                cooldown = snapshot.get("cooldown")
                if cooldown and cooldown not in ("READY", None, "NONE"):
                    issues.append(f"cooldown:{cooldown}")
                
                consecutive_losses = snapshot.get("consecutive_losses", 0)
                if consecutive_losses >= 3:
                    issues.append(f"consecutive_losses:{consecutive_losses}")
                
                if issues:
                    return DiagnosticResult(
                        name="risk_limits",
                        status="warning" if len(issues) <= 2 else "error",
                        message="Risk limits triggered",
                        details={"issues": issues},
                    )
                
                return DiagnosticResult(
                    name="risk_limits",
                    status="ok",
                    message="All risk limits OK",
                )
            
            return DiagnosticResult(
                name="risk_limits",
                status="ok",
                message="Risk limits OK",
            )
        except Exception as e:
            return DiagnosticResult(
                name="risk_limits",
                status="error",
                message=f"Risk check error: {str(e)}",
                details={"exception": str(e)},
            )

    async def _check_system_resources(self) -> DiagnosticResult:
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            issues: list[str] = []
            
            if cpu_percent > 90:
                issues.append(f"cpu_high:{cpu_percent:.1f}%")
            
            if memory.percent > 90:
                issues.append(f"memory_high:{memory.percent:.1f}%")
            
            if disk.percent > 95:
                issues.append(f"disk_high:{disk.percent:.1f}%")
            
            status = "ok"
            if issues:
                status = "warning" if len(issues) == 1 else "error"
            
            return DiagnosticResult(
                name="system_resources",
                status=status,
                message="System resources healthy" if status == "ok" else f"Resource issues: {', '.join(issues)}",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "issues": issues,
                },
            )
        except Exception as e:
            return DiagnosticResult(
                name="system_resources",
                status="ok",
                message="System resources check skipped (psutil unavailable)",
            )

    async def _check_api_latency(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "snapshot"):
                snapshot = self.bot.snapshot
                latency_ms = snapshot.get("api_latency_p95_ms", 0) or snapshot.get("api_latency_ms", 0)
                
                if latency_ms < 500:
                    return DiagnosticResult(
                        name="api_latency",
                        status="ok",
                        message=f"API latency healthy ({latency_ms:.0f}ms)",
                        details={"latency_ms": latency_ms},
                    )
                elif latency_ms < 2000:
                    return DiagnosticResult(
                        name="api_latency",
                        status="warning",
                        message=f"API latency elevated ({latency_ms:.0f}ms)",
                        details={"latency_ms": latency_ms},
                    )
                else:
                    return DiagnosticResult(
                        name="api_latency",
                        status="error",
                        message=f"API latency high ({latency_ms:.0f}ms)",
                        details={"latency_ms": latency_ms},
                    )
            
            return DiagnosticResult(
                name="api_latency",
                status="ok",
                message="API latency OK",
            )
        except Exception as e:
            return DiagnosticResult(
                name="api_latency",
                status="warning",
                message=f"API latency check error: {str(e)}",
            )

    async def _check_error_rate(self) -> DiagnosticResult:
        try:
            if self.bot and hasattr(self.bot, "observability"):
                obs = self.bot.observability
                error_count = getattr(obs, "error_count", 0)
                
                if error_count < 5:
                    return DiagnosticResult(
                        name="error_rate",
                        status="ok",
                        message=f"Error rate low ({error_count} errors)",
                        details={"error_count": error_count},
                    )
                elif error_count < 20:
                    return DiagnosticResult(
                        name="error_rate",
                        status="warning",
                        message=f"Error rate elevated ({error_count} errors)",
                        details={"error_count": error_count},
                    )
                else:
                    return DiagnosticResult(
                        name="error_rate",
                        status="error",
                        message=f"Error rate high ({error_count} errors)",
                        details={"error_count": error_count},
                    )
            
            return DiagnosticResult(
                name="error_rate",
                status="ok",
                message="Error rate OK",
            )
        except Exception as e:
            return DiagnosticResult(
                name="error_rate",
                status="ok",
                message="Error rate check skipped",
            )

    def create_support_ticket(
        self,
        title: str,
        description: str,
        priority: str = "medium",
        category: str = "bug",
        include_logs: bool = True,
        include_diagnostics: bool = True,
    ) -> SupportTicket:
        ticket_id = hashlib.sha256(
            f"{title}{time.time()}".encode()
        ).hexdigest()[:12]
        
        diagnostics = []
        if include_diagnostics:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    diagnostics = []
                else:
                    diagnostics = list(self._diagnostics_cache.values())
            except Exception:
                diagnostics = list(self._diagnostics_cache.values())
        
        logs: list[str] = []
        if include_logs:
            log_file = Path("logs") / "reco_trading.log"
            if log_file.exists():
                try:
                    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        logs = [line.strip() for line in lines[-100:]]
                except Exception:
                    pass
        
        return SupportTicket(
            ticket_id=ticket_id,
            title=title,
            description=description,
            priority=priority,
            category=category,
            system_info=self.get_system_info(),
            diagnostics=diagnostics,
            logs=logs,
        )

    def export_ticket(self, ticket: SupportTicket) -> str:
        data = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "description": ticket.description,
            "priority": ticket.priority,
            "category": ticket.category,
            "status": ticket.status,
            "created_at": ticket.created_at,
            "system_info": {
                "os_name": ticket.system_info.os_name,
                "os_version": ticket.system_info.os_version,
                "python_version": ticket.system_info.python_version,
                "cpu_count": ticket.system_info.cpu_count,
                "memory_total_gb": ticket.system_info.memory_total_gb,
                "disk_total_gb": ticket.system_info.disk_total_gb,
                "hostname": ticket.system_info.hostname,
            },
            "diagnostics": [
                {
                    "name": d.name,
                    "status": d.status,
                    "message": d.message,
                    "details": d.details,
                    "timestamp": d.timestamp,
                }
                for d in ticket.diagnostics
            ],
            "logs": ticket.logs[:50],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def get_help_for_issue(self, issue: str) -> dict[str, Any]:
        help_docs: dict[str, dict[str, Any]] = {
            "database": {
                "title": "Database Connection Issues",
                "solutions": [
                    "Check if database server is running",
                    "Verify DATABASE_URL environment variable",
                    "Check network connectivity",
                    "Verify database credentials",
                    "Check disk space for SQLite databases",
                ],
                "commands": [
                    "python -c \"from reco_trading.database.repository import Repository; import asyncio; asyncio.run(Repository.test_connection())\"",
                ],
            },
            "exchange": {
                "title": "Exchange Connection Issues",
                "solutions": [
                    "Verify BINANCE_API_KEY and BINANCE_API_SECRET are set",
                    "Check if API key has correct permissions",
                    "Verify IP whitelisting if enabled",
                    "Check if testnet mode is correct (BINANCE_TESTNET=true/false)",
                    "Try regenerating API keys",
                ],
                "commands": [
                    "python -c \"from reco_trading.exchange.binance_client import BinanceClient; import asyncio; c = BinanceClient(); asyncio.run(c.test_connection())\"",
                ],
            },
            "market_data": {
                "title": "Market Data Issues",
                "solutions": [
                    "Check internet connection",
                    "Verify symbol is valid (BTCUSDT)",
                    "Check Binance API status",
                    "Try increasing loop_sleep_seconds in config",
                ],
                "commands": [],
            },
            "risk_limits": {
                "title": "Risk Limit Issues",
                "solutions": [
                    "Check drawdown settings (max_drawdown_fraction)",
                    "Review consecutive loss settings (loss_pause_after_consecutive)",
                    "Verify capital allocation is correct",
                    "Check if emergency_stop is active",
                ],
                "commands": [
                    "Use dashboard to clear all blocks: POST /api/control/clear_all_blocks",
                ],
            },
            "positions": {
                "title": "Position Management Issues",
                "solutions": [
                    "Check position manager state",
                    "Verify max_concurrent_trades setting",
                    "Check if positions are stuck in database",
                    "Use force close from dashboard",
                ],
                "commands": [],
            },
            "api_latency": {
                "title": "API Latency Issues",
                "solutions": [
                    "Check network connection quality",
                    "Consider using a VPS closer to exchange servers",
                    "Reduce API call frequency",
                    "Check for rate limiting from exchange",
                ],
                "commands": [],
            },
            "system_resources": {
                "title": "System Resource Issues",
                "solutions": [
                    "Reduce max_concurrent_trades",
                    "Lower ML processing workers",
                    "Close unnecessary applications",
                    "Increase system memory",
                ],
                "commands": [],
            },
            "pause": {
                "title": "Bot Paused Unexpectedly",
                "solutions": [
                    "Use dashboard to resume: POST /api/control/resume",
                    "Use dashboard to clear all blocks: POST /api/control/clear_all_blocks",
                    "Check if emergency_stop is active",
                    "Check if drawdown limit reached",
                    "Check if consecutive loss limit reached",
                    "Check logs for pause reason",
                ],
                "commands": [
                    "curl -X POST http://localhost:9000/api/control/clear_all_blocks",
                ],
            },
        }
        
        for key, doc in help_docs.items():
            if key in issue.lower():
                return doc
        
        return {
            "title": "Unknown Issue",
            "solutions": [
                "Check the logs for more details",
                "Run diagnostics from dashboard",
                "Create a support ticket with system information",
            ],
            "commands": [],
        }