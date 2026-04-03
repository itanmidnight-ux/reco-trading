from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from reco_trading.core.llm_systems.log_analyzer import LLMLogAnalyzer


class AutoFixCoordinator:
    """
    Coordinates hourly error resolution cycles between LLM Log Analyzer and bot engine.
    
    Every hour:
    1. Checks docs/analisis.txt for pending errors
    2. If errors exist, signals the bot to pause
    3. Generates a prompt for the AI to fix the errors
    4. After fixes are applied, triggers a bot refresh
    5. Every 24 hours: cleans up resolved errors
    
    The coordinator works in sync with the LLM Log Analyzer:
    - The analyzer writes errors to docs/analisis.txt
    - The coordinator reads the file and triggers fix cycles
    - Both agree on when to pause/resume the bot for applying fixes
    """

    def __init__(
        self,
        log_analyzer: LLMLogAnalyzer | None = None,
        pause_callback: Callable[[], None] | None = None,
        resume_callback: Callable[[], None] | None = None,
        refresh_callback: Callable[[], None] | None = None,
        check_interval_seconds: float = 3600.0,
        cleanup_interval_seconds: float = 86400.0,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.log_analyzer = log_analyzer or LLMLogAnalyzer()
        self.pause_callback = pause_callback
        self.resume_callback = resume_callback
        self.refresh_callback = refresh_callback
        self.check_interval = check_interval_seconds
        self.cleanup_interval = cleanup_interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None
        self._bot_paused = False
        self._last_check = time.monotonic()
        self._last_cleanup = time.monotonic()
        self._fix_cycles = 0
        self._errors_resolved = 0

    async def start(self) -> None:
        """Start the auto-fix coordinator background loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self.logger.info(
            f"AutoFixCoordinator started (check every {self.check_interval}s, "
            f"cleanup every {self.cleanup_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the auto-fix coordinator."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("AutoFixCoordinator stopped")

    async def _run_loop(self) -> None:
        """Main coordination loop."""
        while self._running:
            try:
                now = time.monotonic()
                
                # Check for errors every hour
                if (now - self._last_check) >= self.check_interval:
                    await self._check_for_errors()
                    self._last_check = now

                # Cleanup resolved errors every 24 hours
                if (now - self._last_cleanup) >= self.cleanup_interval:
                    await self._cleanup_resolved_errors()
                    self._last_cleanup = now

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"AutoFixCoordinator loop error: {exc}")
                await asyncio.sleep(60)

    async def _check_for_errors(self) -> None:
        """Check for pending errors and trigger fix cycle if needed."""
        summary = self.log_analyzer.get_summary()
        pending_count = summary.get("pending", 0)

        if pending_count == 0:
            self.logger.info("AutoFixCoordinator: No pending errors found")
            return

        self.logger.info(
            f"AutoFixCoordinator: Found {pending_count} pending errors, "
            f"starting fix cycle #{self._fix_cycles + 1}"
        )

        # Step 1: Pause the bot
        await self._pause_bot()

        # Step 2: Generate fix prompt
        prompt = self._generate_fix_prompt(summary)
        self.logger.info(f"AutoFixCoordinator: Generated fix prompt ({len(prompt)} chars)")

        # Step 3: Write prompt to file for AI to pick up
        self._write_fix_prompt(prompt)

        # Step 4: Wait for fixes to be applied (with timeout)
        await self._wait_for_fixes(timeout_seconds=300)

        # Step 5: Refresh the bot
        await self._refresh_bot()

        # Step 6: Resume the bot
        await self._resume_bot()

        self._fix_cycles += 1
        self.logger.info(f"AutoFixCoordinator: Fix cycle #{self._fix_cycles} completed")

    async def _pause_bot(self) -> None:
        """Pause the bot for applying fixes."""
        if self._bot_paused:
            return
        self.logger.info("AutoFixCoordinator: Pausing bot for fix application...")
        if self.pause_callback:
            try:
                self.pause_callback()
                self._bot_paused = True
                self.logger.info("AutoFixCoordinator: Bot paused successfully")
            except Exception as exc:
                self.logger.error(f"AutoFixCoordinator: Failed to pause bot: {exc}")
        else:
            self.logger.warning("AutoFixCoordinator: No pause callback configured")

    async def _resume_bot(self) -> None:
        """Resume the bot after fixes are applied."""
        if not self._bot_paused:
            return
        self.logger.info("AutoFixCoordinator: Resuming bot after fixes...")
        if self.resume_callback:
            try:
                self.resume_callback()
                self._bot_paused = False
                self.logger.info("AutoFixCoordinator: Bot resumed successfully")
            except Exception as exc:
                self.logger.error(f"AutoFixCoordinator: Failed to resume bot: {exc}")
        else:
            self.logger.warning("AutoFixCoordinator: No resume callback configured")

    async def _refresh_bot(self) -> None:
        """Trigger a bot refresh to apply fixes."""
        self.logger.info("AutoFixCoordinator: Triggering bot refresh...")
        if self.refresh_callback:
            try:
                self.refresh_callback()
                self.logger.info("AutoFixCoordinator: Bot refreshed successfully")
            except Exception as exc:
                self.logger.error(f"AutoFixCoordinator: Failed to refresh bot: {exc}")
        else:
            self.logger.warning("AutoFixCoordinator: No refresh callback configured")

    def _generate_fix_prompt(self, summary: dict[str, Any]) -> str:
        """Generate a prompt for the AI to fix the errors."""
        pending_errors = summary.get("pending_errors", [])
        
        prompt = (
            "AUTO-FIX PROMPT - Reco-Trading Bot Error Resolution\n"
            "=" * 60 + "\n\n"
            f"Fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Ciclo de correccion: #{self._fix_cycles + 1}\n"
            f"Errores pendientes: {len(pending_errors)}\n\n"
            "INSTRUCCIONES:\n"
            "1. Lee el archivo docs/analisis.txt para ver los errores detallados\n"
            "2. Para cada error enumerado como PENDIENTE:\n"
            "   a. Analiza el codigo fuente para encontrar la causa raiz\n"
            "   b. Aplica la correccion necesaria\n"
            "   c. Verifica que la correccion no rompa otras funcionalidades\n"
            "   d. Marca el error como RESUELTO en docs/analisis.txt\n"
            "3. NO repitas correcciones de errores ya marcados como RESUELTO\n"
            "4. Si un error no se puede resolver, agrega una nota explicativa\n"
            "5. Despues de resolver todos los errores, confirma con un resumen\n\n"
            "ERRORES A RESOLVER:\n"
            + "-" * 40 + "\n"
        )

        for error in pending_errors:
            prompt += f"\n#{error['number']}. {error['description']}\n"

        prompt += "\n" + "-" * 40 + "\n"
        prompt += "\nPor favor, resuelve estos errores de manera profesional y verifica que el programa funcione correctamente.\n"

        return prompt

    def _write_fix_prompt(self, prompt: str) -> None:
        """Write the fix prompt to a file for AI to pick up."""
        import os
        docs_dir = self.log_analyzer.docs_dir
        os.makedirs(docs_dir, exist_ok=True)
        prompt_path = os.path.join(docs_dir, "fix_prompt.txt")
        
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        self.logger.info(f"AutoFixCoordinator: Fix prompt written to {prompt_path}")

    async def _wait_for_fixes(self, timeout_seconds: float = 300) -> None:
        """Wait for fixes to be applied, checking periodically."""
        start = time.monotonic()
        check_interval = 30  # Check every 30 seconds
        
        while (time.monotonic() - start) < timeout_seconds:
            summary = self.log_analyzer.get_summary()
            pending = summary.get("pending", 0)
            
            if pending == 0:
                self.logger.info("AutoFixCoordinator: All errors resolved!")
                return
            
            self.logger.info(
                f"AutoFixCoordinator: Still {pending} errors pending, "
                f"waiting {check_interval}s..."
            )
            await asyncio.sleep(check_interval)
        
        self.logger.warning(
            f"AutoFixCoordinator: Timeout waiting for fixes after {timeout_seconds}s"
        )

    async def _cleanup_resolved_errors(self) -> None:
        """Clean up resolved errors every 24 hours."""
        self.logger.info("AutoFixCoordinator: Starting 24-hour cleanup...")
        removed = self.log_analyzer.cleanup_old_errors()
        self.logger.info(f"AutoFixCoordinator: Removed {removed} resolved errors")

    def get_status(self) -> dict[str, Any]:
        """Get coordinator status."""
        return {
            "running": self._running,
            "bot_paused": self._bot_paused,
            "fix_cycles": self._fix_cycles,
            "errors_resolved": self._errors_resolved,
            "last_check": self._last_check,
            "last_cleanup": self._last_cleanup,
            "check_interval": self.check_interval,
            "cleanup_interval": self.cleanup_interval,
            "error_summary": self.log_analyzer.get_summary(),
        }
