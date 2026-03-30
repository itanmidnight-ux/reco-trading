"""
Worker class for Reco-Trading.
Handles the bot lifecycle and state management.
"""

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

import sdnotify

from reco_trading import __version__
from reco_trading.enums import RunMode, State
from reco_trading.constants import PROCESS_THROTTLE_SECS


logger = logging.getLogger(__name__)


class Worker:
    """
    Worker class that manages the bot lifecycle.
    """

    def __init__(self, config: dict[str, Any], freqtrade: Any) -> None:
        """
        Initialize the worker.
        
        Args:
            config: Configuration dictionary
            freqtrade: The bot instance
        """
        logger.info(f"Starting worker {__version__}")
        
        self._config = config
        self.freqtrade = freqtrade
        
        internals_config = self._config.get("internals", {})
        self._throttle_secs = internals_config.get("process_throttle_secs", PROCESS_THROTTLE_SECS)
        self._heartbeat_interval = internals_config.get("heartbeat_interval", 60)
        
        self._sd_notify = None
        if self._config.get("internals", {}).get("sd_notify", False):
            try:
                self._sd_notify = sdnotify.SystemdNotifier()
            except Exception as e:
                logger.warning(f"Systemd notify not available: {e}")
        
        self._heartbeat_msg: float = 0
        self._heartbeat_msg = 0

    def _notify(self, message: str) -> None:
        """
        Send notification to systemd if enabled.
        
        Args:
            message: Message to send
        """
        if self._sd_notify:
            logger.debug(f"sd_notify: {message}")
            self._sd_notify.notify(message)

    async def run(self) -> None:
        """Main run loop."""
        state = None
        while True:
            state = await self._worker(old_state=state)
            if state == State.RELOAD_CONFIG:
                await self._reconfigure()

    async def _worker(self, old_state: State | None) -> State:
        """
        Main routine that handles the states.
        
        Args:
            old_state: Previous state
            
        Returns:
            Current state
        """
        state = self.freqtrade.state

        if state != old_state:
            if old_state is not None and old_state != State.RELOAD_CONFIG:
                self.freqtrade.notify_status(f"{state.value}")
                
            logger.info(
                f"Changing state{f' from {old_state.value}' if old_state else ''} to: {state.value}"
            )
            
            if state in (State.RUNNING, State.PAUSED) and old_state not in (
                State.RUNNING,
                State.PAUSED,
            ):
                await self.freqtrade.startup()

            if state == State.STOPPED:
                await self.freqtrade.check_for_open_trades()

            self._heartbeat_msg = 0

        if state == State.STOPPED:
            if self._sd_notify:
                self._notify("WATCHDOG=1\nSTATUS=State: STOPPED.")
            
            await self._throttle(
                self._process_stopped, 
                throttle_secs=self._throttle_secs
            )

        elif state in (State.RUNNING, State.PAUSED):
            state_str = "RUNNING" if state == State.RUNNING else "PAUSED"
            if self._sd_notify:
                self._notify(f"WATCHDOG=1\nSTATUS=State: {state_str}.")

            await self._throttle(
                self._process_running,
                throttle_secs=self._throttle_secs,
            )

        if self._heartbeat_interval:
            now = time.time()
            if (now - self._heartbeat_msg) > self._heartbeat_interval:
                logger.info(
                    f"Bot heartbeat. PID=, version='{__version__}', state='{state.value}'"
                )
                self._heartbeat_msg = now

        return state

    async def _throttle(
        self,
        func: Callable[..., Any],
        throttle_secs: float,
        *args,
        **kwargs,
    ) -> Any:
        """
        Throttle the given callable.
        
        Args:
            func: Function to throttle
            throttle_secs: Minimum execution time
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of function execution
        """
        last_throttle_start_time = time.time()
        logger.debug("========================================")
        
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
            
        time_passed = time.time() - last_throttle_start_time
        sleep_duration = throttle_secs - time_passed
        
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
            
        return result

    async def _process_stopped(self) -> None:
        """Process when bot is stopped."""
        await self.freqtrade.processStopped()

    async def _process_running(self) -> None:
        """Process when bot is running."""
        await self.freqtrade.process()

    async def _reconfigure(self) -> None:
        """Reconfigure the bot."""
        logger.info("Reconfiguring bot...")
        self.freqtrade.config_updated()
        self._config = self.freqtrade.get_config()


def start_worker(config: dict[str, Any], freqtrade: Any) -> Worker:
    """
    Create and return a worker instance.
    
    Args:
        config: Configuration dictionary
        freqtrade: Bot instance
        
    Returns:
        Worker instance
    """
    return Worker(config, freqtrade)
