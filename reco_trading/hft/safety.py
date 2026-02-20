from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any

from loguru import logger

from reco_trading.monitoring.alert_manager import AlertManager


@dataclass(slots=True)
class SafetyEvent:
    event_type: str
    exchange: str
    severity: str
    message: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExchangeSafetyState:
    last_heartbeat_ts: float = 0.0
    heartbeat_lag_seconds: float = float('inf')
    heartbeat_ok: bool = False
    consecutive_desyncs: int = 0
    consecutive_latency_spikes: int = 0
    is_blocked: bool = False
    block_reason: str = ''


@dataclass(slots=True)
class SafetyState:
    auto_disable_arbitrage: bool = False
    operating_mode: str = 'normal'
    degraded_exchanges: set[str] = field(default_factory=set)
    blocked_exchanges: set[str] = field(default_factory=set)
    exchange_states: dict[str, ExchangeSafetyState] = field(default_factory=dict)
    capital_isolation: dict[str, float] = field(default_factory=dict)


class HFTSafetyMonitor:
    def __init__(
        self,
        *,
        alert_manager: AlertManager,
        heartbeat_timeout_seconds: float = 5.0,
        book_ticker_desync_bps: float = 20.0,
        latency_spike_std_factor: float = 2.0,
        latency_spike_floor_ms: float = 150.0,
        anomaly_persistence_threshold: int = 3,
        capital_isolation: dict[str, float] | None = None,
    ) -> None:
        self.alert_manager = alert_manager
        self.heartbeat_timeout_seconds = max(heartbeat_timeout_seconds, 0.1)
        self.book_ticker_desync_bps = max(book_ticker_desync_bps, 0.0)
        self.latency_spike_std_factor = max(latency_spike_std_factor, 0.1)
        self.latency_spike_floor_ms = max(latency_spike_floor_ms, 1.0)
        self.anomaly_persistence_threshold = max(anomaly_persistence_threshold, 1)
        self.state = SafetyState(capital_isolation=capital_isolation or {})
        self._latency_windows: dict[str, deque[float]] = {}
        self._events: deque[SafetyEvent] = deque(maxlen=200)

    def _exchange_state(self, exchange: str) -> ExchangeSafetyState:
        if exchange not in self.state.exchange_states:
            self.state.exchange_states[exchange] = ExchangeSafetyState()
        return self.state.exchange_states[exchange]

    def _record_event(
        self,
        *,
        event_type: str,
        exchange: str,
        severity: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = SafetyEvent(
            event_type=event_type,
            exchange=exchange,
            severity=severity,
            message=message,
            timestamp=time.time(),
            payload=payload or {},
        )
        self._events.append(event)
        logger.bind(component='hft_safety', exchange=exchange, event_type=event_type, severity=severity).warning(message)
        if severity in {'warning', 'critical'}:
            self.alert_manager.emit(
                f'hft_safety:{event_type}',
                message,
                severity=severity,
                exchange=exchange,
                payload=event.payload,
            )

    def update_heartbeat(self, exchange: str, heartbeat_ts: float | None = None) -> None:
        state = self._exchange_state(exchange)
        now = time.time()
        state.last_heartbeat_ts = heartbeat_ts if heartbeat_ts is not None else now
        state.heartbeat_lag_seconds = max(0.0, now - state.last_heartbeat_ts)
        state.heartbeat_ok = state.heartbeat_lag_seconds <= self.heartbeat_timeout_seconds

    def evaluate_heartbeats(self, now: float | None = None) -> None:
        current = now if now is not None else time.time()
        for exchange, exchange_state in self.state.exchange_states.items():
            exchange_state.heartbeat_lag_seconds = max(0.0, current - exchange_state.last_heartbeat_ts)
            exchange_state.heartbeat_ok = exchange_state.heartbeat_lag_seconds <= self.heartbeat_timeout_seconds
            if not exchange_state.heartbeat_ok:
                self.state.degraded_exchanges.add(exchange)
                self._record_event(
                    event_type='heartbeat_timeout',
                    exchange=exchange,
                    severity='warning',
                    message=f'Heartbeat timeout ({exchange_state.heartbeat_lag_seconds:.2f}s)',
                    payload={'lag_seconds': exchange_state.heartbeat_lag_seconds},
                )
        self._refresh_operating_mode()

    def detect_book_ticker_desync(self, exchange: str, order_book: dict[str, Any], ticker: dict[str, Any]) -> None:
        exchange_state = self._exchange_state(exchange)
        bids = order_book.get('bids') or []
        asks = order_book.get('asks') or []
        if not bids or not asks:
            return

        book_mid = (float(bids[0][0]) + float(asks[0][0])) / 2
        ticker_price = float(ticker.get('last') or ticker.get('close') or 0.0)
        if book_mid <= 0 or ticker_price <= 0:
            return

        drift_bps = abs(book_mid - ticker_price) / book_mid * 10_000
        if drift_bps >= self.book_ticker_desync_bps:
            exchange_state.consecutive_desyncs += 1
            self.state.degraded_exchanges.add(exchange)
            self._record_event(
                event_type='book_ticker_desync',
                exchange=exchange,
                severity='warning',
                message=f'Desync detectado ({drift_bps:.2f} bps)',
                payload={'drift_bps': drift_bps},
            )
            if exchange_state.consecutive_desyncs >= self.anomaly_persistence_threshold:
                self._block_exchange(exchange, reason='persistent_book_ticker_desync')
        else:
            exchange_state.consecutive_desyncs = 0

        self._refresh_operating_mode()

    def detect_latency_spike(self, exchange: str, latency_ms: float) -> None:
        exchange_state = self._exchange_state(exchange)
        window = self._latency_windows.setdefault(exchange, deque(maxlen=100))

        spike = False
        if len(window) >= 10:
            mean_latency = fmean(window)
            std_latency = pstdev(window) if len(window) > 1 else 0.0
            dynamic_threshold = max(self.latency_spike_floor_ms, mean_latency + self.latency_spike_std_factor * std_latency)
            spike = latency_ms >= dynamic_threshold
        else:
            spike = latency_ms >= self.latency_spike_floor_ms

        window.append(latency_ms)

        if spike:
            exchange_state.consecutive_latency_spikes += 1
            self.state.degraded_exchanges.add(exchange)
            self._record_event(
                event_type='latency_spike',
                exchange=exchange,
                severity='warning',
                message=f'Pico de latencia detectado ({latency_ms:.2f}ms)',
                payload={'latency_ms': latency_ms},
            )
            if exchange_state.consecutive_latency_spikes >= self.anomaly_persistence_threshold:
                self._block_exchange(exchange, reason='persistent_latency_spike')
        else:
            exchange_state.consecutive_latency_spikes = 0

        self._refresh_operating_mode()

    def _block_exchange(self, exchange: str, reason: str) -> None:
        state = self._exchange_state(exchange)
        state.is_blocked = True
        state.block_reason = reason
        self.state.blocked_exchanges.add(exchange)
        self._record_event(
            event_type='exchange_blocked',
            exchange=exchange,
            severity='critical',
            message=f'Exchange bloqueado por seguridad: {reason}',
            payload={'reason': reason},
        )

    def allowed_capital_fraction(self, exchange: str) -> float:
        if exchange in self.state.blocked_exchanges:
            return 0.0

        base_limit = self.state.capital_isolation.get(exchange, 1.0)
        if exchange in self.state.degraded_exchanges:
            return max(0.0, base_limit * 0.5)
        return max(0.0, min(1.0, base_limit))

    def _refresh_operating_mode(self) -> None:
        if self.state.blocked_exchanges:
            self.state.operating_mode = 'restricted'
        elif self.state.degraded_exchanges:
            self.state.operating_mode = 'degraded'
        else:
            self.state.operating_mode = 'normal'

        self.state.auto_disable_arbitrage = len(self.state.blocked_exchanges) >= 2

    def health_snapshot(self) -> dict[str, Any]:
        self.evaluate_heartbeats()
        return {
            'auto_disable_arbitrage': self.state.auto_disable_arbitrage,
            'operating_mode': self.state.operating_mode,
            'blocked_exchanges': sorted(self.state.blocked_exchanges),
            'degraded_exchanges': sorted(self.state.degraded_exchanges),
            'exchange_states': {
                name: {
                    'heartbeat_ok': st.heartbeat_ok,
                    'heartbeat_lag_seconds': st.heartbeat_lag_seconds,
                    'consecutive_desyncs': st.consecutive_desyncs,
                    'consecutive_latency_spikes': st.consecutive_latency_spikes,
                    'is_blocked': st.is_blocked,
                    'block_reason': st.block_reason,
                    'allowed_capital_fraction': self.allowed_capital_fraction(name),
                }
                for name, st in self.state.exchange_states.items()
            },
            'recent_events': [
                {
                    'event_type': ev.event_type,
                    'exchange': ev.exchange,
                    'severity': ev.severity,
                    'message': ev.message,
                    'timestamp': ev.timestamp,
                    'payload': ev.payload,
                }
                for ev in list(self._events)[-25:]
            ],
        }
