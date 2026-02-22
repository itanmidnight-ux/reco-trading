from __future__ import annotations

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from reco_trading.ui.visual_snapshot import VisualSnapshot


class TerminalDashboard:
    def __init__(self, refresh_per_second: int = 4) -> None:
        self.refresh_per_second = refresh_per_second
        self._live: Live | None = None
        self._last_snapshot: VisualSnapshot | None = None

    def _render(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(f'Precio: {snapshot.price:,.4f}', f'Equity: {snapshot.equity:,.2f}')
        table.add_row(f'PnL: {snapshot.pnl:+,.2f}', f'Decisión: {snapshot.decision}')
        table.add_row(f'Confianza: {snapshot.confidence:.2%}', f'Régimen: {snapshot.regime}')
        table.add_row(f'Riesgo: {snapshot.risk_state}', f'Ejecución: {snapshot.execution_state}')
        score_text = ' '.join(f'{k}={v:.3f}' for k, v in snapshot.scores.items()) if snapshot.scores else 'n/a'
        body = Group(table, f'Scores: {score_text}', f'Reason: {snapshot.reason}')
        return Panel(body, title='RECO Trading Dashboard', border_style='cyan')

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(auto_refresh=True, refresh_per_second=self.refresh_per_second)
        self._live.start()

    def update(self, snapshot: VisualSnapshot) -> None:
        self._last_snapshot = snapshot
        if self._live is not None:
            self._live.update(self._render(snapshot), refresh=True)

    def stop(self) -> None:
        if self._live is None:
            return
        if self._last_snapshot is not None:
            self._live.update(self._render(self._last_snapshot), refresh=True)
        self._live.stop()
        self._live = None
