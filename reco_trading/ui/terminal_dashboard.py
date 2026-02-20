from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table


@dataclass(slots=True)
class VisualSnapshot:
    capital: float
    balance: float
    pnl_total: float
    pnl_diario: float
    drawdown: float
    riesgo_activo: float
    exposicion: float
    trades: int
    win_rate: float
    expectancy: float
    sharpe_rolling: float
    regimen: str
    senal: str
    latencia_ms: float
    ultimo_precio: float
    estado_binance: str
    estado_sistema: str


class TerminalDashboard:
    def __init__(self, refresh_per_second: int = 4) -> None:
        self.refresh_per_second = refresh_per_second
        self._live: Live | None = None
        self._last_snapshot: VisualSnapshot | None = None

    @staticmethod
    def _status_style(status: str) -> str:
        normalized = status.upper()
        if normalized == 'OK':
            return 'bold green'
        if normalized == 'RISK':
            return 'bold yellow'
        if normalized == 'BLOCKED':
            return 'bold red'
        return 'bold white'

    def _render(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify='left')
        table.add_column(justify='right')

        table.add_row('Capital', f"{snapshot.capital:,.2f} USDT")
        table.add_row('Balance', f"{snapshot.balance:,.2f} USDT")
        table.add_row('PnL Total', f"{snapshot.pnl_total:+,.2f} USDT")
        table.add_row('PnL Diario', f"{snapshot.pnl_diario:+,.2f} USDT")
        table.add_row('Drawdown', f"{snapshot.drawdown:.2%}")
        table.add_row('Riesgo Activo', f"{snapshot.riesgo_activo:.2%}")
        table.add_row('Exposición', f"{snapshot.exposicion:,.2f} USDT")
        table.add_row('Trades', str(snapshot.trades))
        table.add_row('Win Rate', f"{snapshot.win_rate:.2%}")
        table.add_row('Expectancy', f"{snapshot.expectancy:+.4f}")
        table.add_row('Sharpe Rolling', f"{snapshot.sharpe_rolling:+.3f}")
        table.add_row('Régimen', snapshot.regimen)
        table.add_row('Señal', snapshot.senal)
        table.add_row('Latencia', f"{snapshot.latencia_ms:.1f} ms")
        table.add_row('Último Precio', f"{snapshot.ultimo_precio:,.2f}")
        table.add_row('Binance', f"[{self._status_style(snapshot.estado_binance)}]{snapshot.estado_binance}[/]")
        table.add_row('Sistema', f"[{self._status_style(snapshot.estado_sistema)}]{snapshot.estado_sistema}[/]")

        risk_progress = Progress(
            TextColumn('[bold]Riesgo[/bold]'),
            BarColumn(bar_width=30),
            TextColumn('{task.percentage:>5.1f}%'),
            expand=True,
        )
        risk_progress.add_task('riesgo', total=100.0, completed=max(0.0, min(snapshot.riesgo_activo * 100.0, 100.0)))

        dd_progress = Progress(
            TextColumn('[bold]Drawdown[/bold]'),
            BarColumn(bar_width=30),
            TextColumn('{task.percentage:>5.1f}%'),
            expand=True,
        )
        dd_progress.add_task('drawdown', total=100.0, completed=max(0.0, min(snapshot.drawdown * 100.0, 100.0)))

        content = Group(table, risk_progress, dd_progress)
        return Panel(content, title='RECO Terminal Dashboard', border_style=self._status_style(snapshot.estado_sistema))

    def start(self) -> None:
        if self._live is not None:
            return
        self._live = Live(auto_refresh=True, refresh_per_second=self.refresh_per_second)
        self._live.start()

    def update(self, snapshot: VisualSnapshot) -> None:
        self._last_snapshot = snapshot
        if self._live is None:
            return
        self._live.update(self._render(snapshot), refresh=True)

    def stop(self) -> None:
        if self._live is None:
            return
        if self._last_snapshot is not None:
            self._live.update(self._render(self._last_snapshot), refresh=True)
        self._live.stop()
        self._live = None
