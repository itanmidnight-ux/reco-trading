from __future__ import annotations

from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


from reco_trading.ui.visual_snapshot import VisualSnapshot

class TerminalDashboard:
    def __init__(self, refresh_per_second: int = 4) -> None:
        self.refresh_per_second = refresh_per_second
        self._live: Live | None = None
        self._last_snapshot: VisualSnapshot | None = None

    @staticmethod
    def _status_style(status: str) -> str:
        normalized = status.upper()
        if normalized in {'OK', 'IN_POSITION'}:
            return 'bold green'
        if normalized in {'RISK', 'ANALYZING_MARKET', 'WAITING_FOR_DATA', 'WAITING_EDGE', 'COOLDOWN', 'LEARNING_MARKET'}:
            return 'bold yellow'
        if normalized in {'BLOCKED', 'BLOCKED_BY_RISK', 'ERROR'}:
            return 'bold red'
        if normalized == 'SENDING_ORDER':
            return 'bold cyan'
        return 'bold white'

    @staticmethod
    def _pnl_style(value: float) -> str:
        return 'bold green' if value >= 0 else 'bold red'

    @staticmethod
    def _signal_style(signal: str) -> str:
        normalized = signal.upper()
        if normalized == 'BUY':
            return 'bold green'
        if normalized == 'SELL':
            return 'bold red'
        if normalized == 'HOLD':
            return 'bold yellow'
        return 'bold white'

    def _render_summary(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify='left', style='cyan')
        table.add_column(justify='right')
        table.add_row('Capital base', f'{snapshot.capital:,.2f} USDT')
        table.add_row('Balance Binance', f'{snapshot.balance:,.2f} USDT')
        table.add_row('Exposición', f'{snapshot.exposicion:,.2f} USDT')
        table.add_row('Último Precio', f'{snapshot.ultimo_precio:,.2f}')
        table.add_row('Latencia', f'{snapshot.latencia_ms:.1f} ms')
        return Panel(table, title='[bold]Resumen de Cuenta[/bold]', border_style='cyan')

    def _render_performance(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify='left', style='magenta')
        table.add_column(justify='right')
        table.add_row('PnL Total', f'[{self._pnl_style(snapshot.pnl_total)}]{snapshot.pnl_total:+,.2f} USDT[/]')
        table.add_row('PnL Diario', f'[{self._pnl_style(snapshot.pnl_diario)}]{snapshot.pnl_diario:+,.2f} USDT[/]')
        table.add_row('Trades', str(snapshot.trades))
        table.add_row('Win Rate', f'{snapshot.win_rate:.2%}')
        table.add_row('Expectancy', f'{snapshot.expectancy:+.4f}')
        table.add_row('Sharpe', f'{snapshot.sharpe_rolling:+.3f}')
        return Panel(table, title='[bold]Performance[/bold]', border_style='magenta')

    def _render_decision(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(justify='left', style='yellow')
        table.add_column(justify='right')
        table.add_row('Régimen', snapshot.regimen)
        table.add_row('Señal Final', f'[{self._signal_style(snapshot.senal)}]{snapshot.senal}[/]')
        table.add_row('Binance', f'[{self._status_style(snapshot.estado_binance)}]{snapshot.estado_binance}[/]')
        table.add_row('Sistema', f'[{self._status_style(snapshot.estado_sistema)}]{snapshot.estado_sistema}[/]')
        table.add_row('Confianza', f'{snapshot.confianza:.2%}')
        table.add_row('Scores', f'M={snapshot.score_momentum:.2f} R={snapshot.score_reversion:.2f} G={snapshot.score_regime:.2f}')
        return Panel(table, title='[bold]Estado de Decisión[/bold]', border_style='yellow')

    def _render_risk_progress(self, snapshot: VisualSnapshot) -> Panel:
        risk_progress = Progress(
            TextColumn('[bold]Riesgo activo[/bold]'),
            BarColumn(bar_width=28),
            TextColumn('{task.percentage:>5.1f}%'),
            expand=True,
        )
        risk_progress.add_task('riesgo', total=100.0, completed=max(0.0, min(snapshot.riesgo_activo * 100.0, 100.0)))

        dd_progress = Progress(
            TextColumn('[bold]Drawdown[/bold]'),
            BarColumn(bar_width=28),
            TextColumn('{task.percentage:>5.1f}%'),
            expand=True,
        )
        dd_progress.add_task('drawdown', total=100.0, completed=max(0.0, min(snapshot.drawdown * 100.0, 100.0)))

        activity = Text(f'Actividad: {snapshot.actividad}', style='bold white')
        pos_time = Text(f'Tiempo en posición: {snapshot.tiempo_en_posicion_s:.1f}s', style='bold cyan')
        cooldown = Text(f'Cooldown restante: {snapshot.cooldown_restante_s:.1f}s', style='bold yellow')
        learning = Text(f'Tiempo aprendizaje restante: {snapshot.learning_remaining_seconds:.1f}s', style='bold magenta')
        blocked = Text(f'Motivo bloqueo: {snapshot.motivo_bloqueo}', style='bold red' if snapshot.motivo_bloqueo != 'none' else 'bold green')
        return Panel(Group(risk_progress, dd_progress, activity, pos_time, cooldown, learning, blocked), title='[bold]Riesgo y Actividad[/bold]', border_style='red')

    def _render_header(self, snapshot: VisualSnapshot) -> Text:
        title = Text(' RECO TRADING · TERMINAL LIVE ', style='bold white on blue')
        title.append(f'   Estado: {snapshot.estado_sistema}', style=self._status_style(snapshot.estado_sistema))
        return title

    def _render(self, snapshot: VisualSnapshot) -> Panel:
        top = Columns([
            self._render_summary(snapshot),
            self._render_performance(snapshot),
            self._render_decision(snapshot),
        ], expand=True, equal=True)
        body = Group(self._render_header(snapshot), top, self._render_risk_progress(snapshot))
        return Panel(body, border_style=self._status_style(snapshot.estado_sistema))

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
