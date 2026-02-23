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

    @staticmethod
    def _money_color(value: float) -> str:
        return 'bright_green' if value >= 0 else 'bright_red'

    def _header_panel(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(justify='left')
        table.add_column(justify='center')
        table.add_column(justify='right')
        risk_color = 'yellow' if snapshot.risk_state == 'BLOCKED' else 'bright_green'
        table.add_row(
            f"[bold cyan]Sistema[/]: {snapshot.system_state}",
            f"[bold blue]Régimen[/]: {snapshot.regime}",
            f"[bold {risk_color}]Riesgo[/]: {snapshot.risk_state}",
        )
        return Panel(table, title='Estado', border_style='blue')

    def _capital_panel(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify='right')
        table.add_row('Capital actual', f"[bold]${snapshot.equity:,.2f}[/]")
        table.add_row('PnL total', f"[{self._money_color(snapshot.pnl)}]{snapshot.pnl:+,.2f}[/]")
        table.add_row('PnL diario', f"[{self._money_color(snapshot.daily_pnl)}]{snapshot.daily_pnl:+,.2f}[/]")
        table.add_row('Drawdown', f"[yellow]{snapshot.drawdown:.2%}[/]")
        return Panel(table, title='Capital', border_style='cyan')

    def _analysis_panel(self, snapshot: VisualSnapshot) -> Panel:
        table = Table.grid(expand=True)
        table.add_column()
        table.add_column(justify='right')
        edge_color = 'bright_green' if snapshot.edge >= 0 else 'bright_red'
        table.add_row('Edge actual', f"[{edge_color}]{snapshot.edge:+.5f}[/]")
        table.add_row('Confianza agregada', f"[bright_blue]{snapshot.confidence:.2%}[/]")
        table.add_row('Volatilidad', f"[blue]{snapshot.volatility:.4f}[/]")
        exp_color = 'bright_green' if snapshot.expectancy >= 0 else 'bright_red'
        table.add_row('Expectancy rolling', f"[{exp_color}]{snapshot.expectancy:+.6f}[/]")
        return Panel(table, title='Análisis', border_style='blue')

    def _models_panel(self, snapshot: VisualSnapshot) -> Panel:
        table = Table(show_header=True, header_style='bold blue', expand=True)
        table.add_column('Modelo')
        table.add_column('Peso', justify='right')
        table.add_column('Estabilidad', justify='right')
        table.add_column('EV Neto', justify='right')
        diagnostics = snapshot.model_diagnostics or {}
        if not diagnostics:
            table.add_row('n/a', '-', '-', '-')
        else:
            for model, values in diagnostics.items():
                stability = float(values.get('stability', 0.0))
                ev = float(values.get('ev_net', 0.0))
                ev_color = 'bright_green' if ev >= 0 else 'bright_red'
                table.add_row(
                    model,
                    f"{float(values.get('weight', 0.0)):.2%}",
                    f"{stability:.2%}",
                    f"[{ev_color}]{ev:+.5f}[/]",
                )
        return Panel(table, title='Modelos activos', border_style='cyan')

    def _render(self, snapshot: VisualSnapshot) -> Panel:
        if snapshot.critical_error:
            return Panel(
                f"[bold bright_red]ERROR CRÍTICO[/]\n\n{snapshot.critical_error}",
                border_style='bright_red',
                title='Kernel Recovery Mode',
            )

        reason_panel = Panel(
            f"[bold]Decisión:[/] {snapshot.decision} | [bold]Ejecución:[/] {snapshot.execution_state}\n"
            f"[dim]{snapshot.reason}[/]",
            border_style='yellow' if snapshot.risk_state == 'BLOCKED' else 'blue',
            title='Ejecución',
        )

        body = Group(
            self._header_panel(snapshot),
            Group(self._capital_panel(snapshot), self._analysis_panel(snapshot)),
            self._models_panel(snapshot),
            reason_panel,
        )
        return Panel(body, title='RECO Trading Professional Dashboard', border_style='bright_blue')

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
