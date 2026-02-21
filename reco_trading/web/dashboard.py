from __future__ import annotations

import asyncio
import math
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import asyncpg
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import DictLoader, Environment, select_autoescape

from reco_trading.infra.binance_client import BinanceClient
from reco_trading.web.dashboard_data_contract import CONTRACT


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DashboardMetrics:
    capital: float = 0.0
    binance_balance: float | None = None
    pnl_diario: float = 0.0
    win_rate: float = 0.0
    trades: int = 0
    operaciones_ganadas: int = 0
    operaciones_perdidas: int = 0
    drawdown: float = 0.0
    sharpe: float = 0.0
    ultima_senal: str = 'N/A'
    operacion_actual: str = 'Sin operación activa'
    binance_connected: bool = False
    updated_at: str = ''


class DashboardService:
    def __init__(self) -> None:
        self.postgres_dsn = os.getenv('POSTGRES_DSN', '')
        self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.binance_testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        self._pool: asyncpg.Pool | None = None
        self._binance: BinanceClient | None = None

    async def startup(self) -> None:
        if self.postgres_dsn:
            normalized_dsn = self.postgres_dsn.replace('postgresql+asyncpg://', 'postgresql://', 1)
            try:
                self._pool = await asyncpg.create_pool(dsn=normalized_dsn, min_size=1, max_size=4)
            except Exception as exc:
                logger.warning('dashboard_db_unavailable: %s', exc)
                self._pool = None
        if self.binance_api_key and self.binance_api_secret:
            self._binance = BinanceClient(
                api_key=self.binance_api_key,
                api_secret=self.binance_api_secret,
                testnet=self.binance_testnet,
                confirm_mainnet=os.getenv('CONFIRM_MAINNET', 'false').lower() == 'true',
            )

    async def shutdown(self) -> None:
        if self._pool is not None:
            await self._pool.close()
        if self._binance is not None:
            await self._binance.close()

    async def _queryrow(self, query: str) -> asyncpg.Record | None:
        if self._pool is None:
            return None
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query)

    async def _query(self, query: str) -> list[asyncpg.Record]:
        if self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            return await conn.fetch(query)

    async def _fetch_binance_balance(self) -> tuple[float | None, bool]:
        if self._binance is None:
            return None, False
        try:
            balance = await self._binance.fetch_balance()
            total_usdt = float(balance.get('total', {}).get('USDT', 0.0) or 0.0)
            return total_usdt, True
        except Exception:
            return None, False

    async def _fetch_last_signal(self) -> str:
        row = await self._queryrow(CONTRACT.last_signal_query)
        if row is None:
            return 'N/A'
        score = float(row['score']) if row['score'] is not None else 0.0
        signal = str(row['signal'] or 'N/A').upper()
        reason = str(row['reason'] or '').strip()
        return f"{signal} ({score:.2%}){' · ' + reason if reason else ''}"

    async def _fetch_trade_stats(self) -> tuple[float, float, int, int, float]:
        rows = await self._query(CONTRACT.daily_execution_pnls_query)
        pnls = [float(row['pnl']) for row in rows if row['pnl'] is not None]
        if not pnls:
            fill_summary = await self._queryrow(CONTRACT.daily_fill_aggregate_query)
            if fill_summary is None:
                return 0.0, 0.0, 0, 0, 0.0
            buy_notional = float(fill_summary['buy_notional'] or 0.0)
            sell_notional = float(fill_summary['sell_notional'] or 0.0)
            fees = float(fill_summary['fees'] or 0.0)
            pnl_diario = sell_notional - buy_notional - fees
            return pnl_diario, 0.0, 0, 0, 0.0

        trades = len(pnls)
        wins = sum(1 for pnl in pnls if pnl > 0)
        losses = sum(1 for pnl in pnls if pnl <= 0)
        pnl_diario = sum(pnls)
        win_rate = (wins / trades) * 100

        mean = pnl_diario / trades
        variance = sum((value - mean) ** 2 for value in pnls) / max(trades - 1, 1)
        std = math.sqrt(max(variance, 0.0))
        sharpe = (mean / std) * math.sqrt(trades) if std > 0 else 0.0
        return pnl_diario, win_rate, wins, losses, sharpe


    async def _fetch_current_operation(self) -> str:
        row = await self._queryrow(CONTRACT.current_operation_query)
        if row is None:
            return 'Sin operación activa'
        side = str(row['side'] or 'N/A').upper()
        qty = float(row['qty'] or 0.0)
        entry_price = float(row['price'] or 0.0)
        status = str(row['status'] or 'N/A').upper()
        return f"{status} · {side} {qty:.6f} BTC @ {entry_price:.2f}"

    async def _fetch_portfolio_state(self) -> tuple[float, float]:
        row = await self._queryrow(CONTRACT.latest_portfolio_snapshot_query)
        if row is None:
            return 0.0, 0.0
        snapshot = row['snapshot'] if isinstance(row['snapshot'], dict) else {}
        equity = float(snapshot.get('equity') or 0.0)
        initial_equity = float(snapshot.get('initial_equity') or equity or 0.0)
        drawdown = float(snapshot.get('drawdown') or 0.0)
        if drawdown <= 0.0 and initial_equity > 0.0 and equity > 0.0:
            peak = max(initial_equity, equity)
            drawdown = max((peak - equity) / peak, 0.0)
        return equity, drawdown

    async def get_metrics(self) -> DashboardMetrics:
        capital, drawdown = await self._fetch_portfolio_state()
        pnl_diario, win_rate, operaciones_ganadas, operaciones_perdidas, sharpe = await self._fetch_trade_stats()
        trades = operaciones_ganadas + operaciones_perdidas
        ultima_senal = await self._fetch_last_signal()
        operacion_actual = await self._fetch_current_operation()
        binance_balance, binance_connected = await self._fetch_binance_balance()

        return DashboardMetrics(
            capital=capital,
            binance_balance=binance_balance,
            pnl_diario=pnl_diario,
            win_rate=win_rate,
            trades=trades,
            operaciones_ganadas=operaciones_ganadas,
            operaciones_perdidas=operaciones_perdidas,
            drawdown=drawdown * 100,
            sharpe=sharpe,
            ultima_senal=ultima_senal,
            operacion_actual=operacion_actual,
            binance_connected=binance_connected,
            updated_at=datetime.now(UTC).isoformat(),
        )


app = FastAPI(title='Reco Trading Dashboard')
service = DashboardService()

templates = Environment(
    loader=DictLoader(
        {
            'dashboard.html': """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Reco Trading Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <main class="max-w-6xl mx-auto p-6">
    <header class="mb-6 flex items-center justify-between">
      <h1 class="text-3xl font-bold">Reco Trading · Live Dashboard</h1>
      <span id="binance-status" class="px-3 py-1 rounded-full bg-slate-800 text-slate-200 text-sm">Binance: --</span>
    </header>

    <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {% for key, title in cards %}
      <article class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg">
        <h2 class="text-slate-400 text-sm">{{ title }}</h2>
        <p id="{{ key }}" class="text-2xl font-semibold mt-2">--</p>
      </article>
      {% endfor %}
    </section>

    <p id="updated-at" class="text-slate-500 text-xs mt-6">Última actualización: --</p>
  </main>

  <script>
    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${wsProtocol}://${window.location.host}/ws/metrics`);

    function formatCurrency(value) {
      if (value === null || value === undefined) return 'N/A';
      return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'USD' }).format(value);
    }

    function render(payload) {
      document.getElementById('capital').textContent = formatCurrency(payload.capital);
      document.getElementById('binance_balance').textContent = formatCurrency(payload.binance_balance);
      document.getElementById('pnl_diario').textContent = formatCurrency(payload.pnl_diario);
      document.getElementById('win_rate').textContent = `${payload.win_rate.toFixed(2)}%`;
      document.getElementById('trades').textContent = payload.trades;
      document.getElementById('drawdown').textContent = `${payload.drawdown.toFixed(2)}%`;
      document.getElementById('sharpe').textContent = payload.sharpe.toFixed(2);
      document.getElementById('operaciones_ganadas').textContent = payload.operaciones_ganadas;
      document.getElementById('operaciones_perdidas').textContent = payload.operaciones_perdidas;
      document.getElementById('ultima_senal').textContent = payload.ultima_senal;
      document.getElementById('operacion_actual').textContent = payload.operacion_actual;

      const status = document.getElementById('binance-status');
      status.textContent = payload.binance_connected ? 'Binance: conectado' : 'Binance: desconectado';
      status.className = payload.binance_connected
        ? 'px-3 py-1 rounded-full bg-emerald-900 text-emerald-200 text-sm'
        : 'px-3 py-1 rounded-full bg-rose-900 text-rose-200 text-sm';

      document.getElementById('updated-at').textContent = `Última actualización: ${new Date(payload.updated_at).toLocaleString('es-ES')}`;
    }

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      render(payload);
    };
  </script>
</body>
</html>
            """,
        }
    ),
    autoescape=select_autoescape(['html', 'xml']),
)


@app.on_event('startup')
async def on_startup() -> None:
    await service.startup()


@app.on_event('shutdown')
async def on_shutdown() -> None:
    await service.shutdown()


@app.get('/', response_class=HTMLResponse)
async def dashboard(_request: Request) -> HTMLResponse:
    html = templates.get_template('dashboard.html').render(
        cards=[
            ('capital', 'Capital'),
            ('binance_balance', 'Balance real Binance'),
            ('pnl_diario', 'PnL diario'),
            ('win_rate', 'Win rate'),
            ('trades', 'Trades (hoy)'),
            ('drawdown', 'Drawdown'),
            ('sharpe', 'Sharpe'),
            ('operaciones_ganadas', 'Operaciones ganadas'),
            ('operaciones_perdidas', 'Operaciones perdidas'),
            ('ultima_senal', 'Última señal'),
            ('operacion_actual', 'Operación actual/última'),
        ]
    )
    return HTMLResponse(content=html)


@app.get('/api/metrics', response_class=JSONResponse)
async def api_metrics() -> JSONResponse:
    return JSONResponse(asdict(await service.get_metrics()))


@app.websocket('/ws/metrics')
async def metrics_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            metrics = await service.get_metrics()
            await websocket.send_json(asdict(metrics))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('reco_trading.web.dashboard:app', host='localhost', port=8080, reload=False)
