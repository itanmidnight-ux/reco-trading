from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from trading_system.app.dashboard.service import DashboardService
from trading_system.app.dashboard.websocket import DashboardWebSocketStreamer

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / 'templates'))
router = APIRouter()

_dashboard_service: DashboardService | None = None
_websocket_streamer: DashboardWebSocketStreamer | None = None

_ALLOWED_HOSTS = {'127.0.0.1', 'localhost', '::1'}


def configure_dashboard(service: DashboardService, streamer: DashboardWebSocketStreamer) -> None:
    global _dashboard_service, _websocket_streamer
    _dashboard_service = service
    _websocket_streamer = streamer


def _enforce_local_request(host: str | None) -> None:
    if host not in _ALLOWED_HOSTS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='dashboard solo disponible en localhost')


def _service() -> DashboardService:
    if _dashboard_service is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='dashboard no inicializado')
    return _dashboard_service


@router.get('/dashboard', response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    _enforce_local_request(request.client.host if request.client else None)
    return templates.TemplateResponse('dashboard.html', {'request': request})


@router.get('/dashboard/data')
async def dashboard_data(request: Request) -> dict:
    _enforce_local_request(request.client.host if request.client else None)
    return await _service().get_dashboard_payload()


@router.get('/dashboard/trades')
async def dashboard_trades(request: Request) -> dict:
    _enforce_local_request(request.client.host if request.client else None)
    trades = await _service().get_recent_trades()
    return {'trades': trades}


@router.get('/dashboard/equity')
async def dashboard_equity(request: Request) -> dict:
    _enforce_local_request(request.client.host if request.client else None)
    equity = await _service().get_equity_curve()
    return {'equity': equity}


@router.websocket('/ws/dashboard')
async def dashboard_ws(websocket: WebSocket) -> None:
    host = websocket.client.host if websocket.client else None
    if host not in _ALLOWED_HOSTS:
        await websocket.close(code=1008)
        return
    if _websocket_streamer is None:
        await websocket.close(code=1013)
        return
    await _websocket_streamer.stream(websocket)
