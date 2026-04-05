# CODEX MASTER PROMPT v3 — reco-trading: Overhaul Completo y Profesional
> **Para: OpenAI Codex / GPT-4.1 / Claude Code**
> **Instrucción principal:** Lee este documento COMPLETO de principio a fin antes de modificar cualquier archivo. Luego ejecuta cada cambio en el orden indicado. Cada bloque incluye el código original exacto (lo que existe hoy) y el código nuevo exacto (lo que debes escribir). No omitas ningún bloque. No trunces ningún archivo.

---

## 0. REGLAS ABSOLUTAS

1. **Lee todo antes de actuar.** No empieces a editar hasta terminar de leer.
2. **No elimines funcionalidad existente** que no esté marcada con `[ELIMINAR]`.
3. **Cada archivo modificado debe quedar completo** — nunca escribas `# ... resto igual`.
4. **Ejecuta los bloques en orden:** A → B → C → D → E → F → G → H.
5. **Ollama fue removido del proyecto.** No lo uses, no lo importes, no lo menciones en código nuevo.
6. **No cambies** `signal_engine.py`, `indicators.py`, `confidence_model.py`, `regime_filter.py` — no tienen bugs.
7. Después de cada archivo modificado, ejecuta `python -m py_compile <archivo>` para verificar sintaxis.
8. Stack: Python 3.11+, asyncio, Flask (threaded), Rich TUI, ccxt, SQLAlchemy async, pydantic-settings.

---

## A. ARCHIVO: `reco_trading/main.py`

### A1. Tres bugs en este archivo. Aplica los tres juntos reemplazando el archivo completo.

**BUG A1:** `_verify_database_connection` usa `settings.postgres_dsn` directamente → `TypeError` si no hay PostgreSQL configurado.

**BUG A2:** Race condition — Flask arranca **antes** que el bot thread, causando snapshot vacío en los primeros segundos.

**BUG A3:** `state_manager=None` siempre → los logs y controles del web dashboard no se enrutan al bot.

**CÓDIGO ORIGINAL (lo que existe hoy en `reco_trading/main.py`):**
```python
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine
from reco_trading.database.repository import Repository
from web_site import run_in_thread as run_web_dashboard_in_thread
from web_site.dashboard_server import set_bot_instance_getter

_bot_instance: BotEngine | None = None
_bot_runtime_error: Exception | None = None


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def get_bot_instance() -> BotEngine | None:
    return _bot_instance


def _run_bot(settings: Settings, state_manager: object | None) -> None:
    global _bot_instance, _bot_runtime_error
    bot = BotEngine(settings, state_manager=state_manager)
    _bot_instance = bot
    try:
        asyncio.run(bot.run())
    except Exception as exc:
        _bot_runtime_error = exc
        logging.getLogger(__name__).exception("Bot terminated unexpectedly")
    finally:
        _bot_instance = None


def _join_bot_thread_or_exit(bot_thread: threading.Thread, logger: logging.Logger) -> None:
    bot_thread.join()
    if _bot_runtime_error is not None:
        logger.error("Bot stopped unexpectedly: %s", _bot_runtime_error)
        sys.exit(1)


def _start_web_dashboard(logger: logging.Logger) -> None:
    web_host = str(os.getenv("WEB_DASHBOARD_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    web_port_raw = str(os.getenv("WEB_DASHBOARD_PORT", "9000")).strip()
    try:
        web_port = int(web_port_raw)
    except ValueError:
        web_port = 9000
        logger.warning("Invalid WEB_DASHBOARD_PORT=%s; falling back to %s", web_port_raw, web_port)

    set_bot_instance_getter(get_bot_instance)
    run_web_dashboard_in_thread(host=web_host, port=web_port)
    logger.info("Web dashboard thread started at http://%s:%s", web_host, web_port)


async def _verify_database_connection(settings: Settings) -> str:
    repository = Repository(settings.postgres_dsn)
    try:
        await repository.verify_connectivity()
        return settings.postgres_dsn
    finally:
        await repository.close()


def run() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    global _bot_runtime_error
    _bot_runtime_error = None

    settings = Settings()
    if settings.require_api_keys and (not settings.binance_api_key or not settings.binance_api_secret):
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
    if not settings.binance_testnet and not settings.confirm_mainnet:
        raise RuntimeError("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")

    if hasattr(settings, "terminal_tui_enabled") and not bool(getattr(settings, "terminal_tui_enabled", True)):
        logger.warning("terminal_tui_enabled=false detectado; forzando True para iniciar web+terminal dashboard juntos")
        setattr(settings, "terminal_tui_enabled", True)

    _start_web_dashboard(logger)

    try:
        asyncio.run(_verify_database_connection(settings))
    except Exception as exc:
        logger.warning(
            "Database unavailable at startup; the bot will retry when PostgreSQL is available: %s",
            exc,
        )

    bot_thread = threading.Thread(target=_run_bot, args=(settings, None), daemon=True, name="bot-engine")
    bot_thread.start()

    try:
        _join_bot_thread_or_exit(bot_thread, logger)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    run()
```

**CÓDIGO NUEVO (reemplaza todo el archivo `reco_trading/main.py`):**
```python
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading

from reco_trading.config.settings import Settings
from reco_trading.core.bot_engine import BotEngine, _get_database_dsn
from reco_trading.database.repository import Repository
from web_site import run_in_thread as run_web_dashboard_in_thread
from web_site.dashboard_server import set_bot_instance_getter

_bot_instance: BotEngine | None = None
_bot_runtime_error: Exception | None = None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_bot_instance() -> BotEngine | None:
    return _bot_instance


def _run_bot(settings: Settings, state_manager: object | None) -> None:
    global _bot_instance, _bot_runtime_error
    bot = BotEngine(settings, state_manager=state_manager)
    _bot_instance = bot
    try:
        asyncio.run(bot.run())
    except Exception as exc:  # noqa: BLE001
        _bot_runtime_error = exc
        logging.getLogger(__name__).exception("Bot terminated unexpectedly")
    finally:
        _bot_instance = None


def _join_bot_thread_or_exit(bot_thread: threading.Thread, logger: logging.Logger) -> None:
    bot_thread.join()
    if _bot_runtime_error is not None:
        logger.error("Bot stopped unexpectedly: %s", _bot_runtime_error)
        sys.exit(1)


def _start_web_dashboard(logger: logging.Logger) -> None:
    web_host = str(os.getenv("WEB_DASHBOARD_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    web_port_raw = str(os.getenv("WEB_DASHBOARD_PORT", "9000")).strip()
    try:
        web_port = int(web_port_raw)
    except ValueError:
        web_port = 9000
        logger.warning(
            "Invalid WEB_DASHBOARD_PORT=%s; falling back to %s", web_port_raw, web_port
        )

    # El getter ya está registrado antes de llamar a run_web_dashboard_in_thread
    run_web_dashboard_in_thread(host=web_host, port=web_port)
    logger.info("Web dashboard thread started at http://%s:%s", web_host, web_port)


async def _verify_database_connection(settings: Settings) -> str:
    """Verifica la mejor DSN disponible con fallback automático a SQLite."""
    dsn = _get_database_dsn(settings)
    if not dsn:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        db_path = os.path.join(project_root, "data", "reco_trading.db")
        dsn = f"sqlite+aiosqlite:///{db_path}"
    repository = Repository(dsn)
    try:
        await repository.setup()
        return dsn
    finally:
        await repository.close()


def run() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    global _bot_runtime_error
    _bot_runtime_error = None

    settings = Settings()

    if settings.require_api_keys and (
        not settings.binance_api_key or not settings.binance_api_secret
    ):
        raise RuntimeError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
    if not settings.binance_testnet and not settings.confirm_mainnet:
        raise RuntimeError("Mainnet trading blocked: set CONFIRM_MAINNET=true to proceed")

    if hasattr(settings, "terminal_tui_enabled") and not bool(
        getattr(settings, "terminal_tui_enabled", True)
    ):
        logger.warning(
            "terminal_tui_enabled=false detectado; forzando True para web+terminal dashboard"
        )
        setattr(settings, "terminal_tui_enabled", True)

    # FIX A3: Inicializar StateManager antes de arrancar el bot
    state_manager = None
    try:
        from reco_trading.ui.state_manager import StateManager  # noqa: PLC0415
        state_manager = StateManager()
        logger.info("StateManager initialized successfully")
    except ImportError:
        logger.info("StateManager not available, running without it")
    except Exception as exc:  # noqa: BLE001
        logger.warning("StateManager failed to initialize: %s — running without it", exc)

    # FIX A2 + A1: Registrar getter ANTES de cualquier thread; verificar BD con DSN correcto
    set_bot_instance_getter(get_bot_instance)

    try:
        asyncio.run(_verify_database_connection(settings))
        logger.info("Database connection verified successfully")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Database unavailable at startup (bot usará fallback SQLite automático): %s", exc
        )

    # FIX A2: Bot thread arranca ANTES que Flask para que tenga tiempo de inicializarse
    bot_thread = threading.Thread(
        target=_run_bot, args=(settings, state_manager), daemon=True, name="bot-engine"
    )
    bot_thread.start()

    # Web dashboard arranca DESPUÉS: getter ya registrado, bot en proceso de init
    _start_web_dashboard(logger)

    try:
        _join_bot_thread_or_exit(bot_thread, logger)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    run()
```

---

## B. ARCHIVO: `run.sh`

### B1. Dos bugs. Aplica ambos.

**BUG B1:** `python main.py` al final no usa `exec`, así que Ctrl+C no llega limpiamente al proceso Python.

**BUG B2:** La función `upsert_env_var` está definida dos veces en el script (una en el heredoc del runtime_env.sh y otra directamente), lo que causa shadowing y confusión.

**BUG B3:** No hay mensaje al usuario indicando la URL del dashboard web durante el startup.

**INSTRUCCIÓN:** Localiza el bloque al final de `run.sh` que empieza con:
```bash
echo ""
echo "Iniciando Reco-Trading Bot..."
echo ""

python main.py
```

**Reemplázalo exactamente por:**
```bash
WEB_PORT="${WEB_DASHBOARD_PORT:-9000}"

# Verificar si el puerto ya está en uso
if command -v lsof >/dev/null 2>&1; then
  if lsof -i ":${WEB_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Puerto ${WEB_PORT} ya está en uso. El dashboard usará el siguiente disponible."
  fi
elif command -v ss >/dev/null 2>&1; then
  if ss -tlnp 2>/dev/null | grep -q ":${WEB_PORT} "; then
    echo "⚠️  Puerto ${WEB_PORT} ya está en uso. El dashboard usará el siguiente disponible."
  fi
fi

echo ""
echo "══════════════════════════════════════════════════"
echo "  ◈  RECO TRADING BOT  —  Iniciando..."
echo "  📊 Dashboard Web  :  http://127.0.0.1:${WEB_PORT}"
echo "  📡 Dashboard TUI  :  Esta consola"
echo "  ⌨️  Detener        :  Ctrl + C"
echo "══════════════════════════════════════════════════"
echo ""

exec python main.py
```

**INSTRUCCIÓN ADICIONAL para B2:** Busca en `run.sh` la primera definición de `upsert_env_var` (la que está dentro del heredoc `RUNTIME_LIB`). Esa versión dentro del heredoc es solo para el archivo `scripts/lib/runtime_env.sh`. La que debe quedar activa en el script principal es la segunda definición (la que tiene el método con python3 como fallback). No elimines ninguna de las dos — simplemente verifica que el order de ejecución es: primero se genera el heredoc, luego se source el archivo, y finalmente la segunda definición en el script principal sobreescribe la del heredoc. Esto es correcto y no necesita cambio estructural. Solo agrega este comentario en la segunda definición para que sea claro:

```bash
# Definición activa de upsert_env_var (sobreescribe la del runtime_env.sh cargado arriba)
# Usa python3 como método primario para máxima compatibilidad cross-distro
upsert_env_var() {
  # ... (código existente sin cambiar) ...
}
```

---

## C. ARCHIVO: `reco_trading/risk/capital_profile.py`

### C1. Bug: Los perfiles NANO y MICRO tienen `max_trades_per_day=2` y `cooldown_minutes=15/12`, haciendo imposible el objetivo de 20+ trades/día.

**INSTRUCCIÓN:** En `CapitalProfileManager.__init__`, localiza y reemplaza SOLO los primeros dos `CapitalProfile(...)` con los siguientes valores. No cambies los demás perfiles (SMALL, MEDIUM, LARGE, WHALE).

**Perfil NANO — localiza:**
```python
            CapitalProfile(
                name="NANO",
                min_equity=0.0,
                max_equity=25.0,
                reserve_ratio=0.40,
                reserve_buffer_usdt=1.0,
                risk_per_trade_fraction=0.04,
                max_trade_balance_fraction=0.10,
                min_confidence=0.65,
                max_trades_per_day=2,
                cooldown_minutes=15,
                loss_pause_minutes=90,
                loss_pause_after_consecutive=1,
                max_spread_ratio=0.0010,
                min_expected_reward_risk=3.0,
                min_operable_notional_buffer=1.25,
                max_concurrent_trades=1,
                entry_quality_floor=0.75,
                size_multiplier=0.40,
            ),
```

**Reemplaza con:**
```python
            CapitalProfile(
                name="NANO",
                min_equity=0.0,
                max_equity=25.0,
                reserve_ratio=0.40,
                reserve_buffer_usdt=1.0,
                risk_per_trade_fraction=0.04,
                max_trade_balance_fraction=0.10,
                min_confidence=0.65,
                max_trades_per_day=25,          # Límite de seguridad, no el objetivo mínimo
                cooldown_minutes=5,             # Cooldown corto para mayor frecuencia
                loss_pause_minutes=45,          # Reducido para recuperar actividad antes
                loss_pause_after_consecutive=2, # 2 pérdidas seguidas = pausa breve
                max_spread_ratio=0.0010,
                min_expected_reward_risk=2.5,   # RR más alcanzable con capital pequeño
                min_operable_notional_buffer=1.25,
                max_concurrent_trades=1,
                entry_quality_floor=0.70,
                size_multiplier=0.40,
            ),
```

**Perfil MICRO — localiza:**
```python
            CapitalProfile(
                name="MICRO",
                min_equity=25.0,
                max_equity=50.0,
                reserve_ratio=0.35,
                reserve_buffer_usdt=3.0,
                risk_per_trade_fraction=0.03,
                max_trade_balance_fraction=0.12,
                min_confidence=0.62,
                max_trades_per_day=2,
                cooldown_minutes=12,
                loss_pause_minutes=60,
                loss_pause_after_consecutive=2,
                max_spread_ratio=0.0015,
                min_expected_reward_risk=2.8,
                min_operable_notional_buffer=1.20,
                max_concurrent_trades=1,
                entry_quality_floor=0.72,
                size_multiplier=0.50,
            ),
```

**Reemplaza con:**
```python
            CapitalProfile(
                name="MICRO",
                min_equity=25.0,
                max_equity=50.0,
                reserve_ratio=0.35,
                reserve_buffer_usdt=3.0,
                risk_per_trade_fraction=0.03,
                max_trade_balance_fraction=0.12,
                min_confidence=0.62,
                max_trades_per_day=25,          # Límite de seguridad amplio
                cooldown_minutes=5,             # Cooldown reducido
                loss_pause_minutes=35,
                loss_pause_after_consecutive=2,
                max_spread_ratio=0.0015,
                min_expected_reward_risk=2.3,
                min_operable_notional_buffer=1.20,
                max_concurrent_trades=1,
                entry_quality_floor=0.68,
                size_multiplier=0.50,
            ),
```

---

## D. ARCHIVO: `reco_trading/core/bot_engine.py`

### D1. Agregar safety clamps en `_apply_symbol_filter_config`

**INSTRUCCIÓN:** Localiza el método `_apply_symbol_filter_config`. Al final del método, ANTES del `self.logger.info(...)` final, inserta este bloque:

```python
        # ── SAFETY CLAMPS: evitar filtros peligrosos en cualquier dirección ──
        _SAFETY = {
            "adx_threshold":              (8.0,  35.0),
            "rsi_buy_threshold":          (40.0, 70.0),
            "rsi_sell_threshold":         (30.0, 60.0),
            "min_confidence":             (0.45, 0.85),
            "volume_buy_threshold":       (0.50, 2.50),
            "volume_sell_threshold":      (0.30, 1.50),
            "atr_low_threshold":          (0.001, 0.010),
            "atr_high_threshold":         (0.010, 0.060),
            "stop_loss_atr_multiplier":   (1.0,  4.0),
            "take_profit_atr_multiplier": (1.5,  6.0),
        }
        for _k, (_lo, _hi) in _SAFETY.items():
            if _k in self.base_filter_config:
                _orig = self.base_filter_config[_k]
                _clamped = max(_lo, min(_hi, _orig))
                if abs(_clamped - _orig) > 1e-9:
                    self.logger.warning(
                        "Filter safety clamp: %s %.4f → %.4f [%.4f, %.4f]",
                        _k, _orig, _clamped, _lo, _hi,
                    )
                    self.base_filter_config[_k] = _clamped

        self.runtime_filter_config = dict(self.base_filter_config)
        self._filter_auto_adjustment_count = getattr(self, "_filter_auto_adjustment_count", 0) + 1
```

### D2. Agregar `_filter_auto_adjustment_count` en `__init__`

**INSTRUCCIÓN:** En `BotEngine.__init__`, localiza la línea:
```python
        self.runtime_filter_config: dict[str, float] = dict(self.base_filter_config)
```
Inmediatamente después, agrega:
```python
        self._filter_auto_adjustment_count: int = 0
        self._current_capital_profile = None  # Se actualiza en cada ciclo de equity
```

### D3. Exponer nuevos campos en `self.snapshot` dentro de `_sync_ui_state`

**INSTRUCCIÓN:** En el método `_sync_ui_state` (o el método equivalente que actualiza `self.snapshot` al final de cada ciclo), localiza el bloque que actualiza campos del snapshot y agrega al final de ese bloque, antes del `return`:

```python
        # ── Campos nuevos para dashboards ──────────────────────────────────
        # Capital profile objetivo de trades
        _cp = getattr(self, "_current_capital_profile", None)
        if _cp is not None:
            self.snapshot["trades_target_daily"] = getattr(_cp, "max_trades_per_day", 0)
        else:
            self.snapshot.setdefault("trades_target_daily", 0)

        # Estado de filtros adaptativos
        _base_conf = self.base_filter_config.get("min_confidence", 0.55)
        _runtime_conf = self.runtime_filter_config.get("min_confidence", 0.55)
        self.snapshot["filter_relaxation_active"] = bool(_runtime_conf < _base_conf - 0.01)
        self.snapshot["filter_auto_adjustments"] = self._filter_auto_adjustment_count

        # Auto-improver stats extendidos
        _ai = getattr(self, "auto_improver", None)
        if _ai is not None:
            self.snapshot["auto_improve_consecutive_losses"] = int(
                getattr(_ai, "consecutive_losses", 0) or 0
            )
            self.snapshot["auto_improve_optimization_count"] = int(
                getattr(_ai, "optimization_count", 0) or 0
            )
            self.snapshot["auto_optimized_params"] = dict(
                getattr(_ai, "current_params", {}) or {}
            )
```

---

## E. ARCHIVO: `web_site/dashboard_server.py`

### E1. Fix: `_run_async` crea y destruye un event loop por cada llamada (memory leak en SSE)

**CÓDIGO ORIGINAL (localiza exactamente):**
```python
def _run_async(coro: Any) -> Any:
    """
    Ejecuta coroutines de forma segura desde contexto Flask (síncrono).
    NUNCA llamar con coroutines del bot que usen sus objetos internos (cliente, orders).
    Para esas operaciones, usar el sistema de comandos asíncrono del bot.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        logger.error("_run_async error: %s", e)
        raise
```

**CÓDIGO NUEVO (reemplaza la función completa):**
```python
# Loop persistente para operaciones de BD del dashboard (evita crear/destruir loops en cada SSE tick)
_dashboard_db_loop: asyncio.AbstractEventLoop | None = None
_dashboard_db_loop_lock = threading.Lock()


def _get_dashboard_db_loop() -> asyncio.AbstractEventLoop:
    global _dashboard_db_loop
    with _dashboard_db_loop_lock:
        if _dashboard_db_loop is None or _dashboard_db_loop.is_closed():
            _dashboard_db_loop = asyncio.new_event_loop()
            _dashboard_db_loop.set_debug(False)
        return _dashboard_db_loop


def _run_async(coro: Any) -> Any:
    """
    Ejecuta coroutines de forma segura desde contexto Flask (síncrono).
    Usa un loop persistente para evitar memory leaks en polling SSE.
    NUNCA llamar con coroutines que accedan a objetos internos del bot.
    """
    try:
        loop = _get_dashboard_db_loop()
        if loop.is_running():
            # No debería ocurrir en Flask sync, pero por seguridad:
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=10.0)
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error("_run_async error: %s", e)
        raise
```

### E2. Fix: SSE endpoint sin manejo de desconexión → errores infinitos en cliente perdido

**CÓDIGO ORIGINAL (localiza la función `api_stream` dentro de `create_app`):**
```python
    @app.route('/api/stream')
    def api_stream():
        """SSE endpoint para actualizaciones en tiempo real."""
        auth_err = _require_dashboard_auth()
        if auth_err:
            return auth_err

        def event_generator():
            import time
            while True:
                try:
                    data = get_bot_snapshot()
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(1.5)  # actualizar cada 1.5 segundos
                except GeneratorExit:
                    break
                except Exception as e:
                    logger.error("SSE error: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    time.sleep(3)

        return Response(
            event_generator(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            }
        )
```

**CÓDIGO NUEVO:**
```python
    @app.route('/api/stream')
    def api_stream():
        """SSE endpoint para actualizaciones en tiempo real con manejo de desconexión."""
        auth_err = _require_dashboard_auth()
        if auth_err:
            return auth_err

        def event_generator():
            import time
            consecutive_errors = 0
            while True:
                try:
                    data = get_bot_snapshot()
                    yield f"data: {json.dumps(data)}\n\n"
                    consecutive_errors = 0
                    time.sleep(1.5)
                except GeneratorExit:
                    logger.debug("SSE client disconnected cleanly")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    logger.error("SSE error (#%d): %s", consecutive_errors, e)
                    if consecutive_errors >= 5:
                        logger.warning("SSE: demasiados errores consecutivos, cerrando stream")
                        break
                    try:
                        yield f"data: {json.dumps({'error': str(e), 'retry': 3000})}\n\n"
                    except Exception:
                        break
                    time.sleep(3)

        return Response(
            event_generator(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
```

### E3. Agregar endpoint nuevo `/api/capital_profile`

**INSTRUCCIÓN:** Dentro de la función `create_app()`, después del endpoint `/api/health_detailed` (que es el último endpoint antes del `_app = app`), agrega:

```python
    @app.route('/api/capital_profile')
    def api_capital_profile():
        """Expone el perfil de capital activo y estado de filtros adaptativos."""
        unauthorized = _require_dashboard_auth()
        if unauthorized:
            return unauthorized
        snapshot = get_bot_snapshot()
        return jsonify({
            "capital_profile":          snapshot.get("capital_profile", "UNKNOWN"),
            "operable_capital_usdt":    snapshot.get("operable_capital_usdt", 0.0),
            "capital_reserve_ratio":    snapshot.get("capital_reserve_ratio", 0.0),
            "min_cash_buffer_usdt":     snapshot.get("min_cash_buffer_usdt", 0.0),
            "capital_limit_usdt":       snapshot.get("capital_limit_usdt", 0.0),
            "filter_relaxation_active": snapshot.get("filter_relaxation_active", False),
            "filter_auto_adjustments":  snapshot.get("filter_auto_adjustments", 0),
            "trades_target_daily":      snapshot.get("trades_target_daily", 0),
            "trades_today":             snapshot.get("trades_today", 0),
            "autonomous_filters":       snapshot.get("autonomous_filters", {}),
        })
```

---

## F. ARCHIVO: `reco_trading/ui/dashboard.py`

### F1. Reemplazar panel "LLM Gate" por "Trade Engine Stats"

**PASO F1-A:** En el dataclass `DashboardSnapshot`, localiza:
```python
    auto_improve_total_trades: int = 0
    investment_mode: str | None = None
```
Reemplaza con:
```python
    auto_improve_total_trades: int = 0
    investment_mode: str | None = None
    # Nuevos campos: Trade Engine Stats
    auto_improve_consecutive_losses: int = 0
    auto_improve_optimization_count: int = 0
    auto_optimized_params: dict[str, Any] = field(default_factory=dict)
    filter_auto_adjustments: int = 0
    trades_target_daily: int = 0
    filter_relaxation_active: bool = False
```

**PASO F1-B:** En el método `from_mapping`, localiza:
```python
            auto_improve_total_trades=int(data.get("auto_improve_total_trades", 0) or 0),
            investment_mode=_to_text(data.get("investment_mode")),
```
Reemplaza con:
```python
            auto_improve_total_trades=int(data.get("auto_improve_total_trades", 0) or 0),
            investment_mode=_to_text(data.get("investment_mode")),
            auto_improve_consecutive_losses=int(data.get("auto_improve_consecutive_losses", 0) or 0),
            auto_improve_optimization_count=int(data.get("auto_improve_optimization_count", 0) or 0),
            auto_optimized_params=dict(data.get("auto_optimized_params", {}) or {}),
            filter_auto_adjustments=int(data.get("filter_auto_adjustments", 0) or 0),
            trades_target_daily=int(data.get("trades_target_daily", 0) or 0),
            filter_relaxation_active=bool(data.get("filter_relaxation_active", False)),
```

**PASO F1-C:** En el método `render`, localiza el bloque completo del LLM GATE PANEL:
```python
            # ── LLM GATE PANEL ───────────────────────────────────────
            llm = snap.llm_trade_confirmator or {}
            llm_mode = (snap.llm_mode or "base").upper()
            llm_health = llm.get("local_endpoint_healthy")
            health_icon = "🟢" if llm_health is True else "🔴" if llm_health is False else "⚪"
            llm_table = Table.grid(expand=True, padding=(0, 1))
            llm_table.add_column(style="dim", min_width=16)
            llm_table.add_column(style="bold white")
            llm_table.add_row("Mode", Text(llm_mode, style="bold magenta"))
            llm_table.add_row("Ollama", Text(health_icon + (" Online" if llm_health else " Offline" if llm_health is False else " —")))
            llm_table.add_row("Analyzed", str(llm.get("total_analyzed", 0)))
            llm_table.add_row("Confirmed", Text(str(llm.get("confirmed", 0)), style="green"))
            llm_table.add_row("Rejected", Text(str(llm.get("rejected", 0)), style="red"))
            rate = llm.get("confirmation_rate", 0)
            llm_table.add_row("Rate", Text(f"{_fmt_float(rate, 1)}%", style="cyan"))
            llm_table.add_row("Avg Latency", f"{_fmt_float(llm.get('avg_analysis_time_ms', 0), 1)} ms")
```

**[ELIMINAR el bloque anterior]** y reemplaza con:
```python
            # ── TRADE ENGINE STATS PANEL ──────────────────────────────
            import os as _os_env
            _web_port = _os_env.getenv("WEB_DASHBOARD_PORT", "9000")
            _profile_colors = {
                "NANO": "bold red", "MICRO": "bold yellow",
                "SMALL": "bold cyan", "MEDIUM": "bold green",
                "LARGE": "bold bright_green", "WHALE": "bold bright_magenta",
                "INSTITUTIONAL": "bold bright_white",
            }
            _profile_name = (snap.capital_profile or "—").upper()
            _profile_style = _profile_colors.get(_profile_name, "bold white")
            _target = snap.trades_target_daily or 0
            _trades_pct = (snap.trades_today / _target) if _target > 0 else 0
            _trades_style = "bold green" if _trades_pct >= 1.0 else "bold yellow" if _trades_pct >= 0.5 else "white"

            engine_table = Table.grid(expand=True, padding=(0, 1))
            engine_table.add_column(style="dim", min_width=18)
            engine_table.add_column(style="bold white")
            engine_table.add_row("Capital Profile", Text(_profile_name, style=_profile_style))
            engine_table.add_row(
                "Target Trades/día",
                Text(str(_target) if _target > 0 else "—", style="cyan"),
            )
            engine_table.add_row(
                "Trades Hoy",
                Text(f"{snap.trades_today}/{_target if _target > 0 else '—'}", style=_trades_style),
            )
            engine_table.add_row(
                "Win Rate AutoMejora",
                Text(_fmt_pct(snap.auto_improve_win_rate),
                     style="green" if (snap.auto_improve_win_rate or 0) >= 0.55 else "yellow"),
            )
            engine_table.add_row(
                "Pérd. Consecutivas",
                _losses_styled(snap.auto_improve_consecutive_losses),
            )
            engine_table.add_row(
                "Optimizaciones",
                Text(str(snap.auto_improve_optimization_count), style="cyan"),
            )
            _flt_status = "RELAJADO ▼" if snap.filter_relaxation_active else "NORMAL ✓"
            _flt_style = "bold yellow" if snap.filter_relaxation_active else "bold green"
            engine_table.add_row("Filtros Estado", Text(_flt_status, style=_flt_style))
            engine_table.add_row(
                "Ajustes Automáticos",
                Text(str(snap.filter_auto_adjustments), style="dim cyan"),
            )
            _opt = snap.auto_optimized_params or {}
            if _opt.get("min_confidence") is not None:
                engine_table.add_row(
                    "Conf. Optimizada",
                    Text(f"{float(_opt['min_confidence'])*100:.1f}%", style="bright_cyan"),
                )
```

**PASO F1-D:** En el footer, localiza:
```python
            footer = Text.assemble(
                ("◈ Reco Trading TUI  ", "dim"),
                ("│  ", "dim"),
                ("Ctrl+C", "bold white"),
                (" to stop  ", "dim"),
                ("│  ", "dim"),
                ("Web: ", "dim"),
                ("http://localhost:9000", "bright_cyan"),
            )
```
Reemplaza con:
```python
            footer = Text.assemble(
                ("◈ Reco Trading TUI  ", "dim"),
                ("│  ", "dim"),
                ("Ctrl+C", "bold white"),
                (" to stop  ", "dim"),
                ("│  ", "dim"),
                ("Web: ", "dim"),
                (f"http://localhost:{_web_port}", "bright_cyan"),
                ("  │  Filtros: ", "dim"),
                ("AUTO ⚡", "bright_yellow"),
            )
```

**PASO F1-E:** Reemplaza TODAS las referencias a `llm_table` en el layout por `engine_table`, y el título/estilo del panel:

En el layout compacto (ancho < 110), localiza:
```python
                    Layout(Panel(llm_table, title="LLM Gate", border_style="magenta"), ratio=3),
```
Reemplaza con:
```python
                    Layout(Panel(engine_table, title="⚡ Trade Engine", border_style="bright_yellow"), ratio=3),
```

En el layout wide (columna derecha), localiza:
```python
                    Layout(Panel(llm_table, title="🤖 LLM Gate", border_style="magenta"), ratio=4),
```
Reemplaza con:
```python
                    Layout(Panel(engine_table, title="⚡ Trade Engine", border_style="bright_yellow"), ratio=4),
```

### F2. Mejorar panel de filtros — mostrar más datos relevantes

**INSTRUCCIÓN:** Localiza el bloque `# ── FILTER STATUS` y reemplaza su contenido (manteniendo la cabecera del comentario) por:

```python
            # ── FILTER STATUS (mejorado) ──────────────────────────────
            af = snap.autonomous_filters
            filter_table = Table.grid(expand=True, padding=(0, 1))
            filter_table.add_column(style="dim", min_width=16)
            filter_table.add_column(style="bold cyan")
            if af:
                filter_table.add_row("ADX  ≥", _fmt_float(af.get("adx_threshold"), 1))
                filter_table.add_row("RSI Buy  ≥", _fmt_float(af.get("rsi_buy_threshold"), 1))
                filter_table.add_row("RSI Sell ≤", _fmt_float(af.get("rsi_sell_threshold"), 1))
                filter_table.add_row(
                    "Confianza Min",
                    Text(f"{_fmt_float((af.get('min_confidence') or 0) * 100, 1)}%",
                         style="green" if (af.get("min_confidence") or 0) >= 0.60 else "yellow"),
                )
                filter_table.add_row("Vol Buy  ≥", _fmt_float(af.get("volume_buy_threshold"), 2))
                filter_table.add_row("SL ATR  ×", _fmt_float(af.get("stop_loss_atr_multiplier"), 2))
                filter_table.add_row("TP ATR  ×", _fmt_float(af.get("take_profit_atr_multiplier"), 2))
                filter_table.add_row(
                    "ATR Rango",
                    Text(
                        f"{_fmt_float(af.get('atr_low_threshold'), 4)}–{_fmt_float(af.get('atr_high_threshold'), 4)}",
                        style="dim cyan",
                    ),
                )
            else:
                filter_table.add_row("Estado", Text("cargando...", style="dim"))
```

---

## G. ARCHIVO: `web_site/templates/index.html`

### G1. Rediseño completo del dashboard web — profesional, sin dependencias externas

**INSTRUCCIÓN:** Reemplaza el contenido COMPLETO de `web_site/templates/index.html` con el siguiente HTML. Es un archivo autocontenido (CSS y JS inline). No uses CDNs ni librerías externas.

```html
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>◈ Reco Trading — Dashboard</title>
<style>
:root {
  --bg:         #080d18;
  --bg2:        #0f1624;
  --bg3:        #161f30;
  --bg4:        #1c2840;
  --border:     #243045;
  --border2:    #2e3f5c;
  --text:       #e2e8f0;
  --text2:      #94a3b8;
  --text3:      #4a5568;
  --green:      #10b981;
  --green2:     #34d399;
  --red:        #ef4444;
  --red2:       #f87171;
  --yellow:     #f59e0b;
  --cyan:       #06b6d4;
  --blue:       #3b82f6;
  --purple:     #8b5cf6;
  --orange:     #f97316;
  --magenta:    #ec4899;
  --radius:     8px;
  --radius2:    12px;
  --shadow:     0 2px 12px rgba(0,0,0,.45);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code','Consolas',monospace;font-size:13px;min-height:100vh;overflow-x:hidden}
a{color:var(--cyan);text-decoration:none}

/* HEADER */
#header{position:sticky;top:0;z-index:100;background:var(--bg2);border-bottom:1px solid var(--border);padding:10px 20px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
#header .logo{font-size:18px;font-weight:700;color:var(--cyan);letter-spacing:.05em;white-space:nowrap}
#header .pair{font-size:22px;font-weight:700;color:#fff;min-width:120px}
#header .price-big{font-size:20px;font-weight:700;color:#fff}
#header .change{font-size:13px;padding:2px 8px;border-radius:4px;font-weight:600}
#header .change.up{background:rgba(16,185,129,.15);color:var(--green2)}
#header .change.down{background:rgba(239,68,68,.15);color:var(--red2)}
#header .change.flat{background:rgba(100,116,139,.15);color:var(--text2)}
#header .status-badge{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.08em;margin-left:auto}
.badge-running{background:rgba(16,185,129,.2);color:var(--green2);border:1px solid rgba(16,185,129,.3)}
.badge-paused{background:rgba(245,158,11,.2);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.badge-error{background:rgba(239,68,68,.2);color:var(--red2);border:1px solid rgba(239,68,68,.3)}
.badge-waiting{background:rgba(100,116,139,.2);color:var(--text2);border:1px solid var(--border)}
#latency{font-size:11px;color:var(--text3);margin-left:8px}

/* GRID PRINCIPAL */
#main{display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:auto auto auto;gap:12px;padding:14px 20px;max-width:1800px;margin:0 auto}
@media(max-width:1100px){#main{grid-template-columns:1fr 1fr}}
@media(max-width:700px){#main{grid-template-columns:1fr}}

/* CARDS */
.card{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius2);padding:14px 16px;box-shadow:var(--shadow)}
.card-title{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--text3);margin-bottom:12px;display:flex;align-items:center;gap:6px}
.card-title span{font-size:14px}

/* METRIC ROW */
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.metric-row:last-child{border-bottom:none}
.metric-label{color:var(--text2);font-size:12px}
.metric-val{font-weight:600;font-size:13px;color:#fff}
.metric-val.green{color:var(--green)}
.metric-val.red{color:var(--red)}
.metric-val.yellow{color:var(--yellow)}
.metric-val.cyan{color:var(--cyan)}
.metric-val.muted{color:var(--text3)}

/* SIGNAL BADGES */
.sig-badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}
.sig-buy{background:rgba(16,185,129,.18);color:var(--green2);border:1px solid rgba(16,185,129,.3)}
.sig-sell{background:rgba(239,68,68,.18);color:var(--red2);border:1px solid rgba(239,68,68,.3)}
.sig-neutral{background:rgba(100,116,139,.18);color:var(--text2);border:1px solid var(--border)}
.sig-hold{background:rgba(245,158,11,.15);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}

/* PROGRESS BAR */
.progress-wrap{background:rgba(255,255,255,.07);border-radius:3px;height:6px;overflow:hidden;margin-top:4px}
.progress-bar{height:100%;border-radius:3px;transition:width .4s ease}
.progress-green{background:linear-gradient(90deg,var(--green),var(--green2))}
.progress-yellow{background:linear-gradient(90deg,var(--yellow),#fbbf24)}
.progress-red{background:linear-gradient(90deg,var(--red),var(--red2))}
.progress-cyan{background:linear-gradient(90deg,var(--cyan),#67e8f9)}

/* CAPITAL PROFILE BADGE */
.profile-badge{display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;letter-spacing:.05em}
.profile-NANO{background:rgba(239,68,68,.2);color:var(--red2);border:1px solid rgba(239,68,68,.3)}
.profile-MICRO{background:rgba(245,158,11,.2);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.profile-SMALL{background:rgba(6,182,212,.2);color:var(--cyan);border:1px solid rgba(6,182,212,.3)}
.profile-MEDIUM{background:rgba(16,185,129,.2);color:var(--green2);border:1px solid rgba(16,185,129,.3)}
.profile-LARGE{background:rgba(52,211,153,.25);color:#6ee7b7;border:1px solid rgba(52,211,153,.4)}
.profile-WHALE{background:rgba(236,72,153,.2);color:#f9a8d4;border:1px solid rgba(236,72,153,.3)}
.profile-INSTITUTIONAL{background:rgba(255,255,255,.15);color:#fff;border:1px solid rgba(255,255,255,.25)}

/* TRADES TABLE */
#trades-section{grid-column:1/-1}
.table-wrap{overflow-x:auto;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:var(--bg4);color:var(--text2);font-weight:600;text-align:left;padding:8px 10px;white-space:nowrap;border-bottom:1px solid var(--border2)}
td{padding:7px 10px;border-bottom:1px solid rgba(255,255,255,.04);white-space:nowrap}
tr:hover td{background:rgba(255,255,255,.03)}
tr.profit td{border-left:2px solid var(--green)}
tr.loss td{border-left:2px solid var(--red)}
tr.open-trade td{border-left:2px solid var(--cyan)}

/* LOG */
#log-section{grid-column:1/-1}
#log-container{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;height:220px;overflow-y:auto;font-size:11.5px;font-family:inherit}
#log-container::-webkit-scrollbar{width:4px}
#log-container::-webkit-scrollbar-track{background:var(--bg2)}
#log-container::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.log-line{padding:2px 0;border-bottom:1px solid rgba(255,255,255,.03);display:flex;gap:10px}
.log-line:last-child{border-bottom:none}
.log-time{color:var(--text3);flex-shrink:0;width:60px}
.log-level{width:48px;flex-shrink:0;font-weight:700}
.log-INFO .log-level{color:var(--text2)}
.log-WARNING .log-level{color:var(--yellow)}
.log-ERROR .log-level{color:var(--red2)}
.log-DEBUG .log-level{color:var(--text3)}
.log-msg{color:var(--text);word-break:break-word}
.log-WARNING .log-msg{color:var(--yellow)}
.log-ERROR .log-msg{color:var(--red2)}

/* CONTROLES */
#controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.btn{padding:7px 16px;border-radius:6px;border:none;cursor:pointer;font-size:12px;font-weight:700;font-family:inherit;transition:all .15s;letter-spacing:.04em}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn-pause{background:rgba(245,158,11,.18);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.btn-pause:hover:not(:disabled){background:rgba(245,158,11,.3)}
.btn-resume{background:rgba(16,185,129,.18);color:var(--green2);border:1px solid rgba(16,185,129,.3)}
.btn-resume:hover:not(:disabled){background:rgba(16,185,129,.3)}
.btn-emergency{background:rgba(239,68,68,.18);color:var(--red2);border:1px solid rgba(239,68,68,.3)}
.btn-emergency:hover:not(:disabled){background:rgba(239,68,68,.3)}

/* MODAL */
#modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:999;align-items:center;justify-content:center}
#modal-overlay.active{display:flex}
#modal{background:var(--bg3);border:1px solid var(--border2);border-radius:var(--radius2);padding:24px 28px;min-width:300px;max-width:440px;box-shadow:0 8px 32px rgba(0,0,0,.6)}
#modal h3{margin-bottom:12px;color:var(--red2);font-size:15px}
#modal p{color:var(--text2);margin-bottom:18px;line-height:1.6}
#modal .modal-actions{display:flex;gap:10px;justify-content:flex-end}

/* CONNECTION indicator */
#conn-indicator{width:8px;height:8px;border-radius:50%;background:var(--text3);transition:background .3s;flex-shrink:0}
#conn-indicator.connected{background:var(--green)}
#conn-indicator.error{background:var(--red)}

/* PnL coloring */
.pnl-pos{color:var(--green)}
.pnl-neg{color:var(--red)}
.pnl-zero{color:var(--text3)}
</style>
</head>
<body>

<!-- HEADER -->
<div id="header">
  <div class="logo">◈ RECO</div>
  <div class="pair" id="h-pair">BTC/USDT</div>
  <div>
    <div class="price-big" id="h-price">$0.00</div>
  </div>
  <div class="change flat" id="h-change">0.00%</div>
  <span id="h-signal" class="sig-badge sig-hold">HOLD</span>
  <div id="conn-indicator" title="Conexión SSE"></div>
  <span id="latency">latencia: —</span>
  <div style="margin-left:auto;display:flex;gap:10px;align-items:center">
    <span id="h-timeframe" style="color:var(--text3);font-size:12px">—</span>
    <span id="h-status" class="status-badge badge-waiting">WAITING</span>
  </div>
</div>

<!-- MAIN GRID -->
<div id="main">

  <!-- CARD: MERCADO -->
  <div class="card">
    <div class="card-title"><span>📈</span> Mercado</div>
    <div class="metric-row"><span class="metric-label">Precio</span><span class="metric-val" id="m-price">—</span></div>
    <div class="metric-row"><span class="metric-label">Bid / Ask</span><span class="metric-val" id="m-bidask">—</span></div>
    <div class="metric-row"><span class="metric-label">Spread</span><span class="metric-val" id="m-spread">—</span></div>
    <div class="metric-row"><span class="metric-label">Tendencia</span><span class="metric-val" id="m-trend">—</span></div>
    <div class="metric-row"><span class="metric-label">Régimen Vol.</span><span class="metric-val" id="m-regime">—</span></div>
    <div class="metric-row"><span class="metric-label">Order Flow</span><span class="metric-val" id="m-oflow">—</span></div>
    <div class="metric-row"><span class="metric-label">Vol. 24h</span><span class="metric-val" id="m-vol24">—</span></div>
    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px"><span class="metric-label">RSI</span><span class="metric-val" id="m-rsi-val">50</span></div>
      <div class="progress-wrap"><div class="progress-bar progress-cyan" id="m-rsi-bar" style="width:50%"></div></div>
    </div>
    <div style="margin-top:8px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px"><span class="metric-label">ADX</span><span class="metric-val" id="m-adx-val">0</span></div>
      <div class="progress-wrap"><div class="progress-bar progress-yellow" id="m-adx-bar" style="width:0%"></div></div>
    </div>
  </div>

  <!-- CARD: PORTFOLIO -->
  <div class="card">
    <div class="card-title"><span>💼</span> Portfolio</div>
    <div class="metric-row"><span class="metric-label">Balance</span><span class="metric-val" id="p-balance">—</span></div>
    <div class="metric-row"><span class="metric-label">Equity</span><span class="metric-val" id="p-equity">—</span></div>
    <div class="metric-row"><span class="metric-label">Capital Operable</span><span class="metric-val cyan" id="p-operable">—</span></div>
    <div class="metric-row"><span class="metric-label">PnL Sesión</span><span class="metric-val" id="p-session-pnl">—</span></div>
    <div class="metric-row"><span class="metric-label">PnL Diario</span><span class="metric-val" id="p-daily-pnl">—</span></div>
    <div class="metric-row"><span class="metric-label">PnL No Realizado</span><span class="metric-val" id="p-unrealized">—</span></div>
    <div style="margin-top:12px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span class="metric-label">Win Rate</span>
        <span class="metric-val" id="p-winrate-val">0%</span>
      </div>
      <div class="progress-wrap"><div class="progress-bar progress-green" id="p-winrate-bar" style="width:0%"></div></div>
    </div>
    <div class="metric-row" style="margin-top:10px"><span class="metric-label">Trades Hoy</span><span class="metric-val" id="p-trades">0</span></div>
    <div class="metric-row"><span class="metric-label">Señal</span><span id="p-signal" class="sig-badge sig-hold" style="font-size:12px">HOLD</span></div>
    <div class="metric-row"><span class="metric-label">Confianza</span><span class="metric-val" id="p-conf">0%</span></div>
    <div class="metric-row"><span class="metric-label">Cooldown</span><span class="metric-val" id="p-cooldown">READY</span></div>
    <div style="margin-top:12px" id="controls">
      <button class="btn btn-pause" onclick="sendControl('pause')">⏸ Pausar</button>
      <button class="btn btn-resume" onclick="sendControl('resume')">▶ Reanudar</button>
      <button class="btn btn-emergency" onclick="confirmEmergency()">🛑 Emergency</button>
    </div>
  </div>

  <!-- CARD: TRADE ENGINE -->
  <div class="card">
    <div class="card-title"><span>⚡</span> Trade Engine</div>
    <div class="metric-row">
      <span class="metric-label">Capital Profile</span>
      <span id="te-profile" class="profile-badge profile-NANO">NANO</span>
    </div>
    <div class="metric-row"><span class="metric-label">Target Trades/día</span><span class="metric-val cyan" id="te-target">—</span></div>
    <div style="margin-top:8px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span class="metric-label">Progreso Trades</span>
        <span class="metric-val" id="te-trades-prog">0 / —</span>
      </div>
      <div class="progress-wrap"><div class="progress-bar progress-cyan" id="te-trades-bar" style="width:0%"></div></div>
    </div>
    <div class="metric-row" style="margin-top:10px"><span class="metric-label">Win Rate AutoMejora</span><span class="metric-val" id="te-wr">—</span></div>
    <div class="metric-row"><span class="metric-label">Pérd. Consecutivas</span><span class="metric-val" id="te-losses">0</span></div>
    <div class="metric-row"><span class="metric-label">Optimizaciones</span><span class="metric-val cyan" id="te-opts">0</span></div>
    <div class="metric-row"><span class="metric-label">Estado Filtros</span><span class="metric-val" id="te-flt-status">NORMAL</span></div>
    <div class="metric-row"><span class="metric-label">Ajustes Automáticos</span><span class="metric-val muted" id="te-flt-adj">0</span></div>
    <div style="margin-top:14px;padding-top:10px;border-top:1px solid var(--border)">
      <div class="card-title" style="margin-bottom:8px"><span>🔧</span> Filtros Activos</div>
      <div class="metric-row"><span class="metric-label">ADX ≥</span><span class="metric-val" id="f-adx">—</span></div>
      <div class="metric-row"><span class="metric-label">RSI Buy ≥</span><span class="metric-val" id="f-rsi-buy">—</span></div>
      <div class="metric-row"><span class="metric-label">RSI Sell ≤</span><span class="metric-val" id="f-rsi-sell">—</span></div>
      <div class="metric-row"><span class="metric-label">Confianza Min</span><span class="metric-val" id="f-conf">—</span></div>
      <div class="metric-row"><span class="metric-label">Vol. Buy ≥</span><span class="metric-val" id="f-vol">—</span></div>
      <div class="metric-row"><span class="metric-label">SL ATR ×</span><span class="metric-val" id="f-sl">—</span></div>
      <div class="metric-row"><span class="metric-label">TP ATR ×</span><span class="metric-val" id="f-tp">—</span></div>
    </div>
  </div>

  <!-- CARD: POSICIÓN ABIERTA -->
  <div class="card">
    <div class="card-title"><span>🔵</span> Posición Abierta</div>
    <div id="position-content">
      <div style="color:var(--text3);text-align:center;padding:20px 0">Sin posición abierta</div>
    </div>
  </div>

  <!-- CARD: SEÑALES -->
  <div class="card">
    <div class="card-title"><span>📡</span> Señales Multi-Factor</div>
    <div id="signals-grid"></div>
    <div style="margin-top:14px">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span class="metric-label">Confianza Compuesta</span>
        <span class="metric-val" id="sig-conf-val">0%</span>
      </div>
      <div class="progress-wrap"><div class="progress-bar progress-green" id="sig-conf-bar" style="width:0%"></div></div>
    </div>
    <div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--border)">
      <div class="card-title" style="margin-bottom:6px"><span>🔍</span> Decisión</div>
      <div id="decision-reason" style="color:var(--text2);font-size:12px;line-height:1.6">—</div>
    </div>
  </div>

  <!-- CARD: ESTADO SISTEMA -->
  <div class="card">
    <div class="card-title"><span>🖥️</span> Estado Sistema</div>
    <div class="metric-row"><span class="metric-label">Exchange</span><span class="metric-val" id="sys-exchange">—</span></div>
    <div class="metric-row"><span class="metric-label">Base de Datos</span><span class="metric-val" id="sys-db">—</span></div>
    <div class="metric-row"><span class="metric-label">Latencia API</span><span class="metric-val" id="sys-lat">—</span></div>
    <div class="metric-row"><span class="metric-label">Circuit Breaker</span><span class="metric-val" id="sys-cb">0</span></div>
    <div class="metric-row"><span class="metric-label">Reconexiones</span><span class="metric-val" id="sys-reconn">0</span></div>
    <div class="metric-row"><span class="metric-label">Emergency Stop</span><span class="metric-val" id="sys-emg">NO</span></div>
    <div class="metric-row"><span class="metric-label">Modo Inversión</span><span class="metric-val cyan" id="sys-invmode">—</span></div>
    <div class="metric-row"><span class="metric-label">Exit Dinámico</span><span class="metric-val" id="sys-dyn-exit">—</span></div>
    <div class="metric-row"><span class="metric-label">Exit Intel. Score</span><span class="metric-val" id="sys-exit-score">—</span></div>
  </div>

  <!-- TRADES TABLE -->
  <div class="card" id="trades-section">
    <div class="card-title" style="justify-content:space-between">
      <span><span>📋</span> Historial de Trades</span>
      <div style="display:flex;gap:8px;align-items:center">
        <select id="filter-status" onchange="filterTrades()" style="background:var(--bg4);color:var(--text2);border:1px solid var(--border2);border-radius:4px;padding:3px 6px;font-size:11px">
          <option value="">Todos</option>
          <option value="OPEN">Abiertos</option>
          <option value="CLOSED">Cerrados</option>
        </select>
        <span id="trades-count" style="color:var(--text3);font-size:11px">0 trades</span>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Par</th><th>Lado</th><th>Entrada</th><th>Salida</th>
            <th>Qty</th><th>PnL</th><th>PnL%</th><th>Estado</th><th>Duración</th><th>Fecha</th>
          </tr>
        </thead>
        <tbody id="trades-body"><tr><td colspan="11" style="text-align:center;color:var(--text3);padding:20px">Cargando trades...</td></tr></tbody>
      </table>
    </div>
    <div id="trades-pagination" style="display:flex;gap:6px;margin-top:10px;justify-content:flex-end"></div>
  </div>

  <!-- LOG -->
  <div class="card" id="log-section">
    <div class="card-title" style="justify-content:space-between">
      <span><span>📄</span> Log en Tiempo Real</span>
      <button onclick="clearLog()" style="background:none;border:1px solid var(--border2);color:var(--text3);border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer">Limpiar</button>
    </div>
    <div id="log-container"></div>
  </div>

</div><!-- /main -->

<!-- MODAL EMERGENCIA -->
<div id="modal-overlay">
  <div id="modal">
    <h3>🛑 Confirmar Emergency Stop</h3>
    <p>Esta acción detiene toda actividad de trading inmediatamente. Las posiciones abiertas no se cierran automáticamente. ¿Confirmar?</p>
    <div class="modal-actions">
      <button class="btn" onclick="closeModal()" style="background:var(--bg4);border:1px solid var(--border2);color:var(--text2)">Cancelar</button>
      <button class="btn btn-emergency" onclick="doEmergency()">🛑 Confirmar Emergency</button>
    </div>
  </div>
</div>

<script>
// ─── State ─────────────────────────────────────────────────────────────────
let allTrades = [];
let currentPage = 1;
const perPage = 20;
let statusFilter = '';
let _sse = null;

// ─── Helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const fmt = (v, d=2) => (v == null || isNaN(v)) ? '—' : Number(v).toFixed(d);
const fmtK = v => {
  if (v == null || isNaN(v)) return '—';
  v = Number(v);
  if (Math.abs(v) >= 1e9) return (v/1e9).toFixed(2)+'B';
  if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(2)+'M';
  if (Math.abs(v) >= 1e3) return (v/1e3).toFixed(1)+'K';
  return v.toFixed(2);
};
const pnlClass = v => Number(v) > 0 ? 'pnl-pos' : Number(v) < 0 ? 'pnl-neg' : 'pnl-zero';
const pnlSign = v => Number(v) > 0 ? '+' : '';
const sigClass = s => s==='BUY'?'sig-buy':s==='SELL'?'sig-sell':s==='HOLD'?'sig-hold':'sig-neutral';

function setText(id, val) {
  const el = $(id); if (el) el.textContent = val ?? '—';
}
function setClass(id, cls) {
  const el = $(id); if (el) el.className = cls;
}
function setHtml(id, html) {
  const el = $(id); if (el) el.innerHTML = html;
}
function bar(id, pct, colorClass) {
  const el = $(id);
  if (!el) return;
  el.style.width = Math.min(100, Math.max(0, Number(pct)||0)) + '%';
  el.className = 'progress-bar ' + colorClass;
}

// ─── Dashboard Update ───────────────────────────────────────────────────────
function updateDashboard(d) {
  if (!d) return;

  // HEADER
  const pair = d.pair || 'BTC/USDT';
  setText('h-pair', pair);
  setText('h-price', '$' + fmt(d.current_price || d.price, 2));
  setText('h-timeframe', d.timeframe || '—');
  setText('latency', 'latencia: ' + fmt(d.api_latency_p95_ms || d.api_latency_ms, 1) + 'ms');

  const chg = Number(d.change_24h || 0);
  const chgEl = $('h-change');
  if (chgEl) {
    chgEl.textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%';
    chgEl.className = 'change ' + (chg > 0.05 ? 'up' : chg < -0.05 ? 'down' : 'flat');
  }

  const sig = (d.signal || 'HOLD').toUpperCase();
  const sigEl = $('h-signal');
  if (sigEl) { sigEl.textContent = sig; sigEl.className = 'sig-badge ' + sigClass(sig); }

  const st = (d.status || 'WAITING').toUpperCase();
  const stEl = $('h-status');
  if (stEl) {
    stEl.textContent = st;
    stEl.className = 'status-badge ' + (
      st === 'RUNNING' ? 'badge-running' :
      st.includes('PAUSE') || st.includes('STOP') ? 'badge-paused' :
      st.includes('ERROR') ? 'badge-error' : 'badge-waiting'
    );
  }

  // MERCADO
  setText('m-price', '$' + fmt(d.current_price || d.price, 2));
  setText('m-bidask', (d.bid ? '$'+fmt(d.bid,4) : '—') + ' / ' + (d.ask ? '$'+fmt(d.ask,4) : '—'));
  setText('m-spread', fmt(d.spread, 6));
  setText('m-trend', d.trend || '—');
  setText('m-regime', d.volatility_regime || '—');
  setText('m-oflow', d.order_flow || '—');
  setText('m-vol24', fmtK(d.volume_24h));

  const rsi = Number(d.rsi || 50);
  setText('m-rsi-val', rsi.toFixed(1));
  bar('m-rsi-bar', rsi, rsi > 70 ? 'progress-red' : rsi < 30 ? 'progress-green' : 'progress-cyan');

  const adx = Number(d.adx || 0);
  setText('m-adx-val', adx.toFixed(1));
  bar('m-adx-bar', Math.min(adx / 50 * 100, 100), adx >= 25 ? 'progress-green' : 'progress-yellow');

  // PORTFOLIO
  setText('p-balance', '$' + fmt(d.balance, 4));
  setText('p-equity', '$' + fmt(d.equity, 4));
  setText('p-operable', '$' + fmt(d.operable_capital_usdt, 4));
  const spnl = Number(d.session_pnl || 0);
  const dpnl = Number(d.daily_pnl || 0);
  const upnl = Number(d.unrealized_pnl || 0);
  const spEl = $('p-session-pnl');
  if (spEl) { spEl.textContent = pnlSign(spnl) + '$' + fmt(spnl, 4); spEl.className = 'metric-val ' + pnlClass(spnl); }
  const dpEl = $('p-daily-pnl');
  if (dpEl) { dpEl.textContent = pnlSign(dpnl) + '$' + fmt(dpnl, 4); dpEl.className = 'metric-val ' + pnlClass(dpnl); }
  const upEl = $('p-unrealized');
  if (upEl) { upEl.textContent = pnlSign(upnl) + '$' + fmt(upnl, 4); upEl.className = 'metric-val ' + pnlClass(upnl); }

  const wr = Number(d.win_rate || 0) * (d.win_rate > 1 ? 1 : 100);
  const wrDisp = d.win_rate > 1 ? fmt(d.win_rate, 1) : fmt(d.win_rate * 100, 1);
  setText('p-winrate-val', wrDisp + '%');
  bar('p-winrate-bar', wr, wr >= 55 ? 'progress-green' : wr >= 45 ? 'progress-yellow' : 'progress-red');

  setText('p-trades', d.trades_today || 0);
  const psig = (d.signal || 'HOLD').toUpperCase();
  const psigEl = $('p-signal');
  if (psigEl) { psigEl.textContent = psig; psigEl.className = 'sig-badge ' + sigClass(psig); }
  setText('p-conf', fmt((d.confidence || 0) * 100, 1) + '%');

  const cool = d.cooldown || 'READY';
  const coolEl = $('p-cooldown');
  if (coolEl) {
    coolEl.textContent = cool;
    coolEl.className = 'metric-val ' + (['READY','ready'].includes(cool) ? 'green' : 'yellow');
  }

  // TRADE ENGINE
  const profile = (d.capital_profile || 'UNKNOWN').toUpperCase();
  const profEl = $('te-profile');
  if (profEl) { profEl.textContent = profile; profEl.className = 'profile-badge profile-' + profile; }

  const target = Number(d.trades_target_daily || 0);
  setText('te-target', target > 0 ? target : '—');
  const tradesT = Number(d.trades_today || 0);
  setText('te-trades-prog', tradesT + ' / ' + (target > 0 ? target : '—'));
  bar('te-trades-bar', target > 0 ? (tradesT / target * 100) : 0, tradesT >= target && target > 0 ? 'progress-green' : 'progress-cyan');

  const aiwr = Number(d.auto_improve_win_rate || 0);
  const aiwrDisp = aiwr > 1 ? fmt(aiwr, 1) : fmt(aiwr * 100, 1);
  const teWrEl = $('te-wr');
  if (teWrEl) { teWrEl.textContent = aiwrDisp + '%'; teWrEl.className = 'metric-val ' + (aiwr >= 0.55 ? 'green' : 'yellow'); }

  const cl = Number(d.auto_improve_consecutive_losses || 0);
  const clEl = $('te-losses');
  if (clEl) { clEl.textContent = cl; clEl.className = 'metric-val ' + (cl >= 3 ? 'red' : cl >= 1 ? 'yellow' : 'green'); }

  setText('te-opts', d.auto_improve_optimization_count || 0);

  const fltRelax = d.filter_relaxation_active;
  const fltEl = $('te-flt-status');
  if (fltEl) { fltEl.textContent = fltRelax ? 'RELAJADO ▼' : 'NORMAL ✓'; fltEl.className = 'metric-val ' + (fltRelax ? 'yellow' : 'green'); }
  setText('te-flt-adj', d.filter_auto_adjustments || 0);

  // FILTROS
  const af = d.autonomous_filters || {};
  setText('f-adx', af.adx_threshold != null ? fmt(af.adx_threshold, 1) : '—');
  setText('f-rsi-buy', af.rsi_buy_threshold != null ? fmt(af.rsi_buy_threshold, 1) : '—');
  setText('f-rsi-sell', af.rsi_sell_threshold != null ? fmt(af.rsi_sell_threshold, 1) : '—');
  setText('f-conf', af.min_confidence != null ? fmt(af.min_confidence * 100, 1) + '%' : '—');
  setText('f-vol', af.volume_buy_threshold != null ? fmt(af.volume_buy_threshold, 2) : '—');
  setText('f-sl', af.stop_loss_atr_multiplier != null ? fmt(af.stop_loss_atr_multiplier, 2) : '—');
  setText('f-tp', af.take_profit_atr_multiplier != null ? fmt(af.take_profit_atr_multiplier, 2) : '—');

  // POSICIÓN
  updatePosition(d);

  // SEÑALES
  updateSignals(d);

  // SISTEMA
  const exch = (d.exchange_status || '—').toUpperCase();
  const exchEl = $('sys-exchange');
  if (exchEl) { exchEl.textContent = exch; exchEl.className = 'metric-val ' + (['CONNECTED','OK'].includes(exch) ? 'green' : 'red'); }

  const db = (d.database_status || '—').toUpperCase();
  const dbEl = $('sys-db');
  if (dbEl) { dbEl.textContent = db; dbEl.className = 'metric-val ' + (['CONNECTED','SQLITE_FALLBACK'].includes(db) ? (db==='SQLITE_FALLBACK'?'yellow':'green') : 'red'); }

  setText('sys-lat', fmt(d.api_latency_p95_ms || d.api_latency_ms, 1) + ' ms');
  const cb = Number(d.circuit_breaker_trips || 0);
  const cbEl = $('sys-cb'); if (cbEl) { cbEl.textContent = cb; cbEl.className = 'metric-val ' + (cb > 0 ? 'yellow' : 'green'); }
  const rn = Number(d.exchange_reconnections || 0);
  const rnEl = $('sys-reconn'); if (rnEl) { rnEl.textContent = rn; rnEl.className = 'metric-val ' + (rn > 0 ? 'yellow' : 'green'); }
  const emg = d.emergency_stop_active;
  const emgEl = $('sys-emg'); if (emgEl) { emgEl.textContent = emg ? 'ACTIVO 🛑' : 'NO'; emgEl.className = 'metric-val ' + (emg ? 'red' : 'green'); }
  setText('sys-invmode', d.investment_mode || '—');
  const dyn = d.dynamic_exit_enabled;
  const dynEl = $('sys-dyn-exit'); if (dynEl) { dynEl.textContent = dyn ? 'SÍ ✓' : 'NO'; dynEl.className = 'metric-val ' + (dyn ? 'green' : 'muted'); }
  const eis = Number(d.exit_intelligence_score || 0);
  const eisEl = $('sys-exit-score');
  if (eisEl) { eisEl.textContent = fmt(eis, 3); eisEl.className = 'metric-val ' + (eis > 0.7 ? 'red' : eis > 0.4 ? 'yellow' : 'green'); }

  // DECISIÓN
  setText('decision-reason', d.decision_reason || d.exit_intelligence_reason || '—');

  // LOGS
  if (Array.isArray(d.logs) && d.logs.length > 0) updateLogs(d.logs);

  // TRADES
  if (Array.isArray(d.trade_history) && d.trade_history.length > 0) {
    allTrades = d.trade_history;
    renderTrades();
  }
}

function updatePosition(d) {
  const el = $('position-content'); if (!el) return;
  const hasPos = d.has_open_position || d.open_position_side;
  if (!hasPos) {
    el.innerHTML = '<div style="color:var(--text3);text-align:center;padding:20px 0">Sin posición abierta</div>';
    return;
  }
  const side = (d.open_position_side || '').toUpperCase();
  const sideColor = side === 'BUY' ? 'var(--green)' : 'var(--red)';
  const upnl = Number(d.unrealized_pnl || 0);
  el.innerHTML = `
    <div class="metric-row"><span class="metric-label">Lado</span><span class="metric-val" style="color:${sideColor};font-weight:700">${side||'—'}</span></div>
    <div class="metric-row"><span class="metric-label">Cantidad</span><span class="metric-val">${fmt(d.open_position_qty, 6)}</span></div>
    <div class="metric-row"><span class="metric-label">Entrada</span><span class="metric-val">$${fmt(d.open_position_entry, 4)}</span></div>
    <div class="metric-row"><span class="metric-label">Precio Actual</span><span class="metric-val">$${fmt(d.current_price||d.price, 4)}</span></div>
    <div class="metric-row"><span class="metric-label">Stop Loss</span><span class="metric-val" style="color:var(--red)">$${fmt(d.open_position_sl, 4)}</span></div>
    <div class="metric-row"><span class="metric-label">Take Profit</span><span class="metric-val" style="color:var(--green)">$${fmt(d.open_position_tp, 4)}</span></div>
    <div class="metric-row"><span class="metric-label">PnL No Real.</span><span class="metric-val ${pnlClass(upnl)}">${pnlSign(upnl)}$${fmt(upnl,4)}</span></div>
  `;
}

function updateSignals(d) {
  const el = $('signals-grid'); if (!el) return;
  const sigs = d.signals || {};
  const names = {trend:'Tendencia',momentum:'Momentum',volume:'Volumen',volatility:'Volatilidad',structure:'Estructura',order_flow:'Order Flow'};
  el.innerHTML = Object.entries(names).map(([k,label]) => {
    const v = (sigs[k] || 'NEUTRAL').toUpperCase();
    return `<div class="metric-row"><span class="metric-label">${label}</span><span class="sig-badge ${sigClass(v)}" style="font-size:11px">${v}</span></div>`;
  }).join('');
  const conf = Number(d.confidence || 0);
  setText('sig-conf-val', fmt(conf*100,1)+'%');
  bar('sig-conf-bar', conf*100, conf>=0.65?'progress-green':conf>=0.45?'progress-yellow':'progress-red');
}

// ─── Logs ────────────────────────────────────────────────────────────────────
const _logSeen = new Set();
function updateLogs(logs) {
  const container = $('log-container'); if (!container) return;
  let added = false;
  logs.slice(-30).forEach(log => {
    const key = (log.time||'') + (log.message||'');
    if (_logSeen.has(key)) return;
    _logSeen.add(key);
    const lv = (log.level || 'INFO').toUpperCase();
    const div = document.createElement('div');
    div.className = 'log-line log-' + lv;
    div.innerHTML = `<span class="log-time">${(log.time||'--:--').substring(0,8)}</span><span class="log-level">${lv.substring(0,4)}</span><span class="log-msg">${escHtml(String(log.message||'').substring(0,200))}</span>`;
    container.appendChild(div);
    added = true;
  });
  if (added) container.scrollTop = container.scrollHeight;
  while (container.children.length > 200) container.removeChild(container.firstChild);
}

function clearLog() {
  const c = $('log-container'); if (c) { c.innerHTML = ''; _logSeen.clear(); }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Trades ──────────────────────────────────────────────────────────────────
function filterTrades() {
  statusFilter = ($('filter-status').value || '').toUpperCase();
  currentPage = 1;
  renderTrades();
}

function renderTrades() {
  let trades = allTrades;
  if (statusFilter) trades = trades.filter(t => (t.status||'').toUpperCase() === statusFilter);
  setText('trades-count', trades.length + ' trades');
  const total = trades.length;
  const pages = Math.max(1, Math.ceil(total / perPage));
  const start = (currentPage - 1) * perPage;
  const slice = trades.slice(start, start + perPage);

  const tbody = $('trades-body');
  if (!tbody) return;
  if (!slice.length) {
    tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;color:var(--text3);padding:16px">Sin trades</td></tr>';
  } else {
    tbody.innerHTML = slice.map((t,i) => {
      const pnl = Number(t.pnl || 0);
      const pnlPct = Number(t.pnl_percent || 0);
      const rowCls = t.status === 'OPEN' ? 'open-trade' : pnl > 0 ? 'profit' : pnl < 0 ? 'loss' : '';
      const side = (t.side||'').toUpperCase();
      const sideColor = side==='BUY'?'var(--green)':'var(--red)';
      return `<tr class="${rowCls}">
        <td style="color:var(--text3)">${start+i+1}</td>
        <td style="font-weight:600">${t.pair||t.symbol||'—'}</td>
        <td style="color:${sideColor};font-weight:700">${side}</td>
        <td>$${fmt(t.entry||t.entry_price,4)}</td>
        <td>${t.exit||t.exit_price ? '$'+fmt(t.exit||t.exit_price,4) : '—'}</td>
        <td>${fmt(t.size||t.quantity,6)}</td>
        <td class="${pnlClass(pnl)}">${pnlSign(pnl)}$${fmt(pnl,4)}</td>
        <td class="${pnlClass(pnlPct)}">${pnlSign(pnlPct)}${fmt(pnlPct,2)}%</td>
        <td><span class="sig-badge ${t.status==='OPEN'?'sig-buy':pnl>=0?'sig-neutral':'sig-sell'}" style="font-size:10px">${t.status||'—'}</span></td>
        <td style="color:var(--text3)">${t.duration||'—'}</td>
        <td style="color:var(--text3)">${(t.time||t.entry_time||'—').substring(0,16)}</td>
      </tr>`;
    }).join('');
  }

  // Paginación
  const pg = $('trades-pagination');
  if (pg) {
    pg.innerHTML = '';
    for (let p = 1; p <= pages; p++) {
      const btn = document.createElement('button');
      btn.textContent = p;
      btn.style.cssText = `background:${p===currentPage?'var(--bg4)':'var(--bg3)'};border:1px solid var(--border2);color:${p===currentPage?'var(--text)':'var(--text3)'};border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer;font-family:inherit`;
      btn.onclick = () => { currentPage = p; renderTrades(); };
      pg.appendChild(btn);
    }
  }
}

// ─── Controls ────────────────────────────────────────────────────────────────
async function sendControl(action) {
  try {
    const r = await fetch('/api/control/' + action, {method:'POST'});
    const d = await r.json();
    if (!d.success) alert('Error: ' + (d.error || 'Unknown'));
  } catch(e) { alert('Error de conexión: ' + e.message); }
}

function confirmEmergency() { $('modal-overlay').classList.add('active'); }
function closeModal() { $('modal-overlay').classList.remove('active'); }
function doEmergency() { closeModal(); sendControl('emergency'); }

// ─── SSE Connection ──────────────────────────────────────────────────────────
function connectSSE() {
  const ind = $('conn-indicator');
  if (_sse) { try { _sse.close(); } catch(_){} }
  _sse = new EventSource('/api/stream');

  _sse.onopen = () => { if (ind) ind.className = 'connected'; };

  _sse.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.error) return;
      updateDashboard(data);
    } catch(err) { console.warn('SSE parse error', err); }
  };

  _sse.onerror = () => {
    if (ind) ind.className = 'error';
    _sse.close();
    setTimeout(connectSSE, 3000);
  };
}

// Cargar trades del endpoint de BD periódicamente
async function loadTrades() {
  try {
    const r = await fetch('/api/all_trades');
    const d = await r.json();
    if (d.trades && d.trades.length > 0) {
      allTrades = d.trades;
      renderTrades();
    }
  } catch(_) {}
}

// ─── Init ────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  connectSSE();
  loadTrades();
  setInterval(loadTrades, 30000); // Refrescar trades cada 30s desde BD
});
</script>
</body>
</html>
```

---

## H. VERIFICACIÓN FINAL

Después de aplicar todos los cambios, ejecuta en orden:

```bash
# 1. Verificar sintaxis Python
python -m py_compile reco_trading/main.py && echo "OK main.py"
python -m py_compile reco_trading/core/bot_engine.py && echo "OK bot_engine.py"
python -m py_compile reco_trading/ui/dashboard.py && echo "OK dashboard.py"
python -m py_compile web_site/dashboard_server.py && echo "OK dashboard_server.py"
python -m py_compile reco_trading/risk/capital_profile.py && echo "OK capital_profile.py"

# 2. Verificar sintaxis bash
bash -n run.sh && echo "OK run.sh"

# 3. Verificar imports básicos
python -c "from reco_trading.main import run; print('OK imports main')"
python -c "from web_site.dashboard_server import create_app; print('OK imports dashboard_server')"
python -c "from reco_trading.ui.dashboard import TerminalDashboard, DashboardSnapshot; print('OK imports dashboard')"
python -c "from reco_trading.risk.capital_profile import CapitalProfileManager; m=CapitalProfileManager(); print('OK capital profiles:', [p.name for p in m._profiles])"
```

Si algún comando falla, corrige el error antes de continuar.

---

## RESUMEN EJECUTIVO DE CAMBIOS

| Archivo | Cambio | Impacto |
|---|---|---|
| `reco_trading/main.py` | Fix race condition + DSN + StateManager | Web dashboard arranca correctamente con el bot |
| `run.sh` | Fix `exec python` + banner informativo | Ctrl+C funciona; usuario ve URL del dashboard |
| `reco_trading/risk/capital_profile.py` | NANO/MICRO: `max_trades_per_day=25`, cooldown=5min | Habilita objetivo de 20+ trades/día |
| `reco_trading/core/bot_engine.py` | Safety clamps + counter + nuevos campos snapshot | Filtros nunca se vuelven peligrosos; datos visibles en dashboards |
| `web_site/dashboard_server.py` | Loop persistente SSE + manejo desconexión + endpoint capital | Sin memory leaks; SSE estable 24/7 |
| `reco_trading/ui/dashboard.py` | Panel "Trade Engine" reemplaza "LLM Gate" | Información relevante sin referencias a Ollama |
| `web_site/templates/index.html` | Rediseño completo profesional | Dashboard web visual, funcional, con capital profile |
