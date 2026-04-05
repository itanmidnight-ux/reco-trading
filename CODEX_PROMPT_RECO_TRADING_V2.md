# CODEX MASTER PROMPT v2 — reco-trading Full Professional Overhaul
**Para: OpenAI Codex / GPT-4.1 / Claude Code**
*Prompt de ingeniería profesional — Leer COMPLETO antes de modificar cualquier archivo*

---

## ⚠️ REGLAS ABSOLUTAS ANTES DE EMPEZAR

1. **Lee este documento completo** antes de tocar un solo archivo.
2. **No borres funcionalidad existente** que no esté listada como "a eliminar".
3. **No rompas imports existentes** — si mueves código, actualiza todos los imports.
4. **Cada archivo modificado debe quedar 100% funcional** — sin truncar, sin `# ... resto del código`.
5. **Ejecuta en orden estricto**: Bloque A → B → C → D → E → F.
6. **No uses `ollama` ni llamadas HTTP a LLM local** en ningún código nuevo — Ollama fue removido del proyecto. Toda la "confirmación" de trades es rule-based (ya existente en `LLMTradeConfirmator`).
7. El proyecto usa: Python 3.11+, asyncio, Flask (threaded), Rich TUI, ccxt, SQLAlchemy async, pydantic-settings.

---

## CONTEXTO DEL PROYECTO

**reco-trading** es un bot de trading de criptomonedas en Python. Arquitectura:
- `reco_trading/core/bot_engine.py` — motor principal asyncio
- `reco_trading/main.py` — entry point: inicia web dashboard en thread + bot en thread
- `main.py` (raíz) — wrapper de compatibilidad que llama `reco_trading.main.run()`
- `web_site/dashboard_server.py` — Flask dashboard web (port 9000)
- `reco_trading/ui/dashboard.py` — Terminal TUI (Rich `Live`)
- `reco_trading/risk/capital_profile.py` — Perfiles de capital automáticos (NANO→WHALE)
- `reco_trading/core/adaptive_config.py` — Motor de configuración adaptativa por régimen
- `reco_trading/core/bot_engine.py` métodos `_get_default_filter_config`, `_apply_symbol_filter_config`, `_calibrate_filter_config_with_recent_market_data` — sistema de filtros adaptativos
- `run.sh` — script bash de inicio que pide modo (testnet/producción) y luego ejecuta `python main.py`

---

## BLOQUE A — BUGS CRÍTICOS EN `reco_trading/main.py`

### A1. BUG: `_verify_database_connection` usa `settings.postgres_dsn` directamente sin fallback

**Ubicación:** `reco_trading/main.py`, función `_verify_database_connection`

**Problema:** Si `settings.postgres_dsn` es `None` (cuando se usa SQLite o DATABASE_URL), la función falla con `TypeError` y la advertencia al usuario es confusa. Además, el bot arranca igual (el error es silenciado), pero el mensaje de warning dice "Database unavailable" aunque SQLite esté funcionando.

**Corrección:** Reemplazar la función completa por:

```python
async def _verify_database_connection(settings: Settings) -> str:
    """Verifica la mejor DSN disponible con fallback a SQLite."""
    from reco_trading.core.bot_engine import _get_database_dsn
    dsn = _get_database_dsn(settings)
    if not dsn:
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
        db_path = os.path.join(project_root, "data", "reco_trading.db")
        dsn = f"sqlite+aiosqlite:///{db_path}"
    repository = Repository(dsn)
    try:
        await repository.setup()
        return dsn
    except Exception:
        raise
    finally:
        await repository.close()
```

### A2. BUG: El web dashboard se inicia ANTES de que `set_bot_instance_getter` esté completamente registrado, causando que Flask intente servir datos de un bot que aún no existe

**Ubicación:** `reco_trading/main.py`, función `run()`

**Problema:** El orden actual es:
1. `_start_web_dashboard()` → Flask arranca, pero `_global_bot_instance` es `None`
2. `_verify_database_connection()` → puede fallar
3. `bot_thread.start()` → el bot arranca **después** de Flask

Esto significa que en los primeros segundos, el dashboard web devuelve snapshot vacío y el frontend puede quedarse en estado "WAITING" indefinidamente o mostrar errores de conexión.

**Corrección:** Cambiar el orden en `run()` para que el bot_thread se inicie antes y el web dashboard tenga `set_bot_instance_getter` apuntando correctamente:

```python
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

    # 1. Registrar getter ANTES de arrancar Flask para evitar race condition
    set_bot_instance_getter(get_bot_instance)

    # 2. Verificar BD con manejo de errores robusto
    try:
        asyncio.run(_verify_database_connection(settings))
        logger.info("Database connection verified successfully")
    except Exception as exc:
        logger.warning(
            "Database unavailable at startup (bot usará fallback SQLite): %s", exc,
        )

    # 3. Arrancar bot thread primero para que tenga tiempo de inicializarse
    bot_thread = threading.Thread(target=_run_bot, args=(settings, None), daemon=True, name="bot-engine")
    bot_thread.start()

    # 4. Arrancar web dashboard después (getter ya registrado, bot en proceso de init)
    _start_web_dashboard(logger)

    try:
        _join_bot_thread_or_exit(bot_thread, logger)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
```

### A3. BUG: `state_manager=None` siempre — el StateManager nunca se inicializa

**Ubicación:** `reco_trading/main.py`, línea `bot_thread = threading.Thread(target=_run_bot, args=(settings, None), ...)`

**Problema:** El `StateManager` (en `reco_trading/ui/state_manager.py`) nunca se instancia, por lo que:
- `bot.state_manager` es siempre `None`
- Los logs del bot no se agregan al snapshot vía state_manager
- Los controles desde el web dashboard no se enrutan correctamente

**Corrección:** Importar e instanciar el StateManager si existe:

```python
def run() -> None:
    # ... (resto igual que A2) ...
    
    # Intentar inicializar StateManager para conectar TUI con WebDashboard
    state_manager = None
    try:
        from reco_trading.ui.state_manager import StateManager
        state_manager = StateManager()
        logger.info("StateManager initialized successfully")
    except ImportError:
        logger.info("StateManager not available, running without it")
    except Exception as exc:
        logger.warning("StateManager failed to initialize: %s — running without it", exc)

    bot_thread = threading.Thread(target=_run_bot, args=(settings, state_manager), daemon=True, name="bot-engine")
    bot_thread.start()
    # ... resto igual ...
```

---

## BLOQUE B — BUGS CRÍTICOS EN `run.sh`

### B1. BUG: `run.sh` no inicia el web dashboard — solo ejecuta `python main.py`

**Problema:** El `run.sh` ejecuta `python main.py` sin ninguna preparación del entorno web. Aunque `main.py` arranca Flask en thread, no hay mensaje al usuario indicando la URL del dashboard web durante el startup interactivo.

**Corrección:** Al final de `run.sh`, antes de `python main.py`, agregar mensaje informativo y asegurar que el puerto esté libre:

```bash
# --- Verificar que el puerto del dashboard web esté disponible ---
WEB_PORT="${WEB_DASHBOARD_PORT:-9000}"
if command -v lsof >/dev/null 2>&1; then
    if lsof -i ":${WEB_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Puerto ${WEB_PORT} ya está en uso. El dashboard usará el siguiente puerto disponible."
    fi
fi

echo "──────────────────────────────────────────────"
echo "  🚀 Iniciando Reco-Trading Bot"
echo "  📊 Dashboard Web: http://127.0.0.1:${WEB_PORT}"
echo "  📡 Dashboard Terminal: activo en esta consola"
echo "  ⌨️  Ctrl+C para detener"
echo "──────────────────────────────────────────────"
echo ""

exec python main.py
```

**IMPORTANTE:** Cambiar `python main.py` por `exec python main.py` para que las señales (Ctrl+C) lleguen correctamente al proceso Python.

### B2. BUG: Doble definición de `upsert_env_var` en `run.sh`

**Problema:** La función `upsert_env_var` está definida DOS veces en `run.sh`: una vez dentro del heredoc que crea `scripts/lib/runtime_env.sh` y otra vez directamente en el script. La segunda definición sobreescribe a la primera, pero la función del runtime_env.sh ya no está disponible si el script falla antes de sourcing.

**Corrección:** Consolidar: mantener la función definida UNA sola vez, directamente en `run.sh`, ANTES del bloque `ensure_runtime_env_lib_exists`. Eliminar la definición duplicada que está dentro del heredoc (dejar el heredoc solo con `load_dotenv_file`, `build_postgres_dsn_from_config`, `build_sqlite_dsn`).

---

## BLOQUE C — SISTEMA DE FILTROS ADAPTATIVOS (Validación + Correcciones)

### C1. VALIDACIÓN: El sistema de filtros está bien diseñado — correcciones menores

Después del análisis completo del código en `bot_engine.py` (`_get_default_filter_config`, `_apply_symbol_filter_config`, `_calibrate_filter_config_with_recent_market_data`, `_auto_adjust_filters_for_capital`), el sistema es sólido. Se identifican estos problemas:

### C2. BUG: `_get_default_filter_config` no tiene límites de seguridad explícitos documentados

**Ubicación:** `reco_trading/core/bot_engine.py`, método `_get_default_filter_config`

**Problema:** Los valores por defecto son correctos, pero no hay validación en `_apply_symbol_filter_config` que evite que los filtros se relajen peligrosamente con capital alto.

**Corrección:** En `_apply_symbol_filter_config`, agregar clamps de seguridad después de aplicar el config:

```python
def _apply_symbol_filter_config(self, filter_config: dict[str, float]) -> None:
    # ... (código existente igual) ...
    
    # SAFETY CLAMPS: evitar filtros peligrosamente permisivos con capital grande
    SAFETY_LIMITS = {
        "adx_threshold": (8.0, 35.0),          # nunca < 8 ni > 35
        "rsi_buy_threshold": (40.0, 70.0),      # zona válida de compra
        "rsi_sell_threshold": (30.0, 60.0),     # zona válida de venta
        "min_confidence": (0.45, 0.85),         # confianza mínima razonable
        "volume_buy_threshold": (0.50, 2.50),   # ratio de volumen válido
        "volume_sell_threshold": (0.30, 1.50),
        "atr_low_threshold": (0.001, 0.010),    # evitar trades en mercado muerto
        "atr_high_threshold": (0.010, 0.060),   # evitar trades en extrema volatilidad
        "stop_loss_atr_multiplier": (1.0, 4.0),
        "take_profit_atr_multiplier": (1.5, 6.0),
    }
    for key, (lo, hi) in SAFETY_LIMITS.items():
        if key in self.base_filter_config:
            original = self.base_filter_config[key]
            clamped = max(lo, min(hi, original))
            if clamped != original:
                self.logger.warning(
                    "Filter safety clamp: %s %.4f → %.4f (limits [%.4f, %.4f])",
                    key, original, clamped, lo, hi
                )
                self.base_filter_config[key] = clamped
    
    self.runtime_filter_config = dict(self.base_filter_config)
    self.snapshot["autonomous_filters"] = self.runtime_filter_config.copy()
    self.logger.info(f"Applied filter config for {self.symbol}: {self.runtime_filter_config}")
```

### C3. BUG: `_auto_adjust_filters_for_capital` aumenta `min_confidence` para capital grande pero NO ajusta el cooldown ni `max_trades_per_day`

**Problema:** Con capital grande (LARGE/WHALE), los filtros se aprietan (más confianza requerida), pero el `CapitalProfile.max_trades_per_day` sigue siendo el del perfil sin considerar que filtros más estrictos = menos señales = objetivo de 20+ trades/día imposible para capital pequeño.

**Corrección:** En `_auto_adjust_filters_for_capital`, para perfiles NANO/MICRO/SMALL, relajar levemente los filtros (no en la dirección de más ruido, sino en la dirección de más oportunidades válidas):

```python
def _auto_adjust_filters_for_capital(self) -> None:
    """Ajusta filtros según perfil de capital detectado."""
    profile = getattr(self, "_current_capital_profile", None)
    if not profile:
        return
    
    profile_name = getattr(profile, "name", "UNKNOWN")
    
    if profile_name in ("NANO", "MICRO"):
        # Capital pequeño: necesitamos más trades para ser rentables
        # Relajar levemente para capturar más oportunidades VÁLIDAS
        # NUNCA bajar min_confidence por debajo del floor de seguridad del perfil
        _floor = getattr(profile, "min_confidence", 0.62)
        self.runtime_filter_config["min_confidence"] = max(
            _floor,
            min(self.runtime_filter_config.get("min_confidence", _floor), _floor + 0.05)
        )
        # ADX más bajo = más señales en ranging market (útil para scalping)
        self.runtime_filter_config["adx_threshold"] = max(
            10.0, self.runtime_filter_config.get("adx_threshold", 18.0) - 3.0
        )
        self.logger.info(
            "Capital profile %s: relaxed filters for trade frequency. "
            "min_conf=%.2f adx>=%.1f",
            profile_name,
            self.runtime_filter_config["min_confidence"],
            self.runtime_filter_config["adx_threshold"],
        )
    
    elif profile_name in ("LARGE", "WHALE", "INSTITUTIONAL"):
        # Capital grande: priorizar preservación de capital
        # min_confidence mínimo 0.72, ADX mínimo 20
        self.runtime_filter_config["min_confidence"] = max(
            self.runtime_filter_config.get("min_confidence", 0.55), 0.72
        )
        self.runtime_filter_config["adx_threshold"] = max(
            self.runtime_filter_config.get("adx_threshold", 15.0), 20.0
        )
        self.logger.info(
            "Capital profile %s: tightened filters for capital preservation. "
            "min_conf=%.2f adx>=%.1f",
            profile_name,
            self.runtime_filter_config["min_confidence"],
            self.runtime_filter_config["adx_threshold"],
        )
    
    self.snapshot["autonomous_filters"] = self.runtime_filter_config.copy()
```

### C4. MEJORA: Exponer `_current_capital_profile` en snapshot para el dashboard

**Ubicación:** En el método que actualiza el capital profile (buscar `CapitalProfileManager.get_profile` en `bot_engine.py`)

Asegurarse de que cuando se detecta/actualiza el perfil, se guarda en `self._current_capital_profile` y se llama `_auto_adjust_filters_for_capital()`:

```python
# Después de cualquier llamada a capital_profile_manager.get_profile(equity):
new_profile = self.capital_profile_manager.get_profile(current_equity)
if new_profile and (not hasattr(self, "_current_capital_profile") or 
                    getattr(self._current_capital_profile, "name", None) != new_profile.name):
    self._current_capital_profile = new_profile
    self._auto_adjust_filters_for_capital()
    self.logger.info("Capital profile updated: %s (equity=%.2f)", new_profile.name, current_equity)
```

---

## BLOQUE D — DASHBOARD TERMINAL TUI (`reco_trading/ui/dashboard.py`)

### D1. CAMBIO OBLIGATORIO: Eliminar panel "LLM Gate" — Reemplazar por "Trade Engine Stats"

**Problema:** El panel "🤖 LLM Gate" / "LLM Gate" muestra información de Ollama que fue eliminado del proyecto. El espacio debe reutilizarse para mostrar información útil del motor de trading.

**En el archivo `reco_trading/ui/dashboard.py`:**

**Paso 1:** En el dataclass `DashboardSnapshot`, los campos `llm_mode` y `llm_trade_confirmator` son obsoletos pero NO los elimines (puede haber código que los use). Solo deja de renderizarlos.

**Paso 2:** Agregar campos nuevos al dataclass `DashboardSnapshot`:
```python
# Nuevos campos para Trade Engine Stats
auto_improve_consecutive_losses: int = 0
auto_improve_optimization_count: int = 0
auto_optimized_params: dict[str, Any] = field(default_factory=dict)
filter_auto_adjustments: int = 0       # cuántas veces se ajustaron filtros automáticamente
capital_profile_name: str = ""         # nombre del perfil activo
trades_target_daily: int = 0           # objetivo de trades/día del perfil
filter_relaxation_active: bool = False # si los filtros están relajados para más frecuencia
```

**Paso 3:** En el método `from_mapping`, mapear los nuevos campos:
```python
auto_improve_consecutive_losses=int(data.get("auto_improve_consecutive_losses", 0) or 0),
auto_improve_optimization_count=int(data.get("auto_improve_optimization_count", 0) or 0),
auto_optimized_params=dict(data.get("auto_optimized_params", {}) or {}),
filter_auto_adjustments=int(data.get("filter_auto_adjustments", 0) or 0),
capital_profile_name=str(data.get("capital_profile") or ""),
trades_target_daily=int(data.get("trades_target_daily", 0) or 0),
filter_relaxation_active=bool(data.get("filter_relaxation_active", False)),
```

**Paso 4:** Reemplazar la construcción del `llm_table` por el nuevo `engine_table`:

ELIMINAR todo el bloque:
```python
# ── LLM GATE PANEL ───────────────────────────────────────
llm = snap.llm_trade_confirmator or {}
...
llm_table.add_row("Avg Latency", ...)
```

REEMPLAZAR por:
```python
# ── TRADE ENGINE STATS PANEL ─────────────────────────────
engine_table = Table.grid(expand=True, padding=(0, 1))
engine_table.add_column(style="dim", min_width=18)
engine_table.add_column(style="bold white")

# Perfil de capital activo
profile_name = snap.capital_profile_name or snap.capital_profile or "—"
profile_colors = {
    "NANO": "red", "MICRO": "yellow", "SMALL": "cyan",
    "MEDIUM": "green", "LARGE": "bright_green",
    "WHALE": "bright_magenta", "INSTITUTIONAL": "bright_white"
}
profile_color = profile_colors.get(profile_name.upper(), "white")
engine_table.add_row("Capital Profile", Text(profile_name, style=f"bold {profile_color}"))

# Target de trades
target = snap.trades_target_daily
engine_table.add_row(
    "Target Trades/day",
    Text(f"{target}" if target > 0 else "—", style="cyan" if target >= 20 else "yellow")
)

# Trades hoy vs target
trades_color = "green" if snap.trades_today >= max(target, 1) else "yellow" if snap.trades_today >= max(target // 2, 1) else "white"
engine_table.add_row("Trades Today", Text(str(snap.trades_today), style=trades_color))

# Auto-improve
engine_table.add_row("AutoImprove Wins", Text(f"{_fmt_pct(snap.auto_improve_win_rate)}", style="green" if (snap.auto_improve_win_rate or 0) >= 0.55 else "yellow"))
engine_table.add_row("Consec. Losses", _losses_styled(snap.auto_improve_consecutive_losses))
engine_table.add_row("Optimizations", Text(str(snap.auto_improve_optimization_count), style="cyan"))

# Estado de filtros adaptativos
filter_status = "RELAJADO 📈" if snap.filter_relaxation_active else "NORMAL ✓"
filter_style = "yellow" if snap.filter_relaxation_active else "green"
engine_table.add_row("Filtros Auto", Text(filter_status, style=filter_style))
engine_table.add_row("Ajustes Auto", Text(str(snap.filter_auto_adjustments), style="dim cyan"))

# Parámetros optimizados (mostrar los más relevantes)
opt_params = snap.auto_optimized_params or {}
if opt_params:
    conf_opt = opt_params.get("min_confidence")
    if conf_opt is not None:
        engine_table.add_row("Opt. Conf Min", Text(f"{float(conf_opt)*100:.1f}%", style="bright_cyan"))
```

**Paso 5:** En el layout, reemplazar TODAS las referencias a `llm_table` por `engine_table`:
- En layout compacto: `Panel(engine_table, title="⚡ Trade Engine", border_style="bright_yellow")`
- En layout wide right column: `Panel(engine_table, title="⚡ Trade Engine", border_style="bright_yellow")`

### D2. MEJORA: Panel de Filtros activos — mostrar comparación base vs actual

En el panel de filtros (ya existente), agregar indicador visual de si el filtro fue ajustado automáticamente vs el valor base:

```python
# ── FILTER STATUS (mejorado) ──────────────────────────────
af = snap.autonomous_filters
filter_table = Table.grid(expand=True, padding=(0, 1))
filter_table.add_column(style="dim", min_width=16)
filter_table.add_column(style="bold cyan")
filter_table.add_column(style="dim", width=6)  # columna de indicador de cambio
if af:
    def _filter_row(label, key, fmt=".1f", inverted=False):
        val = af.get(key)
        if val is None:
            return label, "—", ""
        formatted = f"{val:{fmt}}"
        # Indicador de si está en zona agresiva (relajado) o conservadora
        # Para ADX y confidence: valores bajos = más permisivo = ▼
        # Para RSI: depende del tipo
        indicator = ""
        return label, formatted, indicator
    
    filter_table.add_row("ADX ≥", _fmt_float(af.get("adx_threshold"), 1), "")
    filter_table.add_row("RSI Buy ≥", _fmt_float(af.get("rsi_buy_threshold"), 1), "")
    filter_table.add_row("RSI Sell ≤", _fmt_float(af.get("rsi_sell_threshold"), 1), "")
    filter_table.add_row("Min Conf", f"{_fmt_float((af.get('min_confidence') or 0) * 100, 1)}%", "")
    filter_table.add_row("Vol Buy ≥", _fmt_float(af.get("volume_buy_threshold"), 2), "")
    filter_table.add_row("SL ATR ×", _fmt_float(af.get("stop_loss_atr_multiplier"), 2), "")
    filter_table.add_row("TP ATR ×", _fmt_float(af.get("take_profit_atr_multiplier"), 2), "")
else:
    filter_table.add_row("Filters", "loading...", "")
```

### D3. MEJORA: Pie de página (footer) — mostrar URL correcta del dashboard

Actualmente el footer siempre dice `http://localhost:9000`. Cambiarlo para leer el puerto real:

```python
import os as _os_dash
_web_port = _os_dash.getenv("WEB_DASHBOARD_PORT", "9000")
footer = Text.assemble(
    ("◈ Reco Trading TUI  ", "dim"),
    ("│  ", "dim"),
    ("Ctrl+C", "bold white"),
    (" to stop  ", "dim"),
    ("│  ", "dim"),
    ("Web: ", "dim"),
    (f"http://localhost:{_web_port}", "bright_cyan"),
    ("  │  Filters: ", "dim"),
    ("AUTO", "bright_yellow"),
)
```

---

## BLOQUE E — DASHBOARD WEB (`web_site/dashboard_server.py` + `web_site/templates/index.html`)

### E1. BUG CRÍTICO: El dashboard web no se actualiza correctamente porque `get_bot_snapshot()` crea un event loop nuevo en cada llamada SSE

**Ubicación:** `web_site/dashboard_server.py`, función `_run_async`

**Problema:** La función `_run_async` crea un `new_event_loop()` en cada llamada y lo cierra después. Esto está bien para operaciones de BD aisladas, pero cuando se llama desde el endpoint SSE `/api/stream` (que hace polling cada 1.5s), se crea y destruye un loop por cada tick — esto puede causar warnings de "coroutine was never awaited" y memory leaks en sistemas corriendo >24h.

**Corrección:** Usar un executor dedicado con un loop persistente para el fallback repository:

```python
import concurrent.futures
_db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashboard-db")
_db_loop: asyncio.AbstractEventLoop | None = None
_db_loop_lock = threading.Lock()

def _get_or_create_db_loop() -> asyncio.AbstractEventLoop:
    global _db_loop
    with _db_loop_lock:
        if _db_loop is None or _db_loop.is_closed():
            _db_loop = asyncio.new_event_loop()
        return _db_loop

def _run_async(coro: Any) -> Any:
    """Ejecuta coroutines de forma segura desde contexto Flask usando loop persistente."""
    try:
        loop = _get_or_create_db_loop()
        if loop.is_running():
            # Si el loop ya corre (no debería en Flask sync), usar run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=10.0)
        else:
            return loop.run_until_complete(coro)
    except Exception as e:
        logger.error("_run_async error: %s", e)
        raise
```

### E2. BUG: El endpoint `/api/stream` (SSE) no tiene manejo de desconexión del cliente

**Corrección:** Agregar detección de cliente desconectado:

```python
@app.route('/api/stream')
def api_stream():
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
                logger.debug("SSE client disconnected")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error("SSE error (#%d): %s", consecutive_errors, e)
                if consecutive_errors >= 5:
                    logger.warning("SSE: too many consecutive errors, stopping stream")
                    break
                yield f"data: {json.dumps({'error': str(e), 'retry': 3000})}\n\n"
                time.sleep(3)

    return Response(
        event_generator(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )
```

### E3. MEJORA: Agregar endpoint `/api/capital_profile` para exponer info de perfiles

```python
@app.route('/api/capital_profile')
def api_capital_profile():
    unauthorized = _require_dashboard_auth()
    if unauthorized:
        return unauthorized
    snapshot = get_bot_snapshot()
    return jsonify({
        "capital_profile": snapshot.get("capital_profile", "UNKNOWN"),
        "operable_capital_usdt": snapshot.get("operable_capital_usdt", 0.0),
        "capital_reserve_ratio": snapshot.get("capital_reserve_ratio", 0.0),
        "min_cash_buffer_usdt": snapshot.get("min_cash_buffer_usdt", 0.0),
        "capital_limit_usdt": snapshot.get("capital_limit_usdt", 0.0),
        "filter_relaxation_active": snapshot.get("filter_relaxation_active", False),
        "trades_target_daily": snapshot.get("trades_target_daily", 0),
        "trades_today": snapshot.get("trades_today", 0),
        "autonomous_filters": snapshot.get("autonomous_filters", {}),
    })
```

### E4. MEJORA VISUAL OBLIGATORIA: Rediseño del `web_site/templates/index.html`

El dashboard web actual es funcional pero básico. Reescribir completamente con diseño profesional. Requisitos:

**Tecnología:** HTML5 + CSS3 + JavaScript vanilla (sin frameworks externos). Usar variables CSS para temas. Todo en un solo archivo.

**Layout (grid CSS):**
```
┌─────────────────────────────────────────────┐
│  HEADER: Logo + Par + Estado + Latencia      │
├──────────┬──────────────┬────────────────────┤
│ MARKET   │  PORTFOLIO   │  TRADE ENGINE      │
│ Price    │  Balance     │  Capital Profile   │
│ RSI/ADX  │  PnL         │  Trades Today/Tgt  │
│ Trend    │  Win Rate    │  AutoImprove Stats │
│ Signals  │  Position    │  Filtros Activos   │
├──────────┴──────────────┴────────────────────┤
│  TRADES RECIENTES (tabla paginada)           │
├─────────────────────────────────────────────┤
│  LOG EN TIEMPO REAL (últimas 20 líneas)      │
└─────────────────────────────────────────────┘
```

**Paleta de colores profesional:**
```css
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2235;
    --bg-card-hover: #1f2d44;
    --border-color: #2a3a55;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-yellow: #f59e0b;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --accent-orange: #f97316;
    /* Gradientes */
    --gradient-profit: linear-gradient(135deg, #10b981, #06b6d4);
    --gradient-loss: linear-gradient(135deg, #ef4444, #dc2626);
    --gradient-neutral: linear-gradient(135deg, #475569, #334155);
}
```

**Componentes obligatorios del HTML:**

1. **Header sticky** con: logo "◈ RECO", par activo (grande), badge de estado (RUNNING/PAUSED/etc con color), precio actual, cambio 24h%, latencia API.

2. **Cards de métricas** (estilo fintech moderno):
   - Balance / Equity / PnL del día / PnL sesión — cada uno con mini sparkline simulado (CSS animado)
   - Win Rate con barra de progreso circular (SVG)
   - Trades hoy vs Target del día

3. **Panel de mercado**: precio grande con animación de cambio, RSI con barra visual, ADX con barra, spread, trend badge animado.

4. **Panel "Trade Engine"** (reemplaza LLM Gate):
   - Capital Profile badge (NANO=rojo, MICRO=naranja, SMALL=cian, MEDIUM=verde, LARGE=verde brillante, WHALE=magenta)
   - Target trades/día del perfil
   - Filtros activos con mini barras visuales para ADX, RSI, Confidence
   - Indicador "Filtros: AUTO" / "Relajados para frecuencia" / "Ajustados para capital"
   - Contador de optimizaciones automáticas

5. **Tabla de trades** con:
   - Paginación
   - Color de fila según PnL (verde=ganancia, rojo=pérdida)
   - Filtros: Estado (OPEN/CLOSED), Par, Fecha
   - Columna duración

6. **Log en tiempo real**: panel oscuro con texto monospace, auto-scroll, colores por nivel (INFO=gris, WARNING=amarillo, ERROR=rojo).

7. **Controles del bot**: botones Pause / Resume / Emergency Stop con confirmación modal.

8. **Conexión SSE automática** con reconnect en caso de desconexión:
```javascript
function connectSSE() {
    const source = new EventSource('/api/stream');
    source.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            updateDashboard(data);
        } catch(err) { console.error('SSE parse error', err); }
    };
    source.onerror = () => {
        source.close();
        setTimeout(connectSSE, 3000); // reconectar en 3s
    };
}
```

9. **Sin dependencias externas** — todo CSS y JS inline en el HTML.

---

## BLOQUE F — SNAPSHOT: Nuevos campos a exponer desde `bot_engine.py`

Para que los dashboards (terminal y web) muestren los nuevos datos, agregar estos campos al `self.snapshot` del bot en el método `_sync_ui_state` o equivalente:

```python
# En _sync_ui_state o donde se actualice self.snapshot:

# Capital profile
if hasattr(self, "_current_capital_profile") and self._current_capital_profile:
    prof = self._current_capital_profile
    self.snapshot["trades_target_daily"] = getattr(prof, "max_trades_per_day", 0)
    self.snapshot["capital_profile"] = getattr(prof, "name", "UNKNOWN")

# Filtros adaptativos
self.snapshot["filter_relaxation_active"] = (
    self.runtime_filter_config.get("min_confidence", 0.55) < 
    self.base_filter_config.get("min_confidence", 0.55)
)
self.snapshot["filter_auto_adjustments"] = getattr(self, "_filter_auto_adjustment_count", 0)

# Auto-improver
if hasattr(self, "auto_improver"):
    ai = self.auto_improver
    self.snapshot["auto_improve_consecutive_losses"] = getattr(ai, "consecutive_losses", 0)
    self.snapshot["auto_improve_optimization_count"] = getattr(ai, "optimization_count", 0)
    self.snapshot["auto_optimized_params"] = dict(getattr(ai, "current_params", {}) or {})
```

También agregar contador en `bot_engine.py`:
```python
# En __init__:
self._filter_auto_adjustment_count: int = 0

# En _apply_symbol_filter_config y _auto_adjust_filters_for_capital:
self._filter_auto_adjustment_count += 1
```

---

## BLOQUE G — CAPITAL PROFILE: Objetivo 20+ trades/día para capital pequeño

### G1. CORRECCIÓN en `capital_profile.py`

Los perfiles NANO y MICRO tienen `max_trades_per_day=2`. Esto hace imposible el objetivo de 20+ trades/día con capital pequeño.

**Solución:** El objetivo de trades no debe depender solo del capital, sino del contexto del mercado. El `max_trades_per_day` en el perfil es un **límite de seguridad** para evitar overtrading — **no es el objetivo**.

Actualizar los perfiles para que el límite de seguridad permita más trades cuando el mercado tiene buenas oportunidades:

```python
# En CapitalProfileManager.__init__:

CapitalProfile(
    name="NANO",
    min_equity=0.0,
    max_equity=25.0,
    reserve_ratio=0.40,
    reserve_buffer_usdt=1.0,
    risk_per_trade_fraction=0.04,
    max_trade_balance_fraction=0.10,
    min_confidence=0.65,
    max_trades_per_day=25,          # ← aumentado: 20+ es el objetivo
    cooldown_minutes=6,             # ← reducido: cooldown más corto
    loss_pause_minutes=45,          # ← reducido
    loss_pause_after_consecutive=2, # ← 2 pérdidas seguidas = pausa
    max_spread_ratio=0.0010,
    min_expected_reward_risk=2.5,   # ← reducido para más señales válidas
    min_operable_notional_buffer=1.25,
    max_concurrent_trades=1,
    entry_quality_floor=0.70,       # ← reducido ligeramente
    size_multiplier=0.40,
),
CapitalProfile(
    name="MICRO",
    min_equity=25.0,
    max_equity=50.0,
    reserve_ratio=0.35,
    reserve_buffer_usdt=3.0,
    risk_per_trade_fraction=0.03,
    max_trade_balance_fraction=0.12,
    min_confidence=0.62,
    max_trades_per_day=25,          # ← aumentado
    cooldown_minutes=5,             # ← reducido
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

**IMPORTANTE:** Para perfiles LARGE y WHALE, el `max_trades_per_day` puede ser menor (10-15) porque la estrategia es calidad sobre cantidad con capital grande, para evitar mover el mercado y gestionar riesgo de manera más conservadora.

---

## RESUMEN DE TODOS LOS ARCHIVOS A MODIFICAR

| Archivo | Bloques | Tipo de cambio |
|---|---|---|
| `reco_trading/main.py` | A1, A2, A3 | Bugfix crítico |
| `run.sh` | B1, B2 | Bugfix + UX mejora |
| `reco_trading/core/bot_engine.py` | C2, C3, C4, F | Bugfix + nuevos campos snapshot |
| `reco_trading/ui/dashboard.py` | D1, D2, D3 | Rediseño panel TUI |
| `web_site/dashboard_server.py` | E1, E2, E3 | Bugfix + nuevo endpoint |
| `web_site/templates/index.html` | E4 | Rediseño completo profesional |
| `reco_trading/risk/capital_profile.py` | G1 | Ajuste de límites de trades |

---

## VERIFICACIÓN FINAL OBLIGATORIA

Después de aplicar todos los cambios, verificar:

1. `python -m py_compile reco_trading/main.py` → sin errores
2. `python -m py_compile reco_trading/core/bot_engine.py` → sin errores
3. `python -m py_compile reco_trading/ui/dashboard.py` → sin errores
4. `python -m py_compile web_site/dashboard_server.py` → sin errores
5. `python -m py_compile reco_trading/risk/capital_profile.py` → sin errores
6. `bash -n run.sh` → sin errores de sintaxis bash
7. El HTML en `index.html` debe ser HTML5 válido con JavaScript sin errores de sintaxis

---

## NOTAS ADICIONALES CRÍTICAS

- **NO instalar ni usar Ollama** en ningún código nuevo. `LLMTradeConfirmator` ya funciona en modo `base` (rule-based).
- **NO cambiar la arquitectura asyncio** — el bot usa `asyncio.run()` en un thread separado; Flask corre en otro thread. Esta arquitectura es correcta.
- **NO modificar** `reco_trading/strategy/signal_engine.py`, `reco_trading/strategy/indicators.py`, ni `reco_trading/strategy/confidence_model.py` — estos archivos no tienen bugs críticos.
- El campo `snapshot` en `BotEngine` es un dict mutable compartido entre el thread del bot y el thread de Flask. Accesos desde Flask siempre deben ser **read-only** y con `dict(snapshot)` para copia shallow.
- El SSE endpoint (`/api/stream`) corre en el thread de Flask y lee `snapshot` solo con lectura — esto es thread-safe para dicts en CPython gracias al GIL, pero siempre usar `snapshot.get()` con defaults.
