# CODEX MASTER PROMPT — reco-trading Full Overhaul
**Instrucciones para OpenAI Codex / GPT-4.1 / Claude Code**
*Prompt de ingeniería de software profesional — Leer completo antes de tocar cualquier archivo*

---

## CONTEXTO DEL PROYECTO

Estás trabajando en **reco-trading**, un bot de trading de criptomonedas en Python con arquitectura asyncio + Flask + Rich TUI. El proyecto usa:
- `ccxt` para exchange Binance
- `SQLAlchemy async` + `aiosqlite/asyncpg` para BD
- `rich` para terminal TUI
- `Flask` (threaded) para dashboard web
- `requests` para llamadas a Ollama (LLM local)
- `pydantic` para settings
- `pandas` para indicadores

**Tu misión es COMPLETA y se divide en 4 bloques que debes ejecutar en orden:**

1. **Bloque A** — Corrección de bugs críticos en lógica del bot
2. **Bloque B** — Corrección de bugs críticos en filtros/señales
3. **Bloque C** — Dashboard Web completamente nuevo (funcional + visual premium)
4. **Bloque D** — Dashboard Terminal TUI mejorado

No omitas ningún bloque. Cada archivo modificado debe quedar completo y funcional.

---

## BLOQUE A — CORRECCIÓN DE BUGS CRÍTICOS (Core / LLM)

### A1. Archivo: `reco_trading/core/llm_trade_confirmator.py`

**BUG 1 — `llm_mode="base"` ignora el score completamente**

Localiza el bloque:
```python
if self.llm_mode == "base":
    confirmed = True
    reasons.append("LLM_MODE=base (sin confirmación LLM final)")
```

Reemplázalo por:
```python
if self.llm_mode == "base":
    confirmed = score >= 50
    reasons.append(f"base_rules score={score:.1f} threshold=50 result={'APPROVED' if score >= 50 else 'REJECTED'}")
```

**BUG 2 — `_avg_time_ms` se actualiza dos veces por llamada**

Localiza el bloque que empieza con:
```python
total_events = self._confirmation_count + self._rejection_count + 1
self._avg_time_ms = ...
if confirmed:
    self._confirmation_count += 1
else:
    self._rejection_count += 1
if self._confirmation_count == 1:
    self._avg_time_ms = analysis_time
else:
    self._avg_time_ms = ((self._avg_time_ms * ...
```

Reemplázalo COMPLETAMENTE por:
```python
if confirmed:
    self._confirmation_count += 1
else:
    self._rejection_count += 1

total_events = self._confirmation_count + self._rejection_count
if total_events == 1:
    self._avg_time_ms = analysis_time
else:
    self._avg_time_ms = (
        (self._avg_time_ms * (total_events - 1)) + analysis_time
    ) / total_events
```

**BUG 3 — `volatility_regime` strings incorrectos**

Localiza:
```python
if volatility_regime in ("NORMAL", "TRENDING"):
    score += 10
elif volatility_regime == "HIGH_VOLATILITY":
    score += 5
```

Reemplaza por:
```python
if volatility_regime in ("NORMAL_VOLATILITY", "LOW_VOLATILITY"):
    score += 10
elif volatility_regime == "HIGH_VOLATILITY":
    score += 5
    reasons.append("High volatility - reduced score bonus")
else:
    reasons.append(f"Unknown volatility regime: {volatility_regime}")
```

**BUG 4 — Warmup de Ollama bloquea `__init__` de forma síncrona**

Localiza en `__init__`:
```python
if self.llm_mode == "llm_local" and self.local_healthcheck_enabled:
    self._local_endpoint_healthy = self._warmup_local_model()
```

Reemplaza por:
```python
self._warmup_done = False
if self.llm_mode == "llm_local" and self.local_healthcheck_enabled:
    import threading
    threading.Thread(
        target=self._warmup_background,
        daemon=True,
        name="ollama-warmup"
    ).start()
```

Agrega el método `_warmup_background` a la clase:
```python
def _warmup_background(self) -> None:
    """Warmup Ollama en background sin bloquear el bot."""
    import time
    time.sleep(2.0)  # dar tiempo al bot a terminar __init__
    self._local_endpoint_healthy = self._warmup_local_model()
    self._warmup_done = True
    if self._local_endpoint_healthy:
        self.logger.info("ollama_warmup_completed_background model=%s", self.local_model)
    else:
        self.logger.warning("ollama_warmup_failed_background model=%s — usando fallback a reglas", self.local_model)
```

**MEJORA — Timeout adaptativo para primera llamada**

En el método `_local_confirm`, localiza:
```python
response = self._http_session.post(
    f"{self.ollama_base_url}/api/generate",
    json=payload,
    timeout=self.local_timeout_seconds,
)
```

Reemplaza con:
```python
# Primera llamada puede necesitar más tiempo (modelo cargando)
effective_timeout = (
    max(self.local_timeout_seconds, 5.0)
    if not self._warmup_done
    else self.local_timeout_seconds
)
response = self._http_session.post(
    f"{self.ollama_base_url}/api/generate",
    json=payload,
    timeout=effective_timeout,
)
```

**MEJORA — Agregar `local_endpoint_healthy` y `warmup_done` al `stats` property:**

```python
@property
def stats(self) -> dict[str, Any]:
    total = self._confirmation_count + self._rejection_count
    return {
        "total_analyzed": total,
        "confirmed": self._confirmation_count,
        "rejected": self._rejection_count,
        "confirmation_rate": (self._confirmation_count / total * 100) if total > 0 else 0.0,
        "avg_analysis_time_ms": round(self._avg_time_ms, 2),
        "local_endpoint_healthy": self._local_endpoint_healthy,
        "warmup_done": self._warmup_done,
        "mode": self.llm_mode,
        "model": self.local_model if self.llm_mode == "llm_local" else self.remote_model,
    }
```

---

### A2. Archivo: `reco_trading/core/bot_engine.py`

**BUG 5 — Import duplicado**

Localiza las líneas 73-74 (dos imports idénticos consecutivos):
```python
from reco_trading.core.trading_modes import TradingModeManager, WebSocketManager
from reco_trading.core.trading_modes import TradingModeManager, WebSocketManager
```
Elimina una de las dos líneas duplicadas. Deja solo una.

**BUG 6 — Filtros LLM sobreescritos por `_apply_symbol_filter_config`**

En el método `_apply_symbol_filter_config`, al final (después de `self.runtime_filter_config = dict(self.base_filter_config)`), agrega:
```python
# Re-aplicar ajustes LLM si está en modo local (se habrían sobreescrito)
if str(getattr(self.settings, "llm_mode", "base")).lower() == "llm_local":
    self._configure_llm_runtime_mode()
    self.logger.info("LLM runtime filters re-applied after symbol config update")
```

**BUG 7 — `history_limit` se reduce silenciosamente sin log**

Localiza:
```python
if bool(getattr(self.settings, "low_ram_mode", True)):
    settings.history_limit = max(120, min(int(settings.history_limit), 220))
```

Reemplaza por:
```python
if bool(getattr(self.settings, "low_ram_mode", True)):
    original_limit = settings.history_limit
    settings.history_limit = max(120, min(int(settings.history_limit), 220))
    if settings.history_limit != original_limit:
        self.logger.warning(
            "history_limit clamped by low_ram_mode: %d → %d. "
            "Indicators requiring >%d candles may be inaccurate.",
            original_limit, settings.history_limit, settings.history_limit
        )
```

**BUG 8 — `refresh_per_second` hardcodeado en `Live()`**

Localiza:
```python
live_context = Live(
    self.dashboard.render(self.snapshot),
    refresh_per_second=8,
    transient=False,
    auto_refresh=False,
    screen=True,
    vertical_overflow="crop",
)
```

Reemplaza por:
```python
import sys
_tui_fps = max(1, min(int(getattr(self, "terminal_tui_refresh_per_second", 4)), 10))
_has_tty = sys.stdout.isatty()
live_context = Live(
    self.dashboard.render(self.snapshot),
    refresh_per_second=_tui_fps,
    transient=False,
    auto_refresh=False,
    screen=_has_tty,  # screen=True solo si hay TTY real
    vertical_overflow="crop",
)
```

---

## BLOQUE B — CORRECCIÓN DE FILTROS Y SEÑALES

### B1. Archivo: `reco_trading/strategy/regime_filter.py`

**BUG CRÍTICO — `allow_trade` siempre es `True` para todos los regímenes**

Reemplaza el método `evaluate` completo por:
```python
def evaluate(self, frame: pd.DataFrame) -> RegimeDecision:
    row = frame.iloc[-1]
    try:
        atr_val = float(row["atr"])
        close_val = float(row["close"])
        if close_val <= 0:
            return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, 0.0, True, 1.0)
        atr_ratio = atr_val / close_val
    except (KeyError, TypeError, ZeroDivisionError):
        return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, 0.0, True, 1.0)

    if atr_ratio < self.low_threshold:
        # Mercado dormido: bajo ATR = sin momentum = no operar
        return RegimeDecision(VolatilityRegime.LOW_VOLATILITY, atr_ratio, allow_trade=False, size_multiplier=0.0)
    if atr_ratio > self.high_threshold:
        # Alta volatilidad: operar con tamaño reducido como protección
        return RegimeDecision(VolatilityRegime.HIGH_VOLATILITY, atr_ratio, allow_trade=True, size_multiplier=0.60)
    # Régimen normal: operación completa
    return RegimeDecision(VolatilityRegime.NORMAL_VOLATILITY, atr_ratio, allow_trade=True, size_multiplier=1.0)
```

### B2. Archivo: `reco_trading/strategy/signal_engine.py`

**BUG — Volumen bajo asignado como señal "SELL"**

Localiza:
```python
if vol_ratio > 1.00:
    volume = "BUY"
elif vol_ratio < 0.85:
    volume = "SELL"
else:
    volume = "NEUTRAL"
```

Reemplaza por:
```python
if vol_ratio > 1.20:
    volume = "BUY"       # Volumen elevado confirma movimiento
elif vol_ratio > 1.00:
    volume = "BUY"       # Volumen sobre media = sesgo alcista moderado
elif vol_ratio < 0.70:
    volume = "NEUTRAL"   # Volumen bajo = sin confirmación (NO es señal bajista)
else:
    volume = "NEUTRAL"
```

**MEJORA — Añadir protección de columnas faltantes en `generate()`**

Al inicio del método `generate()`, después de verificar `len(df5m) < 2`, agrega:
```python
# Verificar que las columnas necesarias existen
required_cols = {"ema20", "ema50", "rsi", "atr", "close", "high", "low", "volume", "vol_ma20"}
missing_5m = required_cols - set(df5m.columns)
missing_15m = required_cols - set(df15m.columns)
if missing_5m or missing_15m:
    import logging
    logging.getLogger(__name__).warning(
        "signal_engine: missing columns 5m=%s 15m=%s — returning NEUTRAL bundle",
        missing_5m, missing_15m
    )
    return SignalBundle(
        trend="NEUTRAL", momentum="NEUTRAL", volume="NEUTRAL",
        volatility="NEUTRAL", structure="NEUTRAL", order_flow="NEUTRAL",
        regime="INSUFFICIENT_DATA", regime_trade_allowed=False,
        size_multiplier=0.0, atr_ratio=0.0,
    )
```

### B3. Archivo: `reco_trading/core/bot_engine.py` — Filtros autónomos

**BUG — Relajación doble de filtros cuando `trades_today == 0`**

En el método `_apply_autonomous_filters()`, localiza el bloque:
```python
if trades_today < 2:
    effective_filters["min_confidence"] = max(0.36, ...)
    effective_filters["adx_threshold"] = max(8.0, ...)
    ...
if trades_today == 0 and daily_pnl >= 0:
    effective_filters["min_confidence"] = max(0.34, ...)
    effective_filters["adx_threshold"] = max(7.0, ...)
```

Reemplaza AMBOS bloques por uno unificado con floor duro:
```python
# Relajación controlada por falta de actividad — un solo ajuste acumulado
if trades_today == 0 and daily_pnl >= 0:
    # Sin trades y sin pérdidas: relajar moderadamente para buscar entrada
    _BASE_CONF = self.base_filter_config.get("min_confidence", 0.45)
    _BASE_ADX = self.base_filter_config.get("adx_threshold", 15.0)
    effective_filters["min_confidence"] = max(_BASE_CONF - 0.06, 0.38)
    effective_filters["adx_threshold"] = max(_BASE_ADX - 3.0, 9.0)
    effective_filters["volume_buy_threshold"] = min(
        effective_filters.get("volume_buy_threshold", 1.0), 0.85
    )
elif trades_today < 2 and daily_pnl >= -10.0:
    # Pocos trades pero sin pérdida grave: relajar levemente
    _BASE_CONF = self.base_filter_config.get("min_confidence", 0.45)
    _BASE_ADX = self.base_filter_config.get("adx_threshold", 15.0)
    effective_filters["min_confidence"] = max(_BASE_CONF - 0.03, 0.42)
    effective_filters["adx_threshold"] = max(_BASE_ADX - 1.5, 11.0)
```

---

## BLOQUE C — DASHBOARD WEB COMPLETO (RECONSTRUCCIÓN TOTAL)

### C1. Archivo: `web_site/dashboard_server.py` — Corrección de bugs de concurrencia

**BUG CRÍTICO — `_run_async` + `force_close_position` = crash de event loop**

Reemplaza el método `_run_async` completo:
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

**CORRECCIÓN — `force_close_position` segura desde Flask**

Localiza en la ruta `/api/control` la llamada:
```python
_run_async(_global_bot_instance.force_close_position())
```

Reemplaza por:
```python
# Usar cola de comandos thread-safe en lugar de llamada directa al event loop del bot
if hasattr(_global_bot_instance, "_control_queue"):
    _global_bot_instance._control_queue.put_nowait({"action": "force_close"})
    logger.warning("force_close command queued via web dashboard")
    return jsonify({"success": True, "message": "Force close command queued"})
elif hasattr(_global_bot_instance, "request_force_close"):
    _global_bot_instance.request_force_close()
    return jsonify({"success": True, "message": "Force close requested"})
else:
    return jsonify({"success": False, "error": "Force close not available in this bot version"})
```

**CORRECCIÓN — Lectura thread-safe del snapshot**

Localiza en `get_bot_snapshot()`:
```python
snapshot = getattr(_global_bot_instance, 'snapshot', {})
if callable(snapshot):
    snapshot = snapshot()
```

Reemplaza por:
```python
# Thread-safe snapshot copy
raw_snapshot = getattr(_global_bot_instance, 'snapshot', {})
if callable(raw_snapshot):
    raw_snapshot = raw_snapshot()
# Hacer copia shallow para evitar race conditions con el loop asyncio del bot
snapshot = dict(raw_snapshot) if isinstance(raw_snapshot, dict) else {}
```

**CORRECCIÓN — Auth habilitada por defecto**

Localiza:
```python
def _is_dashboard_auth_enabled() -> bool:
    return _env_bool("DASHBOARD_AUTH_ENABLED", False)
```

Reemplaza por:
```python
def _is_dashboard_auth_enabled() -> bool:
    # Auth habilitada por defecto para seguridad. Deshabilitar explícitamente con DASHBOARD_AUTH_ENABLED=false
    return _env_bool("DASHBOARD_AUTH_ENABLED", True)
```

**AGREGAR — Endpoint SSE (Server-Sent Events) para actualizaciones en tiempo real**

Agrega esta ruta nueva en la función `create_app()`:
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

---

### C2. Archivo: `web_site/templates/index.html` — RECONSTRUCCIÓN COMPLETA

**Reemplaza el archivo `index.html` COMPLETAMENTE con el siguiente template.**
Este template es un dashboard web premium, responsivo, con tema oscuro, todas las secciones funcionales y settings en acordeón.

```html
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Reco Trading — Dashboard</title>
<style>
/* ═══════════════════════════════════════════════
   DESIGN SYSTEM — Variables y Reset
═══════════════════════════════════════════════ */
:root {
  --bg-0: #0a0e1a;
  --bg-1: #0f1629;
  --bg-2: #141d35;
  --bg-3: #1a2540;
  --bg-card: #111827;
  --bg-card-hover: #1e2d45;
  --border: #1e3a5f;
  --border-bright: #2563eb40;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-green: #10b981;
  --accent-red: #ef4444;
  --accent-yellow: #f59e0b;
  --accent-purple: #8b5cf6;
  --accent-orange: #f97316;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #475569;
  --glow-blue: 0 0 20px #3b82f620;
  --glow-green: 0 0 20px #10b98120;
  --glow-red: 0 0 20px #ef444420;
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --transition: all 0.2s ease;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  background: var(--bg-0);
  color: var(--text-primary);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ═══════════════════════════════════════════════
   TOPBAR
═══════════════════════════════════════════════ */
.topbar {
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  height: 60px;
  background: linear-gradient(90deg, #0a0e1a 0%, #0f1e3d 50%, #0a0e1a 100%);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px;
  backdrop-filter: blur(12px);
}

.topbar-brand {
  display: flex; align-items: center; gap: 12px;
}

.topbar-logo {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 900; color: white;
}

.topbar-title { font-size: 18px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.3px; }
.topbar-subtitle { font-size: 11px; color: var(--text-muted); margin-top: 1px; }

.topbar-status {
  display: flex; align-items: center; gap: 16px;
}

.status-pill {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px; font-weight: 600;
  border: 1px solid;
}
.status-pill.running { background: #10b98115; border-color: #10b98140; color: var(--accent-green); }
.status-pill.paused  { background: #f59e0b15; border-color: #f59e0b40; color: var(--accent-yellow); }
.status-pill.error   { background: #ef444415; border-color: #ef444440; color: var(--accent-red); }
.status-pill.waiting { background: #3b82f615; border-color: #3b82f640; color: var(--accent-blue); }

.status-dot {
  width: 7px; height: 7px; border-radius: 50%;
  animation: pulse-dot 2s infinite;
}
.running .status-dot  { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
.paused .status-dot   { background: var(--accent-yellow); box-shadow: 0 0 6px var(--accent-yellow); }
.error .status-dot    { background: var(--accent-red); box-shadow: 0 0 6px var(--accent-red); }
.waiting .status-dot  { background: var(--accent-blue); box-shadow: 0 0 6px var(--accent-blue); }

@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}

.topbar-clock { font-size: 13px; color: var(--text-secondary); font-family: monospace; }

.emergency-btn {
  padding: 6px 14px; border-radius: 6px;
  background: #ef444420; border: 1px solid #ef444440;
  color: var(--accent-red); font-size: 12px; font-weight: 700;
  cursor: pointer; transition: var(--transition);
  letter-spacing: 0.5px;
}
.emergency-btn:hover { background: #ef444435; border-color: var(--accent-red); box-shadow: var(--glow-red); }

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
.sidebar {
  position: fixed; top: 60px; left: 0; bottom: 0;
  width: 220px;
  background: var(--bg-1);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  padding: 16px 0;
  z-index: 50;
}

.nav-section-label {
  padding: 8px 20px 4px;
  font-size: 10px; font-weight: 700;
  color: var(--text-muted); letter-spacing: 1.2px;
  text-transform: uppercase;
}

.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 20px;
  cursor: pointer;
  color: var(--text-secondary);
  font-size: 13px; font-weight: 500;
  transition: var(--transition);
  border-left: 3px solid transparent;
  user-select: none;
}
.nav-item:hover { background: var(--bg-2); color: var(--text-primary); }
.nav-item.active {
  background: linear-gradient(90deg, #3b82f620, transparent);
  color: var(--accent-blue); border-left-color: var(--accent-blue);
}
.nav-icon { font-size: 16px; width: 20px; text-align: center; }

.sidebar-footer {
  margin-top: auto;
  padding: 16px 20px;
  border-top: 1px solid var(--border);
}
.sidebar-version { font-size: 11px; color: var(--text-muted); }

/* ═══════════════════════════════════════════════
   MAIN CONTENT
═══════════════════════════════════════════════ */
.main {
  margin-left: 220px;
  margin-top: 60px;
  padding: 24px;
  min-height: calc(100vh - 60px);
}

.page { display: none; }
.page.active { display: block; }

/* ═══════════════════════════════════════════════
   CARDS
═══════════════════════════════════════════════ */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px;
  transition: var(--transition);
}
.card:hover { border-color: var(--border-bright); }
.card-title {
  font-size: 12px; font-weight: 700;
  color: var(--text-muted); letter-spacing: 0.8px;
  text-transform: uppercase; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.card-title::before {
  content: ''; width: 3px; height: 14px;
  background: var(--accent-blue); border-radius: 2px;
}

/* GRID LAYOUTS */
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
.grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 20px; }
.grid-auto { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px; }

/* METRIC CARD */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  position: relative; overflow: hidden;
  transition: var(--transition);
}
.metric-card::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--mc-color, var(--accent-blue)), transparent);
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px #00000030; }
.metric-label { font-size: 11px; font-weight: 600; color: var(--text-muted); letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px; }
.metric-value { font-size: 24px; font-weight: 800; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px; }
.metric-sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.metric-badge {
  position: absolute; top: 14px; right: 14px;
  font-size: 18px; opacity: 0.4;
}

.positive { color: var(--accent-green) !important; }
.negative { color: var(--accent-red) !important; }
.neutral  { color: var(--accent-blue) !important; }
.warning  { color: var(--accent-yellow) !important; }

/* SIGNAL CHIPS */
.signal-chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
}
.chip-buy    { background: #10b98120; border: 1px solid #10b98140; color: var(--accent-green); }
.chip-sell   { background: #ef444420; border: 1px solid #ef444440; color: var(--accent-red); }
.chip-hold   { background: #3b82f620; border: 1px solid #3b82f640; color: var(--accent-blue); }
.chip-neutral{ background: #64748b20; border: 1px solid #64748b40; color: var(--text-secondary); }

/* ═══════════════════════════════════════════════
   CONFIDENCE BAR
═══════════════════════════════════════════════ */
.confidence-bar-wrap { margin-top: 8px; }
.confidence-bar-track {
  height: 6px; background: var(--bg-3); border-radius: 3px; overflow: hidden;
}
.confidence-bar-fill {
  height: 100%; border-radius: 3px;
  transition: width 0.5s ease, background 0.3s ease;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
.confidence-bar-fill.high { background: linear-gradient(90deg, var(--accent-green), #34d399); }
.confidence-bar-fill.low  { background: linear-gradient(90deg, var(--accent-red), #f87171); }

/* ═══════════════════════════════════════════════
   TABLES
═══════════════════════════════════════════════ */
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.data-table th {
  text-align: left; padding: 10px 14px;
  font-size: 10px; font-weight: 700; letter-spacing: 0.8px;
  color: var(--text-muted); text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  background: var(--bg-2);
}
.data-table td { padding: 11px 14px; border-bottom: 1px solid var(--border)15; color: var(--text-secondary); }
.data-table tr:hover td { background: var(--bg-2)50; color: var(--text-primary); }
.data-table tr:last-child td { border-bottom: none; }

/* ═══════════════════════════════════════════════
   SIGNALS GRID (Dashboard)
═══════════════════════════════════════════════ */
.signals-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
}
.signal-item {
  background: var(--bg-2); border-radius: var(--radius-sm);
  padding: 12px 14px; border: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
}
.signal-name { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }

/* ═══════════════════════════════════════════════
   LOG FEED
═══════════════════════════════════════════════ */
.log-feed {
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px;
  background: #050a14;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px;
  height: 280px; overflow-y: auto;
}
.log-entry { padding: 3px 0; display: flex; gap: 10px; border-bottom: 1px solid #ffffff05; }
.log-time  { color: var(--text-muted); min-width: 80px; }
.log-level { min-width: 55px; font-weight: 700; }
.log-level.INFO    { color: var(--accent-blue); }
.log-level.WARNING { color: var(--accent-yellow); }
.log-level.ERROR   { color: var(--accent-red); }
.log-level.DEBUG   { color: var(--text-muted); }
.log-msg   { color: var(--text-secondary); word-break: break-all; }
.log-msg.important { color: var(--text-primary); }

/* ═══════════════════════════════════════════════
   SETTINGS — ACCORDION
═══════════════════════════════════════════════ */
.settings-form { display: flex; flex-direction: column; gap: 12px; }

.accordion-group {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-md); overflow: hidden;
}
.accordion-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px; cursor: pointer;
  background: var(--bg-card);
  transition: var(--transition);
  user-select: none;
}
.accordion-header:hover { background: var(--bg-card-hover); }
.accordion-header-left { display: flex; align-items: center; gap: 12px; }
.accordion-icon { font-size: 18px; }
.accordion-title { font-size: 14px; font-weight: 600; color: var(--text-primary); }
.accordion-desc  { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.accordion-arrow {
  color: var(--text-muted); font-size: 12px;
  transition: transform 0.2s ease;
}
.accordion-group.open .accordion-arrow { transform: rotate(180deg); }
.accordion-group.open .accordion-header { border-bottom: 1px solid var(--border); }

.accordion-body {
  display: none; padding: 20px;
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;
}
.accordion-group:not(.open) .accordion-body { display: none; }

/* FORM CONTROLS */
.form-group { display: flex; flex-direction: column; gap: 6px; }
.form-group.full-width { grid-column: 1 / -1; }
.form-label { font-size: 11px; font-weight: 600; color: var(--text-secondary); letter-spacing: 0.3px; }
.form-hint  { font-size: 10px; color: var(--text-muted); margin-top: 2px; }

.form-input, .form-select, .form-textarea {
  background: var(--bg-2); border: 1px solid var(--border);
  color: var(--text-primary); border-radius: var(--radius-sm);
  padding: 9px 12px; font-size: 13px;
  transition: var(--transition); outline: none;
  width: 100%;
}
.form-input:focus, .form-select:focus, .form-textarea:focus {
  border-color: var(--accent-blue); box-shadow: 0 0 0 3px #3b82f615;
}
.form-select { cursor: pointer; }
.form-select option { background: var(--bg-2); }

.toggle-wrap {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px; background: var(--bg-2); border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
.toggle-label { font-size: 13px; color: var(--text-secondary); }
.toggle {
  position: relative; width: 40px; height: 22px;
}
.toggle input { opacity: 0; width: 0; height: 0; }
.toggle-slider {
  position: absolute; cursor: pointer; inset: 0;
  background: var(--bg-3); border-radius: 22px;
  transition: var(--transition);
}
.toggle-slider::before {
  content: ''; position: absolute;
  width: 16px; height: 16px; bottom: 3px; left: 3px;
  background: white; border-radius: 50%;
  transition: var(--transition);
}
.toggle input:checked + .toggle-slider { background: var(--accent-blue); }
.toggle input:checked + .toggle-slider::before { transform: translateX(18px); }

.multi-select {
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 8px; background: var(--bg-2);
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  min-height: 40px;
}
.ms-option {
  padding: 4px 10px; border-radius: 20px; cursor: pointer;
  font-size: 11px; font-weight: 600;
  background: var(--bg-3); border: 1px solid var(--border);
  color: var(--text-secondary); transition: var(--transition);
  user-select: none;
}
.ms-option.selected {
  background: #3b82f620; border-color: #3b82f660;
  color: var(--accent-blue);
}

/* RANGE SLIDER */
.range-wrap { display: flex; flex-direction: column; gap: 4px; }
.range-row   { display: flex; align-items: center; gap: 10px; }
.form-range  {
  flex: 1; -webkit-appearance: none; height: 4px;
  background: var(--bg-3); border-radius: 2px; outline: none;
}
.form-range::-webkit-slider-thumb {
  -webkit-appearance: none; width: 16px; height: 16px;
  background: var(--accent-blue); border-radius: 50%; cursor: pointer;
  box-shadow: 0 0 6px #3b82f640;
}
.range-value {
  min-width: 50px; text-align: right;
  font-family: monospace; font-size: 13px; color: var(--accent-blue);
}

/* ═══════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════ */
.btn {
  padding: 9px 18px; border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 600; cursor: pointer;
  transition: var(--transition); border: 1px solid transparent;
  display: inline-flex; align-items: center; gap: 6px;
}
.btn-primary {
  background: var(--accent-blue); color: white; border-color: var(--accent-blue);
}
.btn-primary:hover { background: #2563eb; box-shadow: 0 4px 12px #3b82f630; }
.btn-secondary {
  background: transparent; color: var(--text-secondary); border-color: var(--border);
}
.btn-secondary:hover { background: var(--bg-2); color: var(--text-primary); }
.btn-danger {
  background: #ef444420; color: var(--accent-red); border-color: #ef444440;
}
.btn-danger:hover { background: #ef444430; box-shadow: var(--glow-red); }
.btn-success {
  background: #10b98120; color: var(--accent-green); border-color: #10b98140;
}
.btn-success:hover { background: #10b98130; }

.btn-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }

/* ═══════════════════════════════════════════════
   POSITION CARD
═══════════════════════════════════════════════ */
.position-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 16px 20px;
  display: grid; grid-template-columns: auto 1fr auto; gap: 16px;
  align-items: center;
}
.pos-side {
  padding: 6px 12px; border-radius: 6px;
  font-weight: 800; font-size: 13px; letter-spacing: 0.5px;
}
.pos-side.BUY  { background: #10b98120; color: var(--accent-green); border: 1px solid #10b98140; }
.pos-side.SELL { background: #ef444420; color: var(--accent-red);   border: 1px solid #ef444440; }

/* ═══════════════════════════════════════════════
   TOASTS
═══════════════════════════════════════════════ */
.toast-container {
  position: fixed; bottom: 24px; right: 24px; z-index: 999;
  display: flex; flex-direction: column; gap: 8px;
}
.toast {
  padding: 12px 18px; border-radius: var(--radius-md);
  font-size: 13px; font-weight: 500;
  border: 1px solid; backdrop-filter: blur(12px);
  animation: slideInToast 0.3s ease;
  min-width: 280px; max-width: 400px;
}
.toast.success { background: #10b98115; border-color: #10b98140; color: var(--accent-green); }
.toast.error   { background: #ef444415; border-color: #ef444440; color: var(--accent-red); }
.toast.info    { background: #3b82f615; border-color: #3b82f640; color: var(--accent-blue); }
@keyframes slideInToast {
  from { transform: translateX(100%); opacity: 0; }
  to   { transform: translateX(0); opacity: 1; }
}

/* ═══════════════════════════════════════════════
   LLM STATUS
═══════════════════════════════════════════════ */
.llm-status-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 16px 20px;
}
.llm-mode-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 20px;
  font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
  background: #8b5cf620; border: 1px solid #8b5cf640; color: var(--accent-purple);
}
.llm-health-dot {
  width: 8px; height: 8px; border-radius: 50%;
  display: inline-block; margin-right: 4px;
}
.llm-health-dot.ok   { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
.llm-health-dot.fail { background: var(--accent-red);   box-shadow: 0 0 6px var(--accent-red); }
.llm-health-dot.unknown { background: var(--text-muted); }

/* ═══════════════════════════════════════════════
   FILTER DISPLAY
═══════════════════════════════════════════════ */
.filter-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 0; border-bottom: 1px solid var(--border)20;
  font-size: 12px;
}
.filter-row:last-child { border-bottom: none; }
.filter-name  { color: var(--text-secondary); font-weight: 500; }
.filter-value { color: var(--accent-blue); font-family: monospace; font-weight: 700; }
.filter-pass  { color: var(--accent-green); }
.filter-fail  { color: var(--accent-red); }

/* ═══════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue)60; }

/* ═══════════════════════════════════════════════
   RESPONSIVE
═══════════════════════════════════════════════ */
@media (max-width: 1200px) { .grid-4 { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 900px) {
  .sidebar { display: none; }
  .main { margin-left: 0; }
  .grid-3, .grid-4 { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 600px) {
  .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
  .signals-grid { grid-template-columns: repeat(2, 1fr); }
  .accordion-body { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ══════════════ TOPBAR ══════════════ -->
<header class="topbar">
  <div class="topbar-brand">
    <div class="topbar-logo">R</div>
    <div>
      <div class="topbar-title">Reco Trading</div>
      <div class="topbar-subtitle" id="topbar-pair">Loading...</div>
    </div>
  </div>
  <div class="topbar-status">
    <div class="status-pill waiting" id="bot-status-pill">
      <span class="status-dot"></span>
      <span id="bot-status-text">CONNECTING</span>
    </div>
    <div class="topbar-clock" id="clock">--:--:--</div>
    <button class="emergency-btn" onclick="sendControl('emergency_stop')">⚡ EMERGENCY STOP</button>
  </div>
</header>

<!-- ══════════════ SIDEBAR ══════════════ -->
<nav class="sidebar">
  <div class="nav-section-label">Overview</div>
  <div class="nav-item active" onclick="showPage('dashboard', this)">
    <span class="nav-icon">📊</span> Dashboard
  </div>
  <div class="nav-item" onclick="showPage('positions', this)">
    <span class="nav-icon">💼</span> Positions
  </div>
  <div class="nav-item" onclick="showPage('trades', this)">
    <span class="nav-icon">📋</span> Trade History
  </div>

  <div class="nav-section-label" style="margin-top:12px">Analysis</div>
  <div class="nav-item" onclick="showPage('signals', this)">
    <span class="nav-icon">📡</span> Signals & Filters
  </div>
  <div class="nav-item" onclick="showPage('ai', this)">
    <span class="nav-icon">🤖</span> AI / LLM Gate
  </div>
  <div class="nav-item" onclick="showPage('risk', this)">
    <span class="nav-icon">🛡️</span> Risk Manager
  </div>

  <div class="nav-section-label" style="margin-top:12px">System</div>
  <div class="nav-item" onclick="showPage('logs', this)">
    <span class="nav-icon">📄</span> Live Logs
  </div>
  <div class="nav-item" onclick="showPage('settings', this)">
    <span class="nav-icon">⚙️</span> Settings
  </div>
  <div class="nav-item" onclick="showPage('system', this)">
    <span class="nav-icon">💻</span> System Health
  </div>

  <div class="sidebar-footer">
    <div class="sidebar-version" id="sidebar-version">reco-trading v—</div>
  </div>
</nav>

<!-- ══════════════ MAIN ══════════════ -->
<main class="main">

<!-- ──────────── PAGE: DASHBOARD ──────────── -->
<div class="page active" id="page-dashboard">

  <!-- KPI Row -->
  <div class="grid-4" id="kpi-grid">
    <div class="metric-card" style="--mc-color: var(--accent-blue)">
      <div class="metric-label">Price</div>
      <div class="metric-value" id="kpi-price">—</div>
      <div class="metric-sub" id="kpi-change">24h change</div>
      <div class="metric-badge">₿</div>
    </div>
    <div class="metric-card" style="--mc-color: var(--accent-green)">
      <div class="metric-label">Equity</div>
      <div class="metric-value" id="kpi-equity">—</div>
      <div class="metric-sub" id="kpi-balance-sub">Available balance</div>
      <div class="metric-badge">💰</div>
    </div>
    <div class="metric-card" style="--mc-color: var(--accent-cyan)">
      <div class="metric-label">Session P&L</div>
      <div class="metric-value" id="kpi-pnl">—</div>
      <div class="metric-sub" id="kpi-unrealized">Unrealized —</div>
      <div class="metric-badge">📈</div>
    </div>
    <div class="metric-card" style="--mc-color: var(--accent-purple)">
      <div class="metric-label">Win Rate</div>
      <div class="metric-value" id="kpi-winrate">—</div>
      <div class="metric-sub" id="kpi-trades-today">0 trades today</div>
      <div class="metric-badge">🎯</div>
    </div>
  </div>

  <!-- Market + Signals + Position -->
  <div class="grid-3">
    <!-- Market State -->
    <div class="card">
      <div class="card-title">📈 Market State</div>
      <div style="display:flex; flex-direction:column; gap:10px;">
        <div class="filter-row">
          <span class="filter-name">Trend</span>
          <span id="mkt-trend">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">ADX</span>
          <span id="mkt-adx" class="filter-value">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">RSI</span>
          <span id="mkt-rsi" class="filter-value">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Spread</span>
          <span id="mkt-spread" class="filter-value">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Volatility Regime</span>
          <span id="mkt-regime">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Order Flow</span>
          <span id="mkt-flow">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Timeframe</span>
          <span id="mkt-tf" class="filter-value">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Cooldown</span>
          <span id="mkt-cooldown">—</span>
        </div>
      </div>
    </div>

    <!-- Signal Summary -->
    <div class="card">
      <div class="card-title">📡 Signal Bundle</div>
      <div style="margin-bottom:14px; display:flex; align-items:center; justify-content:space-between;">
        <span style="font-size:12px; color:var(--text-muted)">Signal</span>
        <span id="sig-main" class="signal-chip chip-neutral">HOLD</span>
      </div>
      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
          <span style="color:var(--text-muted)">Confidence</span>
          <span id="sig-conf-pct" style="color:var(--accent-blue); font-weight:700">0%</span>
        </div>
        <div class="confidence-bar-track">
          <div class="confidence-bar-fill" id="sig-conf-bar" style="width:0%"></div>
        </div>
      </div>
      <div class="signals-grid" id="signals-grid">
        <div class="signal-item"><span class="signal-name">Trend</span><span id="sig-trend" class="signal-chip chip-neutral">—</span></div>
        <div class="signal-item"><span class="signal-name">Momentum</span><span id="sig-momentum" class="signal-chip chip-neutral">—</span></div>
        <div class="signal-item"><span class="signal-name">Volume</span><span id="sig-volume" class="signal-chip chip-neutral">—</span></div>
        <div class="signal-item"><span class="signal-name">Volatility</span><span id="sig-volatility" class="signal-chip chip-neutral">—</span></div>
        <div class="signal-item"><span class="signal-name">Structure</span><span id="sig-structure" class="signal-chip chip-neutral">—</span></div>
        <div class="signal-item"><span class="signal-name">Order Flow</span><span id="sig-orderflow" class="signal-chip chip-neutral">—</span></div>
      </div>
    </div>

    <!-- Current Position -->
    <div class="card">
      <div class="card-title">💼 Open Position</div>
      <div id="position-none" style="color:var(--text-muted); font-size:13px; text-align:center; padding:20px 0;">
        No active position
      </div>
      <div id="position-details" style="display:none; flex-direction:column; gap:10px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
          <span id="pos-side-badge" class="pos-side BUY">BUY</span>
          <span id="pos-symbol" style="font-size:14px; font-weight:700;">BTC/USDT</span>
        </div>
        <div class="filter-row"><span class="filter-name">Entry Price</span><span id="pos-entry" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Current Price</span><span id="pos-mark" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Quantity</span><span id="pos-qty" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Stop Loss</span><span id="pos-sl" style="color:var(--accent-red); font-weight:700;">—</span></div>
        <div class="filter-row"><span class="filter-name">Take Profit</span><span id="pos-tp" style="color:var(--accent-green); font-weight:700;">—</span></div>
        <div class="filter-row">
          <span class="filter-name">Unrealized PnL</span>
          <span id="pos-upnl" style="font-weight:800; font-size:15px;">—</span>
        </div>
        <button class="btn btn-danger" style="width:100%; margin-top:8px;" onclick="sendControl('force_close')">
          🚨 Force Close Position
        </button>
      </div>
    </div>
  </div>

  <!-- Decision Trace + Log -->
  <div class="grid-2">
    <div class="card">
      <div class="card-title">🔍 Decision Trace</div>
      <div id="decision-trace-content" style="display:flex; flex-direction:column; gap:6px;">
        <div style="color:var(--text-muted); font-size:13px;">Waiting for signal...</div>
      </div>
      <div style="margin-top:12px; padding-top:12px; border-top:1px solid var(--border);">
        <div class="filter-row">
          <span class="filter-name">Decision Reason</span>
          <span id="decision-reason" style="font-size:11px; color:var(--text-secondary); text-align:right; max-width:200px;">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Setup Quality</span>
          <span id="setup-quality" class="filter-value">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Capital Profile</span>
          <span id="capital-profile" class="filter-value">—</span>
        </div>
        <div class="filter-row">
          <span class="filter-name">Operable Capital</span>
          <span id="operable-capital" class="filter-value">—</span>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">📄 Live Feed</div>
      <div class="log-feed" id="log-feed-main">
        <div class="log-entry"><span class="log-time">--:--:--</span><span class="log-level INFO">INFO</span><span class="log-msg">Connecting to dashboard...</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ──────────── PAGE: POSITIONS ──────────── -->
<div class="page" id="page-positions">
  <div class="card" style="margin-bottom:20px;">
    <div class="card-title">💼 Open Positions</div>
    <div id="positions-table-wrap">
      <table class="data-table">
        <thead><tr>
          <th>Pair</th><th>Side</th><th>Qty</th><th>Entry</th>
          <th>Mark</th><th>Stop Loss</th><th>Take Profit</th><th>Unrealized PnL</th><th>Action</th>
        </tr></thead>
        <tbody id="positions-tbody">
          <tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">No open positions</td></tr>
        </tbody>
      </table>
    </div>
  </div>
  <div style="display:flex; gap:10px;">
    <button class="btn btn-danger" onclick="sendControl('force_close')">🚨 Force Close All</button>
    <button class="btn btn-secondary" onclick="sendControl('pause')">⏸ Pause Bot</button>
    <button class="btn btn-success" onclick="sendControl('resume')">▶ Resume Bot</button>
  </div>
</div>

<!-- ──────────── PAGE: TRADE HISTORY ──────────── -->
<div class="page" id="page-trades">
  <div class="card">
    <div class="card-title">📋 Trade History</div>
    <div style="margin-bottom:12px; display:flex; gap:10px; flex-wrap:wrap;">
      <select class="form-select" id="filter-side" style="width:auto;" onchange="filterTrades()">
        <option value="">All Sides</option>
        <option value="BUY">BUY</option>
        <option value="SELL">SELL</option>
      </select>
      <select class="form-select" id="filter-result" style="width:auto;" onchange="filterTrades()">
        <option value="">All Results</option>
        <option value="WIN">Wins</option>
        <option value="LOSS">Losses</option>
      </select>
    </div>
    <div style="overflow-x:auto;">
      <table class="data-table" id="trades-table">
        <thead><tr>
          <th>Time</th><th>Pair</th><th>Side</th><th>Entry</th>
          <th>Exit</th><th>Qty</th><th>PnL</th><th>Status</th><th>Duration</th>
        </tr></thead>
        <tbody id="trades-tbody">
          <tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">Loading...</td></tr>
        </tbody>
      </table>
    </div>
    <div style="margin-top:12px; display:flex; gap:20px; flex-wrap:wrap;" id="trades-summary">
      <span style="font-size:12px; color:var(--text-muted)">Total: <span id="stat-total" class="filter-value">0</span></span>
      <span style="font-size:12px; color:var(--text-muted)">Wins: <span id="stat-wins" style="color:var(--accent-green); font-weight:700;">0</span></span>
      <span style="font-size:12px; color:var(--text-muted)">Losses: <span id="stat-losses" style="color:var(--accent-red); font-weight:700;">0</span></span>
      <span style="font-size:12px; color:var(--text-muted)">Total PnL: <span id="stat-total-pnl" style="font-weight:700;">0.00</span></span>
    </div>
  </div>
</div>

<!-- ──────────── PAGE: SIGNALS & FILTERS ──────────── -->
<div class="page" id="page-signals">
  <div class="grid-2">
    <div class="card">
      <div class="card-title">🔧 Active Filter Config</div>
      <div id="filter-config-display" style="display:flex; flex-direction:column; gap:4px;">
        <div class="filter-row"><span class="filter-name">ADX Threshold</span><span id="f-adx" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">RSI Buy Threshold</span><span id="f-rsi-buy" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">RSI Sell Threshold</span><span id="f-rsi-sell" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Vol Buy Threshold</span><span id="f-vol-buy" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Vol Sell Threshold</span><span id="f-vol-sell" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Min Confidence</span><span id="f-conf" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">ATR Low Threshold</span><span id="f-atr-low" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">ATR High Threshold</span><span id="f-atr-high" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">SL ATR Multiplier</span><span id="f-sl-mult" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">TP ATR Multiplier</span><span id="f-tp-mult" class="filter-value">—</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">✅ Last Validation Checks</div>
      <div id="validation-checks" style="display:flex; flex-direction:column; gap:4px;">
        <div style="color:var(--text-muted); font-size:13px;">Waiting for trade evaluation...</div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">🧠 Autonomous Brain</div>
    <div class="grid-3">
      <div class="filter-row"><span class="filter-name">Market Condition</span><span id="auto-condition" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Confluence Score</span><span id="auto-confluence" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Inv. Mode</span><span id="auto-mode" class="filter-value">—</span></div>
    </div>
  </div>
</div>

<!-- ──────────── PAGE: AI / LLM ──────────── -->
<div class="page" id="page-ai">
  <div class="grid-2">
    <div class="card">
      <div class="card-title">🤖 LLM Trade Gate</div>
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <span id="llm-mode-badge" class="llm-mode-badge">BASE</span>
        <span id="llm-model-name" style="font-size:12px; color:var(--text-muted);">—</span>
      </div>
      <div class="filter-row">
        <span class="filter-name">Ollama Health</span>
        <span><span id="llm-health-dot" class="llm-health-dot unknown"></span><span id="llm-health-text" style="font-size:12px;">Unknown</span></span>
      </div>
      <div class="filter-row"><span class="filter-name">Warmup Done</span><span id="llm-warmup" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Total Analyzed</span><span id="llm-total" class="filter-value">0</span></div>
      <div class="filter-row"><span class="filter-name">Confirmed</span><span id="llm-confirmed" style="color:var(--accent-green); font-weight:700;">0</span></div>
      <div class="filter-row"><span class="filter-name">Rejected</span><span id="llm-rejected" style="color:var(--accent-red); font-weight:700;">0</span></div>
      <div class="filter-row"><span class="filter-name">Confirmation Rate</span><span id="llm-rate" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Avg Latency</span><span id="llm-latency" class="filter-value">—</span></div>
    </div>
    <div class="card">
      <div class="card-title">📊 Autonomous Optimizer</div>
      <div class="filter-row"><span class="filter-name">Total Trades Tracked</span><span id="ao-trades" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Win Rate (Optimizer)</span><span id="ao-wr" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Consecutive Losses</span><span id="ao-losses" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Optimization Cycles</span><span id="ao-cycles" class="filter-value">—</span></div>
    </div>
  </div>
</div>

<!-- ──────────── PAGE: RISK ──────────── -->
<div class="page" id="page-risk">
  <div class="grid-4">
    <div class="metric-card" style="--mc-color:var(--accent-red)">
      <div class="metric-label">Daily Loss Limit</div>
      <div class="metric-value" id="r-daily-loss">—</div>
      <div class="metric-sub">max drawdown per day</div>
    </div>
    <div class="metric-card" style="--mc-color:var(--accent-orange)">
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-value" id="r-max-dd">—</div>
      <div class="metric-sub">from equity peak</div>
    </div>
    <div class="metric-card" style="--mc-color:var(--accent-blue)">
      <div class="metric-label">Risk Per Trade</div>
      <div class="metric-value" id="r-risk-pt">—</div>
      <div class="metric-sub">of account equity</div>
    </div>
    <div class="metric-card" style="--mc-color:var(--accent-purple)">
      <div class="metric-label">Open Positions</div>
      <div class="metric-value" id="r-open-pos">—</div>
      <div class="metric-sub">concurrent trades</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">🛡️ Risk Manager Status</div>
    <div class="grid-2">
      <div>
        <div class="filter-row"><span class="filter-name">Drawdown Active</span><span id="r-dd-active">—</span></div>
        <div class="filter-row"><span class="filter-name">Loss Pause Until</span><span id="r-pause-until" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Consecutive Losses</span><span id="r-cons-losses" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Emergency Stop</span><span id="r-emergency">—</span></div>
      </div>
      <div>
        <div class="filter-row"><span class="filter-name">Capital Reserve</span><span id="r-cap-reserve" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Min Cash Buffer</span><span id="r-cash-buf" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Session Recommendation</span><span id="r-session-rec" class="filter-value">—</span></div>
        <div class="filter-row"><span class="filter-name">Equity Peak</span><span id="r-eq-peak" class="filter-value">—</span></div>
      </div>
    </div>
  </div>
</div>

<!-- ──────────── PAGE: LOGS ──────────── -->
<div class="page" id="page-logs">
  <div class="card">
    <div class="card-title">
      📄 Live Bot Logs
      <div style="margin-left:auto; display:flex; gap:8px;">
        <select class="form-select" id="log-level-filter" style="width:auto; font-size:11px; padding:4px 8px;" onchange="filterLogs()">
          <option value="">All Levels</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>
        <button class="btn btn-secondary" style="padding:4px 10px; font-size:11px;" onclick="clearLogs()">Clear</button>
        <button class="btn btn-secondary" style="padding:4px 10px; font-size:11px;" onclick="toggleAutoScroll()">Auto-scroll: <span id="autoscroll-state">ON</span></button>
      </div>
    </div>
    <div class="log-feed" id="log-feed-full" style="height:500px;"></div>
  </div>
</div>

<!-- ──────────── PAGE: SETTINGS ──────────── -->
<div class="page" id="page-settings">
  <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:20px;">
    <div>
      <h2 style="font-size:18px; font-weight:700; color:var(--text-primary);">Settings</h2>
      <p style="font-size:13px; color:var(--text-muted); margin-top:4px;">Changes are applied to the running bot in real time.</p>
    </div>
    <div style="display:flex; gap:10px;">
      <button class="btn btn-secondary" onclick="loadCurrentSettings()">↺ Reload</button>
      <button class="btn btn-primary" onclick="saveSettings()">💾 Apply Settings</button>
    </div>
  </div>

  <div class="settings-form" id="settings-form">

    <!-- ACCORDION: Trading -->
    <div class="accordion-group open" id="acc-trading">
      <div class="accordion-header" onclick="toggleAccordion('acc-trading')">
        <div class="accordion-header-left">
          <span class="accordion-icon">💱</span>
          <div>
            <div class="accordion-title">Trading Configuration</div>
            <div class="accordion-desc">Symbol, timeframes, risk per trade and position sizing</div>
          </div>
        </div>
        <span class="accordion-arrow">▼</span>
      </div>
      <div class="accordion-body">
        <div class="form-group">
          <label class="form-label">Trading Pair</label>
          <select class="form-select" id="s-symbol" name="trading_symbol">
            <option value="BTC/USDT">BTC/USDT</option>
            <option value="ETH/USDT">ETH/USDT</option>
            <option value="SOL/USDT">SOL/USDT</option>
            <option value="BNB/USDT">BNB/USDT</option>
            <option value="XRP/USDT">XRP/USDT</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Multi-Symbol Selection</label>
          <div class="multi-select" id="ms-symbols">
            <span class="ms-option selected" data-value="BTC/USDT">BTC/USDT</span>
            <span class="ms-option" data-value="ETH/USDT">ETH/USDT</span>
            <span class="ms-option" data-value="SOL/USDT">SOL/USDT</span>
            <span class="ms-option" data-value="BNB/USDT">BNB/USDT</span>
            <span class="ms-option" data-value="XRP/USDT">XRP/USDT</span>
            <span class="ms-option" data-value="ADA/USDT">ADA/USDT</span>
            <span class="ms-option" data-value="DOGE/USDT">DOGE/USDT</span>
          </div>
          <div class="form-hint">Select multiple symbols for multi-pair mode</div>
        </div>
        <div class="form-group">
          <label class="form-label">Primary Timeframe</label>
          <select class="form-select" id="s-tf" name="primary_timeframe">
            <option value="1m">1m</option>
            <option value="3m">3m</option>
            <option value="5m" selected>5m</option>
            <option value="15m">15m</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Confirmation Timeframe</label>
          <select class="form-select" id="s-ctf" name="confirmation_timeframe">
            <option value="5m">5m</option>
            <option value="15m" selected>15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Risk Per Trade</label>
          <div class="range-wrap">
            <div class="range-row">
              <input type="range" class="form-range" id="s-risk" name="risk_per_trade_fraction" min="0.001" max="0.05" step="0.001" value="0.01"
                oninput="document.getElementById('s-risk-val').textContent = (parseFloat(this.value)*100).toFixed(1)+'%'">
              <span class="range-value"><span id="s-risk-val">1.0</span>%</span>
            </div>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Max Trades Per Day</label>
          <input type="number" class="form-input" id="s-max-trades" name="max_trades_per_day" value="20" min="1" max="100"/>
        </div>
        <div class="form-group">
          <label class="form-label">Investment Mode</label>
          <select class="form-select" id="s-inv-mode" name="investment_mode">
            <option value="Conservative">Conservative</option>
            <option value="Balanced" selected>Balanced</option>
            <option value="Aggressive">Aggressive</option>
          </select>
        </div>
        <div class="form-group">
          <div class="toggle-wrap">
            <span class="toggle-label">Spot Only Mode</span>
            <label class="toggle"><input type="checkbox" id="s-spot-only" name="spot_only_mode" checked/><span class="toggle-slider"></span></label>
          </div>
        </div>
        <div class="form-group">
          <div class="toggle-wrap">
            <span class="toggle-label">Multi-Symbol Enabled</span>
            <label class="toggle"><input type="checkbox" id="s-multi" name="feature_multi_symbol_enabled"/><span class="toggle-slider"></span></label>
          </div>
        </div>
        <div class="form-group">
          <div class="toggle-wrap">
            <span class="toggle-label">Dynamic Exit</span>
            <label class="toggle"><input type="checkbox" id="s-dyn-exit" name="dynamic_exit_enabled" checked/><span class="toggle-slider"></span></label>
          </div>
        </div>
      </div>
    </div>

    <!-- ACCORDION: Filters -->
    <div class="accordion-group" id="acc-filters">
      <div class="accordion-header" onclick="toggleAccordion('acc-filters')">
        <div class="accordion-header-left">
          <span class="accordion-icon">🔧</span>
          <div>
            <div class="accordion-title">Signal Filters</div>
            <div class="accordion-desc">ADX, RSI, volume and confidence thresholds</div>
          </div>
        </div>
        <span class="accordion-arrow">▼</span>
      </div>
      <div class="accordion-body">
        <div class="form-group">
          <label class="form-label">ADX Threshold</label>
          <div class="range-wrap">
            <div class="range-row">
              <input type="range" class="form-range" id="s-adx" name="adx_threshold" min="5" max="40" step="0.5" value="12"
                oninput="document.getElementById('s-adx-val').textContent = parseFloat(this.value).toFixed(1)">
              <span class="range-value" id="s-adx-val">12.0</span>
            </div>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Min Confidence</label>
          <div class="range-wrap">
            <div class="range-row">
              <input type="range" class="form-range" id="s-minconf" name="min_confidence" min="0.30" max="0.95" step="0.01" value="0.45"
                oninput="document.getElementById('s-minconf-val').textContent = (parseFloat(this.value)*100).toFixed(0)+'%'">
              <span class="range-value" id="s-minconf-val">45%</span>
            </div>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">RSI Buy Threshold</label>
          <input type="number" class="form-input" id="s-rsi-buy" name="rsi_buy_threshold" value="48" min="30" max="70"/>
        </div>
        <div class="form-group">
          <label class="form-label">RSI Sell Threshold</label>
          <input type="number" class="form-input" id="s-rsi-sell" name="rsi_sell_threshold" value="52" min="30" max="70"/>
        </div>
        <div class="form-group">
          <label class="form-label">Volume Buy Threshold</label>
          <input type="number" class="form-input" id="s-vol-buy" name="volume_buy_threshold" value="0.90" min="0.2" max="3.0" step="0.05"/>
        </div>
        <div class="form-group">
          <label class="form-label">Volume Sell Threshold</label>
          <input type="number" class="form-input" id="s-vol-sell" name="volume_sell_threshold" value="0.85" min="0.2" max="3.0" step="0.05"/>
        </div>
        <div class="form-group">
          <label class="form-label">Stop Loss ATR Multiplier</label>
          <input type="number" class="form-input" id="s-sl-atr" name="stop_loss_atr_multiplier" value="1.5" min="0.5" max="5.0" step="0.1"/>
        </div>
        <div class="form-group">
          <label class="form-label">Take Profit ATR Multiplier</label>
          <input type="number" class="form-input" id="s-tp-atr" name="take_profit_atr_multiplier" value="2.5" min="1.0" max="8.0" step="0.1"/>
        </div>
      </div>
    </div>

    <!-- ACCORDION: Risk -->
    <div class="accordion-group" id="acc-risk">
      <div class="accordion-header" onclick="toggleAccordion('acc-risk')">
        <div class="accordion-header-left">
          <span class="accordion-icon">🛡️</span>
          <div>
            <div class="accordion-title">Risk Management</div>
            <div class="accordion-desc">Capital limits, drawdown protection and cooldown</div>
          </div>
        </div>
        <span class="accordion-arrow">▼</span>
      </div>
      <div class="accordion-body">
        <div class="form-group">
          <label class="form-label">Daily Loss Limit (%)</label>
          <input type="number" class="form-input" id="s-daily-loss" name="daily_loss_limit_fraction" value="0.02" min="0.001" max="0.20" step="0.001"/>
          <div class="form-hint">Fraction of balance. 0.02 = 2%</div>
        </div>
        <div class="form-group">
          <label class="form-label">Max Drawdown (%)</label>
          <input type="number" class="form-input" id="s-max-dd" name="max_drawdown_fraction" value="0.10" min="0.01" max="0.50" step="0.01"/>
        </div>
        <div class="form-group">
          <label class="form-label">Capital Limit (USDT)</label>
          <input type="number" class="form-input" id="s-cap-limit" name="capital_limit_usdt" value="0" min="0" step="1"/>
          <div class="form-hint">0 = unlimited</div>
        </div>
        <div class="form-group">
          <label class="form-label">Min Cash Buffer (USDT)</label>
          <input type="number" class="form-input" id="s-cash-buf" name="min_cash_buffer_usdt" value="10" min="0" step="1"/>
        </div>
        <div class="form-group">
          <label class="form-label">Max Concurrent Positions</label>
          <input type="number" class="form-input" id="s-max-pos" name="max_concurrent_trades" value="1" min="1" max="10"/>
        </div>
        <div class="form-group">
          <label class="form-label">Loss Pause Duration (min)</label>
          <input type="number" class="form-input" id="s-pause-min" name="loss_pause_minutes" value="30" min="1" max="480"/>
        </div>
      </div>
    </div>

    <!-- ACCORDION: AI / LLM -->
    <div class="accordion-group" id="acc-llm">
      <div class="accordion-header" onclick="toggleAccordion('acc-llm')">
        <div class="accordion-header-left">
          <span class="accordion-icon">🤖</span>
          <div>
            <div class="accordion-title">AI / LLM Gate</div>
            <div class="accordion-desc">Ollama local model and remote LLM configuration</div>
          </div>
        </div>
        <span class="accordion-arrow">▼</span>
      </div>
      <div class="accordion-body">
        <div class="form-group">
          <label class="form-label">LLM Mode</label>
          <select class="form-select" id="s-llm-mode" name="llm_mode" onchange="updateLLMVisibility()">
            <option value="base">base (rule-based, fastest)</option>
            <option value="llm_local">llm_local (Ollama)</option>
            <option value="llm_remote">llm_remote (OpenAI/API)</option>
          </select>
        </div>
        <div class="form-group" id="llm-local-fields">
          <label class="form-label">Ollama Model</label>
          <input type="text" class="form-input" id="s-llm-model" name="llm_local_model" value="qwen2.5:0.5b"/>
        </div>
        <div class="form-group" id="llm-url-field">
          <label class="form-label">Ollama Base URL</label>
          <input type="text" class="form-input" id="s-ollama-url" name="ollama_base_url" value="http://localhost:11434"/>
        </div>
        <div class="form-group" id="llm-remote-fields" style="display:none;">
          <label class="form-label">Remote Endpoint</label>
          <input type="text" class="form-input" id="s-remote-ep" name="llm_remote_endpoint" value="https://api.openai.com/v1/chat/completions"/>
        </div>
        <div class="form-group" id="llm-remote-model" style="display:none;">
          <label class="form-label">Remote Model</label>
          <input type="text" class="form-input" id="s-remote-model-name" name="llm_remote_model" value="gpt-4o-mini"/>
        </div>
        <div class="form-group" id="llm-apikey" style="display:none;">
          <label class="form-label">Remote API Key</label>
          <input type="password" class="form-input" id="s-remote-key" name="llm_remote_api_key" placeholder="sk-..."/>
        </div>
        <div class="form-group">
          <label class="form-label">Context Window (tokens)</label>
          <input type="number" class="form-input" id="s-llm-ctx" name="llm_local_num_ctx" value="256" min="64" max="4096"/>
          <div class="form-hint">256 = ultrafast, 512 = balanced</div>
        </div>
        <div class="form-group">
          <div class="toggle-wrap">
            <span class="toggle-label">Health Check Enabled</span>
            <label class="toggle"><input type="checkbox" id="s-llm-health" name="llm_local_healthcheck_enabled" checked/><span class="toggle-slider"></span></label>
          </div>
        </div>
      </div>
    </div>

    <!-- ACCORDION: System -->
    <div class="accordion-group" id="acc-system">
      <div class="accordion-header" onclick="toggleAccordion('acc-system')">
        <div class="accordion-header-left">
          <span class="accordion-icon">⚙️</span>
          <div>
            <div class="accordion-title">System & Performance</div>
            <div class="accordion-desc">RAM mode, loop speed, TUI and observability</div>
          </div>
        </div>
        <span class="accordion-arrow">▼</span>
      </div>
      <div class="accordion-body">
        <div class="form-group">
          <div class="toggle-wrap">
            <span class="toggle-label">Low RAM Mode</span>
            <label class="toggle"><input type="checkbox" id="s-low-ram" name="low_ram_mode" checked/><span class="toggle-slider"></span></label>
          </div>
          <div class="form-hint">Clamps history_limit to 220 candles</div>
        </div>
        <div class="form-group">
          <label class="form-label">Max RAM (MB)</label>
          <input type="number" class="form-input" id="s-max-ram" name="max_ram_mb" value="500" min="128" max="4096"/>
        </div>
        <div class="form-group">
          <label class="form-label">Loop Sleep (seconds)</label>
          <input type="number" class="form-input" id="s-loop-sleep" name="loop_sleep_seconds" value="15" min="5" max="120"/>
        </div>
        <div class="form-group">
          <label class="form-label">History Limit (candles)</label>
          <input type="number" class="form-input" id="s-history" name="history_limit" value="300" min="50" max="1000"/>
        </div>
        <div class="form-group">
          <label class="form-label">TUI Refresh Rate (fps)</label>
          <input type="number" class="form-input" id="s-tui-fps" name="terminal_tui_refresh_per_second" value="4" min="1" max="10"/>
        </div>
        <div class="form-group">
          <div class="toggle-wrap">
            <span class="toggle-label">Terminal TUI Enabled</span>
            <label class="toggle"><input type="checkbox" id="s-tui" name="terminal_tui_enabled" checked/><span class="toggle-slider"></span></label>
          </div>
        </div>
      </div>
    </div>

  </div><!-- end settings-form -->

  <div class="btn-row">
    <button class="btn btn-primary" onclick="saveSettings()">💾 Apply to Running Bot</button>
    <button class="btn btn-secondary" onclick="loadCurrentSettings()">↺ Reload Current Values</button>
    <button class="btn btn-danger" onclick="if(confirm('Reset all filters to defaults?')) resetFilters()">🔄 Reset Filters to Default</button>
  </div>
</div>

<!-- ──────────── PAGE: SYSTEM HEALTH ──────────── -->
<div class="page" id="page-system">
  <div class="grid-4">
    <div class="metric-card" style="--mc-color:var(--accent-green)">
      <div class="metric-label">Exchange</div>
      <div class="metric-value" id="sys-exchange" style="font-size:16px;">—</div>
      <div class="metric-sub" id="sys-latency">API Latency —</div>
    </div>
    <div class="metric-card" style="--mc-color:var(--accent-blue)">
      <div class="metric-label">Database</div>
      <div class="metric-value" id="sys-db" style="font-size:16px;">—</div>
      <div class="metric-sub">Connection status</div>
    </div>
    <div class="metric-card" style="--mc-color:var(--accent-purple)">
      <div class="metric-label">Resilience</div>
      <div class="metric-value" id="sys-res" style="font-size:16px;">—</div>
      <div class="metric-sub" id="sys-res-fails">consecutive failures</div>
    </div>
    <div class="metric-card" style="--mc-color:var(--accent-cyan)">
      <div class="metric-label">WS Connected</div>
      <div class="metric-value" id="sys-ws" style="font-size:16px;">—</div>
      <div class="metric-sub">WebSocket stream</div>
    </div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">📡 Observability</div>
      <div class="filter-row"><span class="filter-name">Circuit Breaker Trips</span><span id="obs-cbt" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Exchange Reconnections</span><span id="obs-reconn" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Stale Data Ratio</span><span id="obs-stale" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">API Latency P95</span><span id="obs-lat" class="filter-value">—</span></div>
    </div>
    <div class="card">
      <div class="card-title">🔄 Bot Lifecycle</div>
      <div class="filter-row"><span class="filter-name">Uptime</span><span id="sys-uptime" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">All Modules Init</span><span id="sys-modules" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Trading Mode</span><span id="sys-tmode" class="filter-value">—</span></div>
      <div class="filter-row"><span class="filter-name">Emergency Stop</span><span id="sys-emstop" class="filter-value">—</span></div>
    </div>
  </div>
</div>

</main><!-- end main -->

<!-- TOASTS -->
<div class="toast-container" id="toast-container"></div>

<script>
/* ═══════════════════════════════════════════════
   STATE & CONFIG
═══════════════════════════════════════════════ */
let _state = {};
let _allLogs = [];
let _allTrades = [];
let _autoScroll = true;
let _sseSource = null;

/* ═══════════════════════════════════════════════
   NAVIGATION
═══════════════════════════════════════════════ */
function showPage(name, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const page = document.getElementById('page-' + name);
  if (page) page.classList.add('active');
  if (el) el.classList.add('active');
}

/* ═══════════════════════════════════════════════
   ACCORDION
═══════════════════════════════════════════════ */
function toggleAccordion(id) {
  const el = document.getElementById(id);
  el.classList.toggle('open');
  const body = el.querySelector('.accordion-body');
  if (body) body.style.display = el.classList.contains('open') ? 'grid' : 'none';
}

// Initialize accordion state
document.querySelectorAll('.accordion-group').forEach(g => {
  const body = g.querySelector('.accordion-body');
  if (body) body.style.display = g.classList.contains('open') ? 'grid' : 'none';
});

/* ═══════════════════════════════════════════════
   MULTI-SELECT
═══════════════════════════════════════════════ */
document.querySelectorAll('.ms-option').forEach(opt => {
  opt.addEventListener('click', () => opt.classList.toggle('selected'));
});

function getMultiSelectValues(containerId) {
  return [...document.querySelectorAll(`#${containerId} .ms-option.selected`)]
    .map(o => o.dataset.value);
}

/* ═══════════════════════════════════════════════
   CLOCK
═══════════════════════════════════════════════ */
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toUTCString().split(' ').slice(4, 5)[0] + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

/* ═══════════════════════════════════════════════
   TOAST NOTIFICATIONS
═══════════════════════════════════════════════ */
function showToast(msg, type = 'info', duration = 4000) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

/* ═══════════════════════════════════════════════
   SIGNAL CHIP HELPER
═══════════════════════════════════════════════ */
function signalChip(val) {
  val = (val || 'NEUTRAL').toUpperCase();
  const cls = val === 'BUY' ? 'chip-buy' : val === 'SELL' ? 'chip-sell' : val === 'HOLD' ? 'chip-hold' : 'chip-neutral';
  return `<span class="signal-chip ${cls}">${val}</span>`;
}

function setText(id, val, fallback = '—') {
  const el = document.getElementById(id);
  if (el) el.textContent = val !== undefined && val !== null && val !== '' ? val : fallback;
}

function setHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function fNum(n, dec = 2) {
  if (n === null || n === undefined || n === '') return '—';
  return parseFloat(n).toFixed(dec);
}

function fPct(n) {
  if (n === null || n === undefined) return '—';
  return (parseFloat(n) * 100).toFixed(1) + '%';
}

function colorPnl(n) {
  const v = parseFloat(n);
  if (isNaN(v)) return '—';
  const cls = v > 0 ? 'positive' : v < 0 ? 'negative' : '';
  const sign = v > 0 ? '+' : '';
  return `<span class="${cls}">${sign}${v.toFixed(4)}</span>`;
}

/* ═══════════════════════════════════════════════
   UPDATE UI FROM SNAPSHOT
═══════════════════════════════════════════════ */
function updateUI(s) {
  _state = s;

  // Topbar
  const statusMap = {
    RUNNING: 'running', WAITING_MARKET_DATA: 'waiting', PAUSED: 'paused',
    INITIALIZING: 'waiting', ERROR: 'error', COOLDOWN: 'waiting',
    CONNECTING_EXCHANGE: 'waiting', POSITION_OPEN: 'running',
  };
  const statusKey = (s.status || 'WAITING').toUpperCase().replace(/ /g,'_');
  const pillClass = statusMap[statusKey] || 'waiting';
  const pill = document.getElementById('bot-status-pill');
  pill.className = 'status-pill ' + pillClass;
  setText('bot-status-text', statusKey);
  setText('topbar-pair', (s.pair || 'BTC/USDT') + ' · ' + (s.timeframe || '5m/15m'));

  // KPIs
  setText('kpi-price', fNum(s.price, 2) + ' USDT');
  const chg = parseFloat(s.change_24h || 0);
  const chgEl = document.getElementById('kpi-change');
  if (chgEl) { chgEl.innerHTML = `<span class="${chg >= 0 ? 'positive' : 'negative'}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%</span> 24h`; }

  const eq = parseFloat(s.equity || s.balance || 0);
  setText('kpi-equity', fNum(eq, 2) + ' USDT');
  document.getElementById('kpi-balance-sub').textContent = 'Balance: ' + fNum(s.balance, 4);

  const pnl = parseFloat(s.daily_pnl || s.session_pnl || 0);
  const kpiPnl = document.getElementById('kpi-pnl');
  if (kpiPnl) kpiPnl.innerHTML = colorPnl(pnl) + ' USDT';
  document.getElementById('kpi-unrealized').innerHTML = 'Unrealized: ' + colorPnl(s.unrealized_pnl);

  const wr = parseFloat(s.win_rate || 0);
  setText('kpi-winrate', (wr * 100).toFixed(1) + '%');
  setText('kpi-trades-today', (s.trades_today || 0) + ' trades today');

  // Market state
  setHtml('mkt-trend', signalChip(s.trend));
  setText('mkt-adx', fNum(s.adx, 2));
  setText('mkt-rsi', fNum(s.rsi, 2));
  setText('mkt-spread', fNum(s.spread, 6));
  setText('mkt-regime', s.volatility_regime || '—');
  setHtml('mkt-flow', signalChip(s.order_flow));
  setText('mkt-tf', s.timeframe || '—');
  setText('mkt-cooldown', s.cooldown || '—');

  // Signals
  const sig = s.signal || 'HOLD';
  setHtml('sig-main', signalChip(sig));
  const conf = parseFloat(s.confidence || 0);
  setText('sig-conf-pct', (conf * 100).toFixed(1) + '%');
  const bar = document.getElementById('sig-conf-bar');
  if (bar) {
    bar.style.width = (conf * 100) + '%';
    bar.className = 'confidence-bar-fill' + (conf >= 0.7 ? ' high' : conf < 0.45 ? ' low' : '');
  }
  const sigs = s.signals || {};
  setHtml('sig-trend', signalChip(sigs.trend || s.trend));
  setHtml('sig-momentum', signalChip(sigs.momentum));
  setHtml('sig-volume', signalChip(sigs.volume));
  setHtml('sig-volatility', signalChip(sigs.volatility));
  setHtml('sig-structure', signalChip(sigs.structure));
  setHtml('sig-orderflow', signalChip(sigs.order_flow || s.order_flow));

  // Position
  const hasPos = s.has_open_position || s.open_position_side;
  document.getElementById('position-none').style.display = hasPos ? 'none' : 'block';
  document.getElementById('position-details').style.display = hasPos ? 'flex' : 'none';
  if (hasPos) {
    const side = (s.open_position_side || '').toUpperCase();
    const sideBadge = document.getElementById('pos-side-badge');
    sideBadge.textContent = side;
    sideBadge.className = 'pos-side ' + side;
    setText('pos-symbol', s.pair || 'BTC/USDT');
    setText('pos-entry', fNum(s.open_position_entry, 4));
    setText('pos-mark', fNum(s.price, 4));
    setText('pos-qty', fNum(s.open_position_qty, 6));
    setText('pos-sl', fNum(s.open_position_sl, 4));
    setText('pos-tp', fNum(s.open_position_tp, 4));
    document.getElementById('pos-upnl').innerHTML = colorPnl(s.unrealized_pnl) + ' USDT';
  }

  // Decision trace
  const dt = s.decision_trace || {};
  const factors = dt.factor_scores || {};
  const gating = s.decision_gating || {};
  const dtLines = [];
  for (const [k, v] of Object.entries(factors)) {
    const vf = parseFloat(v);
    const cls = vf > 0 ? 'positive' : vf < 0 ? 'negative' : '';
    dtLines.push(`<div class="filter-row"><span class="filter-name">factor:${k}</span><span class="${cls}" style="font-weight:700;">${vf >= 0 ? '+' : ''}${vf.toFixed(3)}</span></div>`);
  }
  for (const [k, v] of Object.entries(gating)) {
    dtLines.push(`<div class="filter-row"><span class="filter-name">gate:${k}</span><span class="filter-value">${v}</span></div>`);
  }
  if (dtLines.length > 0) setHtml('decision-trace-content', dtLines.join(''));
  setText('decision-reason', s.decision_reason);
  setText('setup-quality', fNum(s.signal_quality_score, 3));
  setText('capital-profile', s.capital_profile);
  setText('operable-capital', fNum(s.operable_capital_usdt, 2) + ' USDT');

  // Logs
  const logs = s.logs || [];
  if (logs.length > 0) {
    logs.forEach(l => {
      if (!_allLogs.find(x => x.time === l.time && x.message === l.message)) {
        _allLogs.push(l);
      }
    });
    if (_allLogs.length > 500) _allLogs = _allLogs.slice(-500);
    renderLogs();
  }

  // LLM
  const llm = s.llm_trade_confirmator || {};
  const llmMode = (s.llm_mode || 'base').toLowerCase();
  setText('llm-mode-badge', llmMode.toUpperCase());
  setText('llm-model-name', llm.model || (llmMode === 'llm_local' ? s.llm_local_model || 'qwen2.5:0.5b' : llmMode === 'llm_remote' ? s.llm_remote_model || 'gpt-4o-mini' : '—'));
  const healthy = llm.local_endpoint_healthy;
  const healthDot = document.getElementById('llm-health-dot');
  if (healthDot) {
    healthDot.className = 'llm-health-dot ' + (healthy === true ? 'ok' : healthy === false ? 'fail' : 'unknown');
    setText('llm-health-text', healthy === true ? 'Online' : healthy === false ? 'Offline' : 'Not checked');
  }
  setText('llm-warmup', llm.warmup_done === true ? 'Yes' : llm.warmup_done === false ? 'Pending' : '—');
  setText('llm-total', llm.total_analyzed || 0);
  setText('llm-confirmed', llm.confirmed || 0);
  setText('llm-rejected', llm.rejected || 0);
  setText('llm-rate', fNum(llm.confirmation_rate, 1) + '%');
  setText('llm-latency', fNum(llm.avg_analysis_time_ms, 1) + 'ms');

  // Autonomous Optimizer
  setText('ao-trades', s.auto_improve_total_trades || 0);
  setText('ao-wr', fPct(s.auto_improve_win_rate));
  setText('ao-losses', s.auto_improve_consecutive_losses || 0);
  setText('ao-cycles', s.auto_improve_optimization_count || 0);

  // Filter config
  const filters = s.autonomous_filters || s.runtime_filter_config || {};
  setText('f-adx', fNum(filters.adx_threshold, 1));
  setText('f-rsi-buy', fNum(filters.rsi_buy_threshold, 1));
  setText('f-rsi-sell', fNum(filters.rsi_sell_threshold, 1));
  setText('f-vol-buy', fNum(filters.volume_buy_threshold, 2));
  setText('f-vol-sell', fNum(filters.volume_sell_threshold, 2));
  setText('f-conf', fPct(filters.min_confidence));
  setText('f-atr-low', fNum(filters.atr_low_threshold, 4));
  setText('f-atr-high', fNum(filters.atr_high_threshold, 4));
  setText('f-sl-mult', fNum(filters.stop_loss_atr_multiplier, 1));
  setText('f-tp-mult', fNum(filters.take_profit_atr_multiplier, 1));

  // Validation checks
  const vChecks = (s.decision_trace || {}).validation_checks || [];
  if (vChecks.length > 0) {
    const lines = vChecks.map(c => {
      const passed = c.passed;
      const cls = passed ? 'filter-pass' : 'filter-fail';
      const icon = passed ? '✅' : '❌';
      return `<div class="filter-row"><span class="filter-name">${icon} ${c.name}</span><span class="${cls}" style="font-size:11px;">${c.value !== undefined ? c.value : ''}</span></div>`;
    }).join('');
    setHtml('validation-checks', lines);
  }

  // Autonomous brain
  setText('auto-condition', s.autonomous_market_condition || '—');
  setText('auto-confluence', fNum(s.confluence_score, 3));
  setText('auto-mode', s.investment_mode || s.runtime_investment_mode || '—');

  // Risk page
  setText('r-daily-loss', fPct(s.daily_loss_limit_fraction || s.live_trade_risk_fraction));
  setText('r-risk-pt', fPct(s.live_trade_risk_fraction));
  const riskMetrics = s.risk_metrics || {};
  const dd = riskMetrics.current_drawdown;
  setText('r-max-dd', dd !== undefined ? fPct(dd) : '—');
  setText('r-open-pos', (s.open_positions || []).length);
  setText('r-dd-active', s.trading_paused_by_drawdown ? '🔴 YES' : '🟢 No');
  setText('r-pause-until', s.pause_trading_until || '—');
  setText('r-cons-losses', s.consecutive_losses || 0);
  setText('r-emergency', s.emergency_stop_active ? '🔴 ACTIVE' : '🟢 Inactive');
  setText('r-cap-reserve', fPct(s.capital_reserve_ratio));
  setText('r-cash-buf', fNum(s.min_cash_buffer_usdt, 2) + ' USDT');
  setText('r-session-rec', s.session_recommendation || '—');
  setText('r-eq-peak', fNum(riskMetrics.equity_peak, 4) + ' USDT');

  // System
  const exchStat = (s.exchange_status || 'UNKNOWN').toUpperCase();
  setText('sys-exchange', exchStat);
  document.getElementById('sys-exchange').className = 'metric-value ' + (exchStat === 'CONNECTED' ? 'positive' : 'negative');
  setText('sys-latency', 'Latency P95: ' + fNum(s.api_latency_p95_ms, 1) + 'ms');
  const dbStat = (s.database_status || 'UNKNOWN').toUpperCase();
  setText('sys-db', dbStat);
  document.getElementById('sys-db').className = 'metric-value ' + (dbStat.includes('CONNECT') ? 'positive' : 'warning');
  setText('sys-res', s.resilience_healthy === true ? 'Healthy' : s.resilience_healthy === false ? 'Degraded' : '—');
  setText('sys-res-fails', (s.resilience_consecutive_failures || 0) + ' consecutive failures');
  setText('sys-ws', s.ws_connected === true ? 'Connected' : s.ws_connected === false ? 'Disconnected' : '—');

  setText('obs-cbt', s.circuit_breaker_trips || 0);
  setText('obs-reconn', s.exchange_reconnections || 0);
  setText('obs-stale', fPct(s.stale_market_data_ratio));
  setText('obs-lat', fNum(s.api_latency_p95_ms, 2) + 'ms');
  setText('sys-modules', s.all_modules_initialized ? '✅ Yes' : '⏳ Initializing');
  setText('sys-tmode', s.trading_mode || '—');
  setText('sys-emstop', s.emergency_stop_active ? '🔴 ACTIVE' : '🟢 Inactive');

  // Positions table
  updatePositionsTable(s);

  // Trades table
  if (s.trade_history && s.trade_history.length > 0) {
    _allTrades = s.trade_history;
    renderTrades();
  }
}

/* ═══════════════════════════════════════════════
   POSITIONS TABLE
═══════════════════════════════════════════════ */
function updatePositionsTable(s) {
  const tbody = document.getElementById('positions-tbody');
  if (!tbody) return;
  const positions = s.open_positions || [];
  if (positions.length === 0 && !s.has_open_position) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">No open positions</td></tr>';
    return;
  }
  const rows = positions.length > 0 ? positions : [{
    side: s.open_position_side, quantity: s.open_position_qty,
    entry_price: s.open_position_entry, stop_loss: s.open_position_sl,
    take_profit: s.open_position_tp, unrealized_pnl: s.unrealized_pnl,
    symbol: s.pair
  }];
  tbody.innerHTML = rows.map(p => {
    const side = (p.side || '').toUpperCase();
    const sideCls = side === 'BUY' ? 'chip-buy' : 'chip-sell';
    return `<tr>
      <td style="font-weight:600;">${p.symbol || s.pair || '—'}</td>
      <td><span class="signal-chip ${sideCls}">${side}</span></td>
      <td style="font-family:monospace;">${fNum(p.quantity, 6)}</td>
      <td style="font-family:monospace;">${fNum(p.entry_price, 4)}</td>
      <td style="font-family:monospace;">${fNum(s.price, 4)}</td>
      <td style="color:var(--accent-red); font-family:monospace;">${fNum(p.stop_loss, 4)}</td>
      <td style="color:var(--accent-green); font-family:monospace;">${fNum(p.take_profit, 4)}</td>
      <td>${colorPnl(p.unrealized_pnl)}</td>
      <td><button class="btn btn-danger" style="padding:4px 10px; font-size:11px;" onclick="sendControl('force_close')">Close</button></td>
    </tr>`;
  }).join('');
}

/* ═══════════════════════════════════════════════
   TRADES TABLE
═══════════════════════════════════════════════ */
function renderTrades() {
  const side = document.getElementById('filter-side')?.value || '';
  const result = document.getElementById('filter-result')?.value || '';
  let trades = _allTrades;
  if (side) trades = trades.filter(t => (t.side||'').toUpperCase() === side);
  if (result === 'WIN') trades = trades.filter(t => parseFloat(t.pnl||0) > 0);
  if (result === 'LOSS') trades = trades.filter(t => parseFloat(t.pnl||0) < 0);

  const tbody = document.getElementById('trades-tbody');
  if (!tbody) return;
  if (trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; color:var(--text-muted); padding:30px;">No trades found</td></tr>';
    return;
  }
  tbody.innerHTML = [...trades].reverse().map(t => {
    const pnl = parseFloat(t.pnl || 0);
    const side = (t.side||'').toUpperCase();
    const sideCls = side === 'BUY' ? 'chip-buy' : 'chip-sell';
    return `<tr>
      <td style="font-size:11px; color:var(--text-muted); white-space:nowrap;">${t.time || '—'}</td>
      <td style="font-weight:600;">${t.pair || '—'}</td>
      <td><span class="signal-chip ${sideCls}">${side}</span></td>
      <td style="font-family:monospace;">${fNum(t.entry, 4)}</td>
      <td style="font-family:monospace;">${fNum(t.exit, 4)}</td>
      <td style="font-family:monospace;">${fNum(t.size, 6)}</td>
      <td>${colorPnl(pnl)}</td>
      <td><span class="signal-chip ${pnl > 0 ? 'chip-buy' : pnl < 0 ? 'chip-sell' : 'chip-neutral'}">${t.status || (pnl > 0 ? 'WIN' : pnl < 0 ? 'LOSS' : '—')}</span></td>
      <td style="font-size:11px; color:var(--text-muted);">${t.duration || '—'}</td>
    </tr>`;
  }).join('');

  // Summary stats
  const wins = trades.filter(t => parseFloat(t.pnl||0) > 0).length;
  const losses = trades.filter(t => parseFloat(t.pnl||0) < 0).length;
  const totalPnl = trades.reduce((sum, t) => sum + parseFloat(t.pnl||0), 0);
  setText('stat-total', trades.length);
  setText('stat-wins', wins);
  setText('stat-losses', losses);
  const tpEl = document.getElementById('stat-total-pnl');
  if (tpEl) tpEl.innerHTML = colorPnl(totalPnl);
}

function filterTrades() { renderTrades(); }

/* ═══════════════════════════════════════════════
   LOGS
═══════════════════════════════════════════════ */
function renderLogs() {
  const levelFilter = document.getElementById('log-level-filter')?.value || '';
  const logs = levelFilter ? _allLogs.filter(l => (l.level||'').toUpperCase() === levelFilter) : _allLogs;

  function buildLogHtml(logList) {
    return logList.slice(-200).map(l => {
      const lvl = (l.level||'INFO').toUpperCase();
      const important = lvl === 'ERROR' || lvl === 'WARNING';
      return `<div class="log-entry">
        <span class="log-time">${l.time || '--:--:--'}</span>
        <span class="log-level ${lvl}">${lvl}</span>
        <span class="log-msg ${important ? 'important' : ''}">${(l.message||'').substring(0, 300)}</span>
      </div>`;
    }).join('');
  }

  const mainFeed = document.getElementById('log-feed-main');
  if (mainFeed) { mainFeed.innerHTML = buildLogHtml(_allLogs.slice(-20)); }

  const fullFeed = document.getElementById('log-feed-full');
  if (fullFeed) {
    fullFeed.innerHTML = buildLogHtml(logs);
    if (_autoScroll) fullFeed.scrollTop = fullFeed.scrollHeight;
  }
}

function filterLogs() { renderLogs(); }
function clearLogs() { _allLogs = []; renderLogs(); }
function toggleAutoScroll() {
  _autoScroll = !_autoScroll;
  setText('autoscroll-state', _autoScroll ? 'ON' : 'OFF');
}

/* ═══════════════════════════════════════════════
   BOT CONTROLS
═══════════════════════════════════════════════ */
async function sendControl(action) {
  if (action === 'emergency_stop' && !confirm('Activate EMERGENCY STOP? This will halt all trading immediately.')) return;
  if (action === 'force_close' && !confirm('Force close all open positions?')) return;
  try {
    const res = await fetch('/api/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action })
    });
    const data = await res.json();
    showToast(data.message || (data.success ? 'Command sent' : 'Error: ' + data.error),
      data.success ? 'success' : 'error');
  } catch (e) {
    showToast('Connection error: ' + e.message, 'error');
  }
}

/* ═══════════════════════════════════════════════
   SETTINGS
═══════════════════════════════════════════════ */
function updateLLMVisibility() {
  const mode = document.getElementById('s-llm-mode')?.value || 'base';
  const isLocal = mode === 'llm_local';
  const isRemote = mode === 'llm_remote';
  const show = (id, cond) => { const el = document.getElementById(id); if (el) el.style.display = cond ? '' : 'none'; };
  show('llm-local-fields', isLocal);
  show('llm-url-field', isLocal);
  show('llm-remote-fields', isRemote);
  show('llm-remote-model', isRemote);
  show('llm-apikey', isRemote);
}

function loadCurrentSettings() {
  fetch('/api/snapshot')
    .then(r => r.json())
    .then(data => {
      const s = data.data || data;
      const rt = s.runtime_settings || {};
      // Populate fields from snapshot runtime_settings
      const setVal = (id, val) => { const el = document.getElementById(id); if (el && val !== undefined && val !== null) el.value = val; };
      const setCheck = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
      setVal('s-symbol', s.pair || 'BTC/USDT');
      setVal('s-llm-mode', s.llm_mode || rt.llm_mode || 'base');
      setVal('s-llm-model', rt.llm_local_model || 'qwen2.5:0.5b');
      setVal('s-ollama-url', rt.ollama_base_url || 'http://localhost:11434');
      setVal('s-max-trades', rt.max_trades_per_day || 20);
      setVal('s-max-pos', rt.max_concurrent_trades || 1);
      setVal('s-loop-sleep', rt.loop_sleep_seconds || 15);
      setVal('s-tui-fps', rt.terminal_tui_refresh_per_second || 4);
      setVal('s-max-ram', rt.max_ram_mb || 500);
      setCheck('s-spot-only', rt.spot_only_mode !== false);
      setCheck('s-low-ram', rt.low_ram_mode !== false);
      setCheck('s-tui', rt.terminal_tui_enabled !== false);
      setCheck('s-multi', rt.feature_multi_symbol_enabled === true);
      setCheck('s-dyn-exit', rt.dynamic_exit_enabled !== false);

      // Filter config
      const f = s.autonomous_filters || s.runtime_filter_config || {};
      const setRange = (id, valId, val, mult) => {
        const el = document.getElementById(id);
        if (el && val !== undefined) { el.value = val; if (valId) setText(valId, mult ? (val * mult).toFixed(mult === 100 ? 1 : 0) + '%' : val); }
      };
      setRange('s-adx', 's-adx-val', f.adx_threshold || 12);
      setRange('s-minconf', 's-minconf-val', f.min_confidence || 0.45, 100);
      setVal('s-rsi-buy', f.rsi_buy_threshold || 48);
      setVal('s-rsi-sell', f.rsi_sell_threshold || 52);
      setVal('s-vol-buy', f.volume_buy_threshold || 0.90);
      setVal('s-vol-sell', f.volume_sell_threshold || 0.85);
      setVal('s-sl-atr', f.stop_loss_atr_multiplier || 1.5);
      setVal('s-tp-atr', f.take_profit_atr_multiplier || 2.5);

      updateLLMVisibility();
      showToast('Settings loaded from bot', 'info');
    })
    .catch(e => showToast('Failed to load settings: ' + e.message, 'error'));
}

async function saveSettings() {
  // Collect all form values
  const payload = {
    // Trading
    trading_symbol: document.getElementById('s-symbol')?.value,
    trading_symbols: getMultiSelectValues('ms-symbols'),
    primary_timeframe: document.getElementById('s-tf')?.value,
    confirmation_timeframe: document.getElementById('s-ctf')?.value,
    risk_per_trade_fraction: parseFloat(document.getElementById('s-risk')?.value || 0.01),
    max_trades_per_day: parseInt(document.getElementById('s-max-trades')?.value || 20),
    investment_mode: document.getElementById('s-inv-mode')?.value,
    spot_only_mode: document.getElementById('s-spot-only')?.checked,
    feature_multi_symbol_enabled: document.getElementById('s-multi')?.checked,
    dynamic_exit_enabled: document.getElementById('s-dyn-exit')?.checked,
    // Filters
    filter_config: {
      adx_threshold: parseFloat(document.getElementById('s-adx')?.value || 12),
      min_confidence: parseFloat(document.getElementById('s-minconf')?.value || 0.45),
      rsi_buy_threshold: parseFloat(document.getElementById('s-rsi-buy')?.value || 48),
      rsi_sell_threshold: parseFloat(document.getElementById('s-rsi-sell')?.value || 52),
      volume_buy_threshold: parseFloat(document.getElementById('s-vol-buy')?.value || 0.90),
      volume_sell_threshold: parseFloat(document.getElementById('s-vol-sell')?.value || 0.85),
      stop_loss_atr_multiplier: parseFloat(document.getElementById('s-sl-atr')?.value || 1.5),
      take_profit_atr_multiplier: parseFloat(document.getElementById('s-tp-atr')?.value || 2.5),
    },
    // Risk
    daily_loss_limit_fraction: parseFloat(document.getElementById('s-daily-loss')?.value || 0.02),
    max_drawdown_fraction: parseFloat(document.getElementById('s-max-dd')?.value || 0.10),
    capital_limit_usdt: parseFloat(document.getElementById('s-cap-limit')?.value || 0),
    min_cash_buffer_usdt: parseFloat(document.getElementById('s-cash-buf')?.value || 10),
    max_concurrent_trades: parseInt(document.getElementById('s-max-pos')?.value || 1),
    loss_pause_minutes: parseInt(document.getElementById('s-pause-min')?.value || 30),
    // AI
    llm_mode: document.getElementById('s-llm-mode')?.value || 'base',
    llm_local_model: document.getElementById('s-llm-model')?.value,
    ollama_base_url: document.getElementById('s-ollama-url')?.value,
    llm_remote_endpoint: document.getElementById('s-remote-ep')?.value,
    llm_remote_model: document.getElementById('s-remote-model-name')?.value,
    llm_remote_api_key: document.getElementById('s-remote-key')?.value,
    llm_local_num_ctx: parseInt(document.getElementById('s-llm-ctx')?.value || 256),
    llm_local_healthcheck_enabled: document.getElementById('s-llm-health')?.checked,
    // System
    low_ram_mode: document.getElementById('s-low-ram')?.checked,
    max_ram_mb: parseInt(document.getElementById('s-max-ram')?.value || 500),
    loop_sleep_seconds: parseInt(document.getElementById('s-loop-sleep')?.value || 15),
    history_limit: parseInt(document.getElementById('s-history')?.value || 300),
    terminal_tui_refresh_per_second: parseInt(document.getElementById('s-tui-fps')?.value || 4),
    terminal_tui_enabled: document.getElementById('s-tui')?.checked,
  };

  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      showToast('✅ Settings applied to running bot!', 'success');
    } else {
      showToast('❌ Error: ' + (data.error || 'Unknown error'), 'error');
    }
  } catch (e) {
    showToast('Connection error: ' + e.message, 'error');
  }
}

async function resetFilters() {
  try {
    const res = await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reset_filters: true })
    });
    const data = await res.json();
    showToast(data.success ? '✅ Filters reset to default' : '❌ ' + data.error,
      data.success ? 'success' : 'error');
    if (data.success) loadCurrentSettings();
  } catch (e) {
    showToast('Error: ' + e.message, 'error');
  }
}

/* ═══════════════════════════════════════════════
   SSE CONNECTION (real-time updates)
═══════════════════════════════════════════════ */
function connectSSE() {
  if (_sseSource) _sseSource.close();
  _sseSource = new EventSource('/api/stream');
  _sseSource.onmessage = e => {
    try { updateUI(JSON.parse(e.data)); } catch(err) { console.error('SSE parse error', err); }
  };
  _sseSource.onerror = () => {
    // SSE failed, fall back to polling
    _sseSource.close();
    _sseSource = null;
    setTimeout(startPolling, 2000);
  };
}

function startPolling() {
  fetch('/api/snapshot')
    .then(r => r.json())
    .then(d => updateUI(d.data || d))
    .catch(e => console.error('Polling error', e));
  setInterval(() => {
    fetch('/api/snapshot')
      .then(r => r.json())
      .then(d => updateUI(d.data || d))
      .catch(() => {});
  }, 2000);
}

/* ═══════════════════════════════════════════════
   STARTUP
═══════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  // Try SSE first, fall back to polling
  connectSSE();
  // Load initial settings after 1s
  setTimeout(loadCurrentSettings, 1500);
  updateLLMVisibility();
});
</script>
</body>
</html>
```

---

## BLOQUE D — DASHBOARD TERMINAL TUI (Rich) — RECONSTRUCCIÓN

### D1. Archivo: `reco_trading/ui/dashboard.py` — REEMPLAZAR COMPLETAMENTE

Reemplaza el archivo completo con la siguiente implementación mejorada que muestra todos los datos correctamente, tiene mejor layout y maneja errores:

```python
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from rich.align import Align
from rich.box import ROUNDED, SIMPLE_HEAD, MINIMAL_DOUBLE_HEAD
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


@dataclass
class DashboardSnapshot:
    state: str = "INITIALIZING"
    pair: str = ""
    timeframe: str = ""
    price: float | None = None
    spread: float | None = None
    bid: float | None = None
    ask: float | None = None
    trend: str | None = None
    adx: float | None = None
    rsi: float | None = None
    volatility_regime: str | None = None
    order_flow: str | None = None
    signal: str | None = None
    confidence: float | None = None
    balance: float | None = None
    equity: float | None = None
    daily_pnl: float | None = None
    session_pnl: float | None = None
    operable_capital_usdt: float | None = None
    capital_profile: str | None = None
    trades_today: int = 0
    win_rate: float | None = None
    last_trade: str | None = None
    cooldown: str | None = None
    consecutive_losses: int = 0
    signals: dict[str, str] = field(default_factory=dict)
    decision_trace: dict[str, Any] = field(default_factory=dict)
    decision_gating: dict[str, Any] = field(default_factory=dict)
    decision_reason: str | None = None
    autonomous_filters: dict[str, Any] = field(default_factory=dict)
    autonomous_market_condition: str | None = None
    api_latency_p95_ms: float | None = None
    stale_market_data_ratio: float | None = None
    exchange_reconnections: int = 0
    circuit_breaker_trips: int = 0
    database_status: str | None = None
    exchange_status: str | None = None
    exit_intelligence_score: float | None = None
    exit_intelligence_reason: str | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)
    unrealized_pnl: float | None = None
    open_position_side: str | None = None
    open_position_entry: float | None = None
    open_position_qty: float | None = None
    open_position_sl: float | None = None
    open_position_tp: float | None = None
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    llm_mode: str | None = None
    llm_trade_confirmator: dict[str, Any] = field(default_factory=dict)
    session_recommendation: str | None = None
    auto_improve_win_rate: float | None = None
    auto_improve_total_trades: int = 0
    investment_mode: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DashboardSnapshot":
        return cls(
            state=str(data.get("status", "INITIALIZING")),
            pair=str(data.get("pair", "")),
            timeframe=str(data.get("timeframe", "")),
            price=_to_float(data.get("price")),
            spread=_to_float(data.get("spread")),
            bid=_to_float(data.get("bid")),
            ask=_to_float(data.get("ask")),
            trend=_to_text(data.get("trend")),
            adx=_to_float(data.get("adx")),
            rsi=_to_float(data.get("rsi")),
            volatility_regime=_to_text(data.get("volatility_regime")),
            order_flow=_to_text(data.get("order_flow")),
            signal=_to_text(data.get("signal")),
            confidence=_to_float(data.get("confidence")),
            balance=_to_float(data.get("balance")),
            equity=_to_float(data.get("equity")),
            daily_pnl=_to_float(data.get("daily_pnl")),
            session_pnl=_to_float(data.get("session_pnl")),
            operable_capital_usdt=_to_float(data.get("operable_capital_usdt")),
            capital_profile=_to_text(data.get("capital_profile")),
            trades_today=int(data.get("trades_today", 0) or 0),
            win_rate=_to_float(data.get("win_rate")),
            last_trade=_to_text(data.get("last_trade")),
            cooldown=_to_text(data.get("cooldown")),
            consecutive_losses=int(data.get("consecutive_losses", 0) or 0),
            signals=dict(data.get("signals", {}) or {}),
            decision_trace=dict(data.get("decision_trace", {}) or {}),
            decision_gating=dict(data.get("decision_gating", {}) or {}),
            decision_reason=_to_text(data.get("decision_reason")),
            autonomous_filters=dict(data.get("autonomous_filters", {}) or {}),
            autonomous_market_condition=_to_text(data.get("autonomous_market_condition")),
            api_latency_p95_ms=_to_float(data.get("api_latency_p95_ms")),
            stale_market_data_ratio=_to_float(data.get("stale_market_data_ratio")),
            exchange_reconnections=int(data.get("exchange_reconnections", 0) or 0),
            circuit_breaker_trips=int(data.get("circuit_breaker_trips", 0) or 0),
            database_status=_to_text(data.get("database_status")),
            exchange_status=_to_text(data.get("exchange_status")),
            exit_intelligence_score=_to_float(data.get("exit_intelligence_score")),
            exit_intelligence_reason=_to_text(data.get("exit_intelligence_reason")),
            logs=[dict(i) for i in (data.get("logs") or []) if isinstance(i, Mapping)],
            unrealized_pnl=_to_float(data.get("unrealized_pnl")),
            open_position_side=_to_text(data.get("open_position_side")),
            open_position_entry=_to_float(data.get("open_position_entry")),
            open_position_qty=_to_float(data.get("open_position_qty")),
            open_position_sl=_to_float(data.get("open_position_sl")),
            open_position_tp=_to_float(data.get("open_position_tp")),
            open_positions=[dict(i) for i in (data.get("open_positions") or []) if isinstance(i, Mapping)],
            llm_mode=_to_text(data.get("llm_mode")),
            llm_trade_confirmator=dict(data.get("llm_trade_confirmator", {}) or {}),
            session_recommendation=_to_text(data.get("session_recommendation")),
            auto_improve_win_rate=_to_float(data.get("auto_improve_win_rate")),
            auto_improve_total_trades=int(data.get("auto_improve_total_trades", 0) or 0),
            investment_mode=_to_text(data.get("investment_mode")),
        )


class TerminalDashboard:
    """
    Rich TUI Dashboard — Full rebuild.
    Adapts between wide (>= 110 cols) and compact mode.
    """

    def __init__(self) -> None:
        self.console = Console()
        self._width_cache: int = 0

    def render(self, snapshot: DashboardSnapshot | Mapping[str, Any]) -> Any:
        try:
            snap = DashboardSnapshot.from_mapping(snapshot) if isinstance(snapshot, Mapping) else snapshot
            width = self.console.size.width

            # ── HEADER ──────────────────────────────────────────────
            header_grid = Table.grid(expand=True)
            header_grid.add_column(ratio=5)
            header_grid.add_column(ratio=1, justify="right")
            header_grid.add_row(
                Text.assemble(
                    ("◈ RECO TRADING ", "bold bright_cyan"),
                    ("TUI", "bold blue"),
                    ("  │  ", "dim"),
                    (snap.pair or "—", "bold white"),
                    ("  ", ""),
                    (f"[{snap.timeframe or '—'}]", "dim cyan"),
                ),
                _status_badge(snap.state),
            )
            header_grid.add_row(
                Text.assemble(
                    _signal_badge(snap.signal),
                    ("  conf:", "dim"),
                    (f" {(snap.confidence or 0)*100:.1f}%", "bold yellow" if (snap.confidence or 0) < 0.5 else "bold green"),
                    ("  │  market: ", "dim"),
                    (snap.autonomous_market_condition or "—", "cyan"),
                    ("  │  mode: ", "dim"),
                    (snap.investment_mode or "—", "magenta"),
                ),
                Text(f"latency {_fmt(snap.api_latency_p95_ms, 1)}ms", style="dim"),
            )

            # ── MARKET PANEL ────────────────────────────────────────
            market = Table.grid(expand=True, padding=(0, 1))
            market.add_column(style="dim", min_width=14)
            market.add_column(style="bold white")
            market.add_row("Price", f"[bold bright_white]{_fmt(snap.price, 2)}[/]  USDT")
            market.add_row("Bid / Ask", f"[green]{_fmt(snap.bid, 4)}[/] / [red]{_fmt(snap.ask, 4)}[/]")
            market.add_row("Spread", f"[yellow]{_fmt(snap.spread, 6)}[/]")
            market.add_row("ADX", _adx_styled(snap.adx))
            market.add_row("RSI", _rsi_styled(snap.rsi))
            market.add_row("Trend", _trend_badge(snap.trend))
            market.add_row("Regime", _regime_badge(snap.volatility_regime))
            market.add_row("Order Flow", _flow_badge(snap.order_flow))
            market.add_row("Cooldown", Text(snap.cooldown or "READY", style="green" if (snap.cooldown or "READY") in ("READY", "ready") else "yellow"))

            # ── SIGNALS PANEL ────────────────────────────────────────
            sigs = snap.signals or {}
            sig_grid = Table(box=MINIMAL_DOUBLE_HEAD, expand=True, show_header=True, padding=(0, 1))
            sig_grid.add_column("Signal", style="dim", width=12)
            sig_grid.add_column("5m", justify="center", width=8)
            sig_grid.add_column("15m", justify="center", width=8)
            for name, key in [("Trend","trend"),("Momentum","momentum"),("Volume","volume"),
                               ("Volatility","volatility"),("Structure","structure"),("OrderFlow","order_flow")]:
                val = sigs.get(key, "NEUTRAL")
                sig_grid.add_row(name, _signal_cell(val), "—")

            # Confidence bar
            conf_pct = int((snap.confidence or 0) * 20)
            conf_bar = "█" * conf_pct + "░" * (20 - conf_pct)
            conf_color = "green" if (snap.confidence or 0) >= 0.65 else "yellow" if (snap.confidence or 0) >= 0.45 else "red"
            conf_row = Text.assemble(("Conf: ", "dim"), (conf_bar, conf_color), (f" {(snap.confidence or 0)*100:.1f}%", f"bold {conf_color}"))

            # ── PORTFOLIO PANEL ─────────────────────────────────────
            portfolio = Table.grid(expand=True, padding=(0, 1))
            portfolio.add_column(style="dim", min_width=18)
            portfolio.add_column(style="bold white")
            portfolio.add_row("Balance", f"[bold]{_fmt(snap.balance, 4)}[/] USDT")
            portfolio.add_row("Equity", f"[bold]{_fmt(snap.equity, 4)}[/] USDT")
            portfolio.add_row("Operable", f"[cyan]{_fmt(snap.operable_capital_usdt, 4)}[/] USDT")
            pnl = snap.daily_pnl or snap.session_pnl or 0.0
            portfolio.add_row("Session PnL", _styled_pnl(pnl))
            portfolio.add_row("Unrealized", _styled_pnl(snap.unrealized_pnl))
            portfolio.add_row("Win Rate", _fmt_pct(snap.win_rate))
            portfolio.add_row("Trades Today", str(snap.trades_today))
            portfolio.add_row("Consec. Losses", _losses_styled(snap.consecutive_losses))
            portfolio.add_row("Capital Profile", Text(snap.capital_profile or "—", style="magenta"))
            portfolio.add_row("Session Rec.", Text(snap.session_recommendation or "—",
                style="green" if snap.session_recommendation == "TRADE" else "yellow" if snap.session_recommendation == "CAUTION" else "red"))

            # ── OPEN POSITION ────────────────────────────────────────
            pos_table = Table(box=ROUNDED, expand=True, padding=(0, 1))
            pos_table.add_column("Side", justify="center", width=6)
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("Mark", justify="right")
            pos_table.add_column("PnL", justify="right")
            pos_table.add_column("SL", justify="right")
            pos_table.add_column("TP", justify="right")

            positions = snap.open_positions or []
            if not positions and snap.open_position_side:
                positions = [{"side": snap.open_position_side, "quantity": snap.open_position_qty,
                              "entry_price": snap.open_position_entry, "stop_loss": snap.open_position_sl,
                              "take_profit": snap.open_position_tp, "unrealized_pnl": snap.unrealized_pnl}]
            if positions:
                for pos in positions[:3]:
                    side = str(pos.get("side", "")).upper()
                    side_style = "bold green" if side == "BUY" else "bold red"
                    pos_table.add_row(
                        Text(side, style=side_style),
                        _fmt(pos.get("quantity"), 6),
                        _fmt(pos.get("entry_price"), 4),
                        _fmt(snap.price, 4),
                        _styled_pnl(_to_float(pos.get("unrealized_pnl"))),
                        Text(_fmt(pos.get("stop_loss"), 4), style="red"),
                        Text(_fmt(pos.get("take_profit"), 4), style="green"),
                    )
            else:
                pos_table.add_row(Text("—", style="dim"), "—", "—", "—", Text("No position", style="dim"), "—", "—")

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

            # ── FILTER STATUS ─────────────────────────────────────────
            af = snap.autonomous_filters
            filter_table = Table.grid(expand=True, padding=(0, 1))
            filter_table.add_column(style="dim", min_width=16)
            filter_table.add_column(style="bold cyan")
            if af:
                filter_table.add_row("ADX ≥", _fmt_float(af.get("adx_threshold"), 1))
                filter_table.add_row("RSI Buy ≥", _fmt_float(af.get("rsi_buy_threshold"), 1))
                filter_table.add_row("RSI Sell ≤", _fmt_float(af.get("rsi_sell_threshold"), 1))
                filter_table.add_row("Min Conf", f"{_fmt_float((af.get('min_confidence') or 0) * 100, 1)}%")
                filter_table.add_row("Vol Buy ≥", _fmt_float(af.get("volume_buy_threshold"), 2))
            else:
                filter_table.add_row("Filters", "loading...")

            # ── EVENT LOG ────────────────────────────────────────────
            log_table = Table(box=None, expand=True, padding=(0, 0), show_header=False)
            log_table.add_column("T", width=8, no_wrap=True, style="dim")
            log_table.add_column("L", width=8, no_wrap=True)
            log_table.add_column("Msg", overflow="fold")
            for log in (snap.logs or [])[-10:]:
                level = str(log.get("level", "INFO")).upper()
                style = {"WARNING": "yellow", "ERROR": "bold red", "DEBUG": "dim"}.get(level, "white")
                log_table.add_row(
                    str(log.get("time", "--:--"))[:8],
                    Text(level[:4], style=style),
                    Text(str(log.get("message", ""))[:150], style=style if level in ("WARNING","ERROR") else "dim white"),
                )

            # ── SYSTEM STATUS BAR ─────────────────────────────────────
            sys_status = Table.grid(expand=True)
            sys_status.add_column(ratio=1)
            sys_status.add_column(ratio=1)
            sys_status.add_column(ratio=1)
            sys_status.add_column(ratio=1)
            exch_style = "green" if (snap.exchange_status or "").upper() in ("CONNECTED", "OK") else "red"
            db_style = "green" if (snap.database_status or "").upper() in ("CONNECTED", "SQLITE_FALLBACK") else "red"
            sys_status.add_row(
                Text.assemble(("EX:", "dim"), (f" {snap.exchange_status or '—'}", exch_style)),
                Text.assemble(("DB:", "dim"), (f" {snap.database_status or '—'}", db_style)),
                Text.assemble(("CBT:", "dim"), (f" {snap.circuit_breaker_trips}", "yellow" if snap.circuit_breaker_trips > 0 else "green")),
                Text.assemble(("RECONN:", "dim"), (f" {snap.exchange_reconnections}", "yellow" if snap.exchange_reconnections > 0 else "green")),
            )

            footer = Text.assemble(
                ("◈ Reco Trading TUI  ", "dim"),
                ("│  ", "dim"),
                ("Ctrl+C", "bold white"),
                (" to stop  ", "dim"),
                ("│  ", "dim"),
                ("Web: ", "dim"),
                ("http://localhost:9000", "bright_cyan"),
            )

            # ── LAYOUT ASSEMBLY ───────────────────────────────────────
            layout = Layout(name="root")

            if width < 110:
                # Compact / mobile SSH mode
                layout.split_column(
                    Layout(Panel(header_grid, border_style="bright_blue", padding=(0, 1)), size=4),
                    Layout(name="body", ratio=1),
                    Layout(Panel(sys_status, border_style="grey27", padding=(0, 1)), size=3),
                    Layout(Panel(footer, border_style="grey27"), size=3),
                )
                layout["body"].split_column(
                    Layout(Panel(Group(market), title="Market", border_style="cyan"), ratio=3),
                    Layout(Panel(conf_row, border_style="blue"), size=3),
                    Layout(Panel(portfolio, title="Portfolio", border_style="green"), ratio=4),
                    Layout(Panel(pos_table, title="Position", border_style="bright_cyan"), ratio=3),
                    Layout(Panel(llm_table, title="LLM Gate", border_style="magenta"), ratio=3),
                    Layout(Panel(log_table, title="Feed", border_style="white"), ratio=4),
                )
            else:
                # Wide mode: 3-column layout
                layout.split_column(
                    Layout(Panel(header_grid, border_style="bright_blue", padding=(0, 1)), size=4),
                    Layout(name="body", ratio=1),
                    Layout(Panel(sys_status, border_style="grey27", padding=(0, 0)), size=3),
                    Layout(Panel(Align.center(footer), border_style="grey27"), size=3),
                )
                layout["body"].split_row(
                    Layout(name="left", ratio=3),
                    Layout(name="center", ratio=4),
                    Layout(name="right", ratio=3),
                )
                layout["body"]["left"].split_column(
                    Layout(Panel(market, title="📈 Market", border_style="cyan"), ratio=5),
                    Layout(Panel(sig_grid, title="📡 Signals", border_style="blue"), ratio=5),
                )
                layout["body"]["center"].split_column(
                    Layout(Panel(Group(conf_row, portfolio), title="💼 Portfolio", border_style="green"), ratio=5),
                    Layout(Panel(pos_table, title="🔵 Open Position", border_style="bright_cyan"), ratio=3),
                    Layout(Panel(log_table, title="📄 Feed", border_style="white"), ratio=4),
                )
                layout["body"]["right"].split_column(
                    Layout(Panel(llm_table, title="🤖 LLM Gate", border_style="magenta"), ratio=4),
                    Layout(Panel(filter_table, title="🔧 Active Filters", border_style="yellow"), ratio=4),
                    Layout(Panel(_build_decision_panel(snap), title="🔍 Decision", border_style="yellow"), ratio=4),
                )

            return Group(layout)

        except Exception as exc:
            return Panel(
                Text.assemble(
                    ("Dashboard render error:\n", "bold red"),
                    (str(exc), "white"),
                ),
                title="Reco Trading TUI",
                border_style="red",
            )


def _build_decision_panel(snap: DashboardSnapshot) -> Table:
    t = Table.grid(expand=True, padding=(0, 1))
    t.add_column(style="dim", min_width=14)
    t.add_column(style="bold white")
    dt = snap.decision_trace or {}
    factors = dt.get("factor_scores") or {}
    gating = snap.decision_gating or {}
    for k, v in list(factors.items())[:5]:
        vf = _to_float(v) or 0.0
        style = "green" if vf > 0 else "red" if vf < 0 else "dim"
        t.add_row(f"▸ {k}", Text(f"{vf:+.3f}", style=style))
    for k, v in list(gating.items())[:3]:
        t.add_row(f"• {k}", Text(str(v)[:20], style="cyan"))
    t.add_row("reason", Text((snap.decision_reason or "—")[:40], style="white"))
    return t


# ── HELPER FORMATTERS ─────────────────────────────────────────────────────────

def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None

def _to_text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value) if str(value).strip() else None

def _fmt(value: Any, dec: int = 2) -> str:
    f = _to_float(value)
    return f"{f:.{dec}f}" if f is not None else "—"

def _fmt_float(value: Any, dec: int = 2) -> str:
    f = _to_float(value)
    return f"{f:.{dec}f}" if f is not None else "—"

def _fmt_pct(value: Any) -> str:
    f = _to_float(value)
    return f"{f*100:.1f}%" if f is not None else "—"

def _styled_pnl(value: Any) -> Text:
    f = _to_float(value)
    if f is None:
        return Text("—", style="dim")
    sign = "+" if f > 0 else ""
    style = "bold green" if f > 0 else "bold red" if f < 0 else "dim"
    return Text(f"{sign}{f:.4f} USDT", style=style)

def _losses_styled(n: int) -> Text:
    if n == 0:
        return Text("0", style="green")
    if n <= 2:
        return Text(str(n), style="yellow")
    return Text(str(n), style="bold red")

def _adx_styled(adx: float | None) -> Text:
    f = _to_float(adx)
    if f is None:
        return Text("—", style="dim")
    style = "bold green" if f >= 25 else "yellow" if f >= 15 else "red"
    return Text(f"{f:.2f}", style=style)

def _rsi_styled(rsi: float | None) -> Text:
    f = _to_float(rsi)
    if f is None:
        return Text("—", style="dim")
    style = "red" if f > 70 else "green" if f < 30 else "cyan" if f > 55 else "white"
    return Text(f"{f:.1f}", style=style)

def _status_badge(state: str) -> Text:
    styles = {
        "RUNNING": ("RUNNING", "bold green"),
        "PAUSED": ("PAUSED", "bold yellow"),
        "INITIALIZING": ("INIT", "bold blue"),
        "WAITING_MARKET_DATA": ("WAITING", "blue"),
        "CONNECTING_EXCHANGE": ("CONNECTING", "blue"),
        "POSITION_OPEN": ("IN TRADE", "bold bright_green"),
        "COOLDOWN": ("COOLDOWN", "yellow"),
        "ERROR": ("ERROR", "bold red"),
    }
    label, style = styles.get(state.upper(), (state[:10], "white"))
    return Text(f"[{label}]", style=style)

def _signal_badge(signal: str | None) -> Text:
    s = (signal or "HOLD").upper()
    styles = {"BUY": "bold green", "SELL": "bold red", "HOLD": "dim", "NEUTRAL": "dim"}
    return Text(s, style=styles.get(s, "white"))

def _signal_cell(val: str) -> Text:
    v = (val or "NEUTRAL").upper()
    styles = {"BUY": "bold green", "SELL": "bold red", "NEUTRAL": "dim", "HOLD": "dim"}
    return Text(v, style=styles.get(v, "white"))

def _trend_badge(trend: str | None) -> Text:
    t = (trend or "NEUTRAL").upper()
    styles = {"BULLISH": "bold green", "BEARISH": "bold red", "BUY": "green", "SELL": "red", "NEUTRAL": "dim"}
    return Text(t, style=styles.get(t, "white"))

def _regime_badge(regime: str | None) -> Text:
    r = (regime or "NORMAL").upper()
    styles = {
        "LOW_VOLATILITY": "dim cyan",
        "NORMAL_VOLATILITY": "green",
        "HIGH_VOLATILITY": "bold yellow",
    }
    return Text(r.replace("_", " "), style=styles.get(r, "white"))

def _flow_badge(flow: str | None) -> Text:
    f = (flow or "NEUTRAL").upper()
    styles = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "dim"}
    return Text(f, style=styles.get(f, "white"))
```

---

## INSTRUCCIONES FINALES PARA CODEX

1. **Orden de ejecución obligatorio:** A → B → C → D. No saltear pasos.

2. **En cada archivo modificado:** deja el archivo completo, no uses `# ... resto igual`. Si un archivo tiene partes que no se mencionan en este prompt, mantenlas intactas.

3. **Para el archivo `index.html`:** reemplaza el contenido completo, no lo mezcles con el HTML anterior.

4. **Crear el endpoint `/api/settings` en `dashboard_server.py`** si no existe, que acepte POST con JSON y llame a `_global_bot_instance._apply_runtime_settings(payload, persist=True)` de forma thread-safe.

5. **Crear el endpoint `/api/snapshot`** que retorne `{"success": True, "data": get_bot_snapshot()}`.

6. **No eliminar ningún endpoint existente.** Solo agregar o modificar.

7. **Tests:** Después de aplicar todos los cambios, ejecuta `pytest tests/ -x -q` y corrige cualquier fallo que sea consecuencia directa de tus cambios.

8. **Variables de entorno necesarias para el dashboard:**
   - `DASHBOARD_AUTH_ENABLED=true`
   - `DASHBOARD_USERNAME=admin`
   - `DASHBOARD_PASSWORD=<tu_password>`
   - `DASHBOARD_API_TOKEN=<tu_token>`
   - `DASHBOARD_AUTH_MODE=hybrid`

9. **Checklist de verificación antes de terminar:**
   - [ ] `llm_mode="base"` ya no confirma todos los trades automáticamente
   - [ ] `RegimeFilter.evaluate()` retorna `allow_trade=False` para LOW_VOLATILITY
   - [ ] Import duplicado de `TradingModeManager` eliminado
   - [ ] Warmup de Ollama corre en background thread
   - [ ] `refresh_per_second` en `Live()` usa el setting configurado
   - [ ] `screen=True` solo cuando `sys.stdout.isatty()`
   - [ ] Dashboard web carga y muestra datos sin errores de consola
   - [ ] Settings tab envía POST a `/api/settings` correctamente
   - [ ] Multi-select de símbolos funciona
   - [ ] Acordeones abren y cierran correctamente
   - [ ] SSE funciona, fallback a polling si no
   - [ ] TUI muestra ADX estilizado, RSI estilizado, posición y filtros activos
   - [ ] `_avg_time_ms` se actualiza una sola vez por llamada

---
*Fin del prompt. Versión para Codex/GPT-4.1 — reco-trading Full Overhaul 2026*
