# Reco Trading Bot v4.0

Plataforma avanzada de trading algorítmico con ejecución multi-módulo, analítica en tiempo real, panel web protegido y dashboard de terminal tipo **TUI** profesional.

---

## Tabla de contenido

1. [Resumen](#resumen)
2. [Novedades recientes](#novedades-recientes)
3. [Arquitectura](#arquitectura)
4. [Instalación](#instalación)
5. [Configuración `.env`](#configuración-env)
6. [Dashboard web seguro (token/basic)](#dashboard-web-seguro-tokenbasic)
7. [Acceso remoto desde celular u otro dispositivo](#acceso-remoto-desde-celular-u-otro-dispositivo)
8. [Docker y hardening de puertos](#docker-y-hardening-de-puertos)
9. [Dashboards disponibles](#dashboards-disponibles)
10. [Troubleshooting](#troubleshooting)
11. [Desarrollo y pruebas](#desarrollo-y-pruebas)

---

## Resumen

**Reco Trading Bot** integra:

- Motor de trading con control de riesgo y módulos de inteligencia de mercado.
- Integración LLM en 3 modos: `base`, `llm_local`, `llm_remote`.
- Dashboard web (Flask) con fallback a DB cuando no hay bot en memoria.
- API complementaria (FastAPI) para endpoints runtime.
- Dashboard de terminal **TUI** en Rich para operación headless profesional.

---

## Novedades recientes

### 1) Modos LLM integrados
- `base`: confirmación final por reglas.
- `llm_local`: confirmación local con Ollama.
- `llm_remote`: confirmación por proveedor remoto (API compatible OpenAI-style).

### 2) Seguridad de dashboard web por `.env`
- Autenticación configurable por variables de entorno:
  - `DASHBOARD_AUTH_ENABLED`
  - `DASHBOARD_AUTH_MODE` (`token`, `basic`, `hybrid`)
  - `DASHBOARD_API_TOKEN`
  - `DASHBOARD_USERNAME`
  - `DASHBOARD_PASSWORD`

### 3) Instaladores autónomos (Linux / Windows)
- Los instaladores ahora agregan y actualizan automáticamente variables de dashboard y LLM en `.env`.
- Generación automática de token de dashboard.

### 4) Docker endurecido
- Puertos ligados a `127.0.0.1` para reducir superficie de exposición.
- Flujo listo para túnel Cloudflared sobre dashboard web.

### 5) Optimización de latencia y datos en tiempo real
- Integración de caché de ticker en tiempo real vía WebSocket de Binance (`bookTicker`) con fallback REST automático.
- Menor overhead HTTP para precio bid/ask y mejor respuesta de dashboard.

### 6) Control operativo unificado `Stop Trade`
- Botón **Stop Trade** visible cuando existe una posición abierta en App y Web dashboard.
- Endpoint web `/api/control/stop_trade` conectado a cierre forzado de posición.

### 7) Perfil automático para `llm_local` (Ollama)
- Ajuste automático de perfil de baja RAM y filtros mínimos al detectar `LLM_MODE=llm_local`.
- Confirmación LLM local optimizada para respuestas más rápidas con timeouts y opciones de inferencia compacta.

### 8) Auto Stop profesional (Break-even + Trailing + Time Stop)
- Break-even automático configurable por porcentaje de beneficio.
- Trailing Stop adaptativo por régimen de volatilidad (delta distinto en baja/alta volatilidad).
- Cierre automático por tiempo máximo de operación para evitar exposición extendida.

---

## Arquitectura

```text
Reco Trading
├── Motor principal (state machine + loop manager + risk)
├── Capa AI/LLM
│   ├── Confirmador de trade LLM
│   └── Análisis y auto-fix
├── Persistencia
│   ├── SQLite / PostgreSQL / MySQL
│   └── Repository asíncrono
├── Dashboards
│   ├── Desktop App (PySide6)
│   ├── Web Dashboard (Flask)
│   └── Terminal TUI (Rich)
└── API runtime (FastAPI)
```

---

## Instalación

## Linux

```bash
chmod +x install-linux.sh
./install-linux.sh
```

## Windows CMD

```bat
install-windows.bat
```

## Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\install-windows-powershell.ps1
```

## Docker (builder integral)

```bash
chmod +x docker-build.sh
./docker-build.sh
```

---

## Configuración `.env`

Ejemplo mínimo recomendado:

```env
# Exchange
BINANCE_API_KEY=TU_API_KEY
BINANCE_API_SECRET=TU_API_SECRET
BINANCE_TESTNET=true
CONFIRM_MAINNET=false
ENVIRONMENT=testnet
RUNTIME_PROFILE=paper

# Database (elige una)
DATABASE_URL=sqlite+aiosqlite:///./data/reco_trading.db
# POSTGRES_DSN=postgresql+asyncpg://user:pass@localhost:5432/reco_trading_prod
# MYSQL_DSN=mysql+aiomysql://user:pass@localhost:3306/reco_trading_prod

# LLM
LLM_MODE=base
LLM_LOCAL_MODEL=qwen2.5:0.5b
OLLAMA_BASE_URL=http://localhost:11434
LLM_REMOTE_ENDPOINT=https://api.openai.com/v1/chat/completions
LLM_REMOTE_MODEL=gpt-4o-mini
LLM_REMOTE_API_KEY=

# Dashboard Security
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_AUTH_MODE=token
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=admin
DASHBOARD_API_TOKEN=TU_TOKEN_GENERADO
```

> Los instaladores generan/agregan automáticamente estas claves para reducir configuración manual.

---

## Dashboard web seguro (token/basic)

Con `DASHBOARD_AUTH_ENABLED=true`:

- `DASHBOARD_AUTH_MODE=token` → usa token.
- `DASHBOARD_AUTH_MODE=basic` → usuario/clave.
- `DASHBOARD_AUTH_MODE=hybrid` → acepta ambos.

El frontend web usa `authFetch` y envía el token en `X-Dashboard-Token` cuando está disponible.

---

## Acceso remoto desde celular u otro dispositivo

### Opción recomendada: Cloudflared Tunnel

El script `docker-build.sh` puede levantar túnel hacia:

```bash
cloudflared tunnel --url http://localhost:9000
```

### Cómo entrar desde otro dispositivo

1. Abre la URL pública del túnel.
2. Agrega el token en la URL la primera vez:
   - `https://tu-url.trycloudflare.com/?token=TU_TOKEN`
3. El frontend guarda el token localmente para próximas peticiones.

> También puedes usar `Authorization: Bearer <token>` o `X-Dashboard-Token` en clientes personalizados.

---

## Docker y hardening de puertos

- Dashboard web ligado a loopback (`127.0.0.1:9000`).
- API auxiliar, DB, Redis y Ollama también en loopback cuando aplica.
- Menor exposición de red por defecto.

Si necesitas exposición controlada, usa reverse proxy/TLS o túnel autenticado.

---

## Dashboards disponibles

## 1) Web Dashboard
- Control operativo (`pause`, `resume`, `emergency`).
- Trades, analytics, risk y estado de módulos.
- Fallback a DB si el bot no está en memoria.

## 2) Desktop Dashboard (PySide6)
- UI gráfica local con tabs especializadas.

## 3) Terminal TUI (Rich) — **mejorado**
- Layout multi-panel profesional.
- Badges visuales de estado/señal.
- Barras de confianza y visuales de salud del sistema.
- Diseñado para operación headless en VPS/servidor.

### No conecta Ollama en modo local
- Revisa `OLLAMA_BASE_URL`.
- Verifica que el modelo exista (`ollama list`).

## Troubleshooting

### El dashboard devuelve `Unauthorized`
- Verifica:
  - `DASHBOARD_AUTH_ENABLED=true`
  - token correcto en `DASHBOARD_API_TOKEN`
  - modo correcto en `DASHBOARD_AUTH_MODE`

### No conecta Ollama en modo local
- Revisa `OLLAMA_BASE_URL`.
- Verifica que el modelo exista (`ollama list`).

### El bot no inicia en Docker
- Revisa logs:

```bash
docker logs reco-trading --tail 200
```

```bash
pytest -q tests/test_web_dashboard_connection_and_layout.py tests/test_web_dashboard_db_and_balance.py
```

## Desarrollo y pruebas

### Pruebas recomendadas

```bash
pytest -q tests/test_web_dashboard_connection_and_layout.py tests/test_web_dashboard_db_and_balance.py
```

### Validaciones de sintaxis útiles

```bash
python -m py_compile web_site/dashboard_server.py
bash -n install-linux.sh
bash -n docker-build.sh
```

- credenciales reales fuera de repositorio,
- revisión de riesgos antes de habilitar mainnet,
- y validación en testnet previo a producción.

## Nota final

Este proyecto está orientado a operación profesional y evolución continua. Mantén siempre:

- credenciales reales fuera de repositorio,
- revisión de riesgos antes de habilitar mainnet,
- y validación en testnet previo a producción.
