# Auditoría técnica inicial (trading algorítmico) - reco-trading

## Alcance real de esta auditoría
Esta auditoría cubre revisión estática del código fuente, revisión de arquitectura/riesgo operativo, ejecución de test suite, chequeos de lint/compilación y análisis de preparación para operar capital real significativo.

> Nota profesional: una auditoría "línea por línea" estricta de todo el repositorio en un único ciclo no sustituye una auditoría formal de producción (incluye pruebas de estrés exchange real, failover, latencia de red en vivo, chaos testing, validación legal/compliance, y paper-trading prolongado).

---

## Comandos ejecutados
- `pytest -q`
- `pytest -q --durations=10`
- `python -m compileall -q reco_trading main.py`
- `ruff check reco_trading tests main.py`
- inspección manual de módulos críticos (`core/bot_engine.py`, `exchange/order_manager.py`, `exchange/binance_client.py`, `risk/*.py`, `strategy/*.py`, `database/repository.py`, `config/settings.py`).

---

## Hallazgos (enumerados uno por uno)

### Críticos
1. **El bot puede intentar abrir operaciones `SELL` en spot sin validar inventario real del activo base.**
   - En spot estándar no existe short directo; un `SELL` sin BTC disponible rechaza orden o genera comportamiento inconsistente.
   - Evidencia: `execute_trade` envía `side.lower()` directo a mercado, sin gate específico para spot-only buy/open logic.

2. **No hay bloqueo explícito para evitar operar con señales `HOLD`.**
   - `ConfidenceModel` puede devolver `HOLD`, pero en el flujo de ejecución no hay early return para `HOLD`.
   - Resultado: puede entrar en `_pullback_confirmed` por rama no-BUY y terminar intentando vender.

3. **`confirm_mainnet` existe en settings pero no se usa para bloquear arranque en mainnet.**
   - Riesgo operativo grave: un operador puede correr en mainnet por configuración accidental sin “doble confirmación”.

4. **No hay cierre garantizado del cliente de exchange en finalización/errores fatales.**
   - Existe `BinanceClient.close()`, pero el loop principal no asegura `finally` con cleanup.
   - Puede dejar recursos/sesiones colgadas y degradar estabilidad de largo plazo.

### Altos
5. **Dependencia de polling secuencial para datos críticos (OHLCV 2 TF + ticker + orderbook + balance).**
   - Riesgo de latencia acumulada y desfase entre snapshots en mercados rápidos.

6. **No hay idempotencia de órdenes por `clientOrderId` ni reconciliación post-fallo antes de reintentar lógica.**
   - En interrupciones de red entre envío y respuesta, se puede perder trazabilidad exacta del estado real de ejecución.

7. **No se observan frenos de “kill switch” por slippage real ejecutado vs esperado.**
   - Hay control por spread previo, pero no cutoff explícito post-fill ante desviación excesiva.

8. **`safe_db_call` absorbe excepciones y devuelve defaults silenciosos.**
   - Aunque evita caída total, puede enmascarar fallos de persistencia y afectar control de riesgo si datos quedan inconsistentes.

9. **Riesgo de precisión por uso de `math.ceil/floor` sobre floats para step size.**
   - En símbolos con step/tick complejos, el redondeo flotante puede producir rechazo en exchange.

10. **No se observa mecanismo de reconciliación de posiciones al reinicio.**
    - Si el proceso cae con posición abierta, al reiniciar el estado en memoria parte vacío y puede desalinearse con cuenta real.

### Medios
11. **Campos de entorno/operación (`environment`, `runtime_profile`) definidos pero no usados para políticas activas.**
    - Señal de controles incompletos de despliegue por perfil.

12. **Métricas de salud incompletas en UI (valores por defecto “UNKNOWN”/`0.0`).**
    - No afecta ejecución directa, pero limita observabilidad real para capital alto.

13. **No hay validación dura de presencia/formato de API keys previo al arranque de exchange.**
    - Puede fallar más tarde dentro del loop en lugar de fallar rápido con diagnóstico claro.

14. **Política de retries simple (backoff exponencial básico) sin circuit breaker granular por endpoint.**
    - Para producción, conviene distinguir endpoints de lectura/escritura y severidad.

15. **Linting con errores activos (imports no usados, referencia de tipo no resuelta, E402 en tests).**
    - No rompe runtime hoy, pero es deuda técnica que suele correlacionar con regresiones futuras.

---

## Sobre “lag/demora de acción”
- No se detectó lag severo en test unitarios, pero **NO** es evidencia de latencia real de mercado.
- Duraciones locales: tests completos ~1.5–2.1s, sin cuellos de botella de CPU visibles en entorno de CI local.
- El principal vector de demora en vivo será red/exchange + secuencia de llamadas por ciclo.

---

## Comparación con estándares de bots listos para capital alto
Respecto a stacks profesionales (HFT no aplica aquí, pero sí low-frequency institucional/prop), faltan piezas indispensables:

1. Reconciliación robusta de estado (órdenes/posiciones/balance) al arranque y periódicamente.
2. Idempotencia y “exactly-once intent” para órdenes (client IDs, deduplicación, journal de intents).
3. Simulación de slippage/comisiones realista + límites dinámicos por liquidez libro.
4. Kill switch multicapa (latencia, error-rate, drawdown intradía, desconexión de DB/telemetría).
5. Pruebas de resiliencia (chaos tests de exchange caído, DB caída, reloj desfasado, respuestas parciales).
6. Monitoreo operacional completo (Prometheus/Grafana/alerting) y auditoría externa de logs inmutables.
7. Procedimiento de despliegue seguro (feature flags, canary, paper->small capital->scale).

---

## Veredicto de preparación para operar “miles de dólares”
**Estado actual: NO listo para operar capital real significativo sin remediación previa.**

Razones clave: riesgos de ejecución spot con señales SELL/HOLD, ausencia de reconcilación robusta post-reinicio, controles de producción incompletos (mainnet gating efectivo, idempotencia órdenes, kill switches avanzados).

---

## Prioridad de corrección recomendada (antes de tocar estrategia)
1. Bloquear `HOLD` y definir política spot-only (BUY para apertura, SELL solo para cierre de posición existente).
2. Implementar “mainnet safety gate” usando `confirm_mainnet` + validaciones de entorno.
3. Reconciliación de estado en startup (posiciones abiertas, órdenes vivas, balance).
4. Idempotencia de órdenes + journal de intents.
5. Endurecer persistencia/errores DB para que fallos críticos disparen modo seguro.
6. Mejorar normalización de cantidades/precio con `Decimal` end-to-end.
7. Añadir pruebas de integración con mocks de exchange para escenarios de red y reintentos.

