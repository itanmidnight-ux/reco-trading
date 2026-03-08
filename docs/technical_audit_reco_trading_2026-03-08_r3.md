# Auditoría técnica integral `reco_trading` (R3, institucional pre-producción)

Fecha: 2026-03-08  
Rol: Auditor principal de arquitectura, ejecución y lógica cuantitativa.

## Evaluación ejecutiva

**Score técnico actual: 80 / 100**  
**Nivel real:** avanzado pre-producción (todavía no institucional pleno para capital real).

El sistema muestra una mejora clara frente a iteraciones previas (NAV MTM, idempotencia robustecida, `DISABLE_TRADING`, snapshots financieros), pero aún persisten riesgos en trazabilidad contable única, atomización de cierre de ejecución y complejidad de capas de riesgo.

---

## PARTE 1 — Auditoría de arquitectura

### Flujo de información (market -> execution)
1. `main.py` valida entorno y arranca `QuantKernel.run()`.
2. `QuantKernel` orquesta market-data, features, modelos, detección de régimen, cálculo de edge, filtros de riesgo y decisión.
3. `ExecutionEngine` recibe intención (`BUY/SELL`), ejecuta lock global DB, firewall, sanitización, idempotencia y persistencia.
4. `IdempotentOrderService` maneja journal durable + recovery.
5. `BinanceClient` gestiona llamadas gobernadas (rate/time sync/retry) a Binance.

### Validación de dependencias
- Dependencias principales son correctas para flujo productivo.
- `CapitalProtectionController` está integrado al kernel para bloquear condiciones de trade activo.

### Acoplamientos y riesgos de arquitectura
- **Riesgo moderado:** `QuantKernel` sigue concentrando demasiadas responsabilidades (estrategia + riesgo + estado + UI + persistencia cíclica), elevando riesgo de regresión cruzada.
- **Riesgo bajo residual:** aunque se redujo acoplamiento, `ExecutionEngine` continúa con lógica extensa de ejecución y post-procesado que convendría segmentar en `execution_lifecycle_service`.

---

## PARTE 2 — Auditoría de modelos cuantitativos

### Modelos activos
- `MomentumModel.predict_from_snapshot` (mezcla z-momentum + ventana corta).
- `MeanReversionModel.predict_from_snapshot` (direccional).
- `SignalCombiner` para ensamblar momentum/reversión/régimen.

### Cálculo de edge/confianza/expectancy
- Edge operativo deriva de probabilidad fusionada + ajustes por régimen/noise/correlation/volatilidad.
- Confianza se filtra por thresholds dinámicos + fricción + edge mínimo.
- Expectancy rolling mantenida en `_rolling_stats`.

### Hallazgos
1. **Corregido:** momentum sin código muerto.
2. **Corregido:** mean reversion direccional.
3. **Riesgo vigente:** potencial de **sobre-filtrado** por apilamiento de gates (confidence, friction, edge floor, MTF, risk blocks), puede reducir excesivamente frecuencia y capacidad de captura de edge.
4. **Mejora aplicada:** probabilidad de régimen ya no es hardcode puro; ahora incluye confianza del detector y estabilidad del régimen.

---

## PARTE 3 — Auditoría de riesgo

### Componentes revisados
- `risk_per_trade`, kill-switch, drawdown/pérdida diaria, `RiskOfRuinEstimator`, `DISABLE_TRADING`, `ExecutionFirewall`, `InstitutionalRiskManager`, `CapitalGovernor`.

### Estado
- Kill switch y `DISABLE_TRADING` funcionan como barrera global.
- Límites de firewall se recalibran dinámicamente con equity.

### Riesgo residual
- **Semántica distribuida de riesgo:** varias capas con decisiones parcialmente superpuestas pueden causar bloqueos redundantes/no explicables (falso positivo operativo).

---

## PARTE 4 — Auditoría de capital y PnL

### Estado de cálculo
- `exchange_equity` usa NAV spot mark-to-market (USDT total + base_qty*last).
- `pnl` dashboard = `realized + unrealized`.
- `daily_pnl` = `equity_actual - anchor_diario`.
- `initial_equity` se define tras reconciliación startup.

### Caso sospechoso (Capital ~470 / PnL ~+455)
Causa más probable, según diseño actual:
- PnL visible mezcla estado de sesión con reconciliación histórica/posición reanclada; puede producir discrepancias perceptuales si no hubo trades nuevos en la sesión actual pero sí contexto histórico o cambio MTM relevante.
- Sin un ledger contable único de sesión+histórico versionado, la interpretación humana puede ser engañosa.

### Mejora aplicada
- Persistencia periódica de `financial_snapshots` para trazabilidad temporal del estado financiero.

---

## PARTE 5 — Auditoría de base de datos

### Tablas encontradas (relevantes)
- `trades`, `orders`, `fills`, `order_executions`, `execution_idempotency_ledger`, `capital_reservations`, `financial_snapshots`.

### Hallazgos
- Hay buena cobertura forense de ejecución.
- No existe aún una tabla canónica explícita de `positions`/`pnl` con reconciliación determinística por ciclo (se reconstruye desde fills + estado runtime).

### Limitación de entorno
- No se pudo verificar residuos históricos reales (DB live) por ausencia de `POSTGRES_DSN` en el entorno.

---

## PARTE 6 — Auditoría de ejecución de órdenes

Pipeline validado: `decision -> execution_engine -> idempotent_order_service -> binance_client`.

### Verificación
- Lock global (`execution_advisory_lock`) presente.
- Idempotencia con `SUBMISSION_UNCERTAIN` + sondeo previo a resubmit.
- Sanitización de cantidad y controles de firewall activos.

### Riesgo pendiente
- Cierre de ciclo order/fill/ledger/reservation todavía no está encapsulado en una transacción única de negocio idempotente end-to-end.

---

## PARTE 7 — Auditoría de sincronización con Binance

Interfaces verificadas:
- `fetch_balance` ✅
- `fetch_open_orders` ✅
- `fetch_my_trades` ✅
- `fetch_positions` ✅

Mejora aplicada:
- fallos en `fetch_order_by_client_order_id` ahora quedan logueados para diagnóstico de sync.

Riesgo residual:
- la ruta devuelve `None` ante excepción; aunque ahora hay logging, la capa superior no distingue tipología de error para decisiones finas de reconciliación.

---

## PARTE 8 — Auditoría Binance Testnet/Mainnet

- Separación de endpoints correcta.
- Guardrail de mainnet con confirmación explícita.
- Configuración de arranque coherente para entorno.

Conclusión: apto para testnet y operación controlada en mainnet con safeguards.

---

## PARTE 9 — Auditoría del dashboard

- Dashboard refleja capital, pnl, edge, riesgo y estado.
- Con NAV MTM y snapshots financieros, la consistencia mejoró.

Riesgo residual:
- Aún puede haber diferencia entre “PnL intuitivo” y “PnL contable institucional” por falta de libro contable único de ciclo.

---

## PARTE 10 — Detección de errores/puntos potenciales

1. Complejidad excesiva del kernel (bus-factor y regresión cruzada).
2. Falta de “single accounting ledger” institucional para positions/PnL.
3. Cierre de ejecución no totalmente atomizado.
4. Multiplicidad de capas de riesgo con explicabilidad incompleta.
5. Clasificación de errores Binance todavía poco tipada aguas arriba.

---

## PARTE 11 — Mejoras necesarias (exactas)

1. **Ledger contable institucional**
   - Implementar `positions_ledger` + `pnl_ledger` con snapshots por ciclo y reconciliación exchange-first.
2. **Execution lifecycle atómico**
   - Crear procedimiento único idempotente para cerrar orden/fill/ledger/reservation en una transacción lógica.
3. **Risk verdict unificado**
   - Consolidar `firewall + institutional + governor + kernel` en un dictamen único con razones jerárquicas.
4. **Error typing Binance**
   - Retornar códigos/enum de error (network, timeout, not_found, auth, rate-limit) en vez de `None` genérico.
5. **Refactor kernel modular**
   - Separar `DecisionOrchestrator`, `RiskOrchestrator` y `StateAccountingService`.

---

## PARTE 12 — Reporte final

1. **Evaluación técnica:** 80/100.
2. **Errores encontrados:** se listan en partes 10 y 11.
3. **Riesgos potenciales:** contabilidad no plenamente institucional, cierre no atómico, explicabilidad de bloqueo de riesgo.
4. **Nivel real:** avanzado pre-producción.
5. **Falta para trading real:** ledger contable único, cierre transaccional de ejecución, unificación de riesgo y tipado de errores de sincronización.

