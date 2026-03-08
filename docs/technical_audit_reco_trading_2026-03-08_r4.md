# Auditoría técnica integral `reco_trading` (R4)

Fecha: 2026-03-08  
Rol: Auditor principal de arquitectura, ejecución y lógica cuantitativa.

## 1) Evaluación ejecutiva

**Score técnico (0–100): 82/100**  
**Nivel real:** avanzado pre-producción con controles institucionales parciales.  
**Veredicto:** sólido para testnet y paper/live controlado; aún no listo para despliegue institucional full capital sin una capa contable única y cierre transaccional completo de ciclo operativo.

---

## 2) PARTE 1 — Auditoría de arquitectura

### Flujo de información (market -> ejecución)
1. `main.py` valida entorno crítico y arranca `QuantKernel.run()`.
2. `QuantKernel` integra market data, feature/signal pipeline, régimen, edge y filtros.
3. Pasa a `ExecutionEngine` para lock global, firewall, idempotencia y persistencia.
4. `IdempotentOrderService` gestiona estado de orden durable y recuperación.
5. `BinanceClient` ejecuta llamadas gobernadas (rate governor + time sync + retry).

### Módulos clave y dependencias
- `QuantKernel` (orquestación central)
- `ExecutionEngine` (pipeline de ejecución y seguridad)
- `CapitalProtectionController` (bloqueos por estado activo)
- `IdempotentOrderService` (idempotencia y reconciliación)
- `BinanceClient` (conectividad exchange)

### Acoplamientos peligrosos detectados
- Riesgo moderado: `QuantKernel` mantiene demasiado alcance funcional (estrategia + riesgo + estado + UI + snapshots), lo que incrementa superficie de fallo por cambios cruzados.
- Riesgo bajo: aunque bajó el acoplamiento interno, `ExecutionEngine` aún mezcla validación, routing, reconciliación y persistencia forense en un único servicio amplio.

---

## 3) PARTE 2 — Auditoría de modelos cuantitativos

### Modelos activos
- `MomentumModel.predict_from_snapshot` (zscore + señal corta).
- `MeanReversionModel.predict_from_snapshot` (direccional).
- `SignalCombiner` para mezcla momentum/reversión/régimen.

### Cálculo de edge/confianza/expectancy
- Edge derivado de probabilidad combinada y ajustes por estabilidad/noise/correlación/volatilidad.
- Confianza filtrada por umbral dinámico + fricción + edge mínimo.
- Expectancy rolling calculada en `_rolling_stats`.

### Hallazgos
- Se corrigieron defectos previos de señal (código muerto momentum y MR no direccional).
- Mejora aplicada en probabilidad de régimen (ya no constante fija pura).
- Persiste posible **sobre-filtrado** por stack de gates (confidence + friction + edge + MTF + risk blocks), que puede degradar captación de edge real.

---

## 4) PARTE 3 — Auditoría de riesgo

### Revisado
- `risk_per_trade`, risk-of-ruin, drawdown global, pérdida diaria, kill-switch y `DISABLE_TRADING`.

### Estado actual
- Kill-switch operativo y `DISABLE_TRADING` funcional como hard-stop.
- Límites firewall se escalan con equity actual.

### Riesgo residual
- Múltiples capas de riesgo (kernel/firewall/institutional/governor) pueden producir bloqueos redundantes y explicabilidad incompleta del motivo final de veto.

---

## 5) PARTE 4 — Auditoría de capital y PnL

### Cálculos
- Equity usa NAV spot MTM (USDT + base * mark price).
- PnL total dashboard = `realized + unrealized`.
- PnL diario = `equity_actual - anchor_diario`.
- `initial_equity` se fija tras startup reconciliation.

### Caso sospechoso (Capital ~470 / PnL ~+455)
Causa técnica probable:
- diferencia entre percepción de sesión y estado reconciliado (histórico + MTM + reanclajes de posición), especialmente si no hubo trades en la sesión actual pero sí legado histórico o variación de mark price.

### Mejora observada
- snapshots financieros + ledgers (`positions_ledger`, `pnl_ledger`) mejoran trazabilidad temporal y diagnóstico.

---

## 6) PARTE 5 — Auditoría de base de datos

### Tablas relevantes
- `trades`, `orders`, `fills`, `order_executions`, `execution_idempotency_ledger`, `capital_reservations`, `financial_snapshots`, `positions_ledger`, `pnl_ledger`.

### Hallazgos
- Cobertura forense fuerte de orden/fill/ejecución.
- Mejora clara con ledgers explícitos de posición y PnL.

### Límite de entorno
- no se pudo auditar contenido live (residuos históricos/inconsistencias reales) por falta de `POSTGRES_DSN` en este entorno.

---

## 7) PARTE 6 — Auditoría de ejecución de órdenes

Pipeline revisado: `decision -> execution_engine -> idempotent_order_service -> binance_client`.

### Verificación
- Lock global DB presente.
- Idempotencia con estado `SUBMISSION_UNCERTAIN` y reintento prudente.
- Mejoras de finalización más atómica con `finalize_execution_atomically`.

### Riesgo residual
- aunque la finalización mejoró, sigue habiendo rutas fallback y condiciones parciales; para nivel institucional pleno conviene consolidar todas las rutas de cierre en un único flujo obligatorio sin bifurcaciones legacy.

---

## 8) PARTE 7 — Auditoría de sincronización con Binance

Validado:
- `fetch_balance` ✅
- `fetch_open_orders` ✅
- `fetch_my_trades` ✅
- `fetch_positions` ✅

Mejora aplicada:
- lookup detallado por `clientOrderId` con tipado de error (`network`, `rate_limit`, `not_found`, etc.).

Riesgo residual:
- aún existe compatibilidad con métodos legacy que retornan `None`; idealmente unificar toda la capa de reconciliación al contrato tipado.

---

## 9) PARTE 8 — Auditoría Binance Testnet/Mainnet

- Endpoints y separación testnet/mainnet correctos.
- Guardrails de mainnet correctos (confirmación explícita).
- Operación apta para testnet y despliegue controlado en real con barreras activas.

---

## 10) PARTE 9 — Auditoría del dashboard

- El dashboard refleja estado operativo amplio (capital, PnL, riesgo, edge, régimen, motivo de bloqueo).
- Con NAV MTM + snapshots persistidos, la coherencia capital/PnL mejoró significativamente.

Riesgo residual:
- persiste diferencia entre “PnL intuitivo” del operador y “PnL contable institucional” sin una capa de reporting normalizada por evento/ciclo.

---

## 11) PARTE 10 — Errores potenciales detectados (vigentes)

1. Complejidad excesiva de `QuantKernel` (demasiadas responsabilidades).
2. Multiplicidad de capas de riesgo sin verdict final único estandarizado.
3. Rutas fallback legacy en ejecución/sync que reducen determinismo institucional.
4. Falta de un módulo dedicado de reporting contable institucional (señales + ejecución + PnL) con contrato formal de auditoría.

---

## 12) PARTE 11 — Mejoras necesarias

1. **Refactor arquitectónico por bounded contexts**:
   - `DecisionOrchestrator`, `RiskVerdictEngine`, `ExecutionLifecycleService`, `AccountingService`.
2. **Risk verdict unificado**:
   - una sola salida con jerarquía de bloqueos, severidad y acción.
3. **Eliminar rutas fallback en cierre de ejecución**:
   - imponer `finalize_execution_atomically` como única ruta para producción.
4. **Reporting institucional**:
   - vista consolidada `positions_ledger + pnl_ledger + executions + decision_audit` con reconciliación diaria automática.

---

## 13) PARTE 12 — Reporte final

1. **Evaluación técnica:** 82/100.
2. **Errores encontrados:** listados en sección 11.
3. **Riesgos potenciales:** complejidad centralizada, riesgo distribuido sin verdict único, rutas legacy.
4. **Nivel real:** avanzado pre-producción.
5. **Falta para trading real institucional:** refactor modular, verdict de riesgo unificado, ejecución totalmente atómica y reporting contable formal.

