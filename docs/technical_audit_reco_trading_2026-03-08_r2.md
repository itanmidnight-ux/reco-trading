# Auditoría técnica integral `reco_trading` (R2, post-remediación)

Fecha: 2026-03-08  
Rol: Auditor principal de arquitectura, ejecución y lógica cuantitativa.

## Resumen ejecutivo

**Evaluación global:** **76 / 100** (sube desde auditoría anterior por remediaciones implementadas).  
**Estado:** sistema **avanzado pre-producción**, aún **no institucional listo para capital real**.

Mejoras verificadas respecto a la auditoría previa:
- NAV spot ya incorpora mark-to-market de inventario base en el cálculo de equity.
- Existe `DISABLE_TRADING` operativo y conectado al kill switch.
- Idempotencia reforzada con estado `SUBMISSION_UNCERTAIN`.
- Se añadió `fetch_positions` para reconciliación explícita de inventario spot.
- Se corrigieron defectos de modelos (código muerto momentum y MR no direccional).

Persisten riesgos relevantes que impiden salida a producción real (detalle abajo).

---

## PARTE 1 — Auditoría de arquitectura

### Flujo end-to-end
`main.py -> QuantKernel.run() -> market_data/feature_engine/modelos/regime/signal -> risk gates -> ExecutionEngine -> IdempotentOrderService -> BinanceClient -> DB`.

### Validación de dependencias
- `QuantKernel` orquesta correctamente el pipeline y estados operativos.
- `ExecutionEngine` encapsula lock global, firewall, sanitización y persistencia.
- `IdempotentOrderService` mantiene ledger durable + reconciliación.
- `BinanceClient` centraliza time sync/rate governor/retry.

### Acoplamientos peligrosos (restantes)
1. `ExecutionEngine.get_exchange_min_size()` sigue delegando al método privado `_firewall._min_size`, lo que mantiene acoplamiento a internals del firewall.
2. La lógica cuantitativa en `QuantKernel` concentra demasiada responsabilidad (decisión + riesgo + estado + dashboard), elevando riesgo de regresiones cruzadas.

---

## PARTE 2 — Auditoría de modelos cuantitativos

### Estado actual
- Modelos activos en runtime: `MomentumModel.predict_from_snapshot` y `MeanReversionModel.predict_from_snapshot`.
- `SignalCombiner` fusiona momentum/reversion/régimen.
- El kernel vuelve a recalcular pesos/edge con capas adicionales (stability/noise/correlation/volatility).

### Hallazgos
1. **Mejora aplicada:** momentum ya no tiene código muerto.
2. **Mejora aplicada:** mean reversion ahora es direccional.
3. **Riesgo restante:** potencial **sobre-filtrado** por múltiples compuertas consecutivas (confidence threshold, friction, edge floor, MTF, session, risk gates), lo que puede impedir operar incluso con señales válidas.
4. **Riesgo restante:** probabilidad de régimen usada en fusión sigue parcialmente hardcodeada en el kernel (`0.78/0.62/0.55`) y no plenamente calibrada a posterior.

### Expectancy y confianza
- Existe rolling expectancy y métricas derivadas (`_rolling_stats`).
- La confianza final se combina con thresholds dinámicos; funcional, pero compleja y con difícil trazabilidad causal para auditoría cuantitativa externa.

---

## PARTE 3 — Auditoría de riesgo

### Controles presentes
- `risk_per_trade`, drawdown global, pérdida diaria, rechazo por latencia/rechazos, kill-switch y `DISABLE_TRADING`.
- `RiskOfRuinEstimator` integrado para multiplicador de riesgo.

### Hallazgos
1. **Mejora aplicada:** `DISABLE_TRADING` bloquea operativa.
2. **Mejora parcial:** límites de firewall ya se recalculan con equity y ratios.
3. **Riesgo restante:** convivencia de varias capas de riesgo (firewall + institutional risk + capital governor + kernel guards) con distinta semántica puede generar falsos positivos de bloqueo y reduce explicabilidad operativa.

---

## PARTE 4 — Auditoría de capital y PnL

### Cálculos actuales
- `exchange_equity`: NAV spot (USDT total + base_qty*last_price) ✅
- `pnl total dashboard`: `realized_pnl + unrealized_pnl`
- `daily_pnl`: `total_equity - daily_anchor_equity`
- `initial_equity`: anclado tras reconciliation startup.

### Hallazgo clave sobre caso sospechoso (Capital ~470 / PnL ~+455)
Causa más probable post-revisión:
1. `realized_pnl` de sesión se reinicia en startup, pero el valor visible puede combinarse con `unrealized` y con anclajes de sesión (`initial_equity`, `daily_anchor`) en un contexto donde hubo fills históricos o reconciliaciones previas.
2. Si no hubo trades en esa sesión, un `entry_price` reanclado tras discrepancia DB/exchange y cambios de mark-to-market puede producir PnL aparente alto respecto al capital libre.

### Riesgo restante
- Aún no existe un servicio contable único tipo “ledger->PnL/NAV” con snapshots versionados; el kernel mantiene parte del estado en memoria y parte por reconstrucción de fills.

---

## PARTE 5 — Auditoría de base de datos

### Tablas relevantes detectadas en esquema
- `trades`, `orders`, `fills`, `order_executions`, `execution_idempotency_ledger`, `capital_reservations`.
- No hay tablas canónicas explícitas `positions` / `pnl`; se reconstruye desde fills y estado runtime.

### Limitación de entorno
- No se pudo auditar contenido real (residuos históricos, inconsistencias DB vs exchange) porque en este entorno no está definido `POSTGRES_DSN`.

---

## PARTE 6 — Auditoría de ejecución de órdenes

Pipeline validado: `decision -> execution_engine -> idempotent_order_service -> binance_client`.

### Estado
- Hay lock global de ejecución (advisory lock), firewall, reserva/liberación de capital, registro forense y reconciliación.
- Idempotencia mejorada con `SUBMISSION_UNCERTAIN` ante timeout de submit.

### Riesgos restantes
1. La finalización order/fill/reservation/ledger no está completamente encapsulada en una única transacción de negocio idempotente.
2. La recuperación por timeout mejoró, pero sigue dependiente de visibilidad eventual del exchange.

---

## PARTE 7 — Auditoría de sincronización con Binance

### Verificación de interfaces
- `fetch_balance` ✅
- `fetch_open_orders` ✅
- `fetch_my_trades` ✅
- `fetch_positions` ✅ (nuevo wrapper spot)

### Riesgo restante
- `fetch_order_by_client_order_id` captura excepciones y devuelve `None`; oculta la causa concreta (network vs not found), degradando diagnóstico fino en reconciliación.

---

## PARTE 8 — Auditoría Binance Testnet/Mainnet

- Endpoints separados correctamente (testnet/mainnet).
- `mainnet` requiere confirmación explícita.
- Resolución de entorno en `main.py` consistente con guardrails.

Conclusión: configuración de entorno de exchange es robusta para operación controlada.

---

## PARTE 9 — Auditoría del dashboard

- Dashboard muestra estado operativo amplio (capital, pnl, edge, ruin, régimen, razones).
- Con la mejora de NAV en kernel, el capital mostrado es más realista que en versión previa.

Riesgo restante:
- PnL mostrado depende de estado de sesión + reconciliaciones; falta una capa contable única para garantizar trazabilidad institucional de punta a punta.

---

## PARTE 10 — Detección de errores potenciales (actuales)

1. Semántica distribuida de riesgo (varias capas) con potencial de bloqueos redundantes.
2. Probabilidad de régimen parcialmente hardcodeada en la lógica final de edge.
3. Manejo de errores de sync exchange demasiado genérico en algunas rutas (`None` silencioso).
4. Ausencia de tabla/snapshot canónica de posiciones y PnL auditables por ciclo.
5. Acoplamiento residual a internals del firewall desde execution engine.

---

## PARTE 11 — Mejoras necesarias para trading real

1. **Contabilidad institucional**
   - Implementar servicio único de `Equity/PnL Ledger` con snapshots por ciclo y reconciliación determinística.
2. **Riesgo explicable**
   - Unificar decisión final de bloqueo en un “risk verdict” único con motivos jerarquizados.
3. **Ejecución atómica**
   - Consolidar cierre de ciclo de orden en transacción idempotente única (order/fill/ledger/reservation).
4. **Calibración cuantitativa**
   - Sustituir probas de régimen hardcodeadas por calibración online/rolling con validación.
5. **Observabilidad**
   - Exponer métricas de discrepancia DB vs exchange y de “submission uncertainty rate”.

---

## PARTE 12 — Reporte final

1. **Evaluación técnica:** **76/100**.
2. **Errores encontrados (actuales):** ver secciones 10 y 11 (5 núcleos de riesgo pendientes).
3. **Riesgos potenciales:** contabilidad no totalmente institucional, bloqueos redundantes de riesgo, reconciliación parcial bajo fallos de red.
4. **Nivel real del sistema:** **avanzado pre-producción** (no institucional completo).
5. **Qué falta para trading real:** contabilidad unificada, ejecución atómica completa, calibración de régimen robusta, observabilidad y reconciliación más estrictas.

