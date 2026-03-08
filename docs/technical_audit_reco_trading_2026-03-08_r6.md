# Auditoría técnica integral `reco_trading` (R6, foco en brechas de nivel institucional)

Fecha: 2026-03-08  
Rol: Auditor principal de arquitectura, ejecución y lógica cuantitativa.

## 1) Evaluación técnica final

**Puntuación: 85/100**  
**Nivel real:** avanzado pre-producción (institucional parcial).  
**Conclusión:** el sistema es sólido y claramente por encima de prototipo, pero aún no alcanza “institucional pleno” para capital real sostenido por tres razones: (1) orquestación centralizada, (2) rutas legacy/fallback en caminos críticos, (3) reporting contable no completamente normalizado para consumo externo/compliance.

---

## 2) Hallazgos críticos (errores o riesgos concretos)

### H1 — Persistencia asíncrona no acotada por ciclo en dashboard (riesgo de backlog)
- En `_publish_dashboard`, se crean tareas con `asyncio.create_task(...)` en cada ciclo para snapshots financieros/ledger.
- Sin control de concurrencia o cola bounded, una degradación DB puede acumular tareas y afectar memoria/latencia.
- Impacto: degradación progresiva en sesiones largas (objetivo 24/7).

### H2 — Convivencia de rutas tipadas y legacy en reconciliación
- Se avanzó a lookup tipado de Binance, pero aún hay rutas de compatibilidad que pueden retornar `None` sin semántica rica en algunos caminos.
- Impacto: decisiones de reconciliación menos deterministas bajo fallos transitorios; más difícil distinguir “not found real” vs “fallo temporal”.

### H3 — `QuantKernel` aún demasiado monolítico
- El kernel concentra decisión, riesgo, ejecución contextual, reconciliación, dashboard y persistencia.
- Impacto: cambios locales con riesgo sistémico; pruebas de integración más frágiles; mayor MTTR ante incidentes.

### H4 — Rutas fallback en ejecución atómica
- Existe guardrail `require_atomic_finalization`, pero persiste lógica fallback en código para escenarios sin API atómica.
- Impacto: complejidad accidental y riesgo de divergencia de comportamiento entre entornos.

---

## 3) Brechas de implementación para subir de nivel

### B1 — Capa de accounting institucional unificada (faltante)
- Ya existen `financial_snapshots`, `positions_ledger`, `pnl_ledger`, pero falta una vista/materialización oficial de reporting institucional con contrato estable.
- Recomendación exacta:
  1. crear `accounting_view` (SQL view/materialized) con llaves `cycle_ts`, `decision_id`, `exchange_order_id`, `position_state`, `realized`, `unrealized`, `equity`;
  2. job de reconciliación diaria que selle `eod_pnl` y discrepancias exchange-vs-ledger.

### B2 — Risk verdict institucional único (parcial)
- Hay `RiskVerdict`, pero aún coexisten múltiples fuentes de veto distribuidas.
- Recomendación exacta:
  - centralizar en `RiskVerdictEngine.final_verdict(...)` consumido por TODO gate de ejecución;
  - taxonomía formal de motivos (`market`, `latency`, `drawdown`, `exposure`, `compliance`).

### B3 — Contrato tipado único para Binance sync (faltante)
- Ya existe API detallada para order-lookup, pero no está homogeneizada en todo el cliente/sync.
- Recomendación exacta:
  - introducir `ExchangeLookupResult` dataclass/enum para TODAS las rutas `fetch_*` críticas y eliminar retornos ambiguos en producción.

### B4 — Orquestación modular (faltante)
- Recomendación exacta por fases:
  1. extraer `DecisionOrchestrator` del kernel,
  2. extraer `RiskOrchestrator`,
  3. extraer `AccountingRuntimeService`,
  4. dejar `QuantKernel` como coordinador ligero.

---

## 4) Revisión por secciones solicitadas (resumen)

### Arquitectura
- Flujo correcto y robusto, con buenas barreras de ejecución y reconciliación.
- Riesgo principal: centralización excesiva del kernel.

### Modelos cuantitativos
- Momentum/MR direccionales y fusión de régimen mejorada.
- Riesgo residual: sobre-filtrado por exceso de gates secuenciales.

### Riesgo
- `DISABLE_TRADING`, drawdown y kill-switch operativos.
- Falta consolidación final estricta de motivos de bloqueo.

### Capital/PnL
- NAV MTM correcto (avance clave).
- Persistencia financiera mejoró, pero aún falta reporting institucional consolidado.

### Base de datos
- Ledgers y snapshots presentes (avance alto).
- Falta modelo oficial de reporte/auditoría externa por ciclo.

### Ejecución
- Idempotencia fuerte + `SUBMISSION_UNCERTAIN` + guardrails atómicos.
- Aún hay complejidad por coexistencia de caminos fallback.

### Sync Binance
- Buen nivel: `fetch_balance/open_orders/my_trades/positions` + lookup tipado.
- Falta homogeneización completa de contrato tipado en todas las rutas críticas.

### Testnet/Mainnet
- Configuración y guardrails correctos para despliegue controlado.

### Dashboard
- Coherente con estado operativo y métricas clave.
- Riesgo: persistencia por `create_task` sin control de presión.

---

## 5) Respuesta explícita al caso sospechoso (Capital ~470 / PnL ~+455)

Con el estado actual del sistema, la causa más probable sigue siendo:
- mezcla entre contexto reconciliado (histórico + mark-to-market + reanclajes) y percepción de sesión actual;
- sin una vista contable institucional única de sesión vs acumulado, el operador puede ver PnL “alto” sin trades recientes.

El problema ya está mucho mejor acotado que en versiones previas, pero para cerrarlo definitivamente se necesita la capa B1 (accounting view + sellado EOD).

---

## 6) Plan de elevación a nivel institucional (priorizado)

**Prioridad P0 (bloqueante para capital real):**
1. Forzar ruta única de finalización atómica (sin fallback en prod).
2. Implementar `RiskVerdictEngine` único.
3. Implementar `accounting_view` institucional + reconciliación EOD.

**Prioridad P1:**
4. Eliminar ambigüedad de retorno en cliente Binance (contrato tipado total).
5. Introducir control de backpressure para snapshots asíncronos del dashboard.

**Prioridad P2:**
6. Refactor modular del kernel por bounded contexts.

---

## 7) Reporte final requerido

1. **Evaluación técnica:** 85/100.
2. **Errores/problemas encontrados:** H1, H2, H3, H4.
3. **Riesgos potenciales:** backlog asíncrono, reconciliación parcialmente ambigua, complejidad del kernel, divergencia por rutas fallback.
4. **Nivel real:** avanzado pre-producción.
5. **Qué falta para trading real institucional:** P0+P1 (ruta atómica única, verdict único de riesgo, accounting view EOD, contrato tipado Binance completo, backpressure de snapshots).

