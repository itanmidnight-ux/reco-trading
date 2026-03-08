# Auditoría técnica integral `reco_trading` (R5)

Fecha: 2026-03-08  
Rol: Auditor principal de arquitectura, ejecución y lógica cuantitativa.

## Resumen ejecutivo

**Evaluación técnica actual: 84/100**  
**Nivel real:** avanzado pre-producción (institucional parcial).  
**Conclusión:** arquitectura robusta para testnet y operación controlada; todavía requiere cierre de brechas para capital real institucional continuo.

Mejoras constatadas respecto a auditorías previas:
- NAV mark-to-market consolidado.
- Idempotencia con estado `SUBMISSION_UNCERTAIN` y reconciliación más segura.
- Ledgers explícitos (`positions_ledger`, `pnl_ledger`) + `financial_snapshots`.
- Guardrails de finalización atómica en ejecución productiva.
- Mejor diagnóstico de sincronización exchange con lookup detallado tipado.

---

## PARTE 1 — Auditoría de arquitectura

### Flujo end-to-end validado
`main.py -> QuantKernel.run() -> market/feature/models/regime/decision -> risk gates -> ExecutionEngine -> IdempotentOrderService -> BinanceClient -> DB`.

### Módulos y acoplamiento
- `QuantKernel`: orquestación completa (muy potente, pero todavía demasiado centralizada).
- `ExecutionEngine`: ejecución + validación + persistencia (más seguro tras guardrails atómicos).
- `IdempotentOrderService`: journal durable + reconcile loop + recuperación.
- `CapitalProtectionController`: bloqueo por actividad concurrente.
- `BinanceClient`: gobernanza de llamadas + time/rate sync.

### Riesgo arquitectónico residual
- `QuantKernel` concentra demasiadas responsabilidades para un entorno institucional 24/7 (recomendable segmentación por servicios).

---

## PARTE 2 — Auditoría de modelos cuantitativos

### Estado
- Momentum direccional sin código muerto.
- Mean reversion direccional.
- Fusión por `SignalCombiner` + ajustes de régimen/estabilidad.

### Cálculo de edge/confianza/expectancy
- Edge: probabilidad combinada ajustada por noise/correlation/volatilidad.
- Confianza: umbrales dinámicos + fricción + edge floor.
- Expectancy: rolling stats y métricas de rendimiento por régimen.

### Riesgo residual
- Sobre-filtrado potencial por apilamiento de gates (confianza/fricción/edge/MTF/risk layers).

---

## PARTE 3 — Auditoría de riesgo

### Revisado
- `risk_per_trade`, `risk_of_ruin`, drawdown, kill-switch, `DISABLE_TRADING`.

### Estado
- `DISABLE_TRADING` y kill-switch funcionan como barrera global.
- Firewall se recalibra con equity.
- Se introdujo un `RiskVerdict` para diagnóstico más unificado.

### Riesgo residual
- Persisten varias capas de veto con explicabilidad todavía no completamente unificada a nivel institucional (fuente única de truth para block reason final).

---

## PARTE 4 — Auditoría de capital y PnL

### Cálculos revisados
- Equity: NAV spot MTM (USDT + base*mark).
- PnL total: `realized + unrealized`.
- PnL diario: anclado a equity diario.
- `initial_equity`: definido tras reconciliación inicial.

### Caso sospechoso `Capital ~470 / PnL ~+455`
Explicación técnica probable:
- Diferencia entre PnL de sesión percibido vs estado reconciliado histórico + mark-to-market, especialmente tras restarts/reanclajes.
- La nueva persistencia en ledgers mejora trazabilidad, pero aún falta capa contable institucional final para reporting operativo inequívoco.

---

## PARTE 5 — Auditoría de base de datos

### Tablas relevantes
- `trades`, `orders`, `fills`, `order_executions`, `execution_idempotency_ledger`, `capital_reservations`, `financial_snapshots`, `positions_ledger`, `pnl_ledger`.

### Hallazgos
- Muy buena capacidad forense de ejecución y estado financiero histórico.
- Aún faltaría una vista/materialización institucional unificada para auditoría externa (posición + pnl + decisiones + fills por ciclo).

### Limitación
- Sin `POSTGRES_DSN` no fue posible validar residuos/histórico real en este entorno.

---

## PARTE 6 — Auditoría de ejecución de órdenes

Pipeline auditado: `decision -> execution_engine -> idempotent_order_service -> binance_client`.

### Estado
- Lock global activo.
- Idempotencia robusta con `SUBMISSION_UNCERTAIN`.
- Finalización atómica forzada en runtime productivo (`require_atomic_finalization=True`).

### Riesgo residual
- Existen rutas de compatibilidad/fallback en código que conviene retirar en perfil producción para reducir complejidad operativa.

---

## PARTE 7 — Auditoría de sincronización con Binance

Verificado:
- `fetch_balance` ✅
- `fetch_open_orders` ✅
- `fetch_my_trades` ✅
- `fetch_positions` ✅

Mejora aplicada:
- lookup tipado por `clientOrderId` (`network`, `rate_limit`, `not_found`, etc.).

Riesgo residual:
- coexisten rutas legacy (`None`) y tipadas; recomendada convergencia total a contrato tipado.

---

## PARTE 8 — Auditoría Binance Testnet/Mainnet

- Endpoints correctos y separados.
- Guardrails de mainnet correctos.
- Operación segura para testnet y despliegue real controlado con confirmación explícita.

---

## PARTE 9 — Auditoría del dashboard

- Dashboard refleja bien estado operativo, riesgo y métricas clave.
- Con ledgers + snapshots, la coherencia financiera es sustancialmente mejor.

Riesgo residual:
- Falta una capa de reporting institucional consolidada para explicar inequívocamente PnL “de sesión” vs “acumulado reconciliado”.

---

## PARTE 10 — Errores potenciales vigentes

1. Centralización de responsabilidades en `QuantKernel`.
2. Multiplicidad de capas de riesgo con diagnóstico aún parcialmente distribuido.
3. Rutas legacy de compatibilidad en ejecución/sync.
4. Ausencia de módulo dedicado de reporting contable institucional consumible externamente.

---

## PARTE 11 — Mejoras necesarias

1. Refactor modular por bounded contexts (`Decision`, `Risk`, `ExecutionLifecycle`, `Accounting`).
2. Dictamen único de riesgo con taxonomía formal de bloqueos.
3. Eliminar fallbacks legacy en perfil producción.
4. Implementar reporting institucional consolidado sobre ledgers + decisiones + ejecuciones.

---

## PARTE 12 — Reporte final

1. **Evaluación técnica:** 84/100.
2. **Errores encontrados:** listados en PARTE 10.
3. **Riesgos potenciales:** centralización del kernel, complejidad de veto multi-capa, reporting no completamente institucional.
4. **Nivel real:** avanzado pre-producción.
5. **Qué falta para trading real:** refactor modular, risk verdict único, consolidación de rutas productivas y reporting contable institucional completo.

