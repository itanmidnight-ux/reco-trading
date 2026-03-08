# Auditoría técnica integral `reco_trading` (R7)

Fecha: 2026-03-08  
Objetivo: detectar errores, problemas y faltantes de integración; determinar nivel exacto del sistema.

## Nivel exacto actual del programa

**Nivel exacto:** **Avanzado pre‑producción (Institutional-Ready Parcial)**  
**Puntuación técnica exacta:** **86/100**

> No es prototipo. Tampoco es “institucional pleno” todavía para capital real continuo 24/7 sin supervisión.

---

## Estado por dominios (resumen ejecutivo)

1. **Arquitectura núcleo (8.5/10):** robusta, pero `QuantKernel` sigue muy centralizado.
2. **Ejecución e idempotencia (8.7/10):** fuerte, con `SUBMISSION_UNCERTAIN` y guardrails atómicos.
3. **Riesgo (8.2/10):** múltiples barreras activas, pero falta consolidación total de dictamen único.
4. **Capital/PnL/Accounting (8.4/10):** MTM + ledgers + snapshots correctos; falta reporting institucional finalizado.
5. **Sincronización exchange (8.5/10):** bien encaminada con lookup tipado, aún conviven rutas legacy.
6. **Operabilidad institucional (8.1/10):** buena base, faltan cierres de integración para nivel “fondos”.

---

## Errores/problemas detectados (vigentes)

### E1 — Kernel demasiado monolítico
- `QuantKernel` concentra decisión, riesgo, reconciliación, dashboard y persistencia.
- Riesgo: blast radius alto ante cambios y menor mantenibilidad institucional.

### E2 — Integración de riesgo aún no totalmente unificada
- Existe `RiskVerdict`, pero convive con múltiples rutas de bloqueo.
- Riesgo: explicabilidad parcial del motivo final de veto en escenarios complejos.

### E3 — Integración exchange tipada incompleta
- Hay ruta tipada para lookup por `clientOrderId`, pero coexisten caminos legacy con semántica menos rica.
- Riesgo: reconciliaciones menos deterministas en degradación de red.

### E4 — Integración contable institucional incompleta
- Ya existen `financial_snapshots`, `positions_ledger`, `pnl_ledger`, `accounting_view`.
- Faltan integraciones finales de “operational reporting/compliance pack” (EOD snapshots sellados, discrepancy reports automáticos y consumo externo formal).

### E5 — Compatibilidad/fallback residual en ejecución
- Aunque la finalización atómica está reforzada, hay complejidad de compatibilidad en rutas alternativas.
- Riesgo: divergencia de comportamiento entre entornos no homogéneos.

---

## Falta de integraciones para subir al siguiente nivel

1. **RiskVerdictEngine único** (una única salida para todo veto de ejecución).
2. **Contrato tipado único exchange** para todas las operaciones críticas (`fetch_*`).
3. **Reporting institucional completo**:
   - consolidado diario (EOD),
   - discrepancias DB vs exchange,
   - trazabilidad de sesión vs acumulado.
4. **Refactor de orquestación** por dominios (`Decision`, `Risk`, `ExecutionLifecycle`, `Accounting`).

---

## Análisis del caso sospechoso (Capital ~470 / PnL ~+455)

Con el estado actual del sistema, el origen más probable sigue siendo:
- mezcla de estado de sesión con reconciliación histórica y mark-to-market,
- diferencia entre “PnL percibido de sesión” y “PnL reconciliado acumulado”.

Con la infraestructura nueva (ledgers + accounting_view) ya existe base para resolverlo de forma institucional, pero falta cerrar la integración de reporting operacional para que quede inequívoco para operador/compliance.

---

## Recomendación final

Para pasar de **86/100** a **92+ institucional**:
1. eliminar rutas legacy en ejecución/sync productivo,
2. consolidar riesgo en un único motor de dictamen,
3. completar capa de reporting institucional (EOD + reconciliación automatizada + API de auditoría externa),
4. desacoplar `QuantKernel` en servicios orquestadores.

