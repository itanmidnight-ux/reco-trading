# Auditoría técnica integral de `reco_trading` (pre-producción)

Fecha: 2026-03-08  
Alcance: arquitectura, modelos cuantitativos, riesgo, capital/PnL, base de datos, ejecución, sincronización con Binance, testnet/mainnet y dashboard.

## 1) Evaluación ejecutiva

**Veredicto:** el sistema está en un nivel **avanzado/prototipo institucional**, pero **no está listo para capital real** sin correcciones en contabilidad de equity/PnL, consistencia de sizing de riesgo, y controles de sincronización/estado.

**Score técnico global (0-100): 68/100**

- Arquitectura modular y barreras de seguridad existen, con pipeline claro desde decisión hasta ejecución.  
- Hay controles sólidos (firewall, idempotencia, lock global, reconcile startup), pero todavía hay inconsistencias de estado y contabilidad que pueden activar o desactivar trading de forma incorrecta.

## 2) Flujo de arquitectura (mercado -> ejecución)

Flujo observado:

1. `main.py` valida variables críticas y arranca `QuantKernel().run()` en modo live.  
2. `QuantKernel.initialize()` crea `BinanceClient`, `Database`, `MarketDataService`, motores de señal y `ExecutionEngine` (con `ExecutionFirewall`, `IdempotentOrderService`, `ExchangeGateway` y `CapitalGovernor`).  
3. En cada ciclo, el kernel:
   - descarga/valida datos de mercado,
   - genera señales (`SignalEngine` con momentum + mean reversion),
   - detecta régimen,
   - calcula edge/confianza/filtros,
   - pasa por controles de riesgo,
   - ejecuta vía `ExecutionEngine.execute()`.
4. `ExecutionEngine` aplica lock global DB + firewall + saneo de cantidad + idempotencia + persistencia forense (`orders`, `fills`, `order_executions`, `execution_idempotency_ledger`).

Interacción entre módulos críticos:

- `QuantKernel` orquesta y mantiene estado financiero operativo.
- `ExecutionEngine` centraliza ejecución y seguridad operacional.
- `IdempotentOrderService` protege contra duplicaciones (ledger + clientOrderId determinista).
- `CapitalProtectionController` bloquea si hay orden/posición activa.
- `BinanceClient` encapsula endpoints, rate governor y time sync.

**Riesgo de acoplamiento peligroso detectado:**

- `QuantKernel` accede atributos privados de `ExecutionEngine` (`_idempotent_order_service`, `_firewall`), lo que rompe encapsulación y dificulta garantías formales en cambios futuros.

## 3) Auditoría de modelos cuantitativos

### Hallazgos

1. **Momentum con código muerto**: `predict_from_snapshot` retorna una primera fórmula y deja una segunda lógica inaccesible (unreachable code).  
2. **Mean reversion no direccional**: usa magnitud absoluta (`abs(...)`) y logística positiva, tendiendo a producir probabilidad > 0.5 sin signo de reversión explícito.  
3. **Combinación de modelos**: el ensemble existe (`SignalCombiner`), pero luego el kernel vuelve a recalcular pesos/edge manualmente; hay duplicación de lógica y potencial deriva conceptual.  
4. **Regime probability hardcodeada** (`0.78/0.62/0.55`) en vez de probabilidad calibrada del detector.

### Riesgo

- Sobre-filtrado y edge inestable por múltiples gates: confianza mínima, edge > fricción, edge > umbral dinámico, modo/regime/MTF/session filters.
- Difícil atribución causal (por qué se ejecutó/no se ejecutó) por capas redundantes de decisión.

## 4) Auditoría de riesgo

### Hallazgos

1. Kill switch existe y cubre latencia, rechazos, drawdown y pérdida diaria.
2. **Inconsistencia de unidad en exposición**:
   - `Settings.max_total_exposure` y `max_asset_exposure` son ratios [0,1],
   - `ExecutionFirewall` en `QuantKernel.initialize()` se construye con límites absolutos (100000/50000),
   - puede producir límites incoherentes entre capas de riesgo.
3. `DISABLE_TRADING` existe como decisión/salida lógica, pero no hay bandera de entorno explícita única tipo `DISABLE_TRADING=true` en settings como interruptor operacional de emergencia.

### Riesgo

- Falsos positivos/falsos negativos en bloqueo por no tener una única semántica de límites y switch operativo global.

## 5) Auditoría de capital y PnL

### Hallazgos críticos

1. **`exchange_equity` se deriva solo de bucket USDT**, no NAV completo multi-asset (USDT + base asset mark-to-market).  
2. En runtime, `state.equity` se actualiza con **free USDT**; `exchange_equity` con **USDT total**; si hay inventario BTC, el capital real queda sub/sobre representado.
3. `realized_pnl` de sesión se resetea a 0 en startup (correcto para sesión), pero el dashboard muestra `pnl = realized + unrealized`, mientras el capital se apoya en `exchange_equity` basado en USDT solamente.
4. Se mezcla fee estimada (`taker_fee`) en tiempo real con fee almacenada de fills (reconciliación histórica), generando deriva metodológica.

### Explicación del caso sospechoso (Capital ~470 / PnL ~+455 sin trades aparentes)

La combinación más probable, según código:

- `initial_equity` se ancla al equity al iniciar sesión,
- el cálculo visible de PnL y equity mezcla estado de sesión + snapshot de balance USDT + posible inventario previo/reconciliación DB,
- si existen fills históricos o desalineación entre DB y exchange, el dashboard puede mostrar PnL alto sin que en esa sesión se hayan hecho trades nuevos.

Esto es consistente con el diseño actual de recuperación + métricas de sesión sin un servicio canónico de NAV.

## 6) Auditoría de base de datos

### Estado

- El esquema contiene tablas de auditoría clave: `orders`, `fills`, `order_executions`, `execution_idempotency_ledger`, `capital_reservations`, `decision_audit`.
- No se encontró `positions` ni `pnl` como tablas canónicas dedicadas; la posición se reconstruye desde fills.
- La reconciliación startup limpia reservas stale y actualiza estados de idempotencia.

### Limitación de entorno

- No fue posible inspeccionar contenido real (conteos/filas) porque `POSTGRES_DSN` no está disponible en este entorno de auditoría.

## 7) Auditoría de ejecución de órdenes

### Fortalezas

- Lock global (`execution_advisory_lock`) para evitar colisiones cross-process.
- Idempotencia con ledger persistente y reconciliación periódica.
- Sanitización de cantidad según reglas de símbolo y verificación de fill terminal.

### Riesgos

1. `client_order_id` usa bucket por segundo + contexto; en ráfagas puede colisionar semánticamente entre decisiones muy cercanas.
2. Manejo de timeout/retry puede entrar en ventanas de incertidumbre si exchange acepta orden pero la consulta aún no la refleja.
3. Persistencia no es transacción única de “finalización” end-to-end (order/fill/ledger/reservation).

## 8) Auditoría de sincronización con Binance

### Cobertura disponible

- `fetch_balance`, `fetch_open_orders`, `fetch_my_trades`, `fetch_order`, `fetch_order_by_client_order_id` existen y se usan.
- Time sync y retry ante `-1021` implementados.

### Gap crítico

- **No existe `fetch_positions`** en el cliente spot (ni wrapper equivalente explícito), lo cual obliga a inferir posición con balances/fills y aumenta riesgo de desincronización cuando hay operaciones externas/manuales.

## 9) Binance Testnet vs Mainnet

- Endpoints separados y correctos en cliente:
  - testnet: `testnet.binance.vision`
  - mainnet: `api.binance.com`
- Guardrail correcto: mainnet exige `confirm_mainnet=true`.
- `main.py` fuerza coherencia entre CLI y entorno para modo `testnet/real`.

## 10) Auditoría del dashboard

- El dashboard muestra métricas ricas (capital, PnL, edge, confianza, ruin, estado).
- **Problema principal:** no hay garantía de que “capital mostrado” sea NAV real, porque depende de `state.exchange_equity` (USDT total) y no de valuación completa de inventario.

## 11) Lista consolidada de errores/problemas detectados

1. Equity/NAV no canónico (USDT-only) para decisiones de riesgo.
2. Inconsistencia de unidades de exposición entre settings y firewall.
3. Código muerto en momentum model.
4. Mean reversion sin señal direccional robusta.
5. Regime probabilities hardcodeadas.
6. Mezcla de metodología de fees/PnL (estimado runtime vs ledger histórico).
7. Acoplamiento a atributos privados de `ExecutionEngine`.
8. No existe interfaz explícita de `fetch_positions`.
9. Ausencia de switch operacional único `DISABLE_TRADING` en configuración.

## 12) Mejoras necesarias (solución exacta propuesta)

### A. Capital/PnL (bloqueante para producción)

- Crear `PortfolioValuationService` y usarlo como **única fuente de verdad** de equity:
  - `equity = quote_free + quote_used + Σ(base_asset_qty * mark_price)`.
- Reemplazar `_fetch_account_balance()` para devolver NAV completo + desglose por activo.
- Hacer que kill switch / drawdown / daily loss usen ese NAV canónico.

Archivos a tocar: `reco_trading/kernel/quant_kernel.py`, `reco_trading/infra/binance_client.py`, nuevo `reco_trading/core/portfolio_valuation.py`.

### B. Riesgo (bloqueante)

- Unificar semántica de exposición:
  - o todo en ratio [0,1],
  - o todo en notional absoluto.
- Añadir validación en `Settings` para impedir mezclar ambas sin conversión.

Archivos: `reco_trading/config/settings.py`, `reco_trading/execution/execution_firewall.py`, `reco_trading/core/institutional_risk_manager.py`.

### C. Ejecución e idempotencia (alta prioridad)

- Incluir `decision_id` UUID en semilla de `client_order_id` siempre.
- Añadir estado explícito `SUBMISSION_UNCERTAIN` y reconciliación reforzada antes de resubmit.
- Implementar función transaccional única de finalización (order+fill+ledger+reservation).

Archivos: `reco_trading/core/execution_engine.py`, `reco_trading/execution/idempotent_order_service.py`, `reco_trading/infra/database.py`.

### D. Modelos cuantitativos (alta prioridad)

- Eliminar código muerto en momentum.
- Rehacer mean reversion para probabilidad direccional (up/down), no solo magnitud.
- Calibrar `regime_prob` desde salidas reales del detector y reliability tracking.

Archivos: `reco_trading/core/momentum_model.py`, `reco_trading/core/mean_reversion_model.py`, `reco_trading/kernel/quant_kernel.py`, `reco_trading/core/market_regime.py`.

### E. Sincronización exchange/DB (alta prioridad)

- Añadir API explícita de posición/inventario reconciliado (`fetch_positions` lógico para spot: balances + open orders + fills recientes).
- Persistir snapshots periódicos de NAV y posición para auditoría.

Archivos: `reco_trading/infra/binance_client.py`, `reco_trading/infra/database.py`, `reco_trading/kernel/quant_kernel.py`.

---

## Nivel real del sistema

**Nivel actual:** avanzado con componentes institucionales, pero en práctica **pre-producción / paper-trading robusto**.  
**No institucional listo para capital real** todavía.

## Criterio de salida a producción real

Mínimo requerido antes de operar capital real:

1. NAV canónico validado contra exchange cada ciclo.
2. Consistencia de riesgo (unidades) y pruebas de estrés de kill switch.
3. Finalización atómica de ejecución + reconciliación de incertidumbre.
4. Modelos calibrados con artifacts versionados y métricas de drift.
5. Dashboard 100% alineado a NAV/PnL canónicos.
