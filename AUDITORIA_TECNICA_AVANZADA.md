# Auditoría técnica avanzada y plan de acción (post-refactor)

## Objetivo operativo
Preparar el bot para operar con capital configurable (mínimo, medio o alto) sin sacrificar seguridad de ejecución, control de riesgo ni observabilidad de UI/DB.

## Hallazgos detectados en esta iteración
1. **Faltaba un puente runtime entre Settings del dashboard y sizing real del engine.**
2. **El usuario no podía elegir un presupuesto objetivo desde el dashboard (capital limit).**
3. **El motor no aplicaba perfiles de inversión (conservador/balanceado/agresivo/personalizado).**
4. **La UI no inyectaba ajustes de inversión al bot en caliente.**
5. **El dashboard dependía de telemetría viva y necesitaba hidratación robusta desde DB al arrancar.**

## Mejoras implementadas en esta entrega
- Se añadió control de inversión en **Settings tab**:
  - modo de inversión
  - límite de capital (USDT)
  - riesgo por trade (%)
  - asignación máxima por trade (%)
- Se conectó la UI con el engine en runtime mediante una cola segura de `runtime_settings` en `StateManager`.
- `MainWindow` ahora envía cada cambio de settings al `StateManager` para aplicación en vivo.
- `BotEngine` aplica runtime settings sin reinicio:
  - `risk_per_trade_fraction` efectivo
  - `max_trade_balance_fraction` efectivo
  - `capital_limit_usdt` para limitar equity utilizable
  - `investment_mode` en snapshot/UI
- Se reforzó visualización de riesgo para mostrar métricas efectivas en dashboard.
- Se mantiene hidratación de DB al inicio para poblar pestañas dependientes de historial.

## Qué queda recomendado para próximas iteraciones
1. Persistir settings de inversión del dashboard en DB (perfil por usuario/sesión).
2. Añadir validaciones cruzadas de riesgo (capital limit < balance libre, etc.).
3. Añadir panel de simulación de impacto (estimación de exposición antes de aplicar settings).
4. Integrar tracking de slippage por trade en UI analytics.
5. Extender portfolio multi-asset con presupuesto por símbolo.

## Criterio de éxito logrado
- El bot ahora permite controlar **cuánto capital usar** y **qué riesgo aplicar** desde dashboard.
- Los cambios se reflejan en sizing real y validación de órdenes.
- No se alteró funcionalidad no relacionada fuera del alcance necesario.
