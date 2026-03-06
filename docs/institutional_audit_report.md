# Institutional Pre-Live Audit Report (Binance 24/7)

## Scope
- Runtime path reviewed: `run.sh -> main.py -> QuantKernel.run()`.
- Domains reviewed: Financial Integrity, Exchange Synchronization, Execution Safety, Quant Model Validity, Production Stability.
- Method: static code audit + targeted tests (`pytest` subset).

## Executive Risk Verdict
**Status: NOT READY FOR REAL CAPITAL**

The engine has several high-severity control failures that can lead to incorrect risk sizing, state divergence, and accounting distortions under normal live conditions (restart, partial fills, and balance drift). The most critical issue is equity mismeasurement that can mis-govern risk and kill-switch logic.

---

## 1) Architecture Weaknesses

### A1 — Equity model is structurally inconsistent with spot inventory (CRITICAL)
- **Location:** `reco_trading/kernel/quant_kernel.py`
- **Root cause:** `exchange_equity` is sourced from USDT balance only, while total equity is computed as `exchange_equity + unrealized_pnl`. This omits base-asset principal and can under/over-estimate capital depending on position state.
- **Why dangerous:** risk manager, capital governor context, and kill switch can operate on stale/incorrect equity.
- **Recommended fix:** compute mark-to-market NAV using full balance vector (quote + base holdings converted by ticker), not USDT bucket only; define one canonical equity source used by all guards.

### A2 — Financial state split-brain between internal estimators and exchange truth (HIGH)
- **Location:** `reco_trading/kernel/quant_kernel.py`, `reco_trading/infra/database.py`
- **Root cause:** runtime updates fees/pnl via estimated taker fee at execution time, while startup restoration reconstructs realized pnl from DB fills (exchange-reported fee field). Methodologies differ.
- **Why dangerous:** drifting realized pnl/fees, inconsistent daily loss checks, hard-to-explain audit trails.
- **Recommended fix:** move to fill-ledger-as-single-source-of-truth; runtime should append fills and derive pnl/fees from ledger snapshots, not maintain ad-hoc incremental counters.

### A3 — Order lifecycle persistence is not atomic (HIGH)
- **Location:** `reco_trading/core/execution_engine.py`
- **Root cause:** order submission, fill capture, idempotency updates, reservation finalization, and execution persistence are split across multiple calls without transactional envelope.
- **Why dangerous:** crash between steps leaves orphan state (e.g., fill executed but reservation active, or fill persisted without matching status update).
- **Recommended fix:** introduce transactional “execution finalization” function in DB layer with idempotent upserts keyed by exchange order id + client order id.

---

## 2) Execution Risks

### E1 — client_order_id can collide under same-second repeated intents (HIGH)
- **Location:** `reco_trading/core/execution_engine.py`, `reco_trading/execution/idempotent_order_service.py`
- **Root cause:** `decision_context_hash` includes symbol/side/amount and second bucket; burst retries with same side/qty/tag in same second can produce identical ids.
- **Why dangerous:** unintended order dedup across distinct decisions, or wrong recovery linkage after restart.
- **Recommended fix:** include immutable decision UUID from upstream decision audit in client_order_id seed; enforce uniqueness at decision granularity.

### E2 — Partial fill handling can defer accounting until terminal status (HIGH)
- **Location:** `reco_trading/core/execution_engine.py`
- **Root cause:** execution loop records fill only when status becomes closed/filled or terminal cancel/reject; partial fills during timeout window may not be persisted immediately.
- **Why dangerous:** temporary invisible exposure and inaccurate position during network or exchange lag events.
- **Recommended fix:** persist incremental executions on each status poll where `filled > previously_seen`; reconcile cumulative fill quantity idempotently.

### E3 — Retry/resubmit path lacks explicit duplicate-prevention query before re-submit on timeout (MEDIUM)
- **Location:** `reco_trading/execution/idempotent_order_service.py`
- **Root cause:** on timeout, recovery checks by client_order_id then may resubmit; correctness depends on exchange visibility timing.
- **Why dangerous:** in delayed propagation windows, a second submit could occur if first accepted but not yet queryable.
- **Recommended fix:** add bounded delayed re-check loop before re-submit and require exchange ack/timeout proof; persist a “submission uncertainty” state.

### E4 — Firewall can be bypassed by non-canonical adapters outside kernel path (MEDIUM)
- **Location:** `reco_trading/hft/multi_exchange_arbitrage.py`
- **Root cause:** adapters can call CCXT `create_order` directly (guarded only by environment profile string).
- **Why dangerous:** alternate runtime paths can trade without canonical firewall/idempotency chain.
- **Recommended fix:** centralize all live order routing through `ExecutionEngine`/`ExchangeGateway`; enforce hard prohibition in production via dependency wiring, not env string checks.

---

## 3) Financial Accounting Risks

### F1 — Restart PnL reconstruction is incomplete for unmatched exchange events (CRITICAL)
- **Location:** `reco_trading/kernel/quant_kernel.py`, `reco_trading/infra/database.py`
- **Root cause:** startup reconciliation reads DB fills and compares DB position to exchange base quantity, but does not backfill missing exchange fills into DB before pnl reconstruction.
- **Why dangerous:** after outage/manual intervention, pnl and average entry become non-reconstructable from internal ledger alone.
- **Recommended fix:** on startup, fetch recent exchange trades/orders and perform deterministic ledger backfill before restoring state.

### F2 — Drawdown/daily loss may use distorted denominator (CRITICAL)
- **Location:** `reco_trading/kernel/quant_kernel.py`
- **Root cause:** kill-switch drawdown and daily loss ratios are based on `_compute_total_equity()` using inconsistent equity representation.
- **Why dangerous:** false negatives (not stopping when should) or false positives (stopping unnecessarily).
- **Recommended fix:** calculate drawdown from canonical NAV snapshots persisted each cycle and cross-checked against exchange balances.

### F3 — Fee accounting methodology mismatch (HIGH)
- **Location:** `reco_trading/kernel/quant_kernel.py`, `reco_trading/infra/database.py`
- **Root cause:** runtime fees for BUY/SELL are estimated from configured taker fee, while DB stores exchange-reported fee per fill.
- **Why dangerous:** mis-stated realized performance and compliance reporting drift.
- **Recommended fix:** use exchange fill fee only; retain configured fee only for pre-trade expected-cost modeling.

### F4 — Average entry reconstruction ignores SELL overfills and external transfers semantics (MEDIUM)
- **Location:** `reco_trading/infra/database.py`
- **Root cause:** snapshot logic clamps quantity to non-negative and resets avg entry on full close, but does not encode provenance for external deposits/withdrawals or off-system trades.
- **Why dangerous:** wrong entry basis after non-bot account activity.
- **Recommended fix:** introduce inventory adjustment ledger events and require reconciliation policy for external inventory changes.

---

## 4) Quantitative Model Risks

### Q1 — “Model” probabilities in live path are mostly heuristic, not trained outputs (HIGH)
- **Location:** `reco_trading/core/momentum_model.py`, `reco_trading/core/mean_reversion_model.py`, `reco_trading/kernel/quant_kernel.py`
- **Root cause:** live path uses `predict_from_snapshot` heuristics; training methods exist but are not integrated in runtime loop.
- **Why dangerous:** perceived model confidence may be statistically uncalibrated and unstable out-of-sample.
- **Recommended fix:** enforce explicit model artifact loading/versioning; block live trading when calibrated model artifacts are unavailable.

### Q2 — Regime mapping is hard-coded and potentially non-stationary (MEDIUM)
- **Location:** `reco_trading/kernel/quant_kernel.py`, `reco_trading/core/market_regime.py`
- **Root cause:** fixed regime probability constants (e.g., 0.78/0.62/0.55) and simple state map from unsupervised labels.
- **Why dangerous:** regime probability may not correspond to true posterior probability, causing mis-weighted signal fusion.
- **Recommended fix:** calibrate regime probabilities with rolling reliability diagnostics; persist calibration drift alerts.

### Q3 — Potential leakage risk in feature targets if reused outside strict train/predict split (MEDIUM)
- **Location:** `reco_trading/core/feature_engine.py`
- **Root cause:** feature builder adds `target_up` and `target_reversion` directly in same frame with features (uses future return shift).
- **Why dangerous:** easy accidental leakage in downstream experimentation/training if split discipline breaks.
- **Recommended fix:** separate target generation into dedicated training-only pipeline and enforce schema contracts.

---

## 5) Exchange Synchronization Risks

### X1 — Startup reconciliation updates idempotency states but does not auto-cancel/resolve orphan open orders (HIGH)
- **Location:** `reco_trading/kernel/quant_kernel.py`
- **Root cause:** open orders are read and statuses updated, but no deterministic policy to cancel stale orders from previous session.
- **Why dangerous:** latent orders can execute after restart against new strategy state.
- **Recommended fix:** at startup, classify open orders by strategy decision lineage and cancel unknown/stale orders before trading.

### X2 — Reservation cleanup depends on open orders + active ledger only, not exchange trade history (MEDIUM)
- **Location:** `reco_trading/infra/database.py`
- **Root cause:** stale reservation release logic has no direct fill-history confirmation.
- **Why dangerous:** potential under/over-release when exchange state lags or during partial-fill transitions.
- **Recommended fix:** tie reservation lifecycle to confirmed cumulative filled notional from exchange order/trade endpoints.

### X3 — fetch_order_by_client_order_id error swallowing can hide desync conditions (MEDIUM)
- **Location:** `reco_trading/infra/binance_client.py`
- **Root cause:** method returns `None` on any exception.
- **Why dangerous:** masks distinguishable failure modes (network outage vs true not-found), harming reconciliation correctness.
- **Recommended fix:** propagate typed errors or tagged failure codes; reconciliation should branch by error class.

---

## Failure-Scenario Assessment

### Scenario: Restart while holding position
- Current behavior: DB-derived position is compared with exchange base balance and exchange quantity wins.
- Residual risk: avg entry may be stale/approximate when DB missing fills; pnl baseline can be wrong.

### Scenario: Partial fill followed by restart
- Current behavior: idempotency ledger can mark partial/submitted, but fill accounting depends on terminal capture.
- Residual risk: exposure may be underreported until reconciliation catches up.

### Scenario: Network failure during order submission
- Current behavior: retry + timeout recovery by client_order_id exists.
- Residual risk: delayed exchange visibility can still create uncertain submission windows.

### Scenario: Binance timestamp error (-1021)
- Current behavior: retry path forces time resync.
- Residual risk: no explicit metric/alert on repeated skew incidents.

### Scenario: Position mismatch exchange vs DB
- Current behavior: startup applies exchange quantity truth.
- Residual risk: no automatic historical backfill to repair pnl lineage.

---

## Institutional Go-Live Gate (Recommended)
Trading should remain disabled until these are complete:
1. Canonical NAV/equity service with full-balance mark-to-market.
2. Deterministic startup reconciliation with historical fill backfill.
3. Atomic execution finalization transaction and orphan-order cancellation policy.
4. Strict single execution path enforcement (no adapter bypass).
5. Model artifact governance + calibration monitoring in production.

