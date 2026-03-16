# reco-trading Full Deep Audit Report

## Scope and methodology
- Repository-wide file inventory reviewed from `rg --files` (all tracked source/config/test files).
- Static integrity checks executed:
  - `python -m compileall -q .`
  - `mypy reco_trading tests`
  - `pytest -q`
  - `python -m reco_trading.main` (startup probe)
  - custom AST scan for blocking calls inside `async def`
  - custom internal import cycle coarse scan
- Network constraints prevented installation of missing tooling (`flake8`, `pylint`) in this environment.

---

## Phase 1 — Static code audit findings

### High severity
1. **Missing runtime dependency blocks startup (`pydantic`)**
   - Evidence: startup fails with `ModuleNotFoundError: No module named 'pydantic'` at import time from `reco_trading/config/settings.py`.
   - Impact: bot cannot bootstrap in environments that rely on `requirements.txt`.
   - Location: `requirements.txt` (missing `pydantic`, `pydantic-settings`) and `reco_trading/config/settings.py` imports.
   - Safe patch: add `pydantic` and `pydantic-settings` to `requirements.txt` with pinned compatible versions.

2. **Potential thread/Qt cross-thread signal emission risk from bot thread**
   - Evidence: bot runs in `threading.Thread` and directly calls `state_manager.add_trade`, `state_manager.add_log`, `state_manager.update`; these methods emit Qt signals immediately.
   - Impact: UI instability/crash risks under high update rates depending on connection types and object affinity.
   - Locations: `reco_trading/main.py`, `reco_trading/core/bot_engine.py`, `reco_trading/ui/state_manager.py`.
   - Safe patch: marshal all state mutations to Qt main thread via `QMetaObject.invokeMethod`/queued single-shot bridge.

3. **Database errors can be silently swallowed in many critical paths**
   - Evidence: `safe_db_call` catches all exceptions and returns default; used on logging/state/signal methods.
   - Impact: hidden DB write failures reduce observability and can mask operational drift.
   - Location: `reco_trading/database/repository.py`.
   - Safe patch: include structured critical alert path (e.g., circuit-breaker counter + escalation callback) while preserving non-fatal behavior.

### Medium severity
4. **Order close path may create untracked duplicate/ghost exits under uncertain exchange responses**
   - Evidence: `_close_position` retries market close up to 3 attempts; if first request reaches exchange but response is lost, retries can submit additional close orders.
   - Impact: over-closing spot inventory or residual reconciliation noise.
   - Location: `reco_trading/core/bot_engine.py`.
   - Safe patch: after any close error, query order/trade status by `clientOrderId` before retrying; reconcile filled amount vs position.

5. **Position model lacks partial fill reconciliation**
   - Evidence: open/close logic assumes full fill quantity from submitted amount, no explicit fill-status polling.
   - Impact: desync between local position quantity and exchange position after partial fills.
   - Locations: `reco_trading/core/bot_engine.py`, `reco_trading/risk/position_manager.py`.
   - Safe patch: persist `filled`, `remaining`, and reconcile each loop via exchange order/trade fetch.

6. **Market intelligence “support/resistance boost” clamps to 1.0 always**
   - Evidence: `min(support_boost, 1.0)` and `min(resistance_boost, 1.0)` where boosts are `1.1/1.2`.
   - Impact: intended boost never exceeds neutral multiplier, reducing strategy intent.
   - Location: `reco_trading/strategy/market_intelligence.py`.
   - Safe patch: cap with intended upper bound (e.g., `1.2`) or remove cap if risk already bounded upstream.

7. **Type-checking debt in trading-critical modules**
   - Evidence: mypy reports 70 errors including arithmetic/type mismatches in order normalization and strategy/UI modules.
   - Impact: increased probability of runtime edge-case failures and maintenance risk.
   - Locations: `reco_trading/exchange/order_manager.py`, `reco_trading/strategy/market_intelligence.py`, UI modules, tests.
   - Safe patch: prioritize strict typing for exchange/order/risk/repository modules first.

8. **`requirements.txt` contains duplicate packages and drift vs `pyproject.toml`**
   - Evidence: duplicate `PySide6` and `psutil`; dependency sets differ from `[project.dependencies]`.
   - Impact: inconsistent deployments.
   - Locations: `requirements.txt`, `pyproject.toml`.
   - Safe patch: unify dependency source-of-truth and regenerate lock/install artifacts.

### Low severity
9. **EventBus and EngineScheduler appear underutilized in runtime orchestration**
   - Evidence: main engine loop directly orchestrates tasks; dedicated bus/scheduler not integrated.
   - Impact: architectural complexity without active value.
   - Locations: `reco_trading/core/event_bus.py`, `reco_trading/core/scheduler.py`.
   - Safe patch: either integrate fully or deprecate to reduce coupling surface.

10. **Broad `except Exception` blocks in UI rendering and engine sync paths**
   - Evidence: multiple broad catches without typed branches.
   - Impact: difficult root-cause analysis.
   - Locations: `reco_trading/core/bot_engine.py`, `reco_trading/ui/dashboard.py`, others.
   - Safe patch: narrow exception classes for expected failures; keep top-level guard only.

---

## Phase 2 — Architecture audit

### Pipeline integrity (Signal → Confidence → Risk → Execution → Position → Performance)
- Implemented in `BotEngine` in expected sequence:
  1. signal generation (`SignalEngine.generate`)
  2. confidence scoring (`ConfidenceModel.evaluate`)
  3. risk validation (`RiskManager.validate` + `AdvancedRiskManager.evaluate`)
  4. execution (`BinanceClient.create_market_order`)
  5. local position tracking (`PositionManager` open/close)
  6. persistence/performance updates (`Repository` trade writes + snapshot win rate/session PnL)
- Strength: clear top-level control flow and gating.
- Weaknesses:
  - exchange reconciliation is not first-class (local state is optimistic)
  - eventing abstractions exist but core flow is monolithic in one loop
  - DB methods with fallback defaults may hide failed pipeline legs

### Coupling/circular architecture
- Coarse static scan found no direct internal import cycles.
- Coupling is moderate/high around `BotEngine` as orchestration and policy concentration point.

---

## Phase 3 — Trading execution audit (ccxt/Binance)

### Positive controls present
- Rate limit support via `enableRateLimit=True`.
- Time drift handling via `sync_time` and `adjustForTimeDifference` option.
- Retry with exponential backoff for network/timeouts.
- Symbol filter normalization:
  - LOT_SIZE (`minQty`, `stepSize`)
  - MIN_NOTIONAL (`minNotional`)
  - PRICE_FILTER (`tickSize`)

### Risks
1. **No explicit cancel path for stale intents** (market orders usually immediate, but failures can leave ambiguity).
2. **Duplicate order risk on uncertain response retries** (especially close flow retries).
3. **Insufficient-balance handling delegated to generic exchange errors; no explicit balance refresh branch before reattempt.**
4. **No idempotency persistence table for `clientOrderId` intents across restarts.**

Safe patch set:
- Persist intent lifecycle (`NEW`/`ACK`/`FILLED`/`FAILED`) keyed by `clientOrderId`.
- After exceptions on create/close, query by clientOrderId before retry.
- Add explicit branch for `InsufficientFunds` with immediate risk pause and UI alert.

---

## Phase 4 — Position management audit

### Findings
- SL/TP/trailing-stop logic exists and is deterministic in `PositionManager.check_exit`.
- Trailing stop activation tied to ATR/risk distance; reasonable baseline.

### Risks
1. **Partial fills not modeled** (single quantity scalar only).
2. **No startup reconciliation against exchange open orders/trades before managing local positions.**
3. **Manual force-close can repeatedly retry without exchange state reconciliation, causing possible double exits.**

Safe patch set:
- Extend `Position` with filled/remaining and exchange order references.
- Add `reconcile_positions()` at startup and periodically.
- Protect close logic with fill-based position decrement semantics.

---

## Phase 5 — Database layer audit

### Findings
- Async SQLAlchemy usage is structurally correct (`AsyncSession`, commits, async engine disposal).
- Schema migration helpers present for incremental columns.

### Risks
1. **`safe_db_call` hides failures by default-return behavior.**
2. **No explicit transaction bundling for multi-step atomic operations around trade+state updates.**
3. **Potential timezone-naive/aware mismatch handling complexity (`replace(tzinfo=None)` approach).**

Safe patch set:
- Add explicit transactional methods for critical write groups.
- Emit error counters/health metrics when fallback paths trigger.
- Normalize timezone strategy to UTC-aware end-to-end.

---

## Phase 6 — UI dashboard audit

### Findings
- UI launch is isolated from bot thread; GUI failure does not kill bot startup path.
- `StateManager` uses lock-protected state snapshots.

### Risks
1. **Cross-thread Qt signal emissions from bot thread (see high severity #2).**
2. **Potential high-frequency state emission causing UI churn; no adaptive throttling.**
3. **Optional plotting fallback uses broad exception and weak typing in chart modules.**

Safe patch set:
- queue + periodic UI-thread drain timer.
- coalesce state updates (e.g., 100–250ms cadence).
- tighten chart typing and explicit import-failure diagnostics.

---

## Phase 7 — Risk management audit

### Findings
- Position sizing based on risk fraction and stop distance is present.
- Daily loss, max trades/day, drawdown and consecutive-loss gates exist.
- Exchange failure circuit-breaker exists (`exchange_failure_count` + cooldown pause).

### Kill switch status
- **Present (partial):** emergency stop and pause mechanics exist, plus exchange circuit breaker.
- **Gap:** no single explicit global “hard kill switch” flag persisted and checked before any order submission.

Safe patch:
- add persistent `hard_kill_switch` runtime setting and pre-order guard in `execute_trade` and `_close_position` override logic.

---

## Phase 8 — Logging and diagnostics audit

### Findings
- Logs include major state transitions, trade events, risk rejections, exchange/runtime errors.
- Repository persists logs/errors/state changes.

### Gaps
- slippage and normalization events logged, but order intent lifecycle and reconciliation details are limited.
- DB fallback suppression can reduce diagnostics fidelity.

Safe patch:
- structured JSON-like log envelopes for each order lifecycle phase.
- add reconciliation logs (expected qty vs filled qty vs remaining).

---

## Phase 9 — Automated tool execution results

- `pip install -r requirements.txt flake8 mypy pylint` → failed due proxy/index restrictions and missing distribution resolution for `flake8` in this environment.
- `mypy reco_trading tests` → **70 errors in 27 files** (notably type issues in order manager, market intelligence, UI typing, optional dependencies stubs).
- `pytest -q` → **24 passed, 2 skipped**.
- `python -m reco_trading.main` → fails with missing `pydantic`.
- `python -m compileall -q .` → passes (no syntax errors).

---

## Phase 10 — Simulated runtime test

- Full live simulation requiring exchange creds/network not performed safely in this environment.
- Proxy simulation via test suite indicates core units around scheduler, risk controls, normalization, market intelligence, UI bootstrap are currently passing.
- Residual runtime risk remains around exchange uncertainty, partial fills, and reconciliation as noted above.

---

## Phase 11 — Performance analysis

Potential bottlenecks:
1. Repeated dataframe indicator computations (mitigated partly by timestamp cache in engine).
2. Frequent deep-copy emissions in UI `StateManager` for every update/trade/log.
3. Sync `ccxt` calls wrapped in threads can add contention under high frequency.
4. DB per-event commits (`_persist`) may be chatty under burst conditions.

Optimizations:
- batch/queue DB log writes.
- coalesce UI updates and reduce deepcopy scope.
- move from polling-only market data to websocket incremental updates where feasible.

---

## Phase 12 — Production readiness

### Consolidated risk summary
- **Critical blocker:** dependency mismatch prevents clean bootstrap in baseline requirements installation.
- **Major risks before real capital:** no robust exchange reconciliation/idempotency for ambiguous execution outcomes; partial-fill handling incomplete; cross-thread UI signal safety concerns.

### Production readiness score
- **63 / 100**

### Priority fixes before real-money deployment
1. Fix dependency packaging (`requirements.txt` parity with runtime imports).
2. Implement idempotent order-intent persistence + reconciliation-by-clientOrderId.
3. Add partial fill lifecycle handling and position reconciliation loop.
4. Harden UI thread marshaling with queued main-thread dispatch.
5. Tighten DB error escalation and add health metrics for fallback paths.
6. Reduce mypy error baseline in trading-critical modules to near-zero.

