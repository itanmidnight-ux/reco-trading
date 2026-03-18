const loginCard = document.getElementById('login-card');
const dashboard = document.getElementById('dashboard');
const loginForm = document.getElementById('login-form');
const loginError = document.getElementById('login-error');
const sessionUser = document.getElementById('session-user');
const actionFeedback = document.getElementById('action-feedback');
const connectionStatus = document.getElementById('connection-status');
const logsText = document.getElementById('logs-text');

let csrfToken = '';
let refreshTimer = null;
const logs = [];

const pushLog = (message) => {
  const stamp = new Date().toLocaleTimeString('es-ES');
  logs.unshift(`${stamp} · ${message}`);
  logs.splice(0, 30);
  logsText.innerHTML = logs.map((line) => `<li>${line}</li>`).join('');
};

const toNumber = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const firstNumber = (...values) => values.map(toNumber).find((value) => value !== null) ?? null;
const fmtText = (value) => (value === null || value === undefined || value === '' ? 'N/D' : String(value));
const fmtNumber = (value, decimals = 2) => {
  const numeric = toNumber(value);
  if (numeric === null) return 'N/D';
  return new Intl.NumberFormat('es-ES', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(numeric);
};
const fmtCurrency = (value) => {
  const numeric = toNumber(value);
  if (numeric === null) return 'N/D';
  return new Intl.NumberFormat('es-ES', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(numeric);
};
const fmtPercent = (value) => {
  const numeric = toNumber(value);
  if (numeric === null) return 'N/D';
  const normalized = numeric > 1 ? numeric / 100 : numeric;
  return new Intl.NumberFormat('es-ES', { style: 'percent', maximumFractionDigits: 2 }).format(normalized);
};
const fmtDuration = (seconds) => {
  const numeric = toNumber(seconds);
  if (numeric === null) return 'N/D';
  const total = Math.max(Math.floor(numeric), 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h}h ${m}m ${s}s`;
};

async function api(path, options = {}) {
  const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (csrfToken) headers['X-CSRF-Token'] = csrfToken;
  const response = await fetch(path, { ...options, headers, credentials: 'include' });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || data.error || 'Error en solicitud');
  return data;
}

function setActiveTab(tabName) {
  document.querySelectorAll('#tabs-nav button').forEach((button) => {
    button.classList.toggle('active', button.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `tab-${tabName}`);
  });
}

function renderGrid(elementId, pairs) {
  document.getElementById(elementId).innerHTML = pairs
    .map(({ label, value }) => `<div class="metric-item"><dt>${label}</dt><dd>${value}</dd></div>`)
    .join('');
}

function renderCards(model) {
  const cards = [
    { title: 'Capital USDT', value: fmtCurrency(model.balance), tone: 'neutral' },
    { title: 'Equity', value: fmtCurrency(model.equity), tone: 'neutral' },
    { title: 'PnL Diario', value: fmtCurrency(model.dailyPnl), tone: model.dailyPnl >= 0 ? 'good' : 'danger' },
    { title: 'PnL Sesión', value: fmtCurrency(model.sessionPnl), tone: model.sessionPnl >= 0 ? 'good' : 'danger' },
    { title: 'Win Rate', value: fmtPercent(model.winRate), tone: 'neutral' },
    { title: 'Estado Bot', value: fmtText(model.botStatus), tone: model.botStatus === 'ERROR' ? 'danger' : 'good' },
  ];

  document.getElementById('summary-cards').innerHTML = cards
    .map((card) => `
      <article class="card stat-card tone-${card.tone}">
        <p class="stat-title">${card.title}</p>
        <p class="stat-value">${card.value}</p>
      </article>
    `)
    .join('');
}

function normalizeData(data) {
  const snapshot = data.snapshot || data.runtime?.snapshot || {};
  const metrics = data.metrics || {};
  const health = data.health || {};
  const runtime = data.runtime || {};
  const settings = data.settings || {};
  const positions = data.positions || {};

  return {
    snapshot,
    settings,
    positions,
    health,
    runtime,
    botStatus: fmtText(health.bot_status || runtime.bot_status || metrics.status),
    balance: firstNumber(metrics.balance, snapshot.balance),
    equity: firstNumber(metrics.equity, metrics.total_equity, snapshot.equity, snapshot.total_equity),
    dailyPnl: firstNumber(metrics.daily_pnl, snapshot.daily_pnl),
    sessionPnl: firstNumber(metrics.session_pnl, snapshot.session_pnl),
    winRate: firstNumber(metrics.win_rate, snapshot.win_rate),
    tradesToday: firstNumber(metrics.trades_today, snapshot.trades_today),
    price: firstNumber(metrics.price, snapshot.price),
    bid: firstNumber(metrics.bid, snapshot.bid),
    ask: firstNumber(metrics.ask, snapshot.ask),
    spread: firstNumber(metrics.spread, snapshot.spread),
    confidence: firstNumber(metrics.confidence, snapshot.confidence),
    signal: snapshot.signal || metrics.signal,
  };
}

function renderData(data) {
  const model = normalizeData(data);
  renderCards(model);

  renderGrid('health-grid', [
    { label: 'Estado runtime', value: model.botStatus },
    { label: 'Uptime', value: fmtDuration(model.health.uptime_seconds) },
    { label: 'Heartbeat age', value: `${fmtNumber(model.health.heartbeat_age_seconds, 1)} s` },
    { label: 'Reinicios', value: fmtNumber(model.health.restart_count ?? 0, 0) },
    { label: 'Posiciones abiertas', value: fmtNumber(model.health.open_positions ?? 0, 0) },
  ]);

  renderGrid('market-grid', [
    { label: 'Símbolo', value: fmtText(model.positions.symbol || model.snapshot.pair || model.settings.symbol) },
    { label: 'Precio', value: fmtCurrency(model.price) },
    { label: 'Bid / Ask', value: `${fmtCurrency(model.bid)} / ${fmtCurrency(model.ask)}` },
    { label: 'Spread', value: fmtNumber(model.spread, 6) },
    { label: 'Señal', value: fmtText(model.signal) },
    { label: 'Confianza', value: fmtPercent(model.confidence) },
  ]);

  renderGrid('position-grid', [
    { label: 'Hay posición', value: model.positions.has_open_position ? 'Sí' : 'No' },
    { label: 'Detalle posición', value: fmtText(model.positions.open_position || model.snapshot.open_position || 'Sin posición activa') },
    { label: 'Último trade', value: fmtText(model.snapshot.last_trade || '-') },
    { label: 'Cooldown', value: fmtText(model.snapshot.cooldown || 'READY') },
  ]);

  renderGrid('analytics-grid', [
    { label: 'Capital USDT', value: fmtCurrency(model.balance) },
    { label: 'Equity total', value: fmtCurrency(model.equity) },
    { label: 'PnL diario', value: fmtCurrency(model.dailyPnl) },
    { label: 'PnL sesión', value: fmtCurrency(model.sessionPnl) },
    { label: 'Win rate', value: fmtPercent(model.winRate) },
    { label: 'Trades hoy', value: fmtNumber(model.tradesToday ?? 0, 0) },
    { label: 'BTC balance', value: fmtNumber(model.snapshot.btc_balance, 8) },
    { label: 'Valor BTC', value: fmtCurrency(model.snapshot.btc_value) },
  ]);

  renderGrid('risk-grid', [
    { label: 'Riesgo por trade', value: fmtPercent(model.settings.risk_per_trade_fraction) },
    { label: 'Máx. balance por trade', value: fmtPercent(model.settings.max_trade_balance_fraction) },
    { label: 'Límite pérdida diaria', value: fmtPercent(model.settings.daily_loss_limit_fraction) },
    { label: 'Máx. drawdown', value: fmtPercent(model.settings.max_drawdown_fraction) },
    { label: 'Modo inversión', value: fmtText(model.snapshot.investment_mode || 'Balanced') },
    { label: 'Capital limit', value: fmtCurrency(model.snapshot.capital_limit_usdt) },
  ]);

  renderGrid('system-grid', [
    { label: 'Perfil runtime', value: fmtText(model.settings.runtime_profile) },
    { label: 'Entorno', value: fmtText(model.settings.environment) },
    { label: 'Timeframe', value: fmtText(model.settings.timeframe || model.snapshot.timeframe) },
    { label: 'Kill switch', value: model.runtime.kill_switch ? 'ACTIVO' : 'Inactivo' },
    { label: 'Pausa manual', value: model.runtime.manual_pause ? 'Sí' : 'No' },
    { label: 'Exchange pause', value: fmtText(model.runtime.exchange_pause_until || '-') },
  ]);

  document.getElementById('risk_per_trade_fraction').value = model.settings.risk_per_trade_fraction ?? 0.01;
  document.getElementById('max_trade_balance_fraction').value = model.settings.max_trade_balance_fraction ?? 0.2;

  pushLog('Dashboard actualizado correctamente.');
}

async function refreshData() {
  try {
    const data = await api('/dashboard/data');
    connectionStatus.textContent = 'Conectado';
    connectionStatus.className = 'badge badge-good';
    renderData(data);
  } catch (error) {
    connectionStatus.textContent = 'Sin conexión';
    connectionStatus.className = 'badge badge-danger';
    pushLog(`Error actualizando datos: ${error.message}`);
    throw error;
  }
}

async function bootstrap() {
  try {
    const session = await api('/dashboard/session');
    csrfToken = session.csrf_token;
    sessionUser.textContent = `Usuario: ${session.username}`;
    loginCard.classList.add('hidden');
    dashboard.classList.remove('hidden');
    setActiveTab('analytics');
    await refreshData();
    if (refreshTimer) clearInterval(refreshTimer);
    refreshTimer = setInterval(() => {
      refreshData().catch(() => null);
    }, 5000);
  } catch {
    loginCard.classList.remove('hidden');
    dashboard.classList.add('hidden');
  }
}

loginForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  loginError.textContent = '';
  try {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const result = await api('/dashboard/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    csrfToken = result.csrf_token;
    pushLog('Sesión iniciada correctamente.');
    await bootstrap();
  } catch (error) {
    loginError.textContent = error.message;
    pushLog(`Error de login: ${error.message}`);
  }
});

document.getElementById('logout-btn').addEventListener('click', async () => {
  await api('/dashboard/logout', { method: 'POST' });
  location.reload();
});

document.querySelectorAll('#tabs-nav button').forEach((button) => {
  button.addEventListener('click', () => setActiveTab(button.dataset.tab));
});

document.querySelectorAll('[data-action]').forEach((button) => {
  button.addEventListener('click', async () => {
    const action = button.dataset.action;
    try {
      if (action === 'close') {
        const data = await api('/dashboard/data');
        const symbol = data.positions?.symbol || data.snapshot?.pair || 'BTCUSDT';
        await api('/close-position', { method: 'POST', body: JSON.stringify({ symbol }) });
      } else if (action === 'kill') {
        await api('/kill-switch', { method: 'POST' });
      } else {
        await api(`/${action}`, { method: 'POST' });
      }
      actionFeedback.textContent = `Acción ejecutada: ${action}.`;
      pushLog(`Acción ejecutada: ${action}.`);
      await refreshData();
    } catch (error) {
      actionFeedback.textContent = `Error en ${action}: ${error.message}`;
      pushLog(`Error en ${action}: ${error.message}`);
    }
  });
});

document.getElementById('runtime-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = {
    investment_mode: document.getElementById('investment_mode').value,
    risk_per_trade_fraction: Number(document.getElementById('risk_per_trade_fraction').value),
    max_trade_balance_fraction: Number(document.getElementById('max_trade_balance_fraction').value),
    capital_limit_usdt: Number(document.getElementById('capital_limit_usdt').value),
    symbol_capital_limits: {},
  };
  try {
    await api('/runtime-settings', { method: 'POST', body: JSON.stringify(payload) });
    actionFeedback.textContent = 'Configuración aplicada.';
    pushLog('Configuración runtime actualizada.');
    await refreshData();
  } catch (error) {
    actionFeedback.textContent = `No se pudo aplicar: ${error.message}`;
    pushLog(`Error de configuración: ${error.message}`);
  }
});

bootstrap();
