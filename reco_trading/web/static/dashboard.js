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
  logs.splice(8);
  logsText.innerHTML = logs.map((line) => `<li>${line}</li>`).join('');
};

const safeNumber = (value) => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const fmt = (value) => (value === null || value === undefined ? 'N/D' : String(value));
const bin = (value) => {
  const numeric = safeNumber(value);
  if (numeric === null) return 'N/A';
  return Math.trunc(numeric).toString(2);
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

function renderList(elementId, lines) {
  document.getElementById(elementId).innerHTML = lines.map((line) => `<li>${line}</li>`).join('');
}

function renderCards(data) {
  const metrics = data.metrics || {};
  const health = data.health || {};
  const positions = data.positions || {};

  const cards = [
    { title: 'Estado bot', value: fmt(health.bot_status), meta: 'Texto plano' },
    { title: 'Balance', value: fmt(metrics.balance), meta: `Binario: ${bin(metrics.balance)}` },
    { title: 'Equity', value: fmt(metrics.equity), meta: `Binario: ${bin(metrics.equity)}` },
    { title: 'PNL diario', value: fmt(metrics.daily_pnl), meta: `Binario: ${bin(metrics.daily_pnl)}` },
    { title: 'Win rate', value: fmt(metrics.win_rate), meta: `Binario: ${bin(metrics.win_rate)}` },
    { title: 'Símbolo activo', value: fmt(positions.symbol || 'Sin posición'), meta: 'Texto plano' },
  ];

  document.getElementById('summary-cards').innerHTML = cards
    .map((card) => `
      <article class="glass stat-card">
        <div class="stat-title">${card.title}</div>
        <div class="stat-value">${card.value}</div>
        <div class="stat-meta">${card.meta}</div>
      </article>
    `)
    .join('');
}

function renderData(data) {
  const health = data.health || {};
  const metrics = data.metrics || {};
  const positions = data.positions || {};
  const runtime = data.runtime || {};
  const settings = data.settings || {};

  renderCards(data);

  renderList('summary-text', [
    `Estado general del bot: ${fmt(health.bot_status)}.`,
    `Tiempo activo del proceso: ${fmt(health.uptime_seconds)} segundos (${bin(health.uptime_seconds)} en binario).`,
    `Posiciones abiertas detectadas: ${fmt(health.open_positions)} (${bin(health.open_positions)} en binario).`,
    `Latencia de heartbeat: ${fmt(health.heartbeat_age_seconds)} segundos.`,
  ]);

  renderList('market-text', [
    `Símbolo principal monitoreado: ${fmt(settings.symbol)}.`,
    `Marco temporal operativo: ${fmt(settings.timeframe)}.`,
    `Estado de snapshot del mercado: ${fmt(metrics.status)}.`,
    `Perfil de ejecución: ${fmt(settings.runtime_profile)} en entorno ${fmt(settings.environment)}.`,
  ]);

  renderList('analytics-text', [
    `Balance actual: ${fmt(metrics.balance)} (binario: ${bin(metrics.balance)}).`,
    `Capital de equity: ${fmt(metrics.equity)} (binario: ${bin(metrics.equity)}).`,
    `PNL diario acumulado: ${fmt(metrics.daily_pnl)} (binario: ${bin(metrics.daily_pnl)}).`,
    `Tasa de aciertos: ${fmt(metrics.win_rate)} (binario entero: ${bin(metrics.win_rate)}).`,
  ]);

  renderList('trades-text', [
    `¿Existe posición abierta?: ${positions.has_open_position ? 'Sí, existe una posición en mercado.' : 'No hay posición abierta actualmente.'}`,
    `Par asociado a la posición: ${fmt(positions.symbol || 'Sin símbolo activo')}.`,
    `Detalle de posición: ${fmt(positions.open_position || 'No disponible en snapshot')}.`,
  ]);

  renderList('risk-text', [
    `Riesgo por operación: ${fmt(settings.risk_per_trade_fraction)} (binario: ${bin(settings.risk_per_trade_fraction)}).`,
    `Máximo balance por operación: ${fmt(settings.max_trade_balance_fraction)} (binario: ${bin(settings.max_trade_balance_fraction)}).`,
    `Límite de pérdida diaria: ${fmt(settings.daily_loss_limit_fraction)}.`,
    `Máximo drawdown permitido: ${fmt(settings.max_drawdown_fraction)}.`,
  ]);

  renderList('settings-text', [
    `Perfil de runtime activo: ${fmt(settings.runtime_profile)}.`,
    `Entorno configurado: ${fmt(settings.environment)}.`,
    `Timeframe consolidado: ${fmt(settings.timeframe)}.`,
  ]);

  renderList('system-text', [
    `Reinicios registrados: ${fmt(health.restart_count)} (${bin(health.restart_count)} en binario).`,
    `Estado principal del runtime: ${fmt(runtime.bot_status)}.`,
    `Interruptor de seguridad (kill switch): ${runtime.kill_switch ? 'ACTIVO' : 'No activado'}.`,
  ]);

  document.getElementById('risk_per_trade_fraction').value = fmt(settings.risk_per_trade_fraction ?? 0.01);
  document.getElementById('max_trade_balance_fraction').value = fmt(settings.max_trade_balance_fraction ?? 0.2);

  pushLog('Datos del dashboard actualizados con éxito.');
}

async function refreshData() {
  try {
    const data = await api('/dashboard/data');
    connectionStatus.textContent = 'Conexión estable y segura';
    renderData(data);
  } catch (error) {
    connectionStatus.textContent = 'Conexión inestable (reintentando)';
    pushLog(`Fallo de actualización: ${error.message}`);
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
    setActiveTab('dashboard');
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
    pushLog('Inicio de sesión válido.');
    await bootstrap();
  } catch (error) {
    loginError.textContent = error.message;
    pushLog(`Intento de login fallido: ${error.message}`);
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
        const symbol = data.positions?.symbol || 'BTCUSDT';
        await api('/close-position', { method: 'POST', body: JSON.stringify({ symbol }) });
      } else if (action === 'kill') {
        await api('/kill-switch', { method: 'POST' });
      } else {
        await api(`/${action}`, { method: 'POST' });
      }
      actionFeedback.textContent = `Acción ejecutada correctamente: ${action}.`;
      pushLog(`Acción enviada: ${action}.`);
      await refreshData();
    } catch (error) {
      actionFeedback.textContent = `Error ejecutando ${action}: ${error.message}`;
      pushLog(`Error en acción ${action}: ${error.message}`);
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
    actionFeedback.textContent = 'Configuración aplicada al runtime.';
    pushLog('Configuración runtime actualizada.');
  } catch (error) {
    actionFeedback.textContent = `No se pudo aplicar configuración: ${error.message}`;
    pushLog(`Error de configuración: ${error.message}`);
  }
});

bootstrap();
