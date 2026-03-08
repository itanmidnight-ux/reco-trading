let equityChart;
let tradesTable;

const fmt = (v, d = 4) => Number(v ?? 0).toFixed(d);
const money = (v) => `${fmt(v, 2)} USDT`;
const pct = (v) => `${(Number(v ?? 0) * 100).toFixed(2)}%`;

const fmtTs = (ts) => {
  const n = Number(ts ?? 0);
  const ms = n < 10000000000 ? n * 1000 : n;
  return new Date(ms).toLocaleString();
};

function applyClass(el, value) {
  el.classList.remove('positive', 'negative');
  el.classList.add(Number(value) >= 0 ? 'positive' : 'negative');
}

function updateMetrics(data) {
  document.getElementById('capital').textContent = money(data.capital_real_usdt ?? data.capital);
  document.getElementById('account-equity').textContent = money(data.account_equity_usdt ?? data.capital);

  const pnlTotal = document.getElementById('pnl-total');
  pnlTotal.textContent = money(data.pnl_total);
  applyClass(pnlTotal, data.pnl_total);

  const pnlDaily = document.getElementById('pnl-daily');
  pnlDaily.textContent = money(data.pnl_daily);
  applyClass(pnlDaily, data.pnl_daily);

  document.getElementById('drawdown').textContent = money(data.drawdown);
  document.getElementById('win-rate').textContent = pct(data.win_rate);
  document.getElementById('sharpe').textContent = fmt(data.sharpe);
  document.getElementById('signal').textContent = data.signal ?? '-';
  document.getElementById('regime').textContent = data.regime ?? '-';
  document.getElementById('risk-active').textContent = data.risk_active ? 'ON' : 'OFF';
  document.getElementById('exposure').textContent = money(data.active_exposure);
  document.getElementById('latest-price').textContent = money(data.latest_price);
  document.getElementById('binance-status').textContent = data.binance_status ?? 'unknown';
  document.getElementById('latency').textContent = `${fmt(data.latency_ms, 0)} ms`;
  document.getElementById('last-update').textContent = fmtTs(data.server_ts ?? Date.now());
}

function renderChart(points) {
  const labels = points.map((p) => fmtTs(p.timestamp));
  const values = points.map((p) => p.equity);

  if (!equityChart) {
    const ctx = document.getElementById('equity-chart');
    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{ label: 'Equity real (USDT)', data: values, borderColor: '#6ea8ff', tension: 0.25, fill: false }],
      },
      options: { responsive: true, maintainAspectRatio: false },
    });
  } else {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.update('none');
  }
}

function renderActivity(activity) {
  const ul = document.getElementById('activity-feed');
  ul.innerHTML = '';
  for (const item of activity.slice(0, 80)) {
    const li = document.createElement('li');
    li.className = `activity-${item.type || 'info'}`;
    li.innerHTML = `<div class="activity-top"><strong>${item.title}</strong><span>${fmtTs(item.ts)}</span></div><div class="activity-detail">${item.detail}</div>`;
    ul.appendChild(li);
  }
}

async function loadInitialData() {
  const [dataRes, tradesRes, equityRes, activityRes] = await Promise.all([
    fetch('/dashboard/data'),
    fetch('/dashboard/trades'),
    fetch('/dashboard/equity'),
    fetch('/dashboard/activity'),
  ]);

  const data = await dataRes.json();
  const trades = (await tradesRes.json()).trades;
  const equity = (await equityRes.json()).equity;
  const activity = (await activityRes.json()).activity;

  updateMetrics(data);
  renderChart(equity);
  renderActivity(activity);

  if (!tradesTable) {
    tradesTable = $('#trades-table').DataTable({ data: trades, columns: [
      { data: 'ts', render: (v) => fmtTs(v) }, { data: 'symbol' }, { data: 'side' }, { data: 'qty' },
      { data: 'price' }, { data: 'status' }, { data: 'pnl' },
    ] });
  } else {
    tradesTable.clear().rows.add(trades).draw(false);
  }
}

function connectWebsocket() {
  const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/dashboard`);
  ws.onmessage = async (event) => {
    const payload = JSON.parse(event.data);
    updateMetrics(payload);
    if (payload.equity_curve) {
      renderChart(payload.equity_curve);
    }
    const [tradesRes, activityRes] = await Promise.all([
      fetch('/dashboard/trades'),
      fetch('/dashboard/activity'),
    ]);
    const trades = (await tradesRes.json()).trades;
    const activity = (await activityRes.json()).activity;
    tradesTable.clear().rows.add(trades).draw(false);
    renderActivity(activity);
  };
  ws.onclose = () => setTimeout(connectWebsocket, 1500);
}

window.addEventListener('DOMContentLoaded', async () => {
  await loadInitialData();
  connectWebsocket();
});
