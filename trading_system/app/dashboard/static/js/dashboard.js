let equityChart;
let tradesTable;

const fmt = (v, d = 4) => Number(v ?? 0).toFixed(d);
const pct = (v) => `${(Number(v ?? 0) * 100).toFixed(2)}%`;

function applyClass(el, value) {
  el.classList.remove('positive', 'negative');
  el.classList.add(Number(value) >= 0 ? 'positive' : 'negative');
}

function updateMetrics(data) {
  document.getElementById('capital').textContent = fmt(data.capital);
  const pnlTotal = document.getElementById('pnl-total');
  pnlTotal.textContent = fmt(data.pnl_total);
  applyClass(pnlTotal, data.pnl_total);

  const pnlDaily = document.getElementById('pnl-daily');
  pnlDaily.textContent = fmt(data.pnl_daily);
  applyClass(pnlDaily, data.pnl_daily);

  document.getElementById('drawdown').textContent = pct(data.drawdown);
  document.getElementById('win-rate').textContent = pct(data.win_rate);
  document.getElementById('sharpe').textContent = fmt(data.sharpe);
  document.getElementById('signal').textContent = data.signal ?? '-';
  document.getElementById('regime').textContent = data.regime ?? '-';
  document.getElementById('risk-active').textContent = data.risk_active ? 'ON' : 'OFF';
  document.getElementById('exposure').textContent = fmt(data.active_exposure);
  document.getElementById('latest-price').textContent = fmt(data.latest_price, 2);
  document.getElementById('binance-status').textContent = data.binance_status ?? 'unknown';
  document.getElementById('latency').textContent = `${fmt(data.latency_ms, 0)} ms`;
}

function renderChart(points) {
  const labels = points.map((p) => new Date(p.timestamp).toLocaleTimeString());
  const values = points.map((p) => p.equity);

  if (!equityChart) {
    const ctx = document.getElementById('equity-chart');
    equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{ label: 'Equity', data: values, borderColor: '#0ecb81', tension: 0.2 }],
      },
      options: { responsive: true, maintainAspectRatio: false },
    });
  } else {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.update('none');
  }
}

async function loadInitialData() {
  const [dataRes, tradesRes, equityRes] = await Promise.all([
    fetch('/dashboard/data'),
    fetch('/dashboard/trades'),
    fetch('/dashboard/equity'),
  ]);

  const data = await dataRes.json();
  const trades = (await tradesRes.json()).trades;
  const equity = (await equityRes.json()).equity;

  updateMetrics(data);
  renderChart(equity);

  if (!tradesTable) {
    tradesTable = $('#trades-table').DataTable({ data: trades, columns: [
      { data: 'ts' }, { data: 'symbol' }, { data: 'side' }, { data: 'qty' },
      { data: 'price' }, { data: 'status' }, { data: 'pnl' },
    ] });
  } else {
    tradesTable.clear().rows.add(trades).draw(false);
  }
}

function connectWebsocket() {
  const ws = new WebSocket('ws://localhost:8000/ws/dashboard');
  ws.onmessage = async (event) => {
    const payload = JSON.parse(event.data);
    updateMetrics(payload);
    if (payload.equity_curve) {
      renderChart(payload.equity_curve);
    }
    const tradesRes = await fetch('/dashboard/trades');
    const trades = (await tradesRes.json()).trades;
    tradesTable.clear().rows.add(trades).draw(false);
  };
  ws.onclose = () => setTimeout(connectWebsocket, 1500);
}

window.addEventListener('DOMContentLoaded', async () => {
  await loadInitialData();
  connectWebsocket();
});
