// State
let mainChart = null;
let candleSeries = null;
let heroChart = null;
let heroLineSeries = null;
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT = 10;
let currentPair = 'BTC/USDT';
let currentTab = 'overview';
let currentTimeframe = '5m';
let tradesData = [];
let chartInitialized = false;
let heroChartInitialized = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    initTabs();
    initControls();
    initPairSelector();
    initSliders();
    connectWebSocket();
    refreshData();
    setInterval(refreshData, 5000);
});

function checkAuth() {
    const token = localStorage.getItem('auth_token');
    if (!token) {
        window.location.href = '/login';
        return;
    }
    validateToken(token);
}

async function validateToken(token) {
    try {
        const r = await fetch('/api/health', {
            headers: { 'Authorization': 'Bearer ' + token }
        });
        if (r.status === 401) {
            localStorage.removeItem('auth_token');
            window.location.href = '/login';
        }
    } catch (e) {
        console.error('Auth check failed:', e);
    }
}

function getHeaders() {
    return {
        'Authorization': 'Bearer ' + localStorage.getItem('auth_token'),
        'Content-Type': 'application/json'
    };
}

function initTabs() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    currentTab = tabId;
    
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    
    document.querySelector(`.nav-tab[data-tab="${tabId}"]`).classList.add('active');
    document.getElementById(`tab-${tabId}`).classList.add('active');
    
    if (tabId === 'trades') loadTrades();
    if (tabId === 'chart') initMainChart();
    if (tabId === 'ml') loadMLData();
    if (tabId === 'risk') loadRiskData();
    if (tabId === 'strategy') loadStrategyData();
    if (tabId === 'overview') initHeroChart();
}

function initControls() {
    document.getElementById('pairBtn').addEventListener('click', () => {
        document.getElementById('pairDropdown').classList.toggle('active');
    });
    
    document.getElementById('refreshBtn').addEventListener('click', () => {
        refreshData();
        showToast('Refreshing data...', 'success');
    });
    
    document.getElementById('logoutBtn').addEventListener('click', logout);
    
    document.querySelectorAll('.tf-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTimeframe = btn.dataset.tf;
            loadChartData();
        });
    });
    
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadTrades(btn.dataset.filter);
        });
    });
    
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadPerformance(btn.dataset.period);
        });
    });
    
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.pair-selector')) {
            document.getElementById('pairDropdown').classList.remove('active');
        }
    });
}

function initPairSelector() {
    const pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT', 'DOGE/USDT', 'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT'];
    const list = document.getElementById('pairList');
    
    pairs.forEach(pair => {
        const item = document.createElement('div');
        item.className = 'pair-item' + (pair === currentPair ? ' active' : '');
        item.textContent = pair;
        item.addEventListener('click', () => selectPair(pair));
        list.appendChild(item);
    });
    
    document.getElementById('pairSearch').addEventListener('input', (e) => {
        const search = e.target.value.toUpperCase();
        document.querySelectorAll('.pair-item').forEach(item => {
            item.style.display = item.textContent.toUpperCase().includes(search) ? '' : 'none';
        });
    });
}

function selectPair(pair) {
    currentPair = pair;
    document.getElementById('currentPair').textContent = pair;
    document.querySelectorAll('.pair-item').forEach(item => {
        item.classList.toggle('active', item.textContent === pair);
    });
    document.getElementById('pairDropdown').classList.remove('active');
    updatePair(pair);
}

async function updatePair(pair) {
    try {
        await fetch('/api/settings/pair', {
            method: 'POST',
            headers: getHeaders(),
            body: JSON.stringify({ symbol: pair })
        });
        showToast(`Switched to ${pair}`, 'success');
        refreshData();
    } catch (e) {
        showToast('Failed to switch pair', 'error');
    }
}

function initSliders() {
    document.querySelectorAll('.setting-slider').forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + '_value');
        if (valueDisplay) {
            slider.addEventListener('input', () => {
                const val = slider.value;
                let text = val;
                if (slider.id.includes('Risk') || slider.id.includes('Loss') || slider.id.includes('Drawdown') || slider.id.includes('Confidence')) {
                    text = val + '%';
                } else if (slider.id.includes('Atr')) {
                    text = val + 'x ATR';
                } else if (slider.id.includes('trailAct') || slider.id.includes('beTrigger') || slider.id.includes('profitLock')) {
                    text = val + 'R';
                }
                valueDisplay.textContent = text;
            });
        }
    });
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            reconnectAttempts = 0;
            updateConnectionStatus(true);
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleUpdate(data);
            } catch (e) {
                console.error('Parse error:', e);
            }
        };
        
        ws.onerror = (e) => {
            console.error('WebSocket error:', e);
        };
        
        ws.onclose = () => {
            updateConnectionStatus(false);
            if (reconnectAttempts < MAX_RECONNECT) {
                reconnectAttempts++;
                setTimeout(connectWebSocket, 3000 * reconnectAttempts);
            }
        };
    } catch (e) {
        console.error('WebSocket init error:', e);
    }
}

function updateConnectionStatus(connected) {
    const dot = document.getElementById('connectionDot');
    const label = document.getElementById('connectionStatus');
    
    if (connected) {
        dot.style.background = 'var(--accent-green)';
        dot.style.boxShadow = '0 0 8px var(--accent-green)';
        label.textContent = 'Connected';
    } else {
        dot.style.background = 'var(--accent-red)';
        dot.style.boxShadow = '0 0 8px var(--accent-red)';
        label.textContent = 'Disconnected';
    }
}

function handleUpdate(data) {
    if (data.type === 'snapshot') {
        updateDashboard(data);
    } else if (data.type === 'trade') {
        loadTrades();
    } else if (data.type === 'position_update') {
        updatePosition(data);
    }
}

async function refreshData() {
    try {
        const r = await fetch('/api/snapshot', { headers: getHeaders() });
        const d = await r.json();
        if (d.success && d.data) {
            updateDashboard(d.data);
        }
    } catch (e) {
        console.error('Refresh error:', e);
    }
}

function updateDashboard(data) {
    if (!data) return;
    
    // Price
    const price = data.price || data.current_price || 0;
    document.getElementById('heroPrice').textContent = formatPrice(price);
    
    // System status
    const statusEl = document.getElementById('systemStatus');
    const status = (data.status || 'RUNNING').toUpperCase();
    statusEl.textContent = status;
    if (status === 'RUNNING' || status === 'TRADE' || status === 'BUY' || status === 'SELL') {
        statusEl.style.background = 'var(--accent-green-dim)';
        statusEl.style.color = 'var(--accent-green)';
        statusEl.style.borderColor = 'rgba(16, 185, 129, 0.3)';
    } else if (status === 'PAUSED' || status === 'WAITING' || status === 'HOLD') {
        statusEl.style.background = 'var(--accent-yellow-dim)';
        statusEl.style.color = 'var(--accent-yellow)';
        statusEl.style.borderColor = 'rgba(245, 158, 11, 0.3)';
    } else {
        statusEl.style.background = 'var(--accent-red-dim)';
        statusEl.style.color = 'var(--accent-red)';
        statusEl.style.borderColor = 'rgba(239, 68, 68, 0.3)';
    }
    
    // Pair
    if (data.pair) {
        currentPair = data.pair;
        document.getElementById('currentPair').textContent = data.pair;
    }
    
    // 24h Change
    const change = data.change_24h || 0;
    const changeEl = document.getElementById('heroChange');
    changeEl.innerHTML = `<span class="change-value ${change >= 0 ? 'positive' : 'negative'}">${change >= 0 ? '+' : ''}${change.toFixed(2)}%</span><span class="change-period">24h</span>`;
    
    // Signal
    const signal = (data.signal || 'HOLD').toUpperCase();
    const signalRing = document.querySelector('.signal-ring');
    const signalText = document.querySelector('.signal-text');
    
    signalText.textContent = signal;
    signalRing.className = 'signal-ring';
    if (signal === 'BUY') {
        signalRing.classList.add('buy');
    } else if (signal === 'SELL') {
        signalRing.classList.add('sell');
    }
    
    // Confidence
    const conf = Math.round((data.confidence || 0) * 100);
    document.getElementById('confidenceVal').textContent = conf;
    
    // Stats
    document.getElementById('volume24h').textContent = formatVolume(data.volume_24h || 0);
    document.getElementById('atrValue').textContent = (data.atr || 0).toFixed(4);
    document.getElementById('spreadValue').textContent = ((data.spread || 0) * 100).toFixed(3) + '%';
    document.getElementById('volatilityValue').textContent = data.volatility_state || data.volatility_regime || 'NORMAL';
    
    // Account
    const equity = data.equity || data.total_equity || data.balance || 0;
    document.getElementById('totalEquity').textContent = formatCurrency(equity);
    document.getElementById('availableBalance').textContent = formatCurrency(data.balance || 0);
    document.getElementById('inTrade').textContent = formatCurrency(data.in_trade || 0);
    
    const todayPnl = data.daily_pnl || 0;
    const todayPnlEl = document.getElementById('todayPnl');
    todayPnlEl.textContent = (todayPnl >= 0 ? '+' : '') + formatCurrency(todayPnl);
    todayPnlEl.className = 'acc-value ' + (todayPnl >= 0 ? 'positive' : 'negative');
    
    const sessionPnl = data.session_pnl || 0;
    const sessionPnlEl = document.getElementById('sessionPnl');
    sessionPnlEl.textContent = (sessionPnl >= 0 ? '+' : '') + formatCurrency(sessionPnl);
    sessionPnlEl.className = 'acc-value ' + (sessionPnl >= 0 ? 'positive' : 'negative');
    
    // Performance
    const winRate = (data.win_rate || 0) * 100;
    document.getElementById('winRate').textContent = winRate.toFixed(0) + '%';
    document.getElementById('winRateCircle').setAttribute('stroke-dasharray', `${winRate.toFixed(0)}, 100`);
    document.getElementById('totalTrades').textContent = data.total_trades || data.trades_today || 0;
    document.getElementById('winCount').textContent = data.wins || 0;
    document.getElementById('lossCount').textContent = data.losses || 0;
    document.getElementById('avgWin').textContent = formatCurrency(data.avg_win || 0);
    document.getElementById('avgLoss').textContent = formatCurrency(data.avg_loss || 0);
    document.getElementById('profitFactor').textContent = (data.profit_factor || 0).toFixed(2);
    document.getElementById('expectancy').textContent = formatCurrency(data.expectancy || 0);
    
    // Capital Manager
    const cm = data.capital_manager || {};
    document.getElementById('capitalMode').textContent = (cm.capital_mode || 'MEDIUM').toUpperCase();
    const effRisk = cm.effective_params ? (cm.effective_params.risk_per_trade || 0.012) : 0.012;
    document.getElementById('effectiveRisk').textContent = (effRisk * 100).toFixed(1) + '%';
    document.getElementById('riskBar').style.width = Math.min(100, effRisk * 1000) + '%';
    const minConf = cm.effective_params ? (cm.effective_params.min_confidence || 0.62) : 0.62;
    document.getElementById('minConfidence').textContent = Math.round(minConf * 100) + '%';
    document.getElementById('marketCondition').textContent = cm.market_condition || 'NORMAL';
    document.getElementById('streakInfo').textContent = `${cm.win_streak || 0}W / ${cm.loss_streak || 0}L`;
    document.getElementById('dailyTrades').textContent = cm.daily_trades || data.trades_today || 0;
    document.getElementById('maxDailyTrades').textContent = cm.effective_params ? (cm.effective_params.max_trades_per_day || 120) : 120;
    
    // Position
    updatePosition(data);
    
    // Controls
    document.getElementById('loopTime').textContent = (data.loop_time_ms || data.api_latency_ms || 0).toFixed(0) + 'ms';
    document.getElementById('apiLatency').textContent = (data.api_latency_ms || 0).toFixed(0) + 'ms';
    
    // Strategy tab
    updateStrategy(data);
    
    // ML tab
    updateML(data);
    
    // Risk tab
    updateRisk(data);
    
    // Hero chart
    if (data.candles_5m && data.candles_5m.length > 0) {
        updateHeroChart(data.candles_5m);
    }
}

function updatePosition(data) {
    const statusEl = document.getElementById('positionStatus');
    const emptyDiv = document.getElementById('positionContent');
    const detailDiv = document.getElementById('positionDetail');
    
    const hasPos = data.has_open_position || (data.open_positions && data.open_positions.length > 0);
    
    if (hasPos) {
        statusEl.textContent = 'Active';
        statusEl.className = 'position-status active';
        emptyDiv.style.display = 'none';
        detailDiv.style.display = 'block';
        
        const pos = (data.open_positions && data.open_positions[0]) || data;
        
        document.getElementById('posSide').textContent = pos.side || '-';
        document.getElementById('posSide').className = 'pos-value ' + (pos.side === 'BUY' ? 'positive' : 'negative');
        document.getElementById('posSize').textContent = (pos.quantity || 0).toFixed(6);
        document.getElementById('posEntry').textContent = formatPrice(pos.entry_price || 0);
        document.getElementById('posCurrent').textContent = formatPrice(data.price || data.current_price || 0);
        
        const pnl = pos.unrealized_pnl || 0;
        const pnlEl = document.getElementById('posPnl');
        pnlEl.textContent = (pnl >= 0 ? '+' : '') + formatCurrency(pnl);
        pnlEl.className = 'pos-value pnl ' + (pnl >= 0 ? 'positive' : 'negative');
        
        document.getElementById('posSL').textContent = formatPrice(pos.stop_loss || 0);
        document.getElementById('posTP').textContent = formatPrice(pos.take_profit || 0);
    } else {
        statusEl.textContent = 'No Position';
        statusEl.className = 'position-status';
        emptyDiv.style.display = 'flex';
        detailDiv.style.display = 'none';
    }
}

function updateStrategy(data) {
    if (!data) return;
    
    const signals = data.signals || {};
    
    // Individual factor bars
    updateFactorBar('trendBar', 'trendValue', signals.trend || data.trend || 'NEUTRAL');
    updateFactorBar('momentumBar', 'momentumValue', signals.momentum || data.momentum || 'NEUTRAL');
    updateFactorBar('volumeBar', 'volumeValue', signals.volume || 'NEUTRAL');
    updateFactorBar('volatilityBar', 'volatilityValue', signals.volatility || data.volatility_state || 'NEUTRAL');
    updateFactorBar('structureBar', 'structureValue', signals.structure || 'NEUTRAL');
    updateFactorBar('flowBar', 'flowValue', signals.order_flow || data.order_flow || 'NEUTRAL');
    
    // Indicators
    const rsi = data.rsi || 50;
    document.getElementById('rsiValue').textContent = rsi.toFixed(1);
    document.getElementById('rsiBar').style.width = Math.min(100, Math.max(0, rsi)) + '%';
    
    const adx = data.adx || 0;
    document.getElementById('adxValue').textContent = adx.toFixed(1);
    document.getElementById('adxBar').style.width = Math.min(100, adx) + '%';
    
    const macd = data.macd_diff || 0;
    const macdEl = document.getElementById('macdValue');
    macdEl.textContent = macd.toFixed(6);
    macdEl.style.color = macd >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
    
    const emaCross = data.ema_cross || 'NEUTRAL';
    document.getElementById('emaValue').textContent = emaCross;
    
    // Confluence
    const confScore = (data.confluence_score || 0) * 100;
    document.getElementById('confluenceBar').style.width = confScore + '%';
    document.getElementById('confluenceScore').textContent = confScore.toFixed(0) + '%';
    
    // Timeframes
    const tfData = data.timeframe_analysis || {};
    updateTimeframeSignal('tf5mSignal', tfData['5m'] || 'NEUTRAL');
    updateTimeframeSignal('tf15mSignal', tfData['15m'] || 'NEUTRAL');
    updateTimeframeSignal('tf1hSignal', tfData['1h'] || 'NEUTRAL');
    
    // Decision log
    const trace = data.decision_trace || {};
    const reason = data.decision_reason || 'ANALYSIS';
    const log = document.getElementById('decisionLog');
    const now = new Date();
    const timeStr = now.toTimeString().slice(0, 8);
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">${timeStr}</span><span class="log-msg">${reason}</span>`;
    log.insertBefore(entry, log.firstChild);
    
    if (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

function updateFactorBar(barId, valueId, value) {
    const bar = document.getElementById(barId);
    const valueEl = document.getElementById(valueId);
    if (!bar || !valueEl) return;
    
    const v = String(value).toUpperCase();
    bar.style.width = v === 'BUY' ? '100%' : v === 'SELL' ? '100%' : '0%';
    bar.className = 'factor-fill ' + (v === 'BUY' ? 'buy' : v === 'SELL' ? 'sell' : 'neutral');
    valueEl.textContent = v;
    valueEl.style.color = v === 'BUY' ? 'var(--accent-green)' : v === 'SELL' ? 'var(--accent-red)' : 'var(--text-secondary)';
}

function updateTimeframeSignal(elId, signal) {
    const el = document.getElementById(elId);
    if (!el) return;
    const s = String(signal).toUpperCase();
    el.textContent = s;
    el.style.color = s === 'BUY' ? 'var(--accent-green)' : s === 'SELL' ? 'var(--accent-red)' : 'var(--text-secondary)';
}

function updateML(data) {
    if (!data) return;
    
    document.getElementById('mlStatus').textContent = data.ml_status || 'Active';
    document.getElementById('mlPrediction').textContent = data.ml_direction || '-';
    document.getElementById('mlConfidence').textContent = Math.round((data.ml_confidence || 0) * 100) + '%';
    document.getElementById('mlAccuracy').textContent = Math.round((data.ml_accuracy || 0.75) * 100) + '%';
    
    const mlCircle = document.getElementById('mlCircle');
    if (mlCircle) {
        mlCircle.setAttribute('stroke-dasharray', `${Math.round((data.ml_confidence || 0.75) * 100)}, 100`);
    }
}

function updateRisk(data) {
    if (!data) return;
    
    const dailyPnl = data.daily_pnl || 0;
    const balance = data.balance || 1;
    const dailyLoss = Math.abs(Math.min(dailyPnl, 0)) / Math.max(balance, 1) * 100;
    const dailyLimit = (data.daily_loss_limit || 0.03) * 100;
    
    document.getElementById('dailyLossBar').style.width = Math.min(100, (dailyLoss / Math.max(dailyLimit, 1)) * 100) + '%';
    document.getElementById('dailyLossUsed').textContent = dailyLoss.toFixed(1);
    document.getElementById('dailyLossLimit').textContent = dailyLimit.toFixed(1);
    
    const dd = (data.current_drawdown || 0) * 100;
    const maxDD = (data.max_drawdown || 0.10) * 100;
    document.getElementById('drawdownBar').style.width = Math.min(100, (dd / Math.max(maxDD, 1)) * 100) + '%';
    document.getElementById('currentDD').textContent = dd.toFixed(1);
    document.getElementById('maxDD').textContent = maxDD.toFixed(1);
    
    const posSize = (data.position_size_pct || 0) * 100;
    document.getElementById('positionSizeBar').style.width = Math.min(100, posSize / 5 * 100) + '%';
    document.getElementById('currentPosSize').textContent = posSize.toFixed(1);
    
    // Smart stop
    const ss = data.smart_stop_stats || {};
    document.getElementById('activeStops').textContent = ss.active_stops || 0;
    document.getElementById('trailsActivated').textContent = ss.trails_activated || 0;
    document.getElementById('breakEvensHit').textContent = ss.break_evens_hit || 0;
    document.getElementById('profitLocks').textContent = ss.profit_locks || 0;
    document.getElementById('lastStopType').textContent = ss.last_stop_type || '-';
    
    // Blocks
    const blocks = data.active_blocks || [];
    const blocksList = document.getElementById('blocksList');
    if (blocks.length > 0) {
        blocksList.innerHTML = blocks.map(b => `<div class="block-item">${b}</div>`).join('');
    } else {
        blocksList.innerHTML = '<div class="block-none">No active blocks</div>';
    }
}

async function loadTrades(filter = 'all') {
    try {
        let url = '/api/trades?limit=100';
        if (filter === 'open') url += '&status=OPEN';
        else if (filter === 'closed') url += '&status=CLOSED';
        
        const r = await fetch(url, { headers: getHeaders() });
        const d = await r.json();
        tradesData = d.trades || [];
        
        const tbody = document.getElementById('tradesBody');
        tbody.innerHTML = '';
        
        if (tradesData.length === 0) {
            tbody.innerHTML = '<tr class="loading-row"><td colspan="9">No trades found</td></tr>';
            return;
        }
        
        tradesData.forEach(trade => {
            const pnl = trade.pnl || 0;
            const status = (trade.status || 'CLOSED').toLowerCase();
            const side = (trade.side || '').toLowerCase();
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${formatTime(trade.timestamp || trade.created_at)}</td>
                <td class="pair-cell">${trade.symbol || '-'}</td>
                <td><span class="signal-badge sig-${side}">${trade.side || '-'}</span></td>
                <td>${formatPrice(trade.entry_price)}</td>
                <td>${formatPrice(trade.exit_price || trade.current_price)}</td>
                <td>${formatQuantity(trade.quantity)}</td>
                <td class="${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}${formatCurrency(pnl)}</td>
                <td><span class="signal-badge sig-${status}">${trade.status || '-'}</span></td>
                <td><button class="btn btn-sm btn-secondary" onclick='showTradeDetails(${JSON.stringify(trade)})'>Details</button></td>
            `;
            tbody.appendChild(row);
        });
    } catch (e) {
        console.error('Load trades error:', e);
    }
}

function showTradeDetails(trade) {
    const modal = document.getElementById('tradeModal');
    const body = document.getElementById('tradeModalBody');
    
    const pnl = trade.pnl || 0;
    
    body.innerHTML = `
        <div class="trade-detail-grid">
            <div class="detail-row">
                <span class="detail-label">Symbol</span>
                <span class="detail-value">${trade.symbol}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Side</span>
                <span class="detail-value">${trade.side}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Entry Price</span>
                <span class="detail-value">${formatPrice(trade.entry_price)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Exit Price</span>
                <span class="detail-value">${formatPrice(trade.exit_price || '-')}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Quantity</span>
                <span class="detail-value">${formatQuantity(trade.quantity)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">PnL</span>
                <span class="detail-value ${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}${formatCurrency(pnl)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Stop Loss</span>
                <span class="detail-value">${formatPrice(trade.stop_loss)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Take Profit</span>
                <span class="detail-value">${formatPrice(trade.take_profit)}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Opened</span>
                <span class="detail-value">${formatDateTime(trade.timestamp || trade.created_at)}</span>
            </div>
            ${trade.close_timestamp ? `
            <div class="detail-row">
                <span class="detail-label">Closed</span>
                <span class="detail-value">${formatDateTime(trade.close_timestamp)}</span>
            </div>` : ''}
        </div>
        ${trade.status === 'OPEN' ? `
        <div class="modal-actions">
            <button class="btn btn-danger" onclick="closeTrade('${trade.id}')">Close Position</button>
        </div>` : ''}
    `;
    
    modal.classList.add('active');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('active');
}

async function closeTrade(tradeId) {
    if (!confirm('Are you sure you want to close this position?')) return;
    
    try {
        const r = await fetch(`/api/trades/${tradeId}/stop`, {
            method: 'POST',
            headers: getHeaders()
        });
        
        if (r.ok) {
            showToast('Position closed successfully', 'success');
            closeModal('tradeModal');
            loadTrades();
        } else {
            showToast('Failed to close position', 'error');
        }
    } catch (e) {
        showToast('Error closing position', 'error');
    }
}

function initMainChart() {
    if (typeof LightweightCharts === 'undefined') {
        console.error('Lightweight Charts library not loaded');
        return;
    }
    
    const container = document.getElementById('mainChart');
    if (!container) return;
    
    if (mainChart) {
        mainChart.remove();
        mainChart = null;
        candleSeries = null;
    }
    
    mainChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 500,
        layout: {
            background: { color: '#151d2b' },
            textColor: '#a0aec0',
        },
        grid: {
            vertLines: { color: '#2a3441' },
            horzLines: { color: '#2a3441' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
        },
    });
    
    candleSeries = mainChart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });
    
    loadChartData();
    
    window.addEventListener('resize', () => {
        if (mainChart && container) {
            mainChart.applyOptions({ width: container.clientWidth });
        }
    });
}

async function loadChartData() {
    if (!candleSeries) return;
    
    try {
        const r = await fetch(`/api/candles?timeframe=${currentTimeframe}&limit=200`, { headers: getHeaders() });
        const d = await r.json();
        
        if (d.candles && d.candles.length > 0) {
            candleSeries.setData(d.candles);
            mainChart.timeScale().fitContent();
        }
    } catch (e) {
        console.error('Chart data error:', e);
    }
}

function initHeroChart() {
    if (typeof LightweightCharts === 'undefined') return;
    
    const container = document.getElementById('heroChart');
    if (!container) return;
    
    if (heroChart) return;
    
    heroChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: 200,
        layout: {
            background: { color: 'transparent' },
            textColor: '#64748b',
        },
        grid: {
            vertLines: { visible: false },
            horzLines: { visible: false },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        timeScale: {
            visible: true,
            timeVisible: true,
            secondsVisible: false,
            borderVisible: false,
        },
        rightPriceScale: {
            borderVisible: false,
        },
    });
    
    heroLineSeries = heroChart.addAreaSeries({
        topColor: 'rgba(59, 130, 246, 0.4)',
        bottomColor: 'rgba(59, 130, 246, 0.0)',
        lineColor: '#3b82f6',
        lineWidth: 2,
    });
    
    window.addEventListener('resize', () => {
        if (heroChart && container) {
            heroChart.applyOptions({ width: container.clientWidth });
        }
    });
}

function updateHeroChart(candles) {
    if (!heroLineSeries || !heroChart) {
        initHeroChart();
        if (!heroLineSeries) return;
    }
    
    if (!candles || candles.length === 0) return;
    
    const lineData = candles.map(c => ({
        time: c.time,
        value: c.close,
    }));
    
    heroLineSeries.setData(lineData);
    heroChart.timeScale().fitContent();
}

async function loadStrategyData() {
    try {
        const r = await fetch('/api/snapshot', { headers: getHeaders() });
        const d = await r.json();
        if (d.success && d.data) {
            updateStrategy(d.data);
        }
    } catch (e) {
        console.error('Strategy data error:', e);
    }
}

async function loadPerformance(period) {
    try {
        const r = await fetch(`/api/analytics?period=${period}`, { headers: getHeaders() });
        const d = await r.json();
        
        document.getElementById('totalTrades').textContent = d.total_trades || 0;
        document.getElementById('winCount').textContent = d.wins || 0;
        document.getElementById('lossCount').textContent = d.losses || 0;
        
        const winRate = d.total_trades > 0 ? (d.wins / d.total_trades * 100) : 0;
        document.getElementById('winRate').textContent = winRate.toFixed(0) + '%';
        document.getElementById('winRateCircle').setAttribute('stroke-dasharray', `${winRate.toFixed(0)}, 100`);
        
        document.getElementById('avgWin').textContent = formatCurrency(d.avg_win || 0);
        document.getElementById('avgLoss').textContent = formatCurrency(Math.abs(d.avg_loss || 0));
        document.getElementById('profitFactor').textContent = (d.profit_factor || 0).toFixed(2);
        document.getElementById('expectancy').textContent = formatCurrency(d.expectancy || 0);
    } catch (e) {
        console.error('Performance load error:', e);
    }
}

async function loadMLData() {
    try {
        const r = await fetch('/api/ml', { headers: getHeaders() });
        const d = await r.json();
        
        document.getElementById('mlPrediction').textContent = d.last_prediction || '-';
        document.getElementById('mlConfidence').textContent = Math.round((d.last_confidence || 0) * 100) + '%';
        document.getElementById('mlAccuracy').textContent = Math.round((d.accuracy_7d || 0.75) * 100) + '%';
        
        document.getElementById('tftPrecision').textContent = Math.round((d.precision || 72)) + '%';
        document.getElementById('tftRecall').textContent = Math.round((d.recall || 70)) + '%';
        document.getElementById('nbeatsPrecision').textContent = Math.round((d.precision || 70)) + '%';
        document.getElementById('nbeatsRecall').textContent = Math.round((d.recall || 68)) + '%';
        document.getElementById('metaF1').textContent = Math.round((d.f1_score || 71)) + '%';
        document.getElementById('metaAcc').textContent = Math.round((d.accuracy || 74)) + '%';
    } catch (e) {
        console.error('ML data error:', e);
    }
}

async function loadRiskData() {
    try {
        const r = await fetch('/api/risk', { headers: getHeaders() });
        const d = await r.json();
        
        document.getElementById('dailyLossUsed').textContent = (d.daily_used * 100 || 0).toFixed(1);
        document.getElementById('dailyLossLimit').textContent = (d.daily_limit * 100 || 3).toFixed(1);
        document.getElementById('dailyLossBar').style.width = Math.min(100, (d.daily_used / Math.max(d.daily_limit, 0.01) * 100 || 0)) + '%';
        
        document.getElementById('effectiveRisk').textContent = (d.effective_size * 100 || 1).toFixed(1) + '%';
        document.getElementById('riskBar').style.width = Math.min(100, d.effective_size * 1000 || 10) + '%';
    } catch (e) {
        console.error('Risk data error:', e);
    }
}

async function sendControl(action) {
    try {
        const r = await fetch(`/api/control/${action}`, {
            method: 'POST',
            headers: getHeaders()
        });
        
        if (r.ok) {
            const d = await r.json();
            showToast(d.message || `${action} command sent`, 'success');
            refreshData();
        } else {
            showToast(`Failed to ${action}`, 'error');
        }
    } catch (e) {
        showToast(`Error: ${e.message}`, 'error');
    }
}

async function saveSettings() {
    const settings = {
        risk_per_trade: parseFloat(document.getElementById('riskPerTrade').value) / 100,
        max_daily_loss: parseFloat(document.getElementById('maxDailyLoss').value) / 100,
        max_drawdown: parseFloat(document.getElementById('maxDrawdown').value) / 100,
        max_trades_day: parseInt(document.getElementById('maxTradesDay').value),
        min_confidence: parseFloat(document.getElementById('minConfidence').value) / 100,
        adx_threshold: parseFloat(document.getElementById('adxThreshold').value),
        stop_atr_multiplier: parseFloat(document.getElementById('stopAtr').value),
        trailing_activation_r: parseFloat(document.getElementById('trailAct').value),
        breakeven_trigger_r: parseFloat(document.getElementById('beTrigger').value),
        profit_lock_trigger_r: parseFloat(document.getElementById('profitLock').value),
        trading_mode: document.getElementById('tradingMode').value,
        primary_timeframe: document.getElementById('primaryTF').value,
        enable_ml: document.getElementById('enableML').checked,
        loop_interval: parseInt(document.getElementById('loopInterval').value),
    };
    
    try {
        const r = await fetch('/api/settings', {
            method: 'POST',
            headers: getHeaders(),
            body: JSON.stringify(settings)
        });
        
        if (r.ok) {
            showToast('Settings saved successfully', 'success');
        } else {
            showToast('Failed to save settings', 'error');
        }
    } catch (e) {
        showToast('Error saving settings', 'error');
    }
}

function resetSettings() {
    document.getElementById('riskPerTrade').value = 1;
    document.getElementById('maxDailyLoss').value = 3;
    document.getElementById('maxDrawdown').value = 10;
    document.getElementById('maxTradesDay').value = 120;
    document.getElementById('minConfidence').value = 60;
    document.getElementById('adxThreshold').value = 10;
    document.getElementById('stopAtr').value = 1.5;
    document.getElementById('trailAct').value = 0.3;
    document.getElementById('beTrigger').value = 0.2;
    document.getElementById('profitLock').value = 0.5;
    document.getElementById('tradingMode').value = 'normal';
    document.getElementById('primaryTF').value = '5m';
    document.getElementById('enableML').checked = true;
    document.getElementById('loopInterval').value = 15;
    
    initSliders();
    showToast('Settings reset to defaults', 'success');
}

function emergencyStop() {
    if (!confirm('Are you sure you want to trigger an emergency stop? This will halt all trading immediately.')) return;
    sendControl('stop');
}

function exportTrades() {
    window.location.href = '/api/trades/export?' + new URLSearchParams({ format: 'csv' });
}

function logout() {
    localStorage.removeItem('auth_token');
    window.location.href = '/login';
}

function closePosition() {
    if (confirm('Close current position?')) {
        sendControl('force_close');
    }
}

// Utility functions
function formatPrice(price) {
    if (price === null || price === undefined) return '-';
    const num = parseFloat(price);
    if (isNaN(num)) return '-';
    if (num >= 1000) return '$' + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (num >= 1) return '$' + num.toFixed(4);
    return '$' + num.toFixed(8);
}

function formatCurrency(value) {
    const num = parseFloat(value);
    if (isNaN(num)) return '$0.00';
    const sign = num < 0 ? '-' : '';
    return sign + '$' + Math.abs(num).toFixed(2);
}

function formatVolume(vol) {
    const num = parseFloat(vol);
    if (isNaN(num)) return '0';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toFixed(2);
}

function formatQuantity(qty) {
    const num = parseFloat(qty);
    if (isNaN(num)) return '0';
    if (num >= 1) return num.toFixed(6);
    return num.toFixed(8);
}

function formatTime(ts) {
    if (!ts) return '-';
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function formatDateTime(ts) {
    if (!ts) return '-';
    const d = new Date(ts);
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}