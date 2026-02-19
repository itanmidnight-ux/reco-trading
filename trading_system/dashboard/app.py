from __future__ import annotations

import os
import requests
import streamlit as st

st.set_page_config(page_title='Trading Quant Dashboard', layout='wide')
st.title('📊 Dashboard Cuantitativo en Tiempo Real')

api_base = os.getenv('API_BASE', 'http://localhost:8000')

col1, col2, col3 = st.columns(3)

try:
    health = requests.get(f'{api_base}/health', timeout=3).json()
    rate = requests.get(f'{api_base}/metrics/rate-limit', timeout=3).json()
    perf = requests.get(f'{api_base}/metrics/performance', timeout=3).json()
except Exception as exc:  # noqa: BLE001
    st.error(f'No se pudo conectar al API Gateway: {exc}')
    st.stop()

with col1:
    st.subheader('Estado del Sistema')
    st.json(health)

with col2:
    st.subheader('Rate Limit')
    st.metric('Weight 1m', f"{rate.get('usage_1m', 0)}/{rate.get('limit', 0)}")

with col3:
    st.subheader('Rendimiento')
    st.metric('Win Rate', perf.get('win_rate', 0.0))
    st.metric('Profit Factor', perf.get('profit_factor', 0.0))
    st.metric('PnL', perf.get('pnl', 0.0))
    st.metric('Drawdown', perf.get('drawdown', 0.0))

st.caption('Métricas en tiempo real + persistencia SQL para auditoría/backtesting.')
