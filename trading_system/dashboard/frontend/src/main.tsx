import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { createChart } from 'lightweight-charts';

function App() {
  const chartRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<any>({});

  useEffect(() => {
    if (!chartRef.current) return;
    const chart = createChart(chartRef.current, { width: 900, height: 320 });
    const line = chart.addLineSeries();
    line.setData([
      { time: '2026-01-01', value: 0.5 },
      { time: '2026-01-02', value: 0.55 },
    ]);

    const ws = new WebSocket('ws://localhost:8000/ws/status');
    ws.onmessage = (ev) => setStatus(JSON.parse(ev.data));
    const iv = setInterval(() => ws.send('ping'), 1500);

    return () => {
      clearInterval(iv);
      ws.close();
      chart.remove();
    };
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 16 }}>
      <h2>Reco Trading — Dashboard Institucional</h2>
      <div ref={chartRef} />
      <pre>{JSON.stringify(status, null, 2)}</pre>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
