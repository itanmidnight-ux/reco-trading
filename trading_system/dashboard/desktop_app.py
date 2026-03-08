from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from urllib.error import URLError
from urllib.request import urlopen


class DesktopDashboardApp:
    def __init__(self, api_base: str) -> None:
        self.api_base = api_base.rstrip('/')
        self.root = tk.Tk()
        self.root.title('RECO Trading Desktop Dashboard')
        self.root.geometry('1080x760')
        self.root.configure(bg='#0f172a')

        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('Card.TFrame', background='#111827')
        style.configure('Title.TLabel', background='#111827', foreground='#93c5fd', font=('Segoe UI', 11, 'bold'))
        style.configure('Value.TLabel', background='#111827', foreground='#e5e7eb', font=('Segoe UI', 13, 'bold'))
        style.configure('Meta.TLabel', background='#111827', foreground='#9ca3af', font=('Segoe UI', 10))

        self.values: dict[str, tk.StringVar] = {}
        self._build_ui()
        self._schedule_refresh()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, style='Card.TFrame', padding=12)
        top.pack(fill='x', padx=12, pady=(12, 8))
        ttk.Label(top, text='RECO Trading - Desktop Control Center', style='Title.TLabel').grid(row=0, column=0, sticky='w')
        self.values['updated_at'] = tk.StringVar(value='Sin conexión')
        ttk.Label(top, textvariable=self.values['updated_at'], style='Meta.TLabel').grid(row=0, column=1, sticky='e')
        top.columnconfigure(0, weight=1)

        metrics = ttk.Frame(self.root, style='Card.TFrame', padding=12)
        metrics.pack(fill='x', padx=12, pady=8)

        fields = [
            ('Capital real USDT', 'capital_real_usdt'),
            ('Equity cuenta USDT', 'account_equity_usdt'),
            ('PnL total realizado', 'pnl_total'),
            ('PnL total cuenta', 'account_pnl_total'),
            ('PnL diario', 'pnl_daily'),
            ('Drawdown realizado', 'drawdown'),
            ('Win rate', 'win_rate'),
            ('Sharpe', 'sharpe'),
            ('Señal', 'signal'),
            ('Régimen', 'regime'),
            ('Estado Binance', 'binance_status'),
            ('Latencia', 'latency_ms'),
        ]

        for idx, (label, key) in enumerate(fields):
            r, c = divmod(idx, 4)
            card = ttk.Frame(metrics, style='Card.TFrame', padding=8)
            card.grid(row=r, column=c, padx=6, pady=6, sticky='nsew')
            self.values[key] = tk.StringVar(value='-')
            ttk.Label(card, text=label, style='Meta.TLabel').pack(anchor='w')
            ttk.Label(card, textvariable=self.values[key], style='Value.TLabel').pack(anchor='w', pady=(2, 0))
        for c in range(4):
            metrics.columnconfigure(c, weight=1)

        bottom = ttk.Frame(self.root, style='Card.TFrame', padding=12)
        bottom.pack(fill='both', expand=True, padx=12, pady=(8, 12))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(1, weight=1)

        ttk.Label(bottom, text='Actividad (señales + ejecuciones)', style='Title.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(bottom, text='Últimas operaciones', style='Title.TLabel').grid(row=0, column=1, sticky='w')

        self.activity_box = ScrolledText(bottom, bg='#0b1220', fg='#e5e7eb', insertbackground='#e5e7eb', font=('Consolas', 10))
        self.activity_box.grid(row=1, column=0, sticky='nsew', padx=(0, 8), pady=(8, 0))
        self.activity_box.configure(state='disabled')

        self.trades_box = ScrolledText(bottom, bg='#0b1220', fg='#e5e7eb', insertbackground='#e5e7eb', font=('Consolas', 10))
        self.trades_box.grid(row=1, column=1, sticky='nsew', padx=(8, 0), pady=(8, 0))
        self.trades_box.configure(state='disabled')

    def _fetch_json(self, path: str) -> dict:
        with urlopen(f'{self.api_base}{path}', timeout=4) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def _refresh_once(self) -> None:
        try:
            data = self._fetch_json('/dashboard/data')
            activity = self._fetch_json('/dashboard/activity').get('activity', [])
            trades = self._fetch_json('/dashboard/trades').get('trades', [])
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self.root.after(0, lambda: self.values['updated_at'].set(f'Error de conexión: {exc}'))
            return

        def _apply() -> None:
            self.values['updated_at'].set('Actualizado desde API local')
            self.values['capital_real_usdt'].set(f"{float(data.get('capital_real_usdt', 0.0)):.2f} USDT")
            self.values['account_equity_usdt'].set(f"{float(data.get('account_equity_usdt', 0.0)):.2f} USDT")
            self.values['pnl_total'].set(f"{float(data.get('pnl_total', 0.0)):+.4f}")
            self.values['account_pnl_total'].set(f"{float(data.get('account_pnl_total', 0.0)):+.4f}")
            self.values['pnl_daily'].set(f"{float(data.get('pnl_daily', 0.0)):+.4f}")
            self.values['drawdown'].set(f"{float(data.get('drawdown', 0.0)):.4f} USDT")
            self.values['win_rate'].set(f"{100.0*float(data.get('win_rate', 0.0)):.2f}%")
            self.values['sharpe'].set(f"{float(data.get('sharpe', 0.0)):.3f}")
            self.values['signal'].set(str(data.get('signal', '-')))
            self.values['regime'].set(str(data.get('regime', '-')))
            self.values['binance_status'].set(str(data.get('binance_status', '-')))
            self.values['latency_ms'].set(f"{float(data.get('latency_ms', 0.0)):.0f} ms")

            self.activity_box.configure(state='normal')
            self.activity_box.delete('1.0', tk.END)
            for item in activity[:120]:
                self.activity_box.insert(tk.END, f"[{item.get('type','info')}] {item.get('title','')}\n{item.get('detail','')}\n\n")
            self.activity_box.configure(state='disabled')

            self.trades_box.configure(state='normal')
            self.trades_box.delete('1.0', tk.END)
            for trade in trades[:120]:
                self.trades_box.insert(
                    tk.END,
                    f"{trade.get('status','')} {trade.get('side','')} {trade.get('symbol','')} "
                    f"qty={float(trade.get('qty',0.0)):.6f} price={float(trade.get('price',0.0)):.6f} "
                    f"pnl={float(trade.get('pnl',0.0)):+.6f}\n",
                )
            self.trades_box.configure(state='disabled')

        self.root.after(0, _apply)

    def _schedule_refresh(self) -> None:
        threading.Thread(target=self._refresh_once, daemon=True).start()
        self.root.after(2000, self._schedule_refresh)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    api_base = os.getenv('API_BASE', 'http://127.0.0.1:8000')
    DesktopDashboardApp(api_base).run()


if __name__ == '__main__':
    main()
