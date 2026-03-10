from __future__ import annotations

COLORS = {
    "bg": "#0b0f17",
    "bg_alt": "#111827",
    "panel": "#141c2b",
    "panel_alt": "#1b2638",
    "border": "#243247",
    "text": "#e5edf9",
    "muted": "#91a3bf",
    "accent": "#38bdf8",
    "accent_alt": "#2563eb",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "warning": "#f59e0b",
}


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
        font-family: Inter, Segoe UI, Arial;
    }}
    QMainWindow {{ background: {COLORS['bg']}; }}
    QFrame#panelCard, QFrame#metricCard, QFrame#sidebar, QFrame#topBar {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
    }}
    QLabel#sectionTitle {{ font-size: 17px; font-weight: 700; }}
    QLabel#metricLabel {{ color: {COLORS['muted']}; font-size: 11px; text-transform: uppercase; }}
    QLabel#valueLabel, QLabel#metricValue {{ font-size: 18px; font-weight: 700; }}
    QPushButton#navButton {{
        text-align: left;
        border: 1px solid transparent;
        background: transparent;
        color: {COLORS['muted']};
        border-radius: 8px;
        padding: 8px;
    }}
    QPushButton#navButton:checked {{
        background: rgba(56, 189, 248, 0.18);
        color: {COLORS['text']};
        border: 1px solid rgba(56, 189, 248, 0.35);
    }}
    QPushButton#navButton:hover {{ background: rgba(255,255,255,0.05); color: {COLORS['text']}; }}
    QTableWidget, QTextEdit, QListWidget {{
        background: {COLORS['panel_alt']}; border: 1px solid {COLORS['border']}; border-radius: 8px;
        gridline-color: {COLORS['border']};
    }}
    QHeaderView::section {{ background: {COLORS['panel']}; color: {COLORS['muted']}; border: none; padding: 6px; }}
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        background: {COLORS['panel_alt']};
        text-align: center;
        min-height: 11px;
    }}
    QProgressBar::chunk {{ background-color: {COLORS['accent']}; border-radius: 6px; }}
    """


def dashboard_stylesheet() -> str:
    return app_stylesheet()
