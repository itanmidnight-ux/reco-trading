from __future__ import annotations

COLORS = {
    "background": "#0f1117",
    "panel": "#1a1d26",
    "border": "#2a2f3a",
    "text_primary": "#e6e8ee",
    "text_secondary": "#9aa4b2",
    "positive": "#16c784",
    "negative": "#ea3943",
    "warning": "#f0b90b",
    "info": "#3a7afe",
    "neutral": "#667085",
}


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {COLORS['background']};
        color: {COLORS['text_primary']};
        font-family: Inter, Segoe UI, Arial;
    }}
    QMainWindow, QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        background: {COLORS['background']};
    }}
    QTabBar::tab {{
        background: {COLORS['panel']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        padding: 10px 16px;
        margin-right: 4px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    QTabBar::tab:hover {{
        color: {COLORS['text_primary']};
        background: #202534;
    }}
    QTabBar::tab:selected {{
        color: {COLORS['text_primary']};
        background: #232839;
    }}
    QFrame#metricCard, QFrame#panelCard {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
    }}
    QLabel#metricLabel {{
        color: {COLORS['text_secondary']};
        font-size: 11px;
        text-transform: uppercase;
    }}
    QLabel#metricValue {{
        color: {COLORS['text_primary']};
        font-size: 19px;
        font-weight: 700;
    }}
    QLabel#smallMetricValue {{
        color: {COLORS['text_primary']};
        font-size: 14px;
        font-weight: 600;
    }}
    QTableWidget, QTextEdit {{
        background: {COLORS['panel']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        gridline-color: {COLORS['border']};
    }}
    QHeaderView::section {{
        background: #202534;
        color: {COLORS['text_secondary']};
        border: none;
        padding: 6px;
    }}
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        text-align: center;
        background: #141824;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['info']};
        border-radius: 6px;
    }}
    QLineEdit, QComboBox, QSpinBox {{
        background: #141824;
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px;
    }}
    QPushButton {{
        background: #242a3c;
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 8px 12px;
    }}
    QPushButton:hover {{
        background: #2d3550;
    }}
    """


def dashboard_stylesheet() -> str:
    """Backward-compatible alias for the global application stylesheet."""
    return app_stylesheet()
