from __future__ import annotations

BG = "#0f1117"
PANEL = "#1a1d26"
BORDER = "#2a2f3a"
TEXT_PRIMARY = "#e6e8ee"
TEXT_SECONDARY = "#9aa4b2"
POSITIVE = "#16c784"
NEGATIVE = "#ea3943"
WARNING = "#f0b90b"
INFO = "#3a7afe"


def status_color(status: str) -> str:
    mapping = {
        "RUNNING": POSITIVE,
        "WAITING_DATA": WARNING,
        "ERROR": NEGATIVE,
    }
    return mapping.get((status or "").upper(), TEXT_SECONDARY)


def signal_color(signal: str) -> str:
    mapping = {"BUY": POSITIVE, "SELL": NEGATIVE, "NEUTRAL": TEXT_SECONDARY, "HOLD": TEXT_SECONDARY}
    return mapping.get((signal or "").upper(), TEXT_SECONDARY)


def app_stylesheet() -> str:
    return f"""
    QWidget {{
        background-color: {BG};
        color: {TEXT_PRIMARY};
        font-size: 12px;
    }}
    QMainWindow {{ background-color: {BG}; }}
    QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {PANEL}; }}
    QTabBar::tab {{
        background: {PANEL};
        color: {TEXT_SECONDARY};
        padding: 8px 14px;
        margin-right: 4px;
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QTabBar::tab:selected {{ color: {TEXT_PRIMARY}; border-color: {INFO}; }}
    QTabBar::tab:hover {{ color: {TEXT_PRIMARY}; background: #222634; }}
    QFrame#card {{ border: 1px solid {BORDER}; border-radius: 10px; background: {PANEL}; }}
    QLabel#cardTitle {{ color: {TEXT_SECONDARY}; font-size: 11px; }}
    QLabel#cardValue {{ color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 700; }}
    QTextEdit, QPlainTextEdit, QTableWidget, QListWidget {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 8px;
    }}
    QHeaderView::section {{ background: #1e2230; border: 0; color: {TEXT_SECONDARY}; padding: 6px; }}
    QProgressBar {{ border: 1px solid {BORDER}; border-radius: 6px; background: #131722; text-align: center; }}
    QProgressBar::chunk {{ background-color: {INFO}; border-radius: 4px; }}
    QPushButton {{ background: #21283a; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 10px; }}
    QPushButton:hover {{ background: #2a3147; }}
    QLineEdit, QComboBox {{ background: #151a26; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px; }}
    """
