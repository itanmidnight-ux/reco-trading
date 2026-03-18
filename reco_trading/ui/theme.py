from __future__ import annotations

COLORS = {
    "background": "#0b1020",
    "background_alt": "#111933",
    "panel": "#141d35",
    "panel_alt": "#1a2643",
    "border": "#273658",
    "text_primary": "#edf2ff",
    "text_secondary": "#9fb2d9",
    "positive": "#22d39b",
    "negative": "#ff5f7b",
    "warning": "#ffcc66",
    "info": "#5a8dff",
    "neutral": "#7f93bf",
    "accent": "#7b61ff",
}


def app_stylesheet() -> str:
    try:
        style = f"""
        QWidget {{
            background-color: {COLORS['background']};
            color: {COLORS['text_primary']};
            font-family: Inter, Segoe UI, Arial;
            selection-background-color: rgba(122, 97, 255, 0.4);
        }}
        QMainWindow, QTabWidget::pane {{
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLORS['background']}, stop:1 {COLORS['background_alt']});
        }}
        QTabBar::tab {{
            background: {COLORS['panel']};
            color: {COLORS['text_secondary']};
            border: 1px solid {COLORS['border']};
            border-bottom: none;
            padding: 11px 18px;
            margin-right: 6px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            font-weight: 600;
        }}
        QTabBar::tab:hover {{
            color: {COLORS['text_primary']};
            background: {COLORS['panel_alt']};
        }}
        QTabBar::tab:selected {{
            color: white;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['accent']}, stop:1 {COLORS['info']});
        }}
        QFrame#metricCard, QFrame#panelCard {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLORS['panel']}, stop:1 {COLORS['panel_alt']});
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
        }}
        QLabel#sectionTitle {{
            color: {COLORS['text_primary']};
            font-size: 17px;
            font-weight: 700;
            letter-spacing: 0.4px;
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
        QLabel#statusRibbon {{
            color: {COLORS['text_primary']};
            font-size: 15px;
            font-weight: 700;
            padding: 12px 16px;
            border-radius: 14px;
            border: 1px solid rgba(122, 97, 255, 0.35);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(22, 31, 56, 0.95), stop:1 rgba(31, 50, 89, 0.95));
        }}
        QLabel#smallMetricValue {{
            color: {COLORS['text_primary']};
            font-size: 14px;
            font-weight: 600;
        }}
        QLabel#metricBadge {{
            color: {COLORS['text_secondary']};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.6px;
        }}
        QTableWidget, QTextEdit {{
            background: {COLORS['panel']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            gridline-color: {COLORS['border']};
        }}
        QListWidget {{
            background: rgba(20, 29, 53, 0.92);
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 6px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 8px 10px;
            margin: 2px 0;
            border-radius: 8px;
            color: {COLORS['text_primary']};
        }}
        QListWidget::item:selected {{
            background: rgba(90, 141, 255, 0.24);
            border: 1px solid rgba(90, 141, 255, 0.32);
        }}
        QHeaderView::section {{
            background: {COLORS['panel_alt']};
            color: {COLORS['text_secondary']};
            border: none;
            padding: 7px;
        }}
        QTableWidget::item {{
            padding: 6px;
        }}
        QProgressBar {{
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            text-align: center;
            background: rgba(20, 29, 53, 0.8);
            min-height: 12px;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['info']};
            border-radius: 6px;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: #101933;
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 7px;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QCheckBox {{
            spacing: 8px;
            color: {COLORS['text_primary']};
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid {COLORS['border']};
            background: {COLORS['panel']};
        }}
        QCheckBox::indicator:checked {{
            background: {COLORS['accent']};
            border: 1px solid {COLORS['info']};
        }}
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['accent']}, stop:1 {COLORS['info']});
            border: 1px solid {COLORS['border']};
            border-radius: 8px;
            padding: 8px 12px;
            color: white;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['info']}, stop:1 {COLORS['accent']});
            border: 1px solid {COLORS['info']};
        }}
        QPushButton:pressed {{
            padding-top: 9px;
            padding-left: 13px;
            background: {COLORS['panel_alt']};
        }}
        QPushButton:disabled {{
            background: #2a3557;
            color: {COLORS['neutral']};
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 12px;
            margin: 6px 0 6px 0;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(122, 97, 255, 0.35);
            min-height: 24px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(90, 141, 255, 0.55);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        """
        return style if isinstance(style, str) else ""
    except Exception:
        return ""


def dashboard_stylesheet() -> str:
    """Backward-compatible alias for the global application stylesheet."""
    return app_stylesheet()
