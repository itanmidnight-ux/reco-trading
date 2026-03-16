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


def app_stylesheet(variant: str = "Dark") -> str:
    try:
        is_contrast = variant.strip().lower().startswith("dark+")
        border_color = "#3d5b93" if is_contrast else COLORS["border"]
        panel_from = "#182849" if is_contrast else COLORS["panel"]
        panel_to = "#213565" if is_contrast else COLORS["panel_alt"]
        style = f"""
        QWidget {{
            background-color: {COLORS['background']};
            color: {COLORS['text_primary']};
            font-family: Inter, Segoe UI, Arial;
            selection-background-color: rgba(122, 97, 255, 0.4);
        }}
        QMainWindow, QTabWidget::pane {{
            border: 1px solid {border_color};
            border-radius: 12px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLORS['background']}, stop:1 {COLORS['background_alt']});
        }}
        QTabBar::tab {{
            background: {COLORS['panel']};
            color: {COLORS['text_secondary']};
            border: 1px solid {border_color};
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
                stop:0 {panel_from}, stop:1 {panel_to});
            border: 1px solid {border_color};
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
        QLabel#smallMetricValue {{
            color: {COLORS['text_primary']};
            font-size: 14px;
            font-weight: 600;
        }}
        QTableWidget, QTextEdit, QListWidget {{
            background: {COLORS['panel']};
            border: 1px solid {border_color};
            border-radius: 10px;
            gridline-color: {COLORS['border']};
        }}
        QHeaderView::section {{
            background: {COLORS['panel_alt']};
            color: {COLORS['text_secondary']};
            border: none;
            padding: 7px;
        }}
        QProgressBar {{
            border: 1px solid {border_color};
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
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 7px;
        }}
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS['accent']}, stop:1 {COLORS['info']});
            border: 1px solid {border_color};
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
        """
        return style if isinstance(style, str) else ""
    except Exception:
        return ""


def dashboard_stylesheet() -> str:
    """Backward-compatible alias for the global application stylesheet."""
    return app_stylesheet()
