from __future__ import annotations

_DARK_COLORS = {
    "background": "#0b1020",
    "background_alt": "#111933",
    "panel": "#111a2e",
    "panel_alt": "#17243f",
    "border": "#30456f",
    "text_primary": "#edf2ff",
    "text_secondary": "#9fb2d9",
    "positive": "#22d39b",
    "negative": "#ff5f7b",
    "warning": "#ffcc66",
    "info": "#5a8dff",
    "neutral": "#7f93bf",
    "accent": "#7b61ff",
}

_MIDNIGHT_COLORS = {
    "background": "#05070f",
    "background_alt": "#0a1020",
    "panel": "#0c1326",
    "panel_alt": "#121d39",
    "border": "#2b3e67",
    "text_primary": "#f1f5ff",
    "text_secondary": "#9cb0d8",
    "positive": "#22d39b",
    "negative": "#ff6b88",
    "warning": "#ffca70",
    "info": "#60a5ff",
    "neutral": "#7f93bf",
    "accent": "#8b7bff",
}

_LIGHT_COLORS = {
    "background": "#f5f7fb",
    "background_alt": "#e8edf7",
    "panel": "#ffffff",
    "panel_alt": "#f1f5fc",
    "border": "#c7d3ea",
    "text_primary": "#12213f",
    "text_secondary": "#425a85",
    "positive": "#0f9f6e",
    "negative": "#d6405c",
    "warning": "#b7791f",
    "info": "#2c5bd8",
    "neutral": "#6f84b0",
    "accent": "#4b54e6",
}



def get_theme_colors(theme: str = "Dark") -> dict[str, str]:
    normalized = str(theme or "Dark").strip().lower()
    if normalized in {"light", "white", "blanco"}:
        return dict(_LIGHT_COLORS)
    if normalized in {"midnight", "amoled", "true dark"}:
        return dict(_MIDNIGHT_COLORS)
    return dict(_DARK_COLORS)


def app_stylesheet(theme: str = "Dark") -> str:
    try:
        normalized = str(theme or "Dark").strip().lower()
        colors = get_theme_colors(theme)
        style = f"""
        QWidget {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            font-family: Inter, Segoe UI, Arial;
            selection-background-color: rgba(122, 97, 255, 0.4);
        }}
        QMainWindow, QTabWidget::pane, QScrollArea {{
            border: 1px solid {colors['border']};
            border-radius: 12px;
            background: {colors['background']};
        }}
        QScrollArea > QWidget > QWidget {{
            background: transparent;
        }}
        QTabBar::tab {{
            background: {colors['panel']};
            color: {colors['text_secondary']};
            border: 1px solid {colors['border']};
            border-bottom: none;
            padding: 11px 18px;
            margin-right: 6px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            font-weight: 600;
        }}
        QTabBar::tab:hover {{
            color: {colors['text_primary']};
            background: {colors['panel_alt']};
        }}
        QTabBar::tab:selected {{
            color: white;
            background: {colors['accent']};
            border-color: {colors['info']};
        }}
        QFrame#metricCard, QFrame#panelCard {{
            background: {colors['panel']};
            border: 1px solid {colors['border']};
            border-radius: 12px;
        }}
        QLabel#sectionTitle {{
            color: {colors['text_primary']};
            font-size: 17px;
            font-weight: 700;
            letter-spacing: 0.4px;
        }}
        QLabel#metricLabel {{
            color: {colors['text_secondary']};
            font-size: 11px;
            text-transform: uppercase;
        }}
        QLabel#metricValue {{
            color: {colors['text_primary']};
            font-size: 19px;
            font-weight: 700;
        }}
        QLabel#statusRibbon {{
            color: {colors['text_primary']};
            font-size: 15px;
            font-weight: 700;
            padding: 12px 16px;
            border-radius: 14px;
            border: 1px solid rgba(122, 97, 255, 0.35);
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {colors['panel']}, stop:1 {colors['panel_alt']});
        }}
        QLabel#smallMetricValue {{
            color: {colors['text_primary']};
            font-size: 14px;
            font-weight: 600;
        }}
        QLabel#metricBadge {{
            color: {colors['text_secondary']};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.6px;
        }}
        QTableWidget, QTextEdit, QListWidget {{
            background: {colors['panel']};
            border: 1px solid {colors['border']};
            border-radius: 10px;
            gridline-color: {colors['border']};
            alternate-background-color: {colors['panel_alt']};
        }}
        QTableWidget::item, QListWidget::item {{
            padding: 6px 8px;
            border-bottom: 1px solid rgba(127, 147, 191, 0.16);
        }}
        QTableWidget::item:selected, QListWidget::item:selected {{
            background: rgba(90, 141, 255, 0.28);
            color: {colors['text_primary']};
        }}
        QTableWidget::item:hover, QListWidget::item:hover {{
            background: rgba(90, 141, 255, 0.16);
        }}
        QHeaderView::section {{
            background: {colors['panel_alt']};
            color: {colors['text_secondary']};
            border: none;
            padding: 7px;
        }}
        QProgressBar {{
            border: 1px solid {colors['border']};
            border-radius: 6px;
            text-align: center;
            background: {colors['panel']};
            min-height: 12px;
        }}
        QProgressBar::chunk {{
            background-color: {colors['info']};
            border-radius: 6px;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {"#ffffff" if normalized in {"light", "white", "blanco"} else "#0d1528"};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 7px;
        }}
        QPushButton {{
            background: {colors['accent']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 8px 12px;
            color: {colors['text_primary']};
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {colors['info']};
            border: 1px solid {colors['info']};
        }}
        QPushButton:pressed {{
            padding-top: 9px;
            padding-left: 13px;
            background: {colors['panel_alt']};
        }}
        QPushButton:disabled {{
            background: #2a3557;
            color: {colors['neutral']};
        }}

        QStatusBar {{
            background: {colors['panel']};
            color: {colors['text_secondary']};
            border-top: 1px solid {colors['border']};
            padding: 4px;
        }}
        QStatusBar QLabel {{
            color: {colors['text_secondary']};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 4px 8px;
            margin-left: 4px;
            background: {colors['panel_alt']};
            font-size: 11px;
            font-weight: 600;
        }}

        QScrollBar:vertical {{
            background: {colors['panel']};
            width: 10px;
            margin: 4px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: {colors['border']};
            min-height: 24px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {colors['info']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
            height: 0;
        }}
        QToolTip {{
            background: {colors['panel_alt']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
            padding: 6px 8px;
            border-radius: 6px;
        }}
        """
        return style if isinstance(style, str) else ""
    except Exception:
        return ""


def dashboard_stylesheet(theme: str = "Dark") -> str:
    """Backward-compatible alias for the global application stylesheet."""
    return app_stylesheet(theme=theme)
