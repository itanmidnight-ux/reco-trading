from __future__ import annotations

_DARK_COLORS = {
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
    return dict(_LIGHT_COLORS if normalized in {"light", "white", "blanco"} else _DARK_COLORS)


def app_stylesheet(theme: str = "Dark") -> str:
    try:
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
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {colors['background']}, stop:1 {colors['background_alt']});
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
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {colors['accent']}, stop:1 {colors['info']});
        }}
        QFrame#metricCard, QFrame#panelCard {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {colors['panel']}, stop:1 {colors['panel_alt']});
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
        QTableWidget, QTextEdit {{
            background: {colors['panel']};
            border: 1px solid {colors['border']};
            border-radius: 10px;
            gridline-color: {colors['border']};
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
            background: {"#ffffff" if colors.get("background") == _LIGHT_COLORS["background"] else "#101933"};
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 7px;
        }}
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {colors['accent']}, stop:1 {colors['info']});
            border: 1px solid {colors['border']};
            border-radius: 8px;
            padding: 8px 12px;
            color: {colors['text_primary']};
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {colors['info']}, stop:1 {colors['accent']});
            border: 1px solid {colors['info']};
        }}
        QPushButton:pressed {{
            padding-top: 9px;
            padding-left: 13px;
            background: {colors['panel_alt']};
        }}
        QPushButton:disabled {{
            background: {colors['panel_alt']};
            color: {colors['neutral']};
        }}
        """
        return style if isinstance(style, str) else ""
    except Exception:
        return ""


def dashboard_stylesheet(theme: str = "Dark") -> str:
    """Backward-compatible alias for the global application stylesheet."""
    return app_stylesheet(theme=theme)
