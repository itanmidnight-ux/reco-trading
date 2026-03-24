from __future__ import annotations

LANG_EN = "English"
LANG_ES = "Español"
SUPPORTED_LANGUAGES = (LANG_EN, LANG_ES)

_TRANSLATIONS: dict[str, dict[str, str]] = {
    LANG_EN: {
        "window_title": "Reco Trading Professional Terminal",
        "tab.dashboard": "Dashboard",
        "tab.trades": "Trades",
        "tab.market": "Market",
        "tab.analytics": "Analytics",
        "tab.strategy": "Strategy",
        "tab.logs": "Logs",
        "tab.risk": "Risk",
        "tab.settings": "Settings",
        "tab.system": "System",
        "settings.title": "Interface Studio",
        "settings.description": "Customize visual behavior, refresh cadence and session-safe controls",
        "settings.refresh_rate": "Refresh rate (ms)",
        "settings.chart_visibility": "Chart visibility",
        "settings.theme": "Theme",
        "settings.language": "Language",
        "settings.log_verbosity": "Log verbosity",
        "settings.default_pair": "Default pair",
        "settings.default_timeframe": "Default timeframe",
        "settings.investment_mode": "Investment mode",
        "settings.capital_limit": "Capital limit",
        "settings.pair_budget": "Per-pair budget",
        "settings.risk_per_trade": "Risk per trade",
        "settings.max_allocation": "Max allocation",
        "settings.reserve_ratio": "Reserve ratio",
        "settings.cash_buffer": "Cash buffer",
        "settings.load_keys": "Load current keys",
        "settings.save_keys": "Apply to session",
        "settings.apply_now": "Apply now",
        "settings.sim_prefix": "Estimated max order",
        "settings.optimizer_waiting": "Optimizer: waiting for account snapshot",
    },
    LANG_ES: {
        "window_title": "Reco Trading Terminal Profesional",
        "tab.dashboard": "Panel",
        "tab.trades": "Trades",
        "tab.market": "Mercado",
        "tab.analytics": "Analítica",
        "tab.strategy": "Estrategia",
        "tab.logs": "Logs",
        "tab.risk": "Riesgo",
        "tab.settings": "Ajustes",
        "tab.system": "Sistema",
        "settings.title": "Estudio de Interfaz",
        "settings.description": "Personaliza visuales, ritmo de refresco y controles seguros de sesión",
        "settings.refresh_rate": "Frecuencia (ms)",
        "settings.chart_visibility": "Mostrar gráfico",
        "settings.theme": "Tema",
        "settings.language": "Idioma",
        "settings.log_verbosity": "Nivel de logs",
        "settings.default_pair": "Par por defecto",
        "settings.default_timeframe": "Temporalidad por defecto",
        "settings.investment_mode": "Modo de inversión",
        "settings.capital_limit": "Límite de capital",
        "settings.pair_budget": "Presupuesto por par",
        "settings.risk_per_trade": "Riesgo por trade",
        "settings.max_allocation": "Asignación máxima",
        "settings.reserve_ratio": "Ratio de reserva",
        "settings.cash_buffer": "Buffer de efectivo",
        "settings.load_keys": "Cargar llaves actuales",
        "settings.save_keys": "Aplicar a sesión",
        "settings.apply_now": "Aplicar ahora",
        "settings.sim_prefix": "Orden máxima estimada",
        "settings.optimizer_waiting": "Optimizador: esperando snapshot de cuenta",
    },
}


def normalize_language(language: str) -> str:
    return LANG_ES if str(language).strip().lower().startswith("es") else LANG_EN


def tr(key: str, language: str) -> str:
    normalized = normalize_language(language)
    value = _TRANSLATIONS.get(normalized, {}).get(key)
    if value is not None:
        return value
    return _TRANSLATIONS[LANG_EN].get(key, key)
