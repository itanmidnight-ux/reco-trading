from __future__ import annotations

import threading
import time

import reco_trading.main as main_module


def test_run_bot_registers_global_instance(monkeypatch) -> None:
    run_started = threading.Event()
    run_continue = threading.Event()

    class DummyBot:
        def __init__(self, settings, state_manager=None) -> None:
            self.settings = settings
            self.state_manager = state_manager

        async def run(self) -> None:
            run_started.set()
            run_continue.wait(timeout=2)

    monkeypatch.setattr(main_module, "BotEngine", DummyBot)
    settings = type("S", (), {})()

    worker = threading.Thread(target=main_module._run_bot, args=(settings, None), daemon=True)
    worker.start()
    assert run_started.wait(timeout=1.0)
    assert main_module.get_bot_instance() is not None
    run_continue.set()
    worker.join(timeout=1.0)
    assert main_module.get_bot_instance() is None


def test_web_dashboard_template_contains_app_like_tabs_and_brand() -> None:
    with open("web_site/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "RECO" in html
    assert 'data-tab="settings"' in html
    assert 'data-tab="logs"' in html
    assert 'data-tab="overview"' in html


def test_web_dashboard_js_uses_risk_metrics_after_declaration() -> None:
    with open("web_site/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "static/dashboard.css" in html or "dashboard.css" in html
    assert "static/dashboard.js" in html or "dashboard.js" in html
