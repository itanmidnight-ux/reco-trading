from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reco_trading.ui.state_manager import StateManager as StateManagerType


__all__ = ["run_gui", "StateManager"]


def run_gui(state_manager: "StateManagerType") -> int:
    from reco_trading.ui.app import run_gui as _run_gui

    return _run_gui(state_manager)


def StateManager():
    from reco_trading.ui.state_manager import StateManager as _StateManager

    return _StateManager()
