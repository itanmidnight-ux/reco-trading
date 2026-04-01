#!/usr/bin/env python3
"""Reco-Trading Bot - Main entry point.
This file exists for backwards compatibility. Use: python -m reco_trading.main
"""
import sys
import os

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if __name__ == "__main__":
    from reco_trading import main as reco_main
    reco_main.run()