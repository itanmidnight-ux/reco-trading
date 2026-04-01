"""
Exchange module for Reco-Trading.
Provides unified exchange interface.
"""

from reco_trading.exchange.exchange import Exchange, create_exchange
from reco_trading.exchange.binance_client import BinanceClient

# Backward compatibility alias
RobustBinanceClient = BinanceClient

__all__ = ["Exchange", "create_exchange", "BinanceClient", "RobustBinanceClient"]
