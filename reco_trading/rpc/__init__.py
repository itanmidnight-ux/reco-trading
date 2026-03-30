"""
RPC module for Reco-Trading.
Handles communication with external interfaces.
"""

from reco_trading.rpc.manager import RPCManager, RPCProxy
from reco_trading.rpc.notifications import NotificationManager, NotificationType

__all__ = [
    "RPCManager",
    "RPCProxy", 
    "NotificationManager",
    "NotificationType",
]
