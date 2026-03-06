from __future__ import annotations

"""Deprecated compatibility shim.

Use `reco_trading.kernel.capital_governor.CapitalGovernor` as single source of truth.
"""

from reco_trading.kernel.capital_governor import CapitalGovernor, CapitalGovernorState, CapitalTicket

__all__ = ['CapitalGovernor', 'CapitalGovernorState', 'CapitalTicket']
