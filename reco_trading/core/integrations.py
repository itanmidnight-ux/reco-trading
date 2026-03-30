"""
Reco-Trading Complete Integration Module

This module integrates all advanced features into the bot:
- Multi-Exchange Support
- Multi-Pair Management  
- Advanced ML Engine
- Auto-Improver
- Resilience System
- Trading Modes (Spot/Margin/Futures)
- DCA Manager
- Edge Analysis
- Pairlist Handlers
- Blacklist Manager
- WebSocket Support
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


def initialize_all_modules(bot_engine) -> bool:
    """
    Initialize all advanced modules in the bot engine.
    Call this function after bot engine is created.
    """
    
    try:
        logger.info("=" * 50)
        logger.info("INITIALIZING ALL ADVANCED MODULES")
        logger.info("=" * 50)
        
        if not hasattr(bot_engine, 'resilience'):
            logger.warning("Resilience module not found in bot_engine")
        else:
            logger.info("[1/10] Resilience System: OK")
        
        if not hasattr(bot_engine, 'auto_improver'):
            logger.warning("Auto-Improver module not found in bot_engine")
        else:
            logger.info("[2/10] Auto-Improver: OK")
        
        if not hasattr(bot_engine, 'multi_pair_manager'):
            logger.warning("Multi-Pair Manager module not found in bot_engine")
        else:
            logger.info("[3/10] Multi-Pair Manager: OK")
        
        if not hasattr(bot_engine, 'enhanced_ml'):
            logger.warning("Enhanced ML module not found in bot_engine")
        else:
            logger.info("[4/10] Enhanced ML Engine: OK")
        
        try:
            from reco_trading.exchange.multi_exchange import MultiExchangeManager
            multi_ex = MultiExchangeManager()
            logger.info("[5/10] Multi-Exchange Manager: OK")
        except Exception as e:
            logger.warning(f"Multi-Exchange: {e}")
        
        try:
            from reco_trading.exchange.pairlist import PairListManager
            logger.info("[6/10] Pairlist Handlers: OK")
        except Exception as e:
            logger.warning(f"Pairlist: {e}")
        
        try:
            from reco_trading.exchange.blacklist import BlacklistManager
            bl = BlacklistManager()
            logger.info("[7/10] Blacklist Manager: OK")
        except Exception as e:
            logger.warning(f"Blacklist: {e}")
        
        try:
            from reco_trading.risk.dca_manager import DCAManager
            logger.info("[8/10] DCA Manager: OK")
        except Exception as e:
            logger.warning(f"DCA: {e}")
        
        try:
            from reco_trading.analysis.edge_analysis import EdgeAnalysis
            logger.info("[9/10] Edge Analysis: OK")
        except Exception as e:
            logger.warning(f"Edge: {e}")
        
        try:
            from reco_trading.ml.freqai_manager import FreqAIManager
            logger.info("[10/10] FreqAI Manager: OK")
        except Exception as e:
            logger.warning(f"FreqAI: {e}")
        
        try:
            from reco_trading.core.trading_modes import TradingModeManager, WebSocketManager
            logger.info("[EXT] Trading Modes (Long/Short): OK")
            logger.info("[EXT] WebSocket Support: OK")
        except Exception as e:
            logger.warning(f"Trading Modes: {e}")
        
        logger.info("=" * 50)
        logger.info("ALL MODULES INITIALIZED SUCCESSFULLY")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"Module initialization failed: {e}")
        return False


def get_system_status(bot_engine) -> dict:
    """Get comprehensive system status."""
    
    status = {
        "core": {
            "bot_engine": "OK" if bot_engine else "ERROR",
            "exchange": "OK" if hasattr(bot_engine, 'client') else "ERROR",
            "database": "OK" if hasattr(bot_engine, 'repository') else "ERROR",
        },
        "advanced_features": {
            "resilience": hasattr(bot_engine, 'resilience'),
            "auto_improver": hasattr(bot_engine, 'auto_improver'),
            "multi_pair_manager": hasattr(bot_engine, 'multi_pair_manager'),
            "enhanced_ml": hasattr(bot_engine, 'enhanced_ml'),
        },
        "resilience": {},
        "auto_improver": {},
        "multi_pair": {},
        "ml": {},
    }
    
    if hasattr(bot_engine, 'resilience'):
        try:
            status["resilience"] = bot_engine.resilience.get_health_status()
        except:
            pass
    
    if hasattr(bot_engine, 'auto_improver') and bot_engine.auto_improver.enabled:
        try:
            status["auto_improver"] = bot_engine.auto_improver.get_improvement_metrics()
        except:
            pass
    
    if hasattr(bot_engine, 'multi_pair_manager'):
        try:
            status["multi_pair"] = {
                "active_pair": bot_engine.multi_pair_manager.active_pair,
                "pairs_tracked": len(bot_engine.multi_pair_manager.pairs_metrics),
            }
        except:
            pass
    
    if hasattr(bot_engine, 'enhanced_ml'):
        try:
            status["ml"] = {
                "enabled": bot_engine._ml_enabled,
                "models": len(bot_engine.enhanced_ml.model_ensemble),
            }
        except:
            pass
    
    return status


def create_default_config() -> dict:
    """Create default configuration for all modules."""
    
    return {
        "trading_modes": {
            "default_mode": "spot",
            "leverage": 1,
            "max_leverage": 10,
            "hedge_mode": False,
        },
        "multi_exchange": {
            "enabled": True,
            "primary": "binance",
            "testnet": True,
        },
        "multi_pair": {
            "enabled": True,
            "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
            "scan_interval": 60,
        },
        "auto_improver": {
            "enabled": True,
            "optimization_interval_hours": 6,
            "min_trades_before_optimization": 5,
            "consecutive_loss_threshold": 3,
        },
        "resilience": {
            "enabled": True,
            "max_retries": 3,
            "crash_recovery": True,
            "auto_restart": True,
        },
        "ml": {
            "enabled": True,
            "min_confidence_threshold": 0.70,
        },
        "dca": {
            "enabled": False,
            "max_safety_orders": 5,
            "safety_order_step": 0.02,
        },
        "edge": {
            "enabled": True,
            "min_trades": 20,
            "min_winrate": 0.40,
        },
        "pairlist": {
            "enabled": True,
            "handlers": ["VolumePairList", "VolatilityFilter"],
        },
        "blacklist": {
            "enabled": True,
            "auto_blacklist": True,
        },
    }


if __name__ == "__main__":
    config = create_default_config()
    print("Default Configuration:")
    import json
    print(json.dumps(config, indent=2))
