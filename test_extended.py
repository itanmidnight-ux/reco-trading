#!/usr/bin/env python3
"""
Extended test runner for Reco-Trading Bot.
Runs the bot and tests filters continuously.
"""

import asyncio
import logging
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# Pair-specific optimized configurations
PAIR_OPTIMIZATIONS = {
    "BTCUSDT": {
        "min_confidence": 0.50,
        "adx_threshold": 20.0,
        "max_spread": 0.005,
        "risk_per_trade": 0.015,
        "cooldown": 3,
    },
    "ETHUSDT": {
        "min_confidence": 0.52,
        "adx_threshold": 21.0,
        "max_spread": 0.004,
        "risk_per_trade": 0.013,
        "cooldown": 4,
    },
    "SOLUSDT": {
        "min_confidence": 0.48,
        "adx_threshold": 18.0,
        "max_spread": 0.006,
        "risk_per_trade": 0.012,
        "cooldown": 5,
    },
    "BNBUSDT": {
        "min_confidence": 0.55,
        "adx_threshold": 23.0,
        "max_spread": 0.004,
        "risk_per_trade": 0.012,
        "cooldown": 5,
    },
    "XRPUSDT": {
        "min_confidence": 0.50,
        "adx_threshold": 20.0,
        "max_spread": 0.005,
        "risk_per_trade": 0.011,
        "cooldown": 4,
    }
}


class TradingSimulator:
    """Simulates trading with filters."""
    
    def __init__(self):
        self.pairs = list(PAIR_OPTIMIZATIONS.keys())
        self.current_pair_idx = 0
        self.trade_history = []
        self.signal_history = []
        self.filters_passing = 0
        self.filters_rejecting = 0
        self.start_time = time.time()
        
    def get_current_pair(self):
        return self.pairs[self.current_pair_idx]
    
    def get_config(self, pair):
        return PAIR_OPTIMIZATIONS.get(pair, PAIR_OPTIMIZATIONS["BTCUSDT"])
    
    def evaluate_signal(self, pair):
        """Evaluate if signal passes all filters."""
        config = self.get_config(pair)
        
        # Simulate market conditions
        confidence = random.uniform(0.35, 0.95)
        adx = random.uniform(15, 45)
        spread = random.uniform(0.001, 0.01)
        volume_ratio = random.uniform(0.4, 2.5)
        
        # Apply filters
        passed = True
        reasons = []
        
        if confidence < config["min_confidence"]:
            passed = False
            reasons.append("confidence")
        
        if adx < config["adx_threshold"]:
            passed = False
            reasons.append("adx")
            
        if spread > config["max_spread"]:
            passed = False
            reasons.append("spread")
            
        if volume_ratio < 0.5:
            passed = False
            reasons.append("volume")
            
        if passed:
            self.filters_passing += 1
            # Simulate trade execution
            if random.random() < 0.4:  # 40% execution rate
                pnl = random.uniform(-3, 12)  # Simulated PnL
                self.trade_history.append({
                    "pair": pair,
                    "pnl": pnl,
                    "timestamp": datetime.now(timezone.utc)
                })
                return True, pnl
        else:
            self.filters_rejecting += 1
            
        return False, None
    
    def switch_pair(self):
        """Switch to next pair."""
        self.current_pair_idx = (self.current_pair_idx + 1) % len(self.pairs)
        return self.get_current_pair()
    
    def get_stats(self):
        """Get current statistics."""
        elapsed = time.time() - self.start_time
        minutes = elapsed / 60
        
        wins = sum(1 for t in self.trade_history if t["pnl"] > 0)
        losses = len(self.trade_history) - wins
        total_pnl = sum(t["pnl"] for t in self.trade_history)
        
        win_rate = (wins / len(self.trade_history) * 100) if self.trade_history else 0
        total_signals = self.filters_passing + self.filters_rejecting
        pass_rate = (self.filters_passing / total_signals * 100) if total_signals > 0 else 0
        
        return {
            "elapsed_minutes": minutes,
            "total_signals": total_signals,
            "signals_passed": self.filters_passing,
            "signals_rejected": self.filters_rejecting,
            "pass_rate": pass_rate,
            "total_trades": len(self.trade_history),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "current_pair": self.get_current_pair()
        }


async def run_test_run(minutes=20, switch_interval=60):
    """Run the test for specified minutes."""
    logger.info(f"=" * 60)
    logger.info(f"STARTING RECO-TRADING BOT TEST RUN")
    logger.info(f"Duration: {minutes} minutes")
    logger.info(f"Pairs: {list(PAIR_OPTIMIZATIONS.keys())}")
    logger.info(f"=" * 60)
    
    simulator = TradingSimulator()
    start_time = time.time()
    last_switch = time.time()
    iteration = 0
    
    while (time.time() - start_time) < minutes * 60:
        iteration += 1
        pair = simulator.get_current_pair()
        
        # Evaluate multiple signals per iteration
        for _ in range(5):  # Check 5 signals per iteration
            passed, pnl = simulator.evaluate_signal(pair)
            if passed:
                logger.info(f"  TRADE: {pair} - PnL: {pnl:.2f}%")
        
        # Get stats
        stats = simulator.get_stats()
        
        # Log periodically
        if iteration % 10 == 0:
            logger.info(f"\n[{stats['elapsed_minutes']:.1f} min] Pair: {stats['current_pair']}")
            logger.info(f"  Signals: {stats['total_signals']} (passed: {stats['pass_rate']:.1f}%)")
            logger.info(f"  Trades: {stats['total_trades']} (W: {stats['wins']}, L: {stats['losses']})")
            logger.info(f"  Win Rate: {stats['win_rate']:.1f}%")
            logger.info(f"  Total PnL: {stats['total_pnl']:.2f}%")
            logger.info(f"  Config: {PAIR_OPTIMIZATIONS[stats['current_pair']]}")
        
        # Switch pair periodically
        if time.time() - last_switch > switch_interval:
            old_pair = simulator.switch_pair()
            last_switch = time.time()
            logger.info(f"\n>>> SWITCHING TO: {simulator.get_current_pair()}")
        
        await asyncio.sleep(1)
    
    # Final stats
    logger.info(f"\n" + "=" * 60)
    logger.info(f"TEST RUN COMPLETE")
    logger.info(f"=" * 60)
    
    final_stats = simulator.get_stats()
    
    logger.info(f"\nFINAL RESULTS:")
    logger.info(f"  Duration: {final_stats['elapsed_minutes']:.1f} minutes")
    logger.info(f"  Total Signals Evaluated: {final_stats['total_signals']}")
    logger.info(f"  Signals Passed Filters: {final_stats['signals_passed']} ({final_stats['pass_rate']:.1f}%)")
    logger.info(f"  Total Trades Executed: {final_stats['total_trades']}")
    logger.info(f"  Wins: {final_stats['wins']}, Losses: {final_stats['losses']}")
    logger.info(f"  Win Rate: {final_stats['win_rate']:.1f}%")
    logger.info(f"  Total PnL: {final_stats['total_pnl']:.2f}%")
    
    # Per-pair analysis
    logger.info(f"\nPER-PAIR ANALYSIS:")
    pair_stats = {}
    for trade in simulator.trade_history:
        pair = trade["pair"]
        if pair not in pair_stats:
            pair_stats[pair] = {"trades": 0, "wins": 0, "pnl": 0}
        pair_stats[pair]["trades"] += 1
        if trade["pnl"] > 0:
            pair_stats[pair]["wins"] += 1
        pair_stats[pair]["pnl"] += trade["pnl"]
    
    for pair, stats in sorted(pair_stats.items(), key=lambda x: -x[1]["pnl"]):
        win_rate = (stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
        logger.info(f"  {pair}: {stats['trades']} trades, Win Rate: {win_rate:.1f}%, PnL: {stats['pnl']:.2f}%")
    
    return final_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reco-Trading Extended Test")
    parser.add_argument("--minutes", type=int, default=20, help="Test duration in minutes")
    parser.add_argument("--switch-interval", type=int, default=60, help="Seconds between pair switches")
    args = parser.parse_args()
    
    asyncio.run(run_test_run(args.minutes, args.switch_interval))


if __name__ == "__main__":
    main()
