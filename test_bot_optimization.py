#!/usr/bin/env python3
"""
Test and Optimization Script for Reco-Trading Bot.
Tests different configurations for various currency pairs.
"""

import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# Test configurations for different pairs
PAIR_CONFIGS = {
    "BTCUSDT": {
        "min_signal_confidence": 0.50,
        "adx_min_threshold": 20.0,
        "max_spread_ratio": 0.005,
        "risk_per_trade_fraction": 0.015,
        "cooldown_minutes": 3,
        "description": "BTC - Moderate confidence, slightly higher risk"
    },
    "ETHUSDT": {
        "min_signal_confidence": 0.55,
        "adx_min_threshold": 22.0,
        "max_spread_ratio": 0.004,
        "risk_per_trade_fraction": 0.012,
        "cooldown_minutes": 5,
        "description": "ETH - Higher confidence, lower risk"
    },
    "SOLUSDT": {
        "min_signal_confidence": 0.45,
        "adx_min_threshold": 18.0,
        "max_spread_ratio": 0.006,
        "risk_per_trade_fraction": 0.010,
        "cooldown_minutes": 8,
        "description": "SOL - Lower confidence threshold, higher volatility tolerance"
    },
    "BNBUSDT": {
        "min_signal_confidence": 0.55,
        "adx_min_threshold": 23.0,
        "max_spread_ratio": 0.004,
        "risk_per_trade_fraction": 0.012,
        "cooldown_minutes": 5,
        "description": "BNB - Conservative settings"
    },
    "XRPUSDT": {
        "min_signal_confidence": 0.50,
        "adx_min_threshold": 20.0,
        "max_spread_ratio": 0.005,
        "risk_per_trade_fraction": 0.010,
        "cooldown_minutes": 6,
        "description": "XRP - Balanced settings"
    }
}


@dataclass
class TestResult:
    pair: str
    signals_generated: int
    trades_executed: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    avg_profit: float
    avg_loss: float
    filter_rejections: dict
    duration_minutes: float


class BotTester:
    def __init__(self):
        self.results: list[TestResult] = []
        self.current_pair = None
        self.signals_generated = 0
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.filter_rejections = {}
        self.start_time = None
        
    def reset(self, pair: str):
        self.current_pair = pair
        self.signals_generated = 0
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.filter_rejections = {
            "confidence": 0,
            "spread": 0,
            "volume": 0,
            "regime": 0,
            "cooldown": 0,
            "other": 0
        }
        self.start_time = time.time()

    def record_signal(self, passed_filters: bool, rejection_reason: str = None):
        self.signals_generated += 1
        if not passed_filters and rejection_reason:
            if rejection_reason in self.filter_rejections:
                self.filter_rejections[rejection_reason] += 1
            else:
                self.filter_rejections["other"] += 1

    def record_trade(self, pnl: float):
        self.trades_executed += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

    def get_result(self) -> TestResult:
        duration = (time.time() - self.start_time) / 60 if self.start_time else 0
        
        win_rate = (self.wins / self.trades_executed * 100) if self.trades_executed > 0 else 0
        avg_profit = self.total_pnl / self.wins if self.wins > 0 else 0
        avg_loss = self.total_pnl / self.losses if self.losses < 0 else 0
        
        return TestResult(
            pair=self.current_pair,
            signals_generated=self.signals_generated,
            trades_executed=self.trades_executed,
            wins=self.wins,
            losses=self.losses,
            total_pnl=self.total_pnl,
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            filter_rejections=self.filter_rejections.copy(),
            duration_minutes=duration
        )


async def simulate_signal_evaluation(tester: BotTester, pair: str, config: dict, iterations: int = 100):
    """Simulate signal evaluation with filters."""
    logger.info(f"Testing {pair}: {config.get('description', 'No description')}")
    
    tester.reset(pair)
    
    min_confidence = config.get("min_signal_confidence", 0.55)
    adx_threshold = config.get("adx_min_threshold", 22.0)
    max_spread = config.get("max_spread_ratio", 0.004)
    
    for i in range(iterations):
        # Generate random market conditions
        confidence = random.uniform(0.3, 0.95)
        adx = random.uniform(10, 40)
        spread = random.uniform(0.001, 0.01)
        volume_ratio = random.uniform(0.3, 2.0)
        regime = random.choice(["bull", "bear", "sideways", "high_vol"])
        
        # Apply filters
        passed = True
        rejection = None
        
        if confidence < min_confidence:
            passed = False
            rejection = "confidence"
        elif adx < adx_threshold:
            passed = False
            rejection = "adx"
        elif spread > max_spread:
            passed = False
            rejection = "spread"
        elif volume_ratio < 0.5:
            passed = False
            rejection = "volume"
        elif regime == "high_vol":
            passed = False
            rejection = "regime"
        
        tester.record_signal(passed, rejection)
        
        # Simulate trade execution for passed signals
        if passed and random.random() < 0.3:  # 30% execution rate
            pnl = random.uniform(-5, 15)  # Simulated PnL
            tester.record_trade(pnl)
        
        await asyncio.sleep(0.01)
    
    result = tester.get_result()
    return result


async def run_tests():
    """Run tests for all pairs."""
    logger.info("=" * 60)
    logger.info("STARTING RECO-TRADING BOT TEST")
    logger.info("=" * 60)
    
    tester = BotTester()
    results = []
    
    # Test each pair configuration
    for pair, config in PAIR_CONFIGS.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Testing pair: {pair}")
        logger.info(f"Configuration: {config.get('description', '')}")
        logger.info(f"{'='*40}")
        
        result = await simulate_signal_evaluation(tester, pair, config, iterations=100)
        results.append(result)
        
        # Print results
        logger.info(f"\nResults for {pair}:")
        logger.info(f"  Signals generated: {result.signals_generated}")
        logger.info(f"  Trades executed: {result.trades_executed}")
        logger.info(f"  Wins: {result.wins}, Losses: {result.losses}")
        logger.info(f"  Win rate: {result.win_rate:.1f}%")
        logger.info(f"  Total PnL: {result.total_pnl:.2f}%")
        logger.info(f"  Filter rejections: {result.filter_rejections}")
    
    return results


def analyze_results(results: list[TestResult]):
    """Analyze and print final results."""
    logger.info("\n" + "=" * 60)
    logger.info("FINAL ANALYSIS")
    logger.info("=" * 60)
    
    total_signals = sum(r.signals_generated for r in results)
    total_trades = sum(r.trades_executed for r in results)
    total_pnl = sum(r.total_pnl for r in results)
    total_wins = sum(r.wins for r in results)
    total_losses = sum(r.losses for r in results)
    
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    logger.info(f"\nTotal signals: {total_signals}")
    logger.info(f"Total trades: {total_trades}")
    logger.info(f"Total wins: {total_wins}, losses: {total_losses}")
    logger.info(f"Overall win rate: {overall_win_rate:.1f}%")
    logger.info(f"Total PnL: {total_pnl:.2f}%")
    
    # Best performing pair
    best_pair = max(results, key=lambda r: r.total_pnl) if results else None
    if best_pair:
        logger.info(f"\nBest performing pair: {best_pair.pair} (PnL: {best_pair.total_pnl:.2f}%)")
    
    # Filter analysis
    all_rejections = {}
    for r in results:
        for k, v in r.filter_rejections.items():
            all_rejections[k] = all_rejections.get(k, 0) + v
    
    logger.info(f"\nFilter rejection summary:")
    for reason, count in sorted(all_rejections.items(), key=lambda x: -x[1]):
        pct = count / total_signals * 100 if total_signals > 0 else 0
        logger.info(f"  {reason}: {count} ({pct:.1f}%)")
    
    return results


async def continuous_test_run(minutes: int = 20):
    """Run continuous test for specified minutes."""
    logger.info(f"\nStarting continuous test run for {minutes} minutes...")
    
    tester = BotTester()
    pair_order = list(PAIR_CONFIGS.keys())
    current_idx = 0
    
    start_time = time.time()
    iteration = 0
    
    while (time.time() - start_time) < minutes * 60:
        pair = pair_order[current_idx % len(pair_order)]
        config = PAIR_CONFIGS[pair]
        
        iteration += 1
        logger.info(f"\n[Iteration {iteration}] Testing {pair}")
        
        result = await simulate_signal_evaluation(tester, pair, config, iterations=50)
        
        # Print interim results
        logger.info(f"  Signals: {result.signals_generated}, Trades: {result.trades_executed}, PnL: {result.total_pnl:.2f}%")
        
        current_idx += 1
        
        # Small delay between pairs
        await asyncio.sleep(1)
    
    logger.info("\n" + "=" * 60)
    logger.info("CONTINUOUS TEST COMPLETE")
    logger.info("=" * 60)
    
    return analyze_results([tester.get_result()])


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reco-Trading Bot Test Runner")
    parser.add_argument("--mode", choices=["quick", "continuous"], default="quick",
                        help="Test mode: quick (10s) or continuous (20 minutes)")
    parser.add_argument("--minutes", type=int, default=20,
                        help="Minutes to run in continuous mode")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        logger.info("Running quick test mode...")
        results = await run_tests()
        analyze_results(results)
    else:
        logger.info(f"Running continuous test mode for {args.minutes} minutes...")
        await continuous_test_run(args.minutes)


if __name__ == "__main__":
    asyncio.run(main())
