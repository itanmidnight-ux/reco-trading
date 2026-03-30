"""
Backtesting Reports for Reco-Trading.
Generate detailed reports from backtest results.
"""

import json
import logging
from datetime import datetime
from typing import Any

from reco_trading.backtesting.performance_metrics import PerformanceMetrics


logger = logging.getLogger(__name__)


class BacktestReporter:
    """
    Generate reports from backtesting results.
    """
    
    @staticmethod
    def generate_report(results: dict[str, Any]) -> str:
        """
        Generate a text report from backtest results.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Formatted report string
        """
        lines = []
        
        lines.append("=" * 70)
        lines.append(f"{'BACKTESTING RESULTS':^70}")
        lines.append("=" * 70)
        
        lines.append(f"\nStrategy:     {results.get('strategy', 'Unknown')}")
        lines.append(f"Timeframe:    {results.get('timeframe', 'N/A')}")
        lines.append(f"Pairs:        {', '.join(results.get('pairs', []))}")
        lines.append(f"Timerange:    {results.get('timerange', 'N/A')}")
        
        if "start_time" in results:
            lines.append(f"Start Time:   {results.get('start_time')}")
        if "end_time" in results:
            lines.append(f"End Time:     {results.get('end_time')}")
        
        lines.append("\n" + "-" * 70)
        lines.append(f"{'TRADING STATS':^70}")
        lines.append("-" * 70)
        
        lines.append(f"Total Trades:       {results.get('total_trades', 0)}")
        lines.append(f"  Winning Trades:   {results.get('winning_trades', 0)}")
        lines.append(f"  Losing Trades:    {results.get('losing_trades', 0)}")
        lines.append(f"  Win Rate:         {results.get('win_rate', 0):.2%}")
        
        lines.append("\n" + "-" * 70)
        lines.append(f"{'PROFIT STATS':^70}")
        lines.append("-" * 70)
        
        lines.append(f"Total Profit:      {results.get('profit_total', 0):.2f}")
        lines.append(f"Average Profit:    {results.get('profit_mean', 0):.2f}")
        
        if results.get('trades'):
            profits = [t['profit'] for t in results['trades']]
            lines.append(f"Best Trade:         {max(profits):.2f}")
            lines.append(f"Worst Trade:       {min(profits):.2f}")
        
        lines.append("\n" + "-" * 70)
        lines.append(f"{'RISK STATS':^70}")
        lines.append("-" * 70)
        
        lines.append(f"Max Drawdown:      {results.get('max_drawdown', 0):.2%}")
        
        if results.get('avg_trade_duration'):
            lines.append(f"Avg Trade Duration: {results.get('avg_trade_duration', 0):.1f} minutes")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_json_report(results: dict[str, Any], filename: str | None = None) -> str:
        """
        Generate a JSON report from backtest results.
        
        Args:
            results: Backtest results dictionary
            filename: Optional filename to save to
            
        Returns:
            JSON string
        """
        report = {
            "strategy": results.get("strategy"),
            "timeframe": results.get("timeframe"),
            "pairs": results.get("pairs"),
            "timerange": results.get("timerange"),
            "metrics": {
                "total_trades": results.get("total_trades", 0),
                "winning_trades": results.get("winning_trades", 0),
                "losing_trades": results.get("losing_trades", 0),
                "win_rate": results.get("win_rate", 0),
                "profit_total": results.get("profit_total", 0),
                "profit_mean": results.get("profit_mean", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "avg_trade_duration": results.get("avg_trade_duration", 0),
            },
        }
        
        if "trades" in results:
            report["trades"] = [
                {
                    "pair": t["pair"],
                    "entry_price": t["entry_price"],
                    "exit_price": t["exit_price"],
                    "profit": t["profit"],
                    "profit_ratio": t.get("profit_ratio", 0),
                    "duration": t.get("duration", 0),
                    "enter_reason": t.get("enter_reason", ""),
                    "exit_reason": t.get("exit_reason", ""),
                }
                for t in results["trades"]
            ]
        
        json_str = json.dumps(report, indent=2, default=str)
        
        if filename:
            with open(filename, "w") as f:
                f.write(json_str)
            logger.info(f"Report saved to {filename}")
        
        return json_str
    
    @staticmethod
    def generate_summary(results: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a summary dictionary from results.
        
        Args:
            results: Backtest results
            
        Returns:
            Summary dictionary
        """
        return {
            "strategy": results.get("strategy"),
            "total_trades": results.get("total_trades", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_total": results.get("profit_total", 0),
            "max_drawdown": results.get("max_drawdown", 0),
        }
    
    @staticmethod
    def print_results(results: dict[str, Any]) -> None:
        """
        Print results to console.
        
        Args:
            results: Backtest results
        """
        print(BacktestReporter.generate_report(results))


def show_backtest_results(results: dict[str, Any]) -> None:
    """
    Display backtesting results.
    
    Args:
        results: Backtest results
    """
    BacktestReporter.print_results(results)
    
    
def store_backtest_results(results: dict[str, Any], filename: str) -> None:
    """
    Store backtest results to file.
    
    Args:
        results: Backtest results
        filename: Output filename
    """
    BacktestReporter.generate_json_report(results, filename)
