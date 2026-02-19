from trading_system.app.backtesting_engine.engine import BacktestingEngine


def test_backtesting_engine_metrics_shapes():
    engine = BacktestingEngine()
    returns = [0.002, -0.001, 0.003, 0.0, -0.002] * 50
    report = engine.run(returns)
    assert report.max_drawdown >= 0
    assert report.profit_factor >= 0
    wf = engine.walk_forward(returns, train=100, test=50)
    assert isinstance(wf, list)
