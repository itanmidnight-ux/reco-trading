from trading_system.app.backtesting.simulator import BacktestingSimulator

if __name__ == '__main__':
    m = BacktestingSimulator().run(500)
    print(m)
