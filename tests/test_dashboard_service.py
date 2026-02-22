from reco_trading.web.dashboard import DashboardService


def test_extract_binance_usdt_balance_supports_total_bucket():
    svc = DashboardService()
    out = svc._extract_binance_usdt_balance({'total': {'USDT': 123.45}})
    assert out == 123.45


def test_extract_binance_usdt_balance_supports_free_bucket_and_asset_bucket():
    svc = DashboardService()
    assert svc._extract_binance_usdt_balance({'free': {'USDT': 88.0}}) == 88.0
    assert svc._extract_binance_usdt_balance({'USDT': {'free': 77.0}}) == 77.0


def test_extract_binance_usdt_balance_handles_invalid_payload():
    svc = DashboardService()
    assert svc._extract_binance_usdt_balance({'total': {'BTC': 1.0}}) == 0.0
    assert svc._extract_binance_usdt_balance({}) == 0.0
