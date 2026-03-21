from reco_trading.analytics.session_tracker import SessionTracker


def test_win_streak_is_tracked_from_tail() -> None:
    tracker = SessionTracker()
    for pnl in [-1.0, 2.0, 3.0, 4.0]:
        tracker.record(pnl)

    stats = tracker.stats()

    assert stats.current_streak == 3


def test_loss_streak_is_tracked_from_tail() -> None:
    tracker = SessionTracker()
    for pnl in [1.0, -2.0, -3.0]:
        tracker.record(pnl)

    stats = tracker.stats()

    assert stats.current_streak == -2


def test_pause_recommendation_after_three_losses() -> None:
    tracker = SessionTracker()
    for pnl in [-1.0, -2.0, -3.0]:
        tracker.record(pnl)

    stats = tracker.stats()

    assert stats.recommendation == "PAUSE"
    assert stats.size_multiplier == 0.0


def test_reduce_size_recommendation_after_two_losses() -> None:
    tracker = SessionTracker()
    for pnl in [1.0, 1.0, -1.0, -2.0]:
        tracker.record(pnl)

    stats = tracker.stats()

    assert stats.current_streak == -2
    assert stats.recommendation == "REDUCE_SIZE"
    assert stats.size_multiplier == 0.6


def test_profit_factor_and_win_rate_are_computed() -> None:
    tracker = SessionTracker()
    for pnl in [2.0, 3.0, -1.0, -1.0]:
        tracker.record(pnl)

    stats = tracker.stats()

    assert stats.profit_factor > 1.0
    assert stats.win_rate == 0.5


def test_recent_pnls_returns_last_ten_items() -> None:
    tracker = SessionTracker(max_samples=10)
    for pnl in range(15):
        tracker.record(float(pnl))

    assert tracker.recent_pnls == [float(v) for v in range(5, 15)]
