from __future__ import annotations

import pandas as pd


class WalkForward:
    def generate_splits(self, frame: pd.DataFrame, train: int = 1000, test: int = 200, step: int = 200):
        start = 0
        while start + train + test <= len(frame):
            train_slice = frame.iloc[start : start + train]
            test_slice = frame.iloc[start + train : start + train + test]
            yield train_slice, test_slice
            start += step
