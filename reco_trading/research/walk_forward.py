from __future__ import annotations

import pandas as pd


class WalkForward:
    def generate_splits(
        self,
        frame: pd.DataFrame,
        train: int = 1000,
        test: int = 200,
        step: int = 100,
        min_train: int = 400,
    ):
        start = 0
        while start + min_train + test <= len(frame):
            train_end = min(start + train, len(frame) - test)
            train_slice = frame.iloc[start:train_end]
            test_slice = frame.iloc[train_end : train_end + test]
            if len(train_slice) >= min_train and len(test_slice) == test:
                yield train_slice, test_slice
            start += step
