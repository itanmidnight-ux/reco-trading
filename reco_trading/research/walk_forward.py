from __future__ import annotations

import pandas as pd


class WalkForward:
    def generate_splits(self, frame: pd.DataFrame, train: int = 800, test: int = 150):
        i = 0
        while i + train + test <= len(frame):
            yield frame.iloc[i:i+train], frame.iloc[i+train:i+train+test]
            i += test
