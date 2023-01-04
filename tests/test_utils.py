import pytest
import numpy as np
import pandas as pd

from interval_diff.utils import (
    generate_random_intervals,
    DEFAULT_TAGS,
)


@pytest.mark.parametrize("n_intervals", [5, 100])
class TestGenerateRandomIntervals:
    start = 100
    max_len = 100
    min_len = 10
    n_samples = 20

    def test_array(self, n_intervals):
        dataframe = False

        for _ in range(self.n_samples):
            intervals = generate_random_intervals(
                n_intervals,
                start=self.start,
                max_len=self.max_len,
                min_len=self.min_len,
                dataframe=dataframe,
            )

            durations = intervals[:, 1] - intervals[:, 0]
            inter_interval_gaps = intervals[1:, 0] - intervals[:-1, 1]

            assert isinstance(intervals, np.ndarray)
            assert len(intervals) == n_intervals
            assert min(durations) > 0
            assert max(durations) <= self.max_len
            assert min(inter_interval_gaps) > 0

    def test_dataframe(self, n_intervals):
        dataframe = True

        for _ in range(self.n_samples):
            intervals = generate_random_intervals(
                n_intervals,
                start=self.start,
                max_len=self.max_len,
                min_len=self.min_len,
                dataframe=dataframe,
            )

            durations = intervals['end'] - intervals['start']
            inter_interval_gaps = intervals['start'].values[1:] - intervals['end'].values[:-1]

            assert isinstance(intervals, pd.DataFrame)
            assert len(intervals) == n_intervals
            assert min(durations) >= self.min_len
            assert max(durations) <= self.max_len
            assert min(inter_interval_gaps) > 0
            assert set(intervals.tags).issubset(set(DEFAULT_TAGS))
