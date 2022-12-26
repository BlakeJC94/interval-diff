import numpy as np
from numpy.typing import NDArray

from .globals import EMPTY_INTERVALS

def sort_intervals_by_start(intervals: NDArray) -> NDArray:
    """Sort an interval array by interval start."""
    return intervals[np.argsort(intervals[:, 0]), :]


# TODO test
def generate_random_intervals(
    n_intervals: int,
    start: float = 0.0,
    max_len: float = 100.0,
    min_len: float = 10.0,
    precision: int = 2,
) -> NDArray:
    """Generate a specified number of random intervals"""
    if n_intervals < 1:
        return EMPTY_INTERVALS
    data = min_len + max_len * np.random.rand(2 * n_intervals - 1)
    data = np.around(data, precision)
    data = np.append(0.0, data)
    data = start + np.cumsum(data)
    starts, ends = data[0::2], data[1::2]
    return np.stack([starts, ends], axis=1)
