from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .globals import EMPTY_INTERVALS, INTERVAL_COL_NAMES
from .vectorised  import filter_overlapping_intervals

DEFAULT_TAGS = list("QWERTY")


# TODO test
def generate_random_intervals(
    n_intervals: int,
    start: float = 0.0,
    max_len: float = 100.0,
    min_len: float = 10.0,
    precision: int = 2,
    dataframe: bool = False,
) -> NDArray:
    """Generate a specified number of random intervals"""
    if n_intervals < 1:
        return EMPTY_INTERVALS
    data = min_len + max_len * np.random.rand(2 * n_intervals - 1)
    data = np.around(data, precision)
    data = np.append(0.0, data)
    data = start + np.cumsum(data)
    starts, ends = data[0::2], data[1::2]
    results = np.stack([starts, ends], axis=1)
    if not dataframe:
        return results
    results = pd.DataFrame(results, columns=INTERVAL_COL_NAMES)
    results["tags"] = [
        DEFAULT_TAGS[i] for i in np.random.randint(0, len(DEFAULT_TAGS), n_intervals)
    ]
    return results


# TODO deprecate
def complement_intervals(
    intervals: NDArray,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> NDArray:
    """Calculate the complement of an interval array (with optional specification of boundaries)."""
    if len(intervals) == 0:
        if minimum is not None and maximum is not None:
            return np.array([(minimum, maximum)])
        return intervals

    first_start, starts = intervals[0, 0], intervals[1:, 0]
    last_end, ends = intervals[-1, 1], intervals[:-1, 1]

    result = np.stack([ends, starts], axis=1)

    if minimum is not None and minimum < first_start:
        left_complement = np.array([(minimum, first_start)])
        result = np.concatenate([left_complement, result], axis=0)

    if maximum is not None and last_end < maximum:
        right_complement = np.array([(last_end, maximum)])
        result = np.concatenate([result, right_complement], axis=0)

    return result


# TODO deprecate
def drop_total_overlaps(
    intervals_a: NDArray,
    intervals_b: NDArray,
    eps: float = 1e-8,
) -> NDArray:
    """Drop all intervals in A that are completely covered by an interval in B."""
    if len(intervals_a) == 0 or len(intervals_b) == 0:
        return intervals_a

    minimum = min(np.min(intervals_a[:, 0]), np.min(intervals_b[:, 0]))
    maximum = max(np.max(intervals_a[:, 1]), np.max(intervals_b[:, 1]))

    intervals_b = intervals_b.astype(float)
    intervals_b[:, 0] -= eps
    intervals_b[:, 1] += eps

    intervals_b_complement = complement_intervals(
        intervals_b,
        minimum=minimum,
        maximum=maximum,
    )
    a_no_total_overlap, _ = filter_overlapping_intervals(
        intervals_a,
        intervals_b_complement,
    )
    return a_no_total_overlap


# TODO deprecate
def create_intervals_from_points(
    points: NDArray,
    min_len: float = 2e-8,
) -> NDArray:
    """Create an interval array from a vector of points."""
    if len(points) < 2:
        return np.empty((0, 2))
    starts, ends = points[:-1], points[1:]
    intervals = np.stack([starts, ends], axis=1)
    mask = intervals[:, 1] - intervals[:, 0] > min_len
    return intervals[mask, :]
