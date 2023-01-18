from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .globals import INTERVAL_COL_NAMES


# TODO test dataframe inputs
def interval_difference(
    intervals_a: Union[NDArray, pd.DataFrame],
    intervals_b: Union[NDArray, pd.DataFrame],
    min_len: float = 0.0,
) -> NDArray:
    """Chop out sub-intervals from A that overlap with B.

    Args:
        intervals_a: Array representing intervals (col 0/1 represent start/end).
        intervals_b: Array representing intervals (col 0/1 represent start/end).
        min_len: minimum allowable length of intervals to keep, intervals shorter than min_len will
            be dropped.

    Returns:
        Interval difference between intervals_a and intervals_b
    """
    if len(intervals_a) == 0 or len(intervals_b) == 0:
        return intervals_a

    intervals_a_input = intervals_a
    if isinstance(intervals_a, pd.DataFrame):
        intervals_a_input = intervals_a.copy()
        intervals_a = intervals_a[INTERVAL_COL_NAMES].values

    if isinstance(intervals_b, pd.DataFrame):
        intervals_b = intervals_b[INTERVAL_COL_NAMES].values

    atoms, indices = atomize_intervals(
        [intervals_a, intervals_b],
        min_len=min_len,
        drop_gaps=False,
    )
    mask_a_atoms = (indices[:, 0] != -1) & (indices[:, 1] == -1)
    result, indices = atoms[mask_a_atoms], indices[mask_a_atoms, 0]

    if isinstance(intervals_a_input, pd.DataFrame):
        metadata = intervals_a_input.drop(INTERVAL_COL_NAMES, axis=1)
        metadata = metadata.iloc[indices].reset_index(drop=True)
        metadata[INTERVAL_COL_NAMES] = result
        result = metadata[intervals_a_input.columns]

    return result


# TODO test
def points_from_intervals(interval_groups: List[NDArray]) -> Tuple[NDArray]:
    n_interval_groups = len(interval_groups)
    interval_points, interval_indices = [], []
    for i, intervals in enumerate(interval_groups):
        assert not intervals_overlapping(
            intervals
        ), "Expected the intervals within a group to be non-overlapping"
        n_intervals = len(intervals)

        indices = np.zeros((n_intervals, n_interval_groups))
        indices[:, i] = np.arange(n_intervals) + 1
        indices = np.concatenate([indices, -indices], axis=0)

        points = np.concatenate([intervals[:, 0:1], intervals[:, 1:2]], axis=0)

        interval_points.append(points)
        interval_indices.append(indices)

    interval_points = np.concatenate(interval_points, axis=0)
    interval_indices = np.concatenate(interval_indices, axis=0)

    foo = np.argsort(interval_points[:, 0])
    interval_points = interval_points[foo, :]
    interval_indices = interval_indices[foo, :]

    interval_indices = np.cumsum(interval_indices, axis=0) - 1
    return interval_points, interval_indices


def intervals_overlapping(intervals: NDArray) -> bool:
    intervals = intervals[np.argsort(intervals[:, 0]), :]
    starts, ends = intervals[:, 0], intervals[:, 1]
    overlaps = starts[1:] - ends[:-1]
    return (overlaps < 0).any()


# TODO test
def atomize_intervals(
    interval_groups,
    min_len: Optional[float] = 0.0,
    drop_gaps: bool = True,
) -> Tuple[NDArray, NDArray]:
    points, indices = points_from_intervals(interval_groups)
    for i in range(1, len(interval_groups)):
        indices[indices[:, i] != -1, :i] = -1

    starts, ends = points[:-1, 0:1], points[1:, 0:1]
    interval_idxs = indices[:-1].astype(int)
    atomized_intervals = np.concatenate([starts, ends], axis=1)

    if drop_gaps:
        mask_nongap_intervals = (interval_idxs != -1).any(axis=1)

        atomized_intervals = atomized_intervals[mask_nongap_intervals]
        interval_idxs = interval_idxs[mask_nongap_intervals]

    if min_len is not None:
        interval_lengths = atomized_intervals[:, 1] - atomized_intervals[:, 0]
        mask_above_min_len = interval_lengths > min_len

        atomized_intervals = atomized_intervals[mask_above_min_len]
        interval_idxs = interval_idxs[mask_above_min_len]

    return atomized_intervals, interval_idxs
