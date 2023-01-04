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

    intervals_a = sort_intervals_by_start(intervals_a)
    intervals_b = sort_intervals_by_start(intervals_b)

    atoms, indices = atomize_intervals(
        [intervals_a, intervals_b],
        min_len=min_len,
        drop_gaps=False,
    )
    mask_a_atoms = (indices[:, 0] != 0) & (indices[:, 1] == 0)
    result, indices = atoms[mask_a_atoms], (indices[mask_a_atoms, 0] - 1).astype(int)

    if isinstance(intervals_a_input, pd.DataFrame):
        metadata = intervals_a_input.drop(INTERVAL_COL_NAMES, axis=1)
        metadata = metadata.iloc[indices].reset_index(drop=True)
        metadata[INTERVAL_COL_NAMES] = result
        result = metadata[intervals_a_input.columns]

    return result


# TODO test
def points_from_intervals(interval_groups: List[NDArray]):
    interval_points = []
    for i, intervals in enumerate(interval_groups):
        n_intervals = len(intervals)

        index_matrix = np.zeros((n_intervals, len(interval_groups)))
        index_matrix[:, i] = intervals[:, -1]
        index_matrix = np.concatenate([index_matrix, -index_matrix], axis=0)

        points = np.concatenate([intervals[:, 0:1], intervals[:, 1:2]], axis=0)
        points = np.concatenate([points, index_matrix], axis=1)

        interval_points.append(points)

    interval_points = np.concatenate(interval_points, axis=0)
    interval_points = interval_points[np.argsort(interval_points[:, 0]), :]
    interval_points[:, 1:] = np.cumsum(interval_points[:, 1:], axis=0)
    return interval_points


# TODO test
def atomize_intervals(
    interval_groups,
    min_len: Optional[float] = 0.0,
    drop_gaps: bool = True,
):
    for i in range(len(interval_groups)):
        interval_groups[i] = append_interval_idx_column(interval_groups[i])

    points = points_from_intervals(interval_groups)
    for i in range(len(interval_groups), 1, -1):
        points[points[:, i] != 0, 1:i] = 0

    starts, ends = points[:-1, 0:1], points[1:, 0:1]
    start_idxs = points[:-1, 1:]
    atomized_intervals = np.concatenate([starts, ends, start_idxs], axis=1)

    if drop_gaps:
        mask_nongap_intervals = np.sum(atomized_intervals[:, 2:], axis=1) != 0
        atomized_intervals = atomized_intervals[mask_nongap_intervals]

    if min_len is not None:
        interval_lengths = atomized_intervals[:, 1] - atomized_intervals[:, 0]
        mask_above_min_len = interval_lengths > min_len
        atomized_intervals = atomized_intervals[mask_above_min_len]

    atomized_intervals, interval_idxs = (
        atomized_intervals[:, :2],
        atomized_intervals[:, 2:],
    )
    return atomized_intervals, interval_idxs


def sort_intervals_by_start(intervals: NDArray) -> NDArray:
    """Sort an interval array by interval start."""
    return intervals[np.argsort(intervals[:, 0]), :]


# TODO test
def append_interval_idx_column(intervals):
    index = 1 + np.arange(len(intervals))
    return np.concatenate([intervals, index[:, None]], axis=1)
