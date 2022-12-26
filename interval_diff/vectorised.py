from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .vis import plot_intervals


# TODO handle interval metadata as well
# IDEA append another column to intervals_a and intervals_b to track which interval is which
def interval_difference(intervals_a: NDArray, intervals_b: NDArray) -> NDArray:
    """Chop out sub-intervals from A that overlap with B.

    Args:
        intervals_a: Array representing intervals (col 0/1 represent start/end).
        intervals_b: Array representing intervals (col 0/1 represent start/end).

    Returns:
        Interval difference between intervals_a and intervals_b
    """
    if len(intervals_a) == 0:
        return intervals_a

    intervals_b, _ = filter_overlapping_intervals(
        intervals_a=intervals_b,
        intervals_b=intervals_a,
    )
    if len(intervals_b) == 0:
        return intervals_a

    intervals_a, intervals_a_non_overlap = filter_overlapping_intervals(intervals_a, intervals_b)

    atoms, indices = atomize_intervals(intervals_a, intervals_b)
    mask_a_atoms = (indices[:, 0] != 0) & (indices[:, 1] == 0)
    result = atoms[mask_a_atoms]

    return concat_interval_groups([result, intervals_a_non_overlap])


def sort_intervals_by_start(intervals: NDArray) -> NDArray:
    """Sort an interval array by interval start."""
    return intervals[np.argsort(intervals[:, 0]), :]


def concat_interval_groups(
    interval_groups: List[NDArray],
    sort: bool = True,
) -> NDArray:
    """Concatenate a list of interval arrays and sort result by interval start."""
    result = np.concatenate(interval_groups, axis=0)
    if not sort:
        return result
    return sort_intervals_by_start(result)


# TODO test
def points_from_intervals(interval_groups: List[NDArray]):
    interval_points = []
    for i, intervals in enumerate(interval_groups):
        n_intervals = len(intervals)

        index_matrix = np.zeros((n_intervals, len(interval_groups)))
        index_matrix[:, i] = 1 + np.arange(n_intervals)
        index_matrix = np.concatenate([index_matrix, -index_matrix], axis=0)

        points = np.concatenate([intervals[:, 0:1], intervals[:, 1:2]], axis=0)
        points = np.concatenate([points, index_matrix], axis=1)

        interval_points.append(points)

    interval_points = np.concatenate(interval_points, axis=0)
    interval_points = interval_points[np.argsort(interval_points[:, 0]), :]
    interval_points[:, 1:] = np.cumsum(interval_points[:, 1:], axis=0)
    return interval_points


# TODO test
def atomize_intervals(intervals_a, intervals_b, min_len: Optional[float] = 2e-8):
    interval_groups = [intervals_a, intervals_b]

    points = points_from_intervals([intervals_a, intervals_b])

    for i in range(len(interval_groups), 1, -1):
        points[points[:, i] != 0, 1:i] = 0

    starts, ends = points[:-1, 0:1], points[1:, 0:1]
    start_idxs = points[:-1, 1:]
    atomized_intervals = np.concatenate([starts, ends, start_idxs], axis=1)

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


# TODO test
def filter_overlapping_intervals_idxs(
    intervals_a: NDArray,
    intervals_b: NDArray,
) -> Tuple[NDArray, NDArray]:
    # Find the index at which intervals_b starts/end would be inserted in the intervals_a
    start_insert_idxs = np.searchsorted(intervals_b[:, 1], intervals_a[:, 0])
    end_insert_idxs = np.searchsorted(intervals_b[:, 0], intervals_a[:, 1])

    # When the insertion index is the same for both minuend start and end, then the minuend has no
    # overlapping subtrahend intervals
    mask_a_some_overlap = start_insert_idxs != end_insert_idxs
    mask_a_no_overlap = start_insert_idxs == end_insert_idxs

    return mask_a_some_overlap, mask_a_no_overlap


def filter_overlapping_intervals(
    intervals_a: NDArray,
    intervals_b: NDArray,
) -> Tuple[NDArray, NDArray]:
    """Partition set of intervals A to intervals that have some overlap with B and intervals with
    no overlap in B.
    """
    mask_a_some_overlap, mask_a_no_overlap = filter_overlapping_intervals_idxs(
        intervals_a,
        intervals_b,
    )
    a_some_overlap = intervals_a[mask_a_some_overlap, :]
    a_no_overlap = intervals_a[mask_a_no_overlap, :]

    return a_some_overlap, a_no_overlap
