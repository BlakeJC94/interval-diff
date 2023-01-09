import logging
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .globals import EMPTY_INTERVALS, INTERVAL_COL_NAMES


logger = logging.getLogger(__name__)

# TODO test dataframe inputs
def interval_difference(
    intervals_a: Union[NDArray, pd.DataFrame],
    intervals_b: Union[NDArray, pd.DataFrame],
    min_len: float = 0.0,
) -> NDArray:
    """Clip labels in `labels` which intersect with labels in `bounds`.

    Args:
        labels: Labels which need to be clipped to prevent overlap with `bounds`.
        bounds: Labels to clip `labels` around.
        min_len: minimum allowable length of intervals to keep, intervals shorter than min_len will
            be dropped.

    Returns:
        Array of labels that overlap `labels` and complement of `bounds`.
    """
    if len(intervals_a) == 0 or len(intervals_b) == 0:
        return intervals_a

    intervals_a_input = intervals_a
    metadata = None
    if isinstance(intervals_a_input, pd.DataFrame):
        metadata = intervals_a_input.drop(INTERVAL_COL_NAMES, axis=1)
        intervals_a_input = intervals_a.copy()
        intervals_a = intervals_a[INTERVAL_COL_NAMES].values

    if isinstance(intervals_b, pd.DataFrame):
        intervals_b = intervals_b[INTERVAL_COL_NAMES].values

    intervals_a = sort_intervals_by_start(intervals_a)
    intervals_b = sort_intervals_by_start(intervals_b)

    index = np.arange(len(intervals_a))
    intervals_a = np.concatenate([intervals_a, index[:, None]], axis=1)

    final_labels = []
    bound_starts, bound_ends = intervals_b[:, 0], intervals_b[:, 1]

    for i in range(len(intervals_a)):
        label, keep_label = intervals_a[i, :], True

        label_start, label_end, label_idx = label
        # FIXME
        # tag, note, confidence = label.tag, label.note, label.confidence
        # timezone, study_id = label.timezone, label.study_id

        # check overlap of selected `label` with bounding labels in `bounds`
        for bound_start, bound_end in zip(bound_starts, bound_ends):

            # L :          (----)
            # B : (----)
            # If bound is strictly before the selected label, check the next bound
            if bound_end <= label_start:
                continue

            # L : (----)
            # B :          (----)
            # If bound is strictly after the selected label, move on to next label
            if label_end < bound_start:
                break

            # L :     (----)
            # B :  (----------)
            # If bound contains selected label, discard label and move on to the next label
            if bound_start <= label_start and label_end <= bound_end:
                keep_label = False
                break

            # L :       (------)
            # B :   (------)
            # If bound overlaps label start, clip label start and check next bound
            if bound_start <= label_start and bound_end <= label_end:
                label_start = bound_end
                continue

            # L :   (-----------...
            # B :      (----)
            # If bound is contained in label, create new label, clip label, and check next bound
            if label_start <= bound_start and bound_end < label_end:
                if bound_start - label_start > min_len:
                    final_labels.append((label_start, bound_start, label_idx))
                label_start = bound_end
                continue

            # L :   (------)
            # B :       (------)
            # If bound overlaps label end, clip end of label and move onto next label
            if label_start <= bound_start and label_end <= bound_end:
                label_end = bound_start
                break

        if keep_label and label_end - label_start > min_len:
            final_labels.append((label_start, label_end, label_idx))

    if len(final_labels) == 0:
        if metadata is not None:
            return pd.DataFrame(columns=intervals_a_input.columns)
        return EMPTY_INTERVALS

    result = np.array(final_labels)
    if metadata is None:
        return result[:, :2]

    result, indices = result[:, :2], result[:, -1].astype(int)
    metadata = metadata.iloc[indices].reset_index(drop=True)
    metadata[INTERVAL_COL_NAMES] = result
    result = metadata[intervals_a_input.columns]
    return result


def sort_intervals_by_start(intervals: NDArray) -> NDArray:
    """Sort an interval array by interval start."""
    return intervals[np.argsort(intervals[:, 0]), :]
