import numpy as np
from numpy.typing import NDArray

from .utils import sort_intervals_by_start
from .globals import EMPTY_INTERVALS


def interval_difference(
    labels: NDArray,
    bounds: NDArray,
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
    if len(labels) == 0 or len(bounds) == 0:
        return labels

    labels = sort_intervals_by_start(labels)
    bounds = sort_intervals_by_start(bounds)

    final_labels = []
    bound_starts, bound_ends = bounds[:, 0], bounds[:, 1]

    for i in range(len(labels)):
        label, keep_label = labels[i, :], True

        label_start, label_end = label
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
                    final_labels.append((label_start, bound_start))
                label_start = bound_end
                continue

            # L :   (------)
            # B :       (------)
            # If bound overlaps label end, clip end of label and move onto next label
            if label_start <= bound_start and label_end <= bound_end:
                label_end = bound_start
                break

        if keep_label and label_end - label_start > min_len:
            final_labels.append((label_start, label_end))

    if len(final_labels) == 0:
        return EMPTY_INTERVALS
    return np.array(final_labels)
