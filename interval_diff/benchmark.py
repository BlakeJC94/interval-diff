import time
import logging
from itertools import product
from collections import defaultdict
from typing import Optional, List, Callable, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from interval_diff.globals import INTERVAL_COL_NAMES

from interval_diff.vectorised import interval_difference as vec_diff
from interval_diff.non_vectorised import interval_difference as nonvec_diff
from interval_diff.utils import generate_random_intervals
from interval_diff.vis import plot_intervals

np.random.seed(1234)

DEFAULT_N_INTERVALS = [20, 100, 500, 1000, 2000, 5000, 10000]
DEFAULT_N_SAMPLES = 3
DEFAULT_DF = False
DATAFRAME = True


logger = logging.getLogger(__name__)

# TODO test
def benchmark(
    n_intervals: Optional[List[int]] = None,
    n_samples: Optional[int] = None,
    dataframes: Optional[bool] = None,
    inspect: bool = True,
):
    if n_intervals is None:
        n_intervals = DEFAULT_N_INTERVALS

    if n_samples is None:
        n_samples = DEFAULT_N_SAMPLES

    if dataframes is None:
        dataframes = DEFAULT_DF

    times = [defaultdict(list) for _ in n_intervals]

    # pylint: disable=invalid-name
    for (i, n), _ in tqdm(
        product(enumerate(n_intervals), range(n_samples)),
        total=len(n_intervals) * n_samples,
        miniters=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]",
    ):
        intervals_a = generate_random_intervals(n, start=100, max_len=100, dataframe=dataframes)
        intervals_b = generate_random_intervals(n, start=0, max_len=80, dataframe=dataframes)

        results = []
        for vec in [True, False]:
            key = "_".join([("vec" if vec else "nonvec"), ("pd" if dataframes else "np")])
            elapsed, result = _time_func_run(
                vec_diff if vec else nonvec_diff,
                intervals_a,
                intervals_b,
            )
            times[i][key].append(elapsed)
            results.append(result)

        if inspect:
            _inspect_if_unequal(*results, intervals_a, intervals_b)

    _print_table(times, n_intervals, n_samples, dataframes)
    _write_csv(times, n_intervals, n_samples, dataframes)


def _print_table(times, n_intervals, n_samples, df):
    mode = "pd" if df else "np"
    vec_key = "vec_" + mode
    nonvec_key = "nonvec_" + mode
    n_intervals_str = f"[{mode}] Intervals ({n_samples} samples)"
    header = " " + "| ".join(
        [
            f"{n_intervals_str:<28}",
            f"{'Non-vec mean (s)':<20}",
            f"{'Vec mean (s)':<20}",
        ]
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, n in enumerate(n_intervals):
        nonvec_mean = sum(times[i][nonvec_key]) / n_samples
        vec_mean = sum(times[i][vec_key]) / n_samples
        row = " " + "| ".join(
            [
                f"{n:<28}",
                f"{nonvec_mean:<20.6f}",
                f"{vec_mean:<20.6f}",
            ]
        )
        print(row)


def _write_csv(times, n_intervals, n_samples, df):
    mode = "pd" if df else "np"
    vec_key, nonvec_key = f"vec_{mode}", f"nonvec_{mode}"
    with open(f"results_{mode}.csv", "w", encoding="utf-8") as f:
        header = ",".join(
            [
                "n_intervals",
                *[f"vec_{i}" for i in range(n_samples)],
                *[f"nonvec_{i}" for i in range(n_samples)],
            ]
        )
        f.write(header + "\n")
        for n in n_intervals:
            row = ",".join(
                [
                    f"{n}",
                    *[f"{t}" for i in range(len(n_intervals)) for t in times[i][vec_key]],
                    *[f"{t}" for i in range(len(n_intervals)) for t in times[i][nonvec_key]],
                ]
            )
            f.write(row + "\n")


def _time_func_run(func: Callable, *args, **kwargs) -> Tuple[float, Any]:
    tic = time.time()
    result = func(*args, **kwargs)
    toc = time.time()
    return toc - tic, result


# TODO rfc
def _inspect_if_unequal(vec_result, nonvec_result, intervals_a, intervals_b):
    vec_metadata = None
    if isinstance(vec_result, pd.DataFrame):
        vec_metadata = vec_result.drop(INTERVAL_COL_NAMES, axis=1)
        vec_result = vec_result[INTERVAL_COL_NAMES].values

    nonvec_metadata = None
    if isinstance(nonvec_result, pd.DataFrame):
        nonvec_metadata = nonvec_result.drop(INTERVAL_COL_NAMES, axis=1)
        nonvec_result = nonvec_result[INTERVAL_COL_NAMES].values

    if np.array_equal(vec_result, nonvec_result):
        if vec_metadata is None or nonvec_metadata is None:
            return
        if vec_metadata.equals(nonvec_metadata):
            return

    n_vec_results = len(vec_result)
    n_nonvec_results = len(nonvec_result)
    print(f"WARNING: Unequal result {n_vec_results = }, {n_nonvec_results = }")
    print(">>> >>> Starting difference investigation")
    for i in range(min(n_vec_results, n_nonvec_results)):
        if not np.array_equal(vec_result[i, :], nonvec_result[i, :]):
            start_idx, end_idx = i - 2, i + 3
            delta = 300

            vec_subset = vec_result[start_idx:end_idx, :]
            nonvec_subset = nonvec_result[start_idx:end_idx, :]

            min_point = min(vec_result[start_idx, 0], nonvec_result[start_idx, 0])
            max_point = max(vec_result[end_idx, 1], nonvec_result[end_idx, 1])

            intervals_a_mask = (intervals_a[:, 1] > min_point - delta) & (
                intervals_a[:, 0] < max_point + delta
            )
            intervals_a_subset = intervals_a[intervals_a_mask, :]

            intervals_b_mask = (intervals_b[:, 1] > min_point - delta) & (
                intervals_b[:, 0] < max_point + delta
            )
            intervals_b_subset = intervals_b[intervals_b_mask, :]

            print(f"difference at interval {i = }")

            print("Vec result:")
            print(vec_subset)

            print("NonVec result:")
            print(nonvec_subset)

            print("A subset:")
            print(intervals_a_subset)

            print("B subset:")
            print(intervals_b_subset)

            figure = plot_intervals(
                [
                    intervals_a_subset,
                    intervals_b_subset,
                    vec_subset,
                    nonvec_subset,
                ],
                colors=["yellow", "red", "yellow", "yellow"],
                names=["A", "B", "VEC", "NONVEC"],
            )
            figure.show()
            breakpoint()
