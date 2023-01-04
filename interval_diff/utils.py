from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .globals import EMPTY_INTERVALS, INTERVAL_COL_NAMES

DEFAULT_TAGS = list("QWERTY")


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
    data = min_len + (max_len - min_len) * np.random.rand(2 * n_intervals - 1)
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
