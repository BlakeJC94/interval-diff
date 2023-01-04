import pytest
import numpy as np

from interval_diff.utils import (
    complement_intervals,
    drop_total_overlaps,
    create_intervals_from_points,
)


