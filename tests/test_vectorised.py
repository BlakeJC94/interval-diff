import pytest
import numpy as np
import pandas as pd

from interval_diff.vectorised import (
    interval_difference,
    sort_intervals_by_start,
)


def parse_intervals(records, df: bool):
    out = None
    if df:
        if len(records[0]) == 2:
            records = [(*r, "q") for r in records]
        columns = ["start", "end", "tags"]
        out = pd.DataFrame(records, columns=columns)
    else:
        out = np.array([(r[0], r[1]) for r in records])
    return out


def results_equal(output, expected):
    if type(output) != type(expected):
        raise TypeError()
    if not isinstance(output, (pd.DataFrame, np.ndarray)):
        raise ValueError()

    if isinstance(output, pd.DataFrame):
        return output.equals(expected)

    return np.array_equal(output, expected)


@pytest.mark.parametrize("df", [False, True])
class TestIntervalDifference:
    intervals_a = [(100, 200, "q"), (600, 700, "w"), (1100, 1200, "e"), (2000, 2200, "e")]

    # A     : (q---)  (w---)  (e---)  (r---)         (t---) (y-----)
    # B     :    (---------------)      (------)  (----)       (---)
    # A \ B : (q-)               (e)  (r)              (t-) (y)
    def test_doc_example(self, df):
        intervals_a = [
            (100, 200, "q"),
            (300, 400, "w"),
            (500, 600, "e"),
            (700, 800, "r"),
            (1000, 1100, "t"),
            (1250, 1400, "y"),
        ]
        intervals_b = [
            (150, 580),
            (720, 890),
            (930, 1070),
            (1300, 1400),
        ]
        expected = [
            (100, 150, "q"),
            (580, 600, "e"),
            (700, 720, "r"),
            (1070, 1100, "t"),
            (1250, 1300, "y"),
        ]

        intervals_a = parse_intervals(intervals_a, df=df)
        intervals_b = parse_intervals(intervals_b, df=df)
        expected = parse_intervals(expected, df=df)
        result = interval_difference(intervals_a, intervals_b)
        assert results_equal(result, expected)

    # PASS
    @pytest.mark.parametrize(
        "intervals_b, expected",
        [
            # 2 left partial overlaps
            (
                [(80, 120), (580, 620)],
                [
                    (120, 200, "q"),
                    (620, 700, "w"),
                    (1100, 1200, "e"),
                    (2000, 2200, "e"),
                ],
            ),
            # 2 right partial overlaps
            (
                [(180, 220), (680, 720)],
                [
                    (100, 180, 'q'),
                    (600, 680, 'w'),
                    (1100, 1200, 'e'),
                    (2000, 2200, 'e'),
                ],
            ),
            # 1 left, 1 right partial overlaps
            (
                [(80, 120), (680, 720)],
                [
                    (120, 200, 'q'),
                    (600, 680, 'w'),
                    (1100, 1200, 'e'),
                    (2000, 2200, 'e'),
                ],
            ),
        ],
    )
    def test_some_partially_overlapping(self, intervals_b, expected, df):
        expected = parse_intervals(expected, df=df)
        result = interval_difference(
            parse_intervals(self.intervals_a, df=df),
            parse_intervals(intervals_b, df=df),
        )
        assert results_equal(expected, result)

    # PASS
    @pytest.mark.parametrize(
        "intervals_b, expected",
        [
            # 1 total overlap
            (
                np.array([(80, 220)]),
                np.array([(600, 700), (1100, 1200), (2000, 2200)]),
            ),
            # 1 total, 1 left
            (
                np.array([(80, 220), (580, 620)]),
                np.array([(620, 700), (1100, 1200), (2000, 2200)]),
            ),
            # 1 left, 1 total
            (
                np.array([(80, 120), (580, 720)]),
                np.array([(120, 200), (1100, 1200), (2000, 2200)]),
            ),
            # 1 total, 1 right
            (
                np.array([(80, 220), (680, 720)]),
                np.array([(600, 680), (1100, 1200), (2000, 2200)]),
            ),
            # 1 right, 1 total
            (
                np.array([(180, 220), (580, 720)]),
                np.array([(100, 180), (1100, 1200), (2000, 2200)]),
            ),
            # 2 total
            (
                np.array([(80, 220), (580, 720)]),
                np.array([(1100, 1200), (2000, 2200)]),
            ),
        ],
    )
    def test_some_totally_overlapping(self, intervals_b, expected, df):
        result = interval_difference(
            self.intervals_a,
            intervals_b,
        )
        assert results_equal(expected, result)

    # PASS
    @pytest.mark.parametrize(
        "intervals_b",
        [
            # In between intervals
            np.array([(300, 400), (500, 550), (800, 900)]),
            # All strictly left
            np.array([(40, 50), (60, 70), (80, 90)]),
            # All strictly right
            np.array([(2540, 2550), (2560, 2570), (2580, 2590)]),
            # All strictly non-overlapping left and right
            np.array([(40, 50), (60, 70), (2560, 2570), (2580, 2590)]),
        ],
    )
    def test_none_overlapping(self, intervals_b, df):
        expected = self.intervals_a
        result = interval_difference(
            self.intervals_a,
            intervals_b,
        )
        assert results_equal(expected, result)

    @pytest.mark.parametrize(
        "minuend, expected_minuend_result",
        [
            # Minuends have no overlap with subtrahends (return minuend unchanged) PASS
            (
                np.array([(300, 400), (800, 900)]),
                np.array([(300, 400), (800, 900)]),
            ),
            # All minuends overlapped by middle 2 subtrahends (return empty list) PASS
            (
                np.array([(610, 690), (1110, 1190)]),
                np.empty((0, 2)),
            ),
            # All minuends left-overlap first 3 subtrahends (trim minuend_ends)
            (
                np.array([(50, 150), (550, 650), (1050, 1150)]),
                np.array([(50, 100), (550, 600), (1050, 1100)]),
            ),
            # All minuends right-overlap first 3 subtrahends (trim minuend_starts)
            (
                np.array([(150, 250), (650, 750), (1150, 1250)]),
                np.array([(200, 250), (700, 750), (1200, 1250)]),
            ),
            # All minuends overlap first 2 subtrahends (split minuends and trim)
            (
                np.array([(90, 210), (590, 710)]),
                np.array([(90, 100), (200, 210), (590, 600), (700, 710)]),
            ),
            # All minuends overlap multiple subtrahends (split minuends and trim)
            (
                np.array([(90, 710), (1050, 2250)]),
                np.array(
                    [
                        (90, 100),
                        (200, 600),
                        (700, 710),
                        (1050, 1100),
                        (1200, 2000),
                        (2200, 2250),
                    ]
                ),
            ),
        ],
    )
    def test_interval_difference_minuend(
        self,
        minuend,
        expected_minuend_result,
        df,
    ):
        output = interval_difference(
            minuend,
            self.intervals_a,
        )
        assert results_equal(expected_minuend_result, output)

    @pytest.mark.parametrize(
        "subtrahend, expected_subtrahend_result",
        [
            # Subtrahends don't overlap with minuends (return minuend unchanged)
            (
                np.array([(300, 400), (800, 900)]),
                np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)]),
            ),
            # Middle 2 minuends overlap all subtrahends (trim and split minuends, return non-overlap)
            (
                np.array([(610, 690), (1110, 1190)]),
                np.array(
                    [
                        (100, 200),
                        (600, 610),
                        (690, 700),
                        (1100, 1110),
                        (1190, 1200),
                        (2000, 2200),
                    ]
                ),
            ),
            # First 3 minuends left-overlap all subtrahends (trim minuend_ends, return non-overlap)
            (
                np.array([(50, 150), (550, 650), (1050, 1150)]),
                np.array([(150, 200), (650, 700), (1150, 1200), (2000, 2200)]),
            ),
            # First 3 minuends right-overlap all subtrahends (trim minuend_starts, return non-overlap)
            (
                np.array([(150, 250), (650, 750), (1150, 1250)]),
                np.array([(100, 150), (600, 650), (1100, 1150), (2000, 2200)]),
            ),
            # First 2 minuends overlapped by subtrahends (drop overlapped minuends, return non-overlap)
            (
                np.array([(90, 210), (590, 710)]),
                np.array([(1100, 1200), (2000, 2200)]),
            ),
            # All minuends overlapped by 2 subtrahends (drop all minuends)
            (
                np.array([(90, 710), (1050, 2250)]),
                np.empty((0, 2)),
            ),
        ],
    )
    def test_interval_difference_subtrahend(
        self,
        subtrahend,
        expected_subtrahend_result,
        df,
    ):
        output = interval_difference(
            self.intervals_a,
            subtrahend,
        )
        assert results_equal(expected_subtrahend_result, output)


def test_sort_intervals_by_start():
    intervals = np.array([(600, 700), (1100, 1200), (100, 200), (2000, 2200)])
    expected = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])
    result = sort_intervals_by_start(intervals)
    assert (result == expected).all()
