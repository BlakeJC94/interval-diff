import pytest
import numpy as np
import pandas as pd

from interval_diff.vectorised import (
    interval_difference,
    sort_intervals_by_start,
)


@pytest.mark.parametrize("df", [False, True])
class TestIntervalDifference:
    @staticmethod
    def parse_intervals(records, df: bool):
        if isinstance(records, str):
            starts, ends, tags = [], [], []
            default_tag = "-"
            for i, c in enumerate(records):
                if c == "(":
                    starts.append(i * 100)
                    tags.append(records[i + 1] if records[i + 1].isalpha() else default_tag)
                elif c == ")":
                    ends.append(i * 100)

            if df:
                records = list(zip(starts, ends, tags))
            else:
                records = list(zip(starts, ends))

        out = None
        if df:
            if len(records) > 0 and len(records[0]) == 2:
                records = [(*r, "q") for r in records]
            columns = ["start", "end", "tags"]
            out = pd.DataFrame(records, columns=columns)
        else:
            out = np.array([(r[0], r[1]) for r in records])
            if len(out) == 0:
                out = np.empty((0, 2))

        return out

    @staticmethod
    def results_equal(output, expected):
        if type(output) != type(expected):
            raise TypeError()
        if not isinstance(output, (pd.DataFrame, np.ndarray)):
            raise ValueError()

        if isinstance(output, pd.DataFrame):
            if output.empty or expected.empty:
                return (output.columns == expected.columns).all()

            return output.equals(expected)

        return np.array_equal(output, expected)

    def test_doc_example(self, df):
        intervals_a, intervals_b, expected = (
            " (q---)  (w---)  (e---)  (r---)         (t---) (y-----)",
            "    (---------------)      (------)  (----)      (----)",
            " (q-)               (e)  (r)              (t-) (y)",
        )

        intervals_a = self.parse_intervals(intervals_a, df=df)
        intervals_b = self.parse_intervals(intervals_b, df=df)
        expected = self.parse_intervals(expected, df=df)

        result = interval_difference(intervals_a, intervals_b)

        assert self.results_equal(result, expected)

    @pytest.mark.parametrize(
        "intervals_a, intervals_b, expected",
        [
            # 2 left partial overlaps
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (--)     (--)                             ",
                "    (q-)     (w--)    (e---)        (e----)",
            ),
            # 2 right partial overlaps
            (
                "  (q---)    (w---)    (e---)        (e----)",
                "     (--)     (--)                         ",
                "  (q-)      (w)       (e---)        (e----)",
            ),
            # 1 left, 1 right partial overlaps
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (--)         (--)                         ",
                "    (q-)    (w)       (e---)        (e----)",
            ),
        ],
    )
    def test_some_partially_overlapping(self, intervals_a, intervals_b, expected, df):
        intervals_a = self.parse_intervals(intervals_a, df=df)
        intervals_b = self.parse_intervals(intervals_b, df=df)
        expected = self.parse_intervals(expected, df=df)

        result = interval_difference(intervals_a, intervals_b)
        assert self.results_equal(expected, result)

    @pytest.mark.parametrize(
        "intervals_a, intervals_b, expected",
        [
            # 1 total overlap
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (------)                                  ",
                "            (w---)    (e---)        (e----)",
            ),
            # 1 total, 1 left
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (------)  (--)                            ",
                "              (w-)    (e---)        (e----)",
            ),
            # 1 left, 1 total
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (--)     (--------)                       ",
                "    (q-)              (e---)        (e----)",
            ),
            # 1 total, 1 right
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (------) (---)                            ",
                "              (w-)    (e---)        (e----)",
            ),
            # 1 right, 1 total
            (
                "  (q---)    (w---)    (e---)        (e----)",
                "     (--) (--------)                       ",
                "  (q-)                (e---)        (e----)",
            ),
            # 2 total
            (
                "  (q---)    (w---)    (e---)        (e----)",
                " (-------) (--------)                      ",
                "                      (e---)        (e----)",
            ),
            # 2 contained (split and trim)
            (
                " (q-------)(e-------)          (r--)        ",
                "   (----)    (----)    (----)        (-----)",
                " (q)    (q)(e)    (e)          (r--)        ",
            ),
            # Multiple contained
            (
                " (q-----------------)(e------------------)    ",
                "   (----)    (----)    (----)        (-----)  ",
                " (q)    (q---)    (q)(e)    (e-------)        ",
            ),
        ],
    )
    def test_some_totally_overlapping(self, intervals_a, intervals_b, expected, df):
        intervals_a = self.parse_intervals(intervals_a, df=df)
        intervals_b = self.parse_intervals(intervals_b, df=df)
        expected = self.parse_intervals(expected, df=df)

        result = interval_difference(intervals_a, intervals_b)
        assert self.results_equal(expected, result)

    @pytest.mark.parametrize(
        "intervals_a, intervals_b",
        [
            # In between intervals
            (
                "  (q---)    (w---)    (e---)        (e----)",
                "        (--)      (--)       (---)         ",
            ),
            # All strictly left
            (
                "               (q---)    (w---)    (e---)        (e----)",
                " (--) (--) (--)                                         ",
            ),
            # All strictly right
            (
                " (q---)    (w---)    (e---)        (e----)                ",
                "                                           (--) (--) (--) ",
            ),
            # All strictly non-overlapping left and right
            (
                "                (q---)    (w---)    (e---)        (e----)                ",
                " (--) (--) (--)                                           (--) (--) (--) ",
            ),
        ],
    )
    def test_none_overlapping(self, intervals_a, intervals_b, df):
        intervals_a = self.parse_intervals(intervals_a, df=df)
        intervals_b = self.parse_intervals(intervals_b, df=df)

        result = interval_difference(
            intervals_a,
            intervals_b,
        )
        assert self.results_equal(intervals_a, result)


def test_sort_intervals_by_start():
    intervals = np.array([(600, 700), (1100, 1200), (100, 200), (2000, 2200)])
    expected = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])
    result = sort_intervals_by_start(intervals)
    assert (result == expected).all()
