import pytest
import numpy as np

from interval_diff.utils import (
    sort_intervals_by_start,
    complement_intervals,
    drop_total_overlaps,
    create_intervals_from_points,
    concat_interval_groups,
    filter_overlapping_intervals,
)


def test_sort_intervals_by_start():
    intervals = np.array([(600, 700), (1100, 1200), (100, 200), (2000, 2200)])
    expected = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])
    result = sort_intervals_by_start(intervals)
    assert (result == expected).all()


class TestComplementIntervals:
    intervals = [(100, 200), (600, 700), (1100, 1200), (2000, 2200)]
    minimum = 10
    maximum = 2340
    expected = [(200, 600), (700, 1100), (1200, 2000)]
    expected_w_min = [(minimum, 100)]
    expected_w_max = [(2200, maximum)]

    def test_wo_max_or_min(self):
        intervals = np.array(self.intervals)
        result = complement_intervals(intervals)
        expected = np.array(self.expected)
        assert (result == expected).all()

    def test_w_min(self):
        intervals = np.array(self.intervals)
        result = complement_intervals(intervals, minimum=self.minimum)
        expected = np.array(self.expected_w_min + self.expected)
        assert (result == expected).all()

    def test_w_max(self):
        intervals = np.array(self.intervals)
        result = complement_intervals(intervals, maximum=self.maximum)
        expected = np.array(self.expected + self.expected_w_max)
        assert (result == expected).all()

    def test_w_max_and_min(self):
        intervals = np.array(self.intervals)
        result = complement_intervals(
            intervals,
            minimum=self.minimum,
            maximum=self.maximum,
        )
        expected = np.array(self.expected_w_min + self.expected + self.expected_w_max)
        assert (result == expected).all()

    def test_empty_intervals(self):
        intervals = np.empty(shape=(0, 2))

        expected = np.empty(shape=(0, 2))
        expected_w_min = np.empty(shape=(0, 2))
        expected_w_max = np.empty(shape=(0, 2))
        expected_w_min_and_max = np.array([(self.minimum, self.maximum)])

        assert len(complement_intervals(intervals)) == 0
        assert len(complement_intervals(intervals, minimum=self.minimum)) == 0
        assert len(complement_intervals(intervals, maximum=self.maximum)) == 0

        result = complement_intervals(intervals, minimum=self.minimum, maximum=self.maximum)
        assert np.array_equal(result, expected_w_min_and_max)


class TestDropTotalOverlaps:
    intervals_a = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])

    def test_no_overlaps_and_one_super_overlap(self):
        intervals_b = np.array([(300, 400), (500, 800), (1000, 2300)])
        expected = np.array([(100, 200)])

        result = drop_total_overlaps(self.intervals_a, intervals_b)
        assert (result == expected).all()

    def test_one_overlaps_all(self):
        intervals_b = np.array([(50, 2500)])
        result = drop_total_overlaps(self.intervals_a, intervals_b)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "intervals_b",
        [
            # In between intervals
            np.array([(300, 400), (500, 550), (800, 900)]),
            # All left partial overlaps
            np.array([(80, 120), (580, 620), (1080, 1120), (1980, 2020)]),
            # All right partial overlaps
            np.array([(180, 220), (680, 720), (1180, 1220), (2180, 2220)]),
            # Some left, some right partial overlaps
            np.array([(80, 120), (580, 620), (1180, 1220), (2180, 2220)]),
            # All strictly left
            np.array([(40, 50), (60, 70), (80, 90)]),
            # All strictly right
            np.array([(2540, 2550), (2560, 2570), (2580, 2590)]),
            # All strictly non-overlapping left and right
            np.array([(40, 50), (60, 70), (2560, 2570), (2580, 2590)]),
        ],
    )
    def test_no_total_overlaps(self, intervals_b):
        result = drop_total_overlaps(self.intervals_a, intervals_b)
        assert (result == self.intervals_a).all()

    @pytest.mark.parametrize(
        "intervals_b, expected",
        [
            (
                np.array([(100, 200)]),
                np.array([(600, 700), (1100, 1200), (2000, 2200)]),
            ),
            (
                np.array([(100, 200), (600, 700)]),
                np.array([(1100, 1200), (2000, 2200)]),
            ),
        ],
    )
    def test_duplicates(self, intervals_b, expected):
        result = drop_total_overlaps(self.intervals_a, intervals_b)
        assert np.array_equal(result, expected)

    def test_trivial(self):
        result = drop_total_overlaps(self.intervals_a, self.intervals_a)
        assert len(result) == 0

    def test_empty(self):
        empty = np.empty((0, 2))
        result = drop_total_overlaps(empty, self.intervals_a)
        assert np.array_equal(result, empty)

    def test_identity(self):
        empty = np.empty((0, 2))
        result = drop_total_overlaps(self.intervals_a, empty)
        assert np.array_equal(result, self.intervals_a)


def test_create_intervals_from_points():
    points = np.array([100, 200, 250, 375, 600])
    expected = np.array([(100, 200), (200, 250), (250, 375), (375, 600)])
    result = create_intervals_from_points(points)
    assert np.array_equal(result, expected)



class TestConcatIntervalGroups:
    intervals_a = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])
    intervals_b = np.array([(80, 120), (580, 620), (1080, 1120), (1980, 2020)])

    def test_concat_interval_groups_sort(self):
        expected = np.array(
            [
                (80, 120),
                (100, 200),
                (580, 620),
                (600, 700),
                (1080, 1120),
                (1100, 1200),
                (1980, 2020),
                (2000, 2200),
            ]
        )

        result_a = concat_interval_groups(
            [self.intervals_a, self.intervals_b],
            sort=True,
        )
        result_b = concat_interval_groups(
            [self.intervals_b, self.intervals_a],
            sort=True,
        )

        assert np.array_equal(result_a, result_b)
        assert np.array_equal(result_a, expected)

    def test_concat_interval_groups_no_sort(self):
        expected = np.array(
            [
                (100, 200),
                (600, 700),
                (1100, 1200),
                (2000, 2200),
                (80, 120),
                (580, 620),
                (1080, 1120),
                (1980, 2020),
            ]
        )

        result = concat_interval_groups(
            [self.intervals_a, self.intervals_b],
            sort=False,
        )

        assert np.array_equal(result, expected)


class TestFilterOverlappingIntervals:
    intervals_a = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])

    def test_all_duplicates(self):
        intervals_b = np.array([(100, 200), (600, 700), (1100, 1200), (2000, 2200)])
        overlapping, non_overlapping = filter_overlapping_intervals(self.intervals_a, intervals_b)
        assert (overlapping == self.intervals_a).all()
        assert len(non_overlapping) == 0

    def test_some_duplicates(self):
        intervals_b = np.array([(100, 200), (1100, 1200), (2000, 2200)])
        expected_overlapping = np.array([(100, 200), (1100, 1200), (2000, 2200)])
        expected_non_overlapping = np.array([(600, 700)])

        overlapping, non_overlapping = filter_overlapping_intervals(self.intervals_a, intervals_b)

        assert (overlapping == expected_overlapping).all()
        assert (non_overlapping == expected_non_overlapping).all()

    @pytest.mark.parametrize(
        "intervals_b",
        [
            # All left partial overlaps
            np.array([(80, 120), (580, 620), (1080, 1120), (1980, 2020)]),
            # All right partial overlaps
            np.array([(180, 220), (680, 720), (1180, 1220), (2180, 2220)]),
            # Some left, some right partial overlaps
            np.array([(80, 120), (580, 620), (1180, 1220), (2180, 2220)]),
            # All totally overlapped
            np.array([(80, 2220)]),
            # Some totally overlapped, others partially left overlapped
            np.array([(80, 120), (580, 620), (1080, 2020)]),
        ],
    )
    def test_all_overlapping(self, intervals_b):
        overlapping, non_overlapping = filter_overlapping_intervals(self.intervals_a, intervals_b)
        assert (overlapping == self.intervals_a).all()
        assert len(non_overlapping) == 0

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
    def test_none_overlapping(self, intervals_b):
        overlapping, non_overlapping = filter_overlapping_intervals(self.intervals_a, intervals_b)

        assert len(overlapping) == 0
        assert (non_overlapping == self.intervals_a).all()

    @pytest.mark.parametrize(
        "intervals_b",
        [
            # Some left partial overlaps
            np.array([(580, 620), (1080, 1120)]),
            # Some right partial overlaps
            np.array([(680, 720), (1180, 1220)]),
            # left and right partial overlaps
            np.array([(580, 620), (1180, 1220)]),
        ],
    )
    def test_some_partially_overlapping(self, intervals_b):
        expected_overlapping = np.array([(600, 700), (1100, 1200)])
        expected_non_overlapping = np.array([(100, 200), (2000, 2200)])

        overlapping, non_overlapping = filter_overlapping_intervals(self.intervals_a, intervals_b)

        assert (overlapping == expected_overlapping).all()
        assert (non_overlapping == expected_non_overlapping).all()
