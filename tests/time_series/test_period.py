import unittest
import datetime as dt
import re

from unittest.mock import patch, Mock
from typing import Callable
from parameterized import parameterized

import time_series.period as p


class TestNaive(unittest.TestCase):
    """Unit tests for the _naive function."""

    def test_naive_with_tzinfo(self):
        """Test _naive function with a datetime object that has tzinfo."""
        dt_with_tz = dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc)
        result = p._naive(dt_with_tz)
        self.assertIsNone(result.tzinfo)
        self.assertEqual(result, dt.datetime(2023, 10, 10, 10, 0))

    def test_naive_without_tzinfo(self):
        """Test _naive function with a datetime object that has no tzinfo."""
        dt_without_tz = dt.datetime(2023, 10, 10, 10, 0)
        result = p._naive(dt_without_tz)
        self.assertIsNone(result.tzinfo)
        self.assertEqual(result, dt_without_tz)


class TestGregorianSeconds(unittest.TestCase):
    """Unit tests for the _gregorian_seconds function."""

    @parameterized.expand(
        [
            ("Zero", dt.datetime(1, 1, 1, 0, 0, 0), 86_400 + 0),
            ("2 hours", dt.datetime(1, 1, 1, 2, 2, 2), 86_400 + (3600 * 2) + (60 * 2) + 2),
        ]
    )
    def test_gregorian_seconds(self, name, dt_obj, expected_seconds):
        """Test _gregorian_seconds function with various datetime inputs."""
        self.assertEqual(p._gregorian_seconds(dt_obj), expected_seconds)


class TestPeriodRegex(unittest.TestCase):
    """Unit tests for the _period_regex function."""

    def test_period_regex(self):
        """Test regex string returned from _period_regex function."""
        regex = p._period_regex("test")
        self.assertIsInstance(regex, str)
        expected = (
            rf"(?:(?P<test_years>\d+)[Yy])?"
            rf"(?:(?P<test_months>\d+)[Mm])?"
            rf"(?:(?P<test_days>\d+)[Dd])?"
            r"(?:[Tt]?"
            rf"(?:(?P<test_hours>\d+)[Hh])?"
            rf"(?:(?P<test_minutes>\d+)[Mm])?"
            rf"(?:(?P<test_seconds>\d+)"
            rf"(?:\.(?P<test_microseconds>\d{{1,6}}))?"
            r"[Ss])?"
            r")?"
        )
        self.assertEqual(regex, expected)


class TestYearShift(unittest.TestCase):
    """Unit tests for the year_shift function."""

    def test_no_shift(self):
        """Test that the date remains the same when shift_amount is 0."""
        date_time = dt.datetime(2023, 10, 8)
        self.assertEqual(p.year_shift(date_time, 0), date_time)

    def test_positive_shift(self):
        """Test shifting the date forward by a positive number of years."""
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2025, 10, 8)
        self.assertEqual(p.year_shift(date_time, 2), expected)

    def test_negative_shift(self):
        """Test shifting the date backward by a negative number of years."""
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2021, 10, 8)
        self.assertEqual(p.year_shift(date_time, -2), expected)

    def test_leap_year(self):
        """Test shifting a leap day to another leap year."""
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2024, 2, 29)
        self.assertEqual(p.year_shift(date_time, 4), expected)

    def test_non_leap_year(self):
        """Test shifting a leap day to a non-leap year."""
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2021, 2, 28)
        self.assertEqual(p.year_shift(date_time, 1), expected)

    def test_end_of_month(self):
        """Test shifting a date at the end of the month."""
        date_time = dt.datetime(2023, 1, 31)
        expected = dt.datetime(2024, 1, 31)
        self.assertEqual(p.year_shift(date_time, 1), expected)


class TestMonthShift(unittest.TestCase):
    """Unit tests for the month_shift function."""

    def test_no_shift(self):
        """Test that the date remains the same when shift_amount is 0."""
        date_time = dt.datetime(2023, 10, 8)
        self.assertEqual(p.month_shift(date_time, 0), date_time)

    def test_positive_shift(self):
        """Test shifting the date forward by a positive number of months."""
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2024, 4, 8)
        self.assertEqual(p.month_shift(date_time, 6), expected)

    def test_negative_shift(self):
        """Test shifting the date backward by a negative number of months."""
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2023, 4, 8)
        self.assertEqual(p.month_shift(date_time, -6), expected)

    def test_end_of_month(self):
        """Test shifting a date at the end of the month."""
        date_time = dt.datetime(2023, 1, 31)
        expected = dt.datetime(2023, 2, 28)  # February in a non-leap year
        self.assertEqual(p.month_shift(date_time, 1), expected)

    def test_leap_year(self):
        """Test shifting a date in a leap year."""
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2020, 3, 29)
        self.assertEqual(p.month_shift(date_time, 1), expected)

    def test_non_leap_year(self):
        """Test shifting a date from a leap year to a non-leap year."""
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2021, 2, 28)
        self.assertEqual(p.month_shift(date_time, 12), expected)


class TestOfMonthOffset(unittest.TestCase):
    """Unit tests for of_month_offset in the the DateTimeAdjusters class."""

    @parameterized.expand([("15 months", 15), ("12 months", 12), ("6 months", 6)])
    def test_of_month_offset_year_shift(self, name, months):
        """Test DateTimeAdjusters.of_month_offset with a differing number of months"""
        adjusters = p.DateTimeAdjusters.of_month_offset(months)
        self.assertIsInstance(adjusters, p.DateTimeAdjusters)
        self.assertIsInstance(adjusters.advance, Callable)
        self.assertIsInstance(adjusters.retreat, Callable)

    def test_of_month_offset_zero(self):
        """Test DateTimeAdjusters.of_month_offset with a zero shift."""
        adjusters = p.DateTimeAdjusters.of_month_offset(0)
        self.assertIsNone(adjusters)


class TestOfMircorsecondOffset(unittest.TestCase):
    """Unit tests for of_microsecond_offset in the the DateTimeAdjusters class."""

    def test_of_microsecond_offset(self):
        """Test DateTimeAdjusters.of_microsecond_offset with various microsecond offsets."""
        microseconds = 100
        adjusters = p.DateTimeAdjusters.of_microsecond_offset(microseconds)
        self.assertIsInstance(adjusters, p.DateTimeAdjusters)
        self.assertIsInstance(adjusters.advance, Callable)
        self.assertIsInstance(adjusters.retreat, Callable)

        # As lambda functions are returned for the adjuster functions, test their behaviour
        dt_obj = dt.datetime(2023, 10, 10, 10, 0, 0)
        expected_advance = dt_obj + dt.timedelta(microseconds=microseconds)
        expected_retreat = dt_obj - dt.timedelta(microseconds=microseconds)

        self.assertEqual(adjusters.advance(dt_obj), expected_advance)
        self.assertEqual(adjusters.retreat(dt_obj), expected_retreat)

    def test_of_microsecond_offset_zero(self):
        """Test DateTimeAdjusters.of_microsecond_offset with a zero shift."""
        adjusters = p.DateTimeAdjusters.of_microsecond_offset(0)
        self.assertIsNone(adjusters)


class TestOfOffsets(unittest.TestCase):
    """Unit tests for of_offsets in the the DateTimeAdjusters class."""

    @parameterized.expand(
        [
            (
                "15 months, 1000000 microseconds",
                15,
                1_000_000,
                dt.datetime(2024, 4, 1, 0, 0, 1),
                dt.datetime(2021, 9, 30, 23, 59, 59),
            ),
            (
                "12 months, 500000 microseconds",
                12,
                500_000,
                dt.datetime(2024, 1, 1, 0, 0, 0, 500_000),
                dt.datetime(2021, 12, 31, 23, 59, 59, 500000),
            ),
            (
                "6 months, 1000 microseconds",
                6,
                1_000,
                dt.datetime(2023, 7, 1, 0, 0, 0, 1000),
                dt.datetime(2022, 6, 30, 23, 59, 59, 999000),
            ),
            (
                "0 months, 1000 microseconds",
                0,
                1_000,
                dt.datetime(2023, 1, 1, 0, 0, 0, 1000),
                dt.datetime(2022, 12, 31, 23, 59, 59, 999000),
            ),
            (
                "6 months, 0 microseconds",
                6,
                0,
                dt.datetime(
                    2023,
                    7,
                    1,
                    0,
                    0,
                    0,
                ),
                dt.datetime(2022, 7, 1, 0, 0, 0),
            ),
        ]
    )
    def test_of_offsets(self, name, months, microseconds, expected_adv, expected_ret):
        """Test DateTimeAdjusters.of_offsets with various month and microsecond offsets."""
        adjusters = p.DateTimeAdjusters.of_offsets(months, microseconds)
        self.assertIsInstance(adjusters, p.DateTimeAdjusters)
        self.assertIsInstance(adjusters.advance, Callable)
        self.assertIsInstance(adjusters.retreat, Callable)

        dt_obj = dt.datetime(2023, 1, 1, 0, 0, 0)
        self.assertEqual(adjusters.advance(dt_obj), expected_adv)
        self.assertEqual(adjusters.retreat(dt_obj), expected_ret)

    def test_of_offsets_zero(self):
        """Test DateTimeAdjusters.of_offsets with zero month and microsecond offsets."""
        with self.assertRaises(AssertionError):
            p.DateTimeAdjusters.of_offsets(0, 0)


class TestDateTimeAdjustersPostInit(unittest.TestCase):
    """Unit tests for __post_init__ in the the DateTimeAdjusters class."""

    def test_post_init_no_error(self):
        """Test DateTimeAdjusters.__post_init__ does not raise an error when advance and retreat are set"""
        p.DateTimeAdjusters(retreat=lambda x: x, advance=lambda x: x).__post_init__()

    @parameterized.expand(
        [("Both none", None, None), ("Retreat None", None, lambda x: x), ("Advance None", lambda x: x, None)]
    )
    def test_post_init_raises_with_none(self, name, retreat, advance):
        """Test DateTimeAdjusters.__post_init__ to ensure it raises AssertionError when retreat or advance is None."""
        with self.assertRaises(AssertionError):
            p.DateTimeAdjusters(retreat=retreat, advance=advance).__post_init__()


class TestSecondString(unittest.TestCase):
    """Unit tests for the _second_string function."""

    @parameterized.expand(
        [
            ("whole seconds", 10, 0, "10"),
            ("seconds with microseconds", 10, 500000, "10.5"),
            ("seconds with trailing zero microseconds", 10, 5000000, "10.5"),
            ("seconds with no trailing zero microseconds", 10, 500001, "10.500001"),
            ("zero seconds", 0, 0, "0"),
            ("zero seconds with microseconds", 0, 123456, "0.123456"),
        ]
    )
    def test_second_string(self, name, seconds, microseconds, expected):
        """Test _second_string function with various seconds and microseconds."""
        result = p._second_string(seconds, microseconds)
        self.assertEqual(result, expected)


class TestAppendSecondElems(unittest.TestCase):
    """Unit tests for the _append_second_elems function."""

    @parameterized.expand(
        [
            ("no days, hours, minutes, or microseconds", [], 10, 0, ["T", "10", "S"]),
            ("with microseconds", [], 10, 500000, ["T", "10.5", "S"]),
            ("with minutes", [], 70, 0, ["T", "1M", "10", "S"]),
            ("with hours", [], 3660, 0, ["T", "1H", "1M"]),
            ("with days", [], 90000, 0, ["1D", "T", "1H"]),
            ("complex case", ["P"], 90061, 500000, ["P", "1D", "T", "1H", "1M", "1.5", "S"]),
        ]
    )
    def test_append_second_elems(self, name, elems, seconds, microseconds, expected):
        """Test _append_second_elems function with various inputs."""
        result = p._append_second_elems(elems, seconds, microseconds)
        self.assertEqual(result, expected)


class TestAppendMonthElems(unittest.TestCase):
    """Unit tests for the _append_month_elems function."""

    @parameterized.expand(
        [
            ("no months", [], 0, []),
            ("only months", [], 5, ["5M"]),
            ("only years", [], 24, ["2Y"]),
            ("years and months", [], 30, ["2Y", "6M"]),
            ("complex case", ["P"], 18, ["P", "1Y", "6M"]),
        ]
    )
    def test_append_month_elems(self, name, elems, months, expected):
        """Test _append_month_elems function with various inputs."""
        result = p._append_month_elems(elems, months)
        self.assertEqual(result, expected)


class TestGetMicrosecondPeriodName(unittest.TestCase):
    """Unit tests for the _get_microsecond_period_name function."""

    @parameterized.expand(
        [
            ("zero microseconds", 0, "P"),
            ("only seconds", 1_000_000, "PT1S"),
            ("seconds and microseconds", 1_500_000, "PT1.5S"),
            ("only microseconds", 500_000, "PT0.5S"),
            ("complex case", 3_645_034_555, "PT1H45.034555S"),
        ]
    )
    def test_get_microsecond_period_name(self, name, total_microseconds, expected):
        """Test _get_microsecond_period_name function with various inputs."""
        result = p._get_microsecond_period_name(total_microseconds)
        self.assertEqual(result, expected)


class TestGetSecondPeriodName(unittest.TestCase):
    """Unit tests for the _get_second_period_name function."""

    @parameterized.expand(
        [
            ("zero seconds", 0, "P"),
            ("only seconds", 10, "PT10S"),
            ("seconds forming minutes", 70, "PT1M10S"),
            ("seconds forming hours", 3660, "PT1H1M"),
            ("complex case", 90061, "P1DT1H1M1S"),
        ]
    )
    def test_get_second_period_name(self, name, seconds, expected):
        """Test _get_second_period_name function with various inputs."""
        result = p._get_second_period_name(seconds)
        self.assertEqual(result, expected)


class TestGetMonthPeriodName(unittest.TestCase):
    """Unit tests for the _get_month_period_name function."""

    @parameterized.expand(
        [
            ("zero months", 0, "P"),
            ("only months", 5, "P5M"),
            ("only years", 24, "P2Y"),
            ("years and months", 30, "P2Y6M"),
        ]
    )
    def test_get_month_period_name(self, name, months, expected):
        """Test _get_month_period_name function with various inputs."""
        result = p._get_month_period_name(months)
        self.assertEqual(result, expected)


class TestFmtNaiveMicrosecond(unittest.TestCase):
    """Unit tests for the _fmt_naive_microsecond function."""

    @parameterized.expand(
        [
            (
                "standard datetime with dash separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456),
                "-",
                "2023-10-10-10:00:00.123456",
            ),
            (
                "standard datetime with T separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456),
                "T",
                "2023-10-10T10:00:00.123456",
            ),
            (
                "datetime with zero microseconds",
                dt.datetime(2023, 10, 10, 10, 0, 0, 0),
                "-",
                "2023-10-10-10:00:00.000000",
            ),
        ]
    )
    def test_fmt_naive_microsecond(self, name, obj, separator, expected):
        """Test _fmt_naive_microsecond function with various datetime objects and separators."""
        result = p._fmt_naive_microsecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveMillisecond(unittest.TestCase):
    """Unit tests for the _fmt_naive_millisecond function."""

    @parameterized.expand(
        [
            (
                "standard datetime with dash separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456),
                "-",
                "2023-10-10-10:00:00.123",
            ),
            (
                "standard datetime with T separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456),
                "T",
                "2023-10-10T10:00:00.123",
            ),
            ("datetime with zero microseconds", dt.datetime(2023, 10, 10, 10, 0, 0, 0), "-", "2023-10-10-10:00:00.000"),
        ]
    )
    def test_fmt_naive_millisecond(self, name, obj, separator, expected):
        """Test _fmt_naive_millisecond function with various datetime objects and separators."""
        result = p._fmt_naive_millisecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveSecond(unittest.TestCase):
    """Unit tests for the _fmt_naive_second function."""

    @parameterized.expand(
        [
            ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0), "-", "2023-10-10-10:00:00"),
            ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0), "T", "2023-10-10T10:00:00"),
        ]
    )
    def test_fmt_naive_second(self, name, obj, separator, expected):
        """Test _fmt_naive_second function with various datetime objects and separators."""
        result = p._fmt_naive_second(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveMinute(unittest.TestCase):
    """Unit tests for the _fmt_naive_minute function."""

    @parameterized.expand(
        [
            ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0), "-", "2023-10-10-10:00"),
            ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0), "T", "2023-10-10T10:00"),
        ]
    )
    def test_fmt_naive_minute(self, name, obj, separator, expected):
        """Test _fmt_naive_minute function with various datetime objects and separators."""
        result = p._fmt_naive_minute(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveHour(unittest.TestCase):
    """Unit tests for the _fmt_naive_hour function."""

    @parameterized.expand(
        [
            ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0), "-", "2023-10-10-10"),
            ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0), "T", "2023-10-10T10"),
        ]
    )
    def test_fmt_naive_hour(self, name, obj, separator, expected):
        """Test _fmt_naive_hour function with various datetime objects and separators."""
        result = p._fmt_naive_hour(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveDay(unittest.TestCase):
    """Unit tests for the _fmt_naive_day function."""

    def test_fmt_naive_day(self):
        """Test _fmt_naive_day function with various datetime objects."""
        expected = "2023-10-10"
        result = p._fmt_naive_day(dt.datetime(2023, 10, 10, 10, 0, 0))
        self.assertEqual(result, expected)


class TestFmtNaiveMonth(unittest.TestCase):
    """Unit tests for the _fmt_naive_month function."""

    def test_fmt_naive_month(self):
        """Test _fmt_naive_month function with various datetime objects."""
        expected = "2023-10"
        result = p._fmt_naive_month(dt.datetime(2023, 10, 10, 10, 0, 0))
        self.assertEqual(result, expected)


class TestFmtNaiveYear(unittest.TestCase):
    """Unit tests for the _fmt_naive_year function."""

    def test_fmt_naive_year(self):
        """Test _fmt_naive_year function with various datetime objects."""
        expected = "2023"
        result = p._fmt_naive_year(dt.datetime(2023, 10, 10, 10, 0, 0))
        self.assertEqual(result, expected)


class TestFmtTzdelta(unittest.TestCase):
    """Unit tests for the _fmt_tzdelta function."""

    @parameterized.expand(
        [
            ("zero delta", dt.timedelta(0), "Z"),
            ("positive delta", dt.timedelta(hours=5, minutes=30), "+05:30"),
            ("negative delta", dt.timedelta(hours=-5, minutes=-30), "-05:30"),
            ("positive delta with seconds", dt.timedelta(hours=1, minutes=45, seconds=30), "+01:45"),
            ("negative delta with seconds", dt.timedelta(hours=-1, minutes=-45, seconds=-30), "-01:45"),
        ]
    )
    def test_fmt_tzdelta(self, name, delta, expected):
        """Test _fmt_tzdelta function with various timedelta objects."""
        result = p._fmt_tzdelta(delta)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            ("delta exceeding one day", dt.timedelta(days=1, hours=1)),
            ("negative delta exceeding one day", dt.timedelta(days=-1, hours=-1)),
        ]
    )
    def test_fmt_tzdelta_invalid(self, name, delta):
        """Test _fmt_tzdelta function with invalid timedelta objects."""
        with self.assertRaises(ValueError):
            p._fmt_tzdelta(delta)


class TestFmtTzinfo(unittest.TestCase):
    """Unit tests for the _fmt_tzinfo function."""

    @parameterized.expand(
        [
            ("no tzinfo", None, ""),
            ("UTC tzinfo", dt.timezone.utc, "Z"),
            ("positive tzinfo", dt.timezone(dt.timedelta(hours=5, minutes=30)), "+05:30"),
            ("negative tzinfo", dt.timezone(dt.timedelta(hours=-5, minutes=-30)), "-05:30"),
        ]
    )
    def test_fmt_tzinfo(self, name, tz, expected):
        """Test _fmt_tzinfo function with various tzinfo objects."""
        result = p._fmt_tzinfo(tz)
        self.assertEqual(result, expected)


class TestFmtAwareMicrosecond(unittest.TestCase):
    """Unit tests for the _fmt_aware_microsecond function."""

    @parameterized.expand(
        [
            (
                "UTC timezone with dash separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc),
                "-",
                "2023-10-10-10:00:00.123456Z",
            ),
            (
                "UTC timezone with T separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc),
                "T",
                "2023-10-10T10:00:00.123456Z",
            ),
            (
                "positive timezone",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
                "-",
                "2023-10-10-10:00:00.123456+05:30",
            ),
            (
                "negative timezone",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))),
                "-",
                "2023-10-10-10:00:00.123456-05:30",
            ),
        ]
    )
    def test_fmt_aware_microsecond(self, name, obj, separator, expected):
        """Test _fmt_aware_microsecond function with various datetime objects and separators."""
        result = p._fmt_aware_microsecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareMillisecond(unittest.TestCase):
    """Unit tests for the _fmt_aware_millisecond function."""

    @parameterized.expand(
        [
            (
                "UTC timezone with dash separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc),
                "-",
                "2023-10-10-10:00:00.123Z",
            ),
            (
                "UTC timezone with T separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc),
                "T",
                "2023-10-10T10:00:00.123Z",
            ),
            (
                "positive timezone",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
                "-",
                "2023-10-10-10:00:00.123+05:30",
            ),
            (
                "negative timezone",
                dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))),
                "-",
                "2023-10-10-10:00:00.123-05:30",
            ),
        ]
    )
    def test_fmt_aware_millisecond(self, name, obj, separator, expected):
        """Test _fmt_aware_millisecond function with various datetime objects and separators."""
        result = p._fmt_aware_millisecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareSecond(unittest.TestCase):
    """Unit tests for the _fmt_aware_second function."""

    @parameterized.expand(
        [
            (
                "UTC timezone with dash separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone.utc),
                "-",
                "2023-10-10-10:00:00Z",
            ),
            (
                "UTC timezone with T separator",
                dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone.utc),
                "T",
                "2023-10-10T10:00:00Z",
            ),
            (
                "positive timezone",
                dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
                "-",
                "2023-10-10-10:00:00+05:30",
            ),
            (
                "negative timezone",
                dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))),
                "-",
                "2023-10-10-10:00:00-05:30",
            ),
        ]
    )
    def test_fmt_aware_second(self, name, obj, separator, expected):
        """Test _fmt_aware_second function with various datetime objects and separators."""
        result = p._fmt_aware_second(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareMinute(unittest.TestCase):
    """Unit tests for the _fmt_aware_minute function."""

    @parameterized.expand(
        [
            (
                "UTC timezone with dash separator",
                dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc),
                "-",
                "2023-10-10-10:00Z",
            ),
            (
                "UTC timezone with T separator",
                dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc),
                "T",
                "2023-10-10T10:00Z",
            ),
            (
                "positive timezone",
                dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
                "-",
                "2023-10-10-10:00+05:30",
            ),
            (
                "negative timezone",
                dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))),
                "-",
                "2023-10-10-10:00-05:30",
            ),
        ]
    )
    def test_fmt_aware_minute(self, name, obj, separator, expected):
        """Test _fmt_aware_minute function with various datetime objects and separators."""
        result = p._fmt_aware_minute(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareHour(unittest.TestCase):
    """Unit tests for the _fmt_aware_hour function."""

    @parameterized.expand(
        [
            (
                "UTC timezone with dash separator",
                dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone.utc),
                "-",
                "2023-10-10-10Z",
            ),
            (
                "UTC timezone with T separator",
                dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone.utc),
                "T",
                "2023-10-10T10Z",
            ),
            (
                "positive timezone",
                dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
                "-",
                "2023-10-10-10+05:30",
            ),
            (
                "negative timezone",
                dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))),
                "-",
                "2023-10-10-10-05:30",
            ),
        ]
    )
    def test_fmt_aware_hour(self, name, obj, separator, expected):
        """Test _fmt_aware_hour function with various datetime objects and separators."""
        result = p._fmt_aware_hour(obj, separator)
        self.assertEqual(result, expected)


class TestOfYears(unittest.TestCase):
    """Unit tests for the Properties.of_years method."""

    @parameterized.expand(
        [
            ("one year", 1),
            ("two years", 2),
        ]
    )
    def test_valid_years(self, name, no_of_years):
        """Test Properties.of_years method with various year inputs."""
        expected = p.Properties(
            step=p._STEP_MONTHS,
            multiplier=no_of_years * 12,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = p.Properties.of_years(no_of_years)
        self.assertEqual(result, expected)

    def test_zero_years(self):
        """Test Properties.of_years method with various year inputs."""
        with self.assertRaises(AssertionError):
            p.Properties.of_years(0)


class TestOfMonths(unittest.TestCase):
    """Unit tests for the Properties.of_months method."""

    @parameterized.expand(
        [
            ("one month", 1),
            ("twelve months", 12),
        ]
    )
    def test_valid_months(self, name, no_of_months):
        """Test Properties.of_months method with various month inputs."""
        expected = p.Properties(
            step=p._STEP_MONTHS,
            multiplier=no_of_months,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = p.Properties.of_months(no_of_months)
        self.assertEqual(result, expected)

    def test_zero_months(self):
        """Test Properties.of_months method with zero month input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_months(0)


class TestOfDays(unittest.TestCase):
    """Unit tests for the Properties.of_days method."""

    @parameterized.expand(
        [
            ("one day", 1),
            ("seven days", 7),
        ]
    )
    def test_valid_days(self, name, no_of_days):
        """Test Properties.of_days method with various day inputs."""
        expected = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=no_of_days * 86_400,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = p.Properties.of_days(no_of_days)
        self.assertEqual(result, expected)

    def test_zero_days(self):
        """Test Properties.of_days method with zero day input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_days(0)


class TestOfHours(unittest.TestCase):
    """Unit tests for the Properties.of_hours method."""

    @parameterized.expand(
        [
            ("one hour", 1),
            ("twenty-four hours", 24),
        ]
    )
    def test_valid_hours(self, name, no_of_hours):
        """Test Properties.of_hours method with various hour inputs."""
        expected = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=no_of_hours * 3_600,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = p.Properties.of_hours(no_of_hours)
        self.assertEqual(result, expected)

    def test_zero_hours(self):
        """Test Properties.of_hours method with zero hour input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_hours(0)


class TestOfMinutes(unittest.TestCase):
    """Unit tests for the Properties.of_minutes method."""

    @parameterized.expand(
        [
            ("one minute", 1),
            ("sixty minutes", 60),
        ]
    )
    def test_valid_minutes(self, name, no_of_minutes):
        """Test Properties.of_minutes method with various minute inputs."""
        expected = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=no_of_minutes * 60,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = p.Properties.of_minutes(no_of_minutes)
        self.assertEqual(result, expected)

    def test_zero_minutes(self):
        """Test Properties.of_minutes method with zero minute input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_minutes(0)


class TestOfSeconds(unittest.TestCase):
    """Unit tests for the Properties.of_seconds method."""

    @parameterized.expand(
        [
            ("one second", 1),
            ("sixty seconds", 60),
        ]
    )
    def test_valid_seconds(self, name, no_of_seconds):
        """Test Properties.of_seconds method with various second inputs."""
        expected = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=no_of_seconds,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = p.Properties.of_seconds(no_of_seconds)
        self.assertEqual(result, expected)

    def test_zero_seconds(self):
        """Test Properties.of_seconds method with zero second input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_seconds(0)


class TestOfMicroseconds(unittest.TestCase):
    """Unit tests for the Properties.of_microseconds method."""

    @parameterized.expand(
        [
            ("one microsecond", 1),
            ("one million microseconds", 1_000_000),
        ]
    )
    def test_valid_microseconds(self, name, no_of_microseconds):
        """Test Properties.of_microseconds method with various microsecond inputs."""
        seconds, microseconds = divmod(no_of_microseconds, 1_000_000)
        if microseconds == 0:
            expected = p.Properties(
                step=p._STEP_SECONDS,
                multiplier=seconds,
                month_offset=0,
                microsecond_offset=0,
                tzinfo=None,
                ordinal_shift=0,
            )
        else:
            expected = p.Properties(
                step=p._STEP_MICROSECONDS,
                multiplier=no_of_microseconds,
                month_offset=0,
                microsecond_offset=0,
                tzinfo=None,
                ordinal_shift=0,
            )
        result = p.Properties.of_microseconds(no_of_microseconds)
        self.assertEqual(result, expected)

    def test_zero_microseconds(self):
        """Test Properties.of_microseconds method with zero microsecond input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_microseconds(0)


class TestOfStepAndMultiplier(unittest.TestCase):
    """Unit tests for the Properties.of_step_and_multiplier method."""

    @parameterized.expand(
        [
            ("microseconds step", p._STEP_MICROSECONDS, 1_000_000),
            ("seconds step", p._STEP_SECONDS, 60),
            ("months step", p._STEP_MONTHS, 12),
        ]
    )
    def test_valid_step_and_multiplier(self, name, step, multiplier):
        """Test Properties.of_step_and_multiplier method with various step and multiplier inputs."""
        expected = p.Properties(
            step=step, multiplier=multiplier, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        result = p.Properties.of_step_and_multiplier(step, multiplier)
        self.assertEqual(result, expected)

    def test_invalid_step(self):
        """Test Properties.of_step_and_multiplier method with invalid step input."""
        with self.assertRaises(AssertionError):
            p.Properties.of_step_and_multiplier(999, 1)


class TestNormaliseOffsets(unittest.TestCase):
    """Unit tests for the Properties.normalise_offsets method."""

    @parameterized.expand(
        [
            (
                "normalised months",
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=13,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=1,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "normalised seconds",
                p.Properties(
                    step=p._STEP_SECONDS,
                    multiplier=60,
                    month_offset=0,
                    microsecond_offset=61_000_000,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
                p.Properties(
                    step=p._STEP_SECONDS,
                    multiplier=60,
                    month_offset=0,
                    microsecond_offset=1_000_000,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "normalised microseconds",
                p.Properties(
                    step=p._STEP_MICROSECONDS,
                    multiplier=1_000_000,
                    month_offset=0,
                    microsecond_offset=1_000_001,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
                p.Properties(
                    step=p._STEP_MICROSECONDS,
                    multiplier=1_000_000,
                    month_offset=0,
                    microsecond_offset=1,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "already normalised",
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=1,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=1,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    def test_normalise_offsets(self, name, props, expected):
        """Test Properties.normalise_offsets method with various inputs."""
        result = props.normalise_offsets()
        self.assertEqual(result, expected)


class TestWithMonthOffset(unittest.TestCase):
    """Unit tests for the Properties.with_month_offset method."""

    @parameterized.expand(
        [
            (
                "add one month",
                1,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=1,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "add thirteen months",
                13,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=1,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "no change",
                0,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    def test_with_month_offset(self, name, month_amount, expected):
        """Test Properties.with_month_offset method with various month amounts."""
        props = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        result = props.with_month_offset(month_amount)
        self.assertEqual(result, expected)

    def test_invalid_offset(self):
        """Test Properties.of_step_and_multiplier method with invalid step input."""
        props = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        with self.assertRaises(AssertionError):
            props.with_month_offset(-1)


class TestWithMicrosecondOffset(unittest.TestCase):
    """Unit tests for the Properties.with_microsecond_offset method."""

    @parameterized.expand(
        [
            (
                "add one microsecond",
                1,
                p.Properties(
                    step=p._STEP_MICROSECONDS,
                    multiplier=1_000_000,
                    month_offset=0,
                    microsecond_offset=1,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "add one million microseconds",
                1_000_000,
                p.Properties(
                    step=p._STEP_MICROSECONDS,
                    multiplier=1_000_000,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "no change",
                0,
                p.Properties(
                    step=p._STEP_MICROSECONDS,
                    multiplier=1_000_000,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    def test_with_microsecond_offset(self, name, microsecond_amount, expected):
        """Test Properties.with_microsecond_offset method with various microsecond amounts."""
        props = p.Properties(
            step=p._STEP_MICROSECONDS,
            multiplier=1_000_000,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        result = props.with_microsecond_offset(microsecond_amount)
        self.assertEqual(result, expected)

    def test_invalid_offset(self):
        """Test Properties.with_microsecond_offset method with invalid microsecond input."""
        props = p.Properties(
            step=p._STEP_MICROSECONDS,
            multiplier=1_000_000,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        with self.assertRaises(AssertionError):
            props.with_microsecond_offset(-1)


class TestWithTzinfo(unittest.TestCase):
    """Unit tests for the Properties.with_tzinfo method."""

    @parameterized.expand(
        [
            (
                "UTC timezone",
                dt.timezone.utc,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=dt.timezone.utc,
                    ordinal_shift=0,
                ),
            ),
            (
                "None timezone",
                None,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "positive timezone",
                dt.timezone(dt.timedelta(hours=5, minutes=30)),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)),
                    ordinal_shift=0,
                ),
            ),
            (
                "negative timezone",
                dt.timezone(dt.timedelta(hours=-5, minutes=-30)),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30)),
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    def test_with_tzinfo(self, name, tzinfo, expected):
        """Test Properties.with_tzinfo method with various tzinfo objects."""
        props = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        result = props.with_tzinfo(tzinfo)
        self.assertEqual(result, expected)


class TestWithOrdinalShift(unittest.TestCase):
    """Unit tests for the Properties.with_ordinal_shift method."""

    @parameterized.expand(
        [
            (
                "positive shift",
                1,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=1,
                ),
            ),
            (
                "negative shift",
                -1,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=-1,
                ),
            ),
            (
                "zero shift",
                0,
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    def test_with_ordinal_shift(self, name, ordinal_shift, expected):
        """Test Properties.with_ordinal_shift method with various ordinal shift values."""
        props = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        result = props.with_ordinal_shift(ordinal_shift)
        self.assertEqual(result, expected)


class TestWithOffsetPeriodFields(unittest.TestCase):
    """Unit tests for the Properties.with_offset_period_fields method."""

    @parameterized.expand(
        [
            (
                "offset period fields with months and seconds",
                (6, 1_000_000),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=6,
                    microsecond_offset=1_000_000,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "offset period fields with normalised months and seconds",
                (12, 1_000_000),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=1_000_000,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "offset period fields with zero months and seconds",
                (0, 0),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    @patch("time_series.period.PeriodFields")
    def test_with_offset_period_fields(self, name, months_seconds, expected, mock_period_fields):
        """Test Properties.with_offset_period_fields method with various PeriodFields inputs."""
        mock_months_seconds = Mock()
        mock_months_seconds.months = months_seconds[0]
        mock_months_seconds.total_microseconds.return_value = months_seconds[1]
        mock_period_fields.get_months_seconds.return_value = mock_months_seconds

        props = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        result = props.with_offset_period_fields(mock_period_fields)
        self.assertEqual(result, expected)


class TestWithOffsetMonthsSeconds(unittest.TestCase):
    """Unit tests for the Properties.with_offset_months_seconds method."""

    @parameterized.expand(
        [
            (
                "offset period fields with months and seconds",
                (6, 1_000_000),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=6,
                    microsecond_offset=1_000_000,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "offset period fields with normalised months and seconds",
                (12, 1_000_000),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=1_000_000,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
            (
                "offset period fields with zero months and seconds",
                (0, 0),
                p.Properties(
                    step=p._STEP_MONTHS,
                    multiplier=12,
                    month_offset=0,
                    microsecond_offset=0,
                    tzinfo=None,
                    ordinal_shift=0,
                ),
            ),
        ]
    )
    @patch("time_series.period.MonthsSeconds")
    def test_with_offset_months_seconds(self, name, months_seconds, expected, mock_months_seconds):
        """Test Properties.with_offset_months_seconds method with various MonthsSeconds inputs."""
        mock_months_seconds = Mock()
        mock_months_seconds.months = months_seconds[0]
        mock_months_seconds.total_microseconds.return_value = months_seconds[1]

        props = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        result = props.with_offset_months_seconds(mock_months_seconds)
        self.assertEqual(result, expected)


class TestGetIso8601(unittest.TestCase):
    """Unit tests for the get_iso8601 method."""

    def test_get_iso8601_microseconds(self):
        """Test get_iso8601 method with step as _STEP_MICROSECONDS."""
        properties = p.Properties(
            step=p._STEP_MICROSECONDS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        expected = p._get_microsecond_period_name(1)
        self.assertEqual(properties.get_iso8601(), expected)

    def test_get_iso8601_seconds(self):
        """Test get_iso8601 method with step as _STEP_SECONDS."""
        properties = p.Properties(
            step=p._STEP_SECONDS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        expected = p._get_second_period_name(1)
        self.assertEqual(properties.get_iso8601(), expected)

    def test_get_iso8601_months(self):
        """Test get_iso8601 method with step as _STEP_MONTHS."""
        properties = p.Properties(
            step=p._STEP_MONTHS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        expected = p._get_month_period_name(1)
        self.assertEqual(properties.get_iso8601(), expected)


class TestGetTimedelta(unittest.TestCase):
    """Unit tests for the get_timedelta method."""

    def test_get_timedelta_microseconds(self):
        """Test get_timedelta method with step as _STEP_MICROSECONDS."""
        properties = p.Properties(
            step=p._STEP_MICROSECONDS,
            multiplier=1_500_000,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        expected = dt.timedelta(seconds=1, microseconds=500_000)
        self.assertEqual(properties.get_timedelta(), expected)

    def test_get_timedelta_seconds(self):
        """Test get_timedelta method with step as _STEP_SECONDS."""
        properties = p.Properties(
            step=p._STEP_SECONDS, multiplier=3600, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        expected = dt.timedelta(seconds=3600)
        self.assertEqual(properties.get_timedelta(), expected)

    def test_get_timedelta_months(self):
        """Test get_timedelta method with step as _STEP_MONTHS."""
        properties = p.Properties(
            step=p._STEP_MONTHS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        self.assertIsNone(properties.get_timedelta())


class TestAppendStepElems(unittest.TestCase):
    """Unit tests for the _append_step_elems method."""

    def test_append_step_elems_microseconds(self):
        """Test _append_step_elems method with step as _STEP_MICROSECONDS."""
        properties = p.Properties(
            step=p._STEP_MICROSECONDS,
            multiplier=1_500_000,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=0,
        )
        elems = []
        properties._append_step_elems(elems)
        expected = ["T", "1.5", "S"]
        self.assertEqual(elems, expected)

    def test_append_step_elems_seconds(self):
        """Test _append_step_elems method with step as _STEP_SECONDS."""
        properties = p.Properties(
            step=p._STEP_SECONDS, multiplier=3600, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        elems = []
        properties._append_step_elems(elems)
        expected = ["T", "1H"]
        self.assertEqual(elems, expected)

    def test_append_step_elems_months(self):
        """Test _append_step_elems method with step as _STEP_MONTHS."""
        properties = p.Properties(
            step=p._STEP_MONTHS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        elems = []
        properties._append_step_elems(elems)
        expected = ["1M"]
        self.assertEqual(elems, expected)


class TestAppendOffsetElems(unittest.TestCase):
    """Unit tests for the _append_offset_elems method."""

    def test_append_offset_elems_no_offset(self):
        """Test _append_offset_elems method with no offsets."""
        properties = p.Properties(
            step=p._STEP_SECONDS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        elems = []
        properties._append_offset_elems(elems)
        self.assertEqual(elems, [])

    def test_append_offset_elems_month_offset(self):
        """Test _append_offset_elems method with a month offset."""
        properties = p.Properties(
            step=p._STEP_MONTHS, multiplier=1, month_offset=2, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        elems = []
        properties._append_offset_elems(elems)
        expected = ["+", "2M"]
        self.assertEqual(elems, expected)

    def test_append_offset_elems_microsecond_offset(self):
        """Test _append_offset_elems method with a microsecond offset."""
        properties = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=1,
            month_offset=0,
            microsecond_offset=1_500_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        elems = []
        properties._append_offset_elems(elems)
        expected = ["+", "T", "1.5", "S"]
        self.assertEqual(elems, expected)

    def test_append_offset_elems_both_offsets(self):
        """Test _append_offset_elems method with both month and microsecond offsets."""
        properties = p.Properties(
            step=p._STEP_MONTHS,
            multiplier=1,
            month_offset=2,
            microsecond_offset=1_500_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        elems = []
        properties._append_offset_elems(elems)
        expected = ["+", "2M", "T", "1.5", "S"]
        self.assertEqual(elems, expected)


class TestAppendTzElems(unittest.TestCase):
    """Unit tests for the _append_tz_elems method."""

    @parameterized.expand(
        [
            ("no_tzinfo", None, ["[", "]"]),
            ("with_tzinfo", dt.timezone(dt.timedelta(hours=5, minutes=30)), ["[", "+05:30", "]"]),
            ("with_utc", dt.timezone.utc, ["[", "Z", "]"]),
        ]
    )
    def test_append_tz_elems(self, name, tzinfo, expected):
        """Test _append_tz_elems method with various tzinfo values."""
        properties = p.Properties(
            step=p._STEP_SECONDS, multiplier=1, month_offset=0, microsecond_offset=0, tzinfo=tzinfo, ordinal_shift=0
        )
        elems = []
        properties._append_tz_elems(elems)
        self.assertEqual(elems, expected)


class TestAppendShiftElems(unittest.TestCase):
    """Unit tests for the _append_shift_elems method."""

    @parameterized.expand(
        [
            ("no_shift", 0, []),
            ("positive_shift", 5, ["5"]),
            ("negative_shift", -3, ["-3"]),
        ]
    )
    def test_append_shift_elems(self, name, ordinal_shift, expected):
        """Test _append_shift_elems method with various ordinal shifts."""
        properties = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=1,
            month_offset=0,
            microsecond_offset=0,
            tzinfo=None,
            ordinal_shift=ordinal_shift,
        )
        elems = []
        properties._append_shift_elems(elems)
        self.assertEqual(elems, expected)


class TestGetNaiveFormatter(unittest.TestCase):
    """Unit tests for the get_naive_formatter method."""

    def test_get_naive_formatter_microseconds(self):
        """Test get_naive_formatter method with microsecond offset."""
        properties = p.Properties(
            step=p._STEP_MICROSECONDS,
            multiplier=1_500_000,
            month_offset=0,
            microsecond_offset=1,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_microsecond(dt.datetime(2023, 10, 10, 10, 0), "T")
        )

    def test_get_naive_formatter_milliseconds(self):
        """Test get_naive_formatter method with millisecond offset."""
        properties = p.Properties(
            step=p._STEP_MICROSECONDS,
            multiplier=1_000_000,
            month_offset=0,
            microsecond_offset=1_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_millisecond(dt.datetime(2023, 10, 10, 10, 0), "T")
        )

    def test_get_naive_formatter_seconds(self):
        """Test get_naive_formatter method with second offset."""
        properties = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=3600,
            month_offset=0,
            microsecond_offset=1_000_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_second(dt.datetime(2023, 10, 10, 10, 0), "T")
        )

    def test_get_naive_formatter_minutes(self):
        """Test get_naive_formatter method with minute offset."""
        properties = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=60,
            month_offset=0,
            microsecond_offset=60_000_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_minute(dt.datetime(2023, 10, 10, 10, 0), "T")
        )

    def test_get_naive_formatter_hours(self):
        """Test get_naive_formatter method with hour offset."""
        properties = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=3600 * 24,
            month_offset=0,
            microsecond_offset=3_600_000_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_hour(dt.datetime(2023, 10, 10, 10, 0), "T")
        )

    def test_get_naive_formatter_days(self):
        """Test get_naive_formatter method with day offset."""
        properties = p.Properties(
            step=p._STEP_SECONDS,
            multiplier=3600 * 24 * 2,
            month_offset=0,
            microsecond_offset=86_400_000_000,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_day(dt.datetime(2023, 10, 10, 10, 0))
        )

    def test_get_naive_formatter_months(self):
        """Test get_naive_formatter method with month offset."""
        properties = p.Properties(
            step=p._STEP_MONTHS, multiplier=1, month_offset=1, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_month(dt.datetime(2023, 10, 10, 10, 0))
        )

    def test_get_naive_formatter_years(self):
        """Test get_naive_formatter method with year offset."""
        properties = p.Properties(
            step=p._STEP_MONTHS, multiplier=12, month_offset=12, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        formatter = properties.get_naive_formatter("T")
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0)), p._fmt_naive_year(dt.datetime(2023, 10, 10, 10, 0))
        )


class TestGetAwareFormatter(unittest.TestCase):
    """Unit tests for the get_aware_formatter method."""

    @parameterized.expand(
        [
            ("microseconds", p._STEP_MICROSECONDS, 1_500_000, 1, "T", p._fmt_aware_microsecond),
            ("milliseconds", p._STEP_MICROSECONDS, 1_000_000, 1_000, "T", p._fmt_aware_millisecond),
            ("seconds", p._STEP_SECONDS, 3600, 1_000_000, "T", p._fmt_aware_second),
            ("minutes", p._STEP_SECONDS, 60, 60_000_000, "T", p._fmt_aware_minute),
            ("hours", p._STEP_SECONDS, 3600 * 24, 3_600_000_000, "T", p._fmt_aware_hour),
        ]
    )
    def test_get_aware_formatter(self, name, step, multiplier, microsecond_offset, separator, expected_formatter):
        """Test get_aware_formatter method with various steps and multipliers."""
        properties = p.Properties(
            step=step,
            multiplier=multiplier,
            month_offset=0,
            microsecond_offset=microsecond_offset,
            tzinfo=None,
            ordinal_shift=0,
        )
        formatter = properties.get_aware_formatter(separator)
        self.assertEqual(
            formatter(dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc)),
            expected_formatter(dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc), separator),
        )


class TestPlInterval(unittest.TestCase):
    """Unit tests for the pl_interval method."""

    @parameterized.expand(
        [
            ("microseconds", p._STEP_MICROSECONDS, 1_500_000, "1500000us"),
            ("seconds", p._STEP_SECONDS, 3600, "3600s"),
            ("months", p._STEP_MONTHS, 1, "1mo"),
        ]
    )
    def test_pl_interval_valid(self, name, step, multiplier, expected):
        """Test pl_interval method with valid steps and multipliers."""
        properties = p.Properties(
            step=step, multiplier=multiplier, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        self.assertEqual(properties.pl_interval(), expected)


class TestPlOffset(unittest.TestCase):
    """Unit tests for the pl_offset method."""

    @parameterized.expand(
        [
            ("no_offset", p._STEP_SECONDS, 0, 0, "0mo0us"),
            ("month_offset", p._STEP_MONTHS, 2, 0, "2mo0us"),
            ("microsecond_offset", p._STEP_MICROSECONDS, 0, 1_500_000, "0mo1500000us"),
            ("both_offsets", p._STEP_MONTHS, 2, 1_500_000, "2mo1500000us"),
        ]
    )
    def test_pl_offset(self, name, step, month_offset, microsecond_offset, expected):
        """Test pl_offset method with various month and microsecond offsets."""
        properties = p.Properties(
            step=step,
            multiplier=1,
            month_offset=month_offset,
            microsecond_offset=microsecond_offset,
            tzinfo=None,
            ordinal_shift=0,
        )
        self.assertEqual(properties.pl_offset(), expected)


class TestIsEpochAgnostic(unittest.TestCase):
    """Unit tests for the is_epoch_agnostic method."""

    @parameterized.expand(
        [
            ("microseconds_epoch_agnostic", p._STEP_MICROSECONDS, 1, True),
            ("microseconds_not_epoch_agnostic", p._STEP_MICROSECONDS, 1_500_000, False),
            ("seconds_epoch_agnostic", p._STEP_SECONDS, 1, True),
            ("seconds_not_epoch_agnostic", p._STEP_SECONDS, 86_401, False),
            ("months_epoch_agnostic", p._STEP_MONTHS, 1, True),
            ("months_not_epoch_agnostic", p._STEP_MONTHS, 13, False),
        ]
    )
    def test_is_epoch_agnostic(self, name, step, multiplier, expected):
        """Test is_epoch_agnostic method with various steps and multipliers."""
        properties = p.Properties(
            step=step, multiplier=multiplier, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )
        self.assertEqual(properties.is_epoch_agnostic(), expected)


class TestPropertiesStrMethod(unittest.TestCase):
    """Unit tests for the __str__ method."""

    @parameterized.expand(
        [
            ("microseconds", p._STEP_MICROSECONDS, 1_500_000, 0, "PT1.5S"),
            ("seconds", p._STEP_SECONDS, 3600, 0, "PT1H"),
            ("months", p._STEP_MONTHS, 1, 0, "P1M"),
            ("microseconds_with_offset", p._STEP_MICROSECONDS, 1_500_000, 1, "PT1.5S+T0.000001S"),
            ("seconds_with_offset", p._STEP_SECONDS, 3600, 1, "PT1H+T0.000001S"),
            ("months_with_offset", p._STEP_MONTHS, 1, 1, "P1M+T0.000001S"),
        ]
    )
    def test_str_method(self, name, step, multiplier, microsecond_offset, expected):
        """Test __str__ method with various steps, multipliers, and offsets."""
        properties = p.Properties(
            step=step,
            multiplier=multiplier,
            month_offset=0,
            microsecond_offset=microsecond_offset,
            tzinfo=None,
            ordinal_shift=0,
        )
        self.assertEqual(str(properties), expected)


class TestPropertiesReprMethod(unittest.TestCase):
    """Unit tests for the __repr__ method."""

    @parameterized.expand(
        [
            ("microseconds", p._STEP_MICROSECONDS, 1_500_000, 0, None, 0, "PT1.5S[]"),
            ("seconds", p._STEP_SECONDS, 3600, 0, None, 0, "PT1H[]"),
            ("months", p._STEP_MONTHS, 1, 0, None, 0, "P1M[]"),
            ("microseconds_with_offset", p._STEP_MICROSECONDS, 1_500_000, 1, None, 0, "PT1.5S+T0.000001S[]"),
            ("seconds_with_offset", p._STEP_SECONDS, 3600, 1, None, 0, "PT1H+T0.000001S[]"),
            ("months_with_offset", p._STEP_MONTHS, 1, 1, None, 0, "P1M+T0.000001S[]"),
            ("with_tzinfo", p._STEP_SECONDS, 3600, 0, dt.timezone.utc, 0, "PT1H[Z]"),
            ("with_ordinal_shift", p._STEP_SECONDS, 3600, 0, None, 5, "PT1H[]5"),
        ]
    )
    def test_repr_method(self, name, step, multiplier, microsecond_offset, tzinfo, ordinal_shift, expected):
        """Test __repr__ method with various steps, multipliers, offsets, tzinfo, and ordinal shifts."""
        properties = p.Properties(
            step=step,
            multiplier=multiplier,
            month_offset=0,
            microsecond_offset=microsecond_offset,
            tzinfo=tzinfo,
            ordinal_shift=ordinal_shift,
        )
        self.assertEqual(repr(properties), expected)


class TestMonthsSecondsPostInit(unittest.TestCase):
    """Unit tests for the __post_init__ method."""

    @parameterized.expand(
        [
            ("valid_months", "P1M", 1, 0, 0),
        ]
    )
    def test_post_init_valid(self, name, string, months, seconds, microseconds):
        """Test __post_init__ method with valid periods."""
        period = p.MonthsSeconds(string, months, seconds, microseconds)
        self.assertEqual(period.string, string)
        self.assertEqual(period.months, months)
        self.assertEqual(period.seconds, seconds)
        self.assertEqual(period.microseconds, microseconds)

    @parameterized.expand(
        [
            ("invalid_negative_months", "P-1M", -1, 0, 0),
            ("invalid_negative_seconds", "PT-1S", 0, -1, 0),
            ("invalid_negative_microseconds", "PT-1us", 0, 0, -1),
            ("invalid_large_microseconds", "PT1.000001S", 0, 0, 1_000_001),
            ("invalid_zero_period", "P0D", 0, 0, 0),
        ]
    )
    def test_post_init_invalid(self, name, string, months, seconds, microseconds):
        """Test __post_init__ method with invalid periods."""
        with self.assertRaises(ValueError):
            p.MonthsSeconds(string, months, seconds, microseconds)


class TestGetStepAndMultiplier(unittest.TestCase):
    """Unit tests for the get_step_and_multiplier method."""

    @parameterized.expand(
        [
            ("months_only", "P1M", 1, 0, 0, (p._STEP_MONTHS, 1)),
            ("seconds_only", "PT1S", 0, 1, 0, (p._STEP_SECONDS, 1)),
            ("microseconds_only", "PT0.000001S", 0, 0, 1, (p._STEP_MICROSECONDS, 1)),
        ]
    )
    def test_get_step_and_multiplier_valid(self, name, string, months, seconds, microseconds, expected):
        """Test get_step_and_multiplier method with valid periods."""
        period = p.MonthsSeconds(string, months, seconds, microseconds)
        self.assertEqual(period.get_step_and_multiplier(), expected)

    @parameterized.expand(
        [
            ("invalid_combination", "P1M1S", 1, 1, 0),
        ]
    )
    def test_get_step_and_multiplier_invalid(self, name, string, months, seconds, microseconds):
        """Test get_step_and_multiplier method with invalid periods."""
        period = p.MonthsSeconds(string, months, seconds, microseconds)
        with self.assertRaises(ValueError):
            period.get_step_and_multiplier()


class TestTotalMicroseconds(unittest.TestCase):
    """Unit tests for the total_microseconds method."""

    @parameterized.expand(
        [
            ("seconds_only", "PT1S", 0, 1, 0, 1_000_000),
            ("microseconds_only", "PT0.000001S", 0, 0, 1, 1),
            ("seconds_and_microseconds", "PT1.000001S", 0, 1, 1, 1_000_001),
        ]
    )
    def test_total_microseconds(self, name, string, months, seconds, microseconds, expected):
        """Test total_microseconds method with various periods."""
        period = p.MonthsSeconds(string, months, seconds, microseconds)
        self.assertEqual(period.total_microseconds(), expected)


class TestMonthsSecondsGetBaseProperties(unittest.TestCase):
    """Unit tests for the get_base_properties method."""

    @parameterized.expand(
        [
            ("months_only", "P1M", 1, 0, 0, p.Properties(p._STEP_MONTHS, 1, 0, 0, None, 0)),
            ("seconds_only", "PT1S", 0, 1, 0, p.Properties(p._STEP_SECONDS, 1, 0, 0, None, 0)),
            ("microseconds_only", "PT0.000001S", 0, 0, 1, p.Properties(p._STEP_MICROSECONDS, 1, 0, 0, None, 0)),
        ]
    )
    def test_get_base_properties_valid(self, name, string, months, seconds, microseconds, expected):
        """Test get_base_properties method with valid periods."""
        period = p.MonthsSeconds(string, months, seconds, microseconds)
        self.assertEqual(period.get_base_properties(), expected)

    @parameterized.expand(
        [
            ("invalid_combination", "P1M1S", 1, 1, 0),
        ]
    )
    def test_get_base_properties_invalid(self, name, string, months, seconds, microseconds):
        """Test get_base_properties method with invalid periods."""
        period = p.MonthsSeconds(string, months, seconds, microseconds)
        with self.assertRaises(ValueError):
            period.get_base_properties()


class TestPeriodFieldsPostInit(unittest.TestCase):
    """Unit tests for the __post_init__ method."""

    @parameterized.expand(
        [
            ("valid_period", "P1Y1M1DT1H1M1S", 1, 1, 1, 1, 1, 1, 1),
        ]
    )
    def test_post_init_valid(self, name, string, years, months, days, hours, minutes, seconds, microseconds):
        """Test __post_init__ method with valid periods."""
        period = p.PeriodFields(string, years, months, days, hours, minutes, seconds, microseconds)
        self.assertEqual(period.string, string)
        self.assertEqual(period.years, years)
        self.assertEqual(period.months, months)
        self.assertEqual(period.days, days)
        self.assertEqual(period.hours, hours)
        self.assertEqual(period.minutes, minutes)
        self.assertEqual(period.seconds, seconds)
        self.assertEqual(period.microseconds, microseconds)

    @parameterized.expand(
        [
            ("invalid_negative_years", "P-1Y", -1, 0, 0, 0, 0, 0, 0),
            ("invalid_negative_months", "P-1M", 0, -1, 0, 0, 0, 0, 0),
            ("invalid_negative_days", "P-1D", 0, 0, -1, 0, 0, 0, 0),
            ("invalid_negative_hours", "PT-1H", 0, 0, 0, -1, 0, 0, 0),
            ("invalid_negative_minutes", "PT-1M", 0, 0, 0, 0, -1, 0, 0),
            ("invalid_negative_seconds", "PT-1S", 0, 0, 0, 0, 0, -1, 0),
            ("invalid_negative_microseconds", "PT-1us", 0, 0, 0, 0, 0, 0, -1),
            ("invalid_large_microseconds", "PT1.000001S", 0, 0, 0, 0, 0, 0, 1_000_001),
        ]
    )
    def test_post_init_invalid(self, name, string, years, months, days, hours, minutes, seconds, microseconds):
        """Test __post_init__ method with invalid periods."""
        with self.assertRaises(ValueError):
            p.PeriodFields(string, years, months, days, hours, minutes, seconds, microseconds)


class TestGetMonthsSeconds(unittest.TestCase):
    """Unit tests for the get_months_seconds method."""

    @parameterized.expand(
        [
            ("valid_period", "P1Y1M1DT1H1M1S", 1, 1, 1, 1, 1, 1, 1, p.MonthsSeconds("P1Y1M1DT1H1M1S", 13, 90061, 1)),
        ]
    )
    def test_get_months_seconds(
        self, name, string, years, months, days, hours, minutes, seconds, microseconds, expected
    ):
        """Test get_months_seconds method with valid periods."""
        period = p.PeriodFields(string, years, months, days, hours, minutes, seconds, microseconds)
        self.assertEqual(period.get_months_seconds(), expected)


class TestPeriodFieldsGetBaseProperties(unittest.TestCase):
    """Unit tests for the get_base_properties method."""

    @parameterized.expand(
        [
            ("valid_period_months", "P1Y1M", 1, 1, 0, 0, 0, 0, 0, p.Properties(p._STEP_MONTHS, 13, 0, 0, None, 0)),
            (
                "valid_period_seconds",
                "P1DT1H1M1S",
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                p.Properties(p._STEP_SECONDS, 86_400 + 60 * 60 + 60 + 1, 0, 0, None, 0),
            ),
            (
                "valid_period_microseconds",
                "P1.234567S",
                0,
                0,
                0,
                0,
                0,
                1,
                234_567,
                p.Properties(p._STEP_MICROSECONDS, 1_234_567, 0, 0, None, 0),
            ),
        ]
    )
    def test_get_base_properties(
        self, name, string, years, months, days, hours, minutes, seconds, microseconds, expected
    ):
        """Test get_base_properties method with valid periods."""
        period = p.PeriodFields(string, years, months, days, hours, minutes, seconds, microseconds)
        self.assertEqual(period.get_base_properties(), expected)


class TestStr2Int(unittest.TestCase):
    """Unit tests for the _str2int function."""

    @parameterized.expand(
        [
            ("none_default_0", None, 0, 0),
            ("none_default_5", None, 5, 5),
            ("string_number", "10", 0, 10),
            ("string_negative_number", "-10", 0, -10),
            ("string_zero", "0", 0, 0),
        ]
    )
    def test_str2int(self, name, num, default, expected):
        """Test _str2int function with various inputs."""
        self.assertEqual(p._str2int(num, default), expected)


class TestStr2Microseconds(unittest.TestCase):
    """Unit tests for the _str2microseconds function."""

    @parameterized.expand(
        [
            ("none_default_0", None, 0, 0),
            ("none_default_5", None, 5, 5),
            ("microseconds_3_digits", "123", 0, 123000),
            ("microseconds_6_digits", "123456", 0, 123456),
            ("microseconds_less_than_6_digits", "123", 0, 123000),
            ("microseconds_more_than_6_digits", "1234567", 0, 123456),
        ]
    )
    def test_str2microseconds(self, name, num, default, expected):
        """Test _str2microseconds function with various inputs."""
        self.assertEqual(p._str2microseconds(num, default), expected)


class TestPeriodMatch(unittest.TestCase):
    """Unit tests for the _period_match function."""

    def test_period_match_valid(self):
        """Test _period_match function with a valid match."""
        pattern = re.compile(
            r"(?P<prefix_years>\d+)?-?(?P<prefix_months>\d+)?-?(?P<prefix_days>\d+)?T?(?P<prefix_hours>\d+)?:(?P<prefix_minutes>\d+)?:(?P<prefix_seconds>\d+)?\.?(?P<prefix_microseconds>\d+)?"
        )
        matcher = pattern.match("2023-10-25T01:02:03.456789")
        expected = p.PeriodFields("2023-10-25T01:02:03.456789", 2023, 10, 25, 1, 2, 3, 456789)
        self.assertEqual(p._period_match("prefix", matcher), expected)


class TestTotalMicroseconds(unittest.TestCase):
    """Unit tests for the _total_microseconds function."""

    @parameterized.expand(
        [
            ("one_second", dt.timedelta(seconds=1), 1_000_000),
            ("one_day", dt.timedelta(days=1), 86_400_000_000),
            ("one_day_one_second", dt.timedelta(days=1, seconds=1), 86_401_000_000),
            ("one_day_one_microsecond", dt.timedelta(days=1, microseconds=1), 86_400_000_001),
        ]
    )
    def test_total_microseconds(self, name, delta, expected):
        """Test _total_microseconds function with various inputs."""
        self.assertEqual(p._total_microseconds(delta), expected)


if __name__ == "__main__":
    unittest.main()
