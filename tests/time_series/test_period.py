import unittest
import datetime as dt

from unittest.mock import patch, Mock
from typing import Callable
from parameterized import parameterized

from time_series.period import (
    _naive,
    _gregorian_seconds,
    _period_regex,
    year_shift,
    month_shift,
    DateTimeAdjusters,
    _second_string,
    _append_second_elems,
    _append_month_elems,
    _get_microsecond_period_name,
    _get_second_period_name,
    _get_month_period_name,
    _STEP_MICROSECONDS,
    _STEP_SECONDS,
    _STEP_MONTHS,
    _fmt_naive_microsecond,
    _fmt_naive_millisecond,
    _fmt_naive_second,
    _fmt_naive_minute,
    _fmt_naive_hour,
    _fmt_naive_day,
    _fmt_naive_month,
    _fmt_naive_year,
    _fmt_tzdelta,
    _fmt_tzinfo,
    _fmt_aware_microsecond,
    _fmt_aware_millisecond,
    _fmt_aware_second,
    _fmt_aware_minute,
    _fmt_aware_hour,
    Properties
)


class TestNaive(unittest.TestCase):
    """ Unit tests for the _naive function.
    """

    def test_naive_with_tzinfo(self):
        """ Test _naive function with a datetime object that has tzinfo.
        """
        dt_with_tz = dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc)
        result = _naive(dt_with_tz)
        self.assertIsNone(result.tzinfo)
        self.assertEqual(result, dt.datetime(2023, 10, 10, 10, 0))

    def test_naive_without_tzinfo(self):
        """ Test _naive function with a datetime object that has no tzinfo.
        """
        dt_without_tz = dt.datetime(2023, 10, 10, 10, 0)
        result = _naive(dt_without_tz)
        self.assertIsNone(result.tzinfo)
        self.assertEqual(result, dt_without_tz)


class TestGregorianSeconds(unittest.TestCase):
    """ Unit tests for the _gregorian_seconds function.
    """

    @parameterized.expand([
        ("Zero", dt.datetime(1, 1, 1, 0, 0, 0), 0), 
        ("2 hours", dt.datetime(1, 1, 1, 2, 2, 2), (3600 * 2) + (60 * 2) + 2), 
    ])
    def test_gregorian_seconds(self, name, dt_obj, expected_seconds):
        """ Test _gregorian_seconds function with various datetime inputs.
        """
        self.assertEqual(_gregorian_seconds(dt_obj), expected_seconds)


class TestPeriodRegex(unittest.TestCase):
    """ Unit tests for the _period_regex function.
    """

    def test_period_regex(self):
        """ Test regex string returned from _period_regex function.
        """
        regex = _period_regex("test")
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
    """ Unit tests for the year_shift function.
    """

    def test_no_shift(self):
        """ Test that the date remains the same when shift_amount is 0.
        """
        date_time = dt.datetime(2023, 10, 8)
        self.assertEqual(year_shift(date_time, 0), date_time)

    def test_positive_shift(self):
        """ Test shifting the date forward by a positive number of years.
        """
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2025, 10, 8)
        self.assertEqual(year_shift(date_time, 2), expected)

    def test_negative_shift(self):
        """ Test shifting the date backward by a negative number of years.
        """
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2021, 10, 8)
        self.assertEqual(year_shift(date_time, -2), expected)

    def test_leap_year(self):
        """ Test shifting a leap day to another leap year.
        """
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2024, 2, 29)
        self.assertEqual(year_shift(date_time, 4), expected)

    def test_non_leap_year(self):
        """ Test shifting a leap day to a non-leap year.
        """
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2021, 2, 28)
        self.assertEqual(year_shift(date_time, 1), expected)

    def test_end_of_month(self):
        """ Test shifting a date at the end of the month.
        """
        date_time = dt.datetime(2023, 1, 31)
        expected = dt.datetime(2024, 1, 31)
        self.assertEqual(year_shift(date_time, 1), expected)


class TestMonthShift(unittest.TestCase):
    """ Unit tests for the month_shift function.
    """

    def test_no_shift(self):
        """ Test that the date remains the same when shift_amount is 0.
        """
        date_time = dt.datetime(2023, 10, 8)
        self.assertEqual(month_shift(date_time, 0), date_time)

    def test_positive_shift(self):
        """ Test shifting the date forward by a positive number of months.
        """
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2024, 4, 8)
        self.assertEqual(month_shift(date_time, 6), expected)

    def test_negative_shift(self):
        """ Test shifting the date backward by a negative number of months.
        """
        date_time = dt.datetime(2023, 10, 8)
        expected = dt.datetime(2023, 4, 8)
        self.assertEqual(month_shift(date_time, -6), expected)

    def test_end_of_month(self):
        """ Test shifting a date at the end of the month.
        """
        date_time = dt.datetime(2023, 1, 31)
        expected = dt.datetime(2023, 2, 28)  # February in a non-leap year
        self.assertEqual(month_shift(date_time, 1), expected)

    def test_leap_year(self):
        """ Test shifting a date in a leap year.
        """
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2020, 3, 29)
        self.assertEqual(month_shift(date_time, 1), expected)

    def test_non_leap_year(self):
        """ Test shifting a date from a leap year to a non-leap year.
        """
        date_time = dt.datetime(2020, 2, 29)
        expected = dt.datetime(2021, 2, 28)
        self.assertEqual(month_shift(date_time, 12), expected)


class TestOfMonthOffset(unittest.TestCase):
    """ Unit tests for of_month_offset in the the DateTimeAdjusters class.
    """

    @parameterized.expand([
        ("15 months", 15), 
        ("12 months", 12), 
        ("6 months", 6)
    ])
    def test_of_month_offset_year_shift(self, name, months):
        """ Test DateTimeAdjusters.of_month_offset with a differing number of months
        """
        adjusters = DateTimeAdjusters.of_month_offset(months)
        self.assertIsInstance(adjusters, DateTimeAdjusters)
        self.assertIsInstance(adjusters.advance, Callable)
        self.assertIsInstance(adjusters.retreat, Callable)

    def test_of_month_offset_zero(self):
        """ Test DateTimeAdjusters.of_month_offset with a zero shift.
        """
        adjusters = DateTimeAdjusters.of_month_offset(0)
        self.assertIsNone(adjusters)


class TestOfMircorsecondOffset(unittest.TestCase):
    """ Unit tests for of_microsecond_offset in the the DateTimeAdjusters class.
    """

    def test_of_microsecond_offset(self):
        """ Test DateTimeAdjusters.of_microsecond_offset with various microsecond offsets.
        """
        microseconds = 100
        adjusters = DateTimeAdjusters.of_microsecond_offset(microseconds)
        self.assertIsInstance(adjusters, DateTimeAdjusters)
        self.assertIsInstance(adjusters.advance, Callable)
        self.assertIsInstance(adjusters.retreat, Callable)

        # As lambda functions are returned for the adjuster functions, test their behaviour
        dt_obj = dt.datetime(2023, 10, 10, 10, 0, 0)
        expected_advance = dt_obj + dt.timedelta(microseconds=microseconds)
        expected_retreat = dt_obj - dt.timedelta(microseconds=microseconds)

        self.assertEqual(adjusters.advance(dt_obj), expected_advance)
        self.assertEqual(adjusters.retreat(dt_obj), expected_retreat)

    def test_of_microsecond_offset_zero(self):
        """ Test DateTimeAdjusters.of_microsecond_offset with a zero shift.
        """
        adjusters = DateTimeAdjusters.of_microsecond_offset(0)
        self.assertIsNone(adjusters)


class TestOfOffsets(unittest.TestCase):
    """ Unit tests for of_offsets in the the DateTimeAdjusters class.
    """

    @parameterized.expand([
        ("15 months, 1000000 microseconds", 15, 1_000_000, dt.datetime(2024, 4, 1, 0, 0, 1), dt.datetime(2021, 9, 30, 23, 59, 59)),
        ("12 months, 500000 microseconds", 12, 500_000, dt.datetime(2024, 1, 1, 0, 0, 0, 500_000), dt.datetime(2021, 12, 31, 23, 59, 59, 500000)),
        ("6 months, 1000 microseconds", 6, 1_000, dt.datetime(2023, 7, 1, 0, 0, 0, 1000), dt.datetime(2022, 6, 30, 23, 59, 59, 999000)),
        ("0 months, 1000 microseconds", 0, 1_000, dt.datetime(2023, 1, 1, 0, 0, 0, 1000), dt.datetime(2022, 12, 31, 23, 59, 59, 999000)),
        ("6 months, 0 microseconds", 6, 0, dt.datetime(2023, 7, 1, 0, 0, 0,),  dt.datetime(2022, 7, 1, 0, 0, 0))
    ])
    def test_of_offsets(self, name, months, microseconds, expected_adv, expected_ret):
        """ Test DateTimeAdjusters.of_offsets with various month and microsecond offsets.
        """
        adjusters = DateTimeAdjusters.of_offsets(months, microseconds)
        self.assertIsInstance(adjusters, DateTimeAdjusters)
        self.assertIsInstance(adjusters.advance, Callable)
        self.assertIsInstance(adjusters.retreat, Callable)

        dt_obj = dt.datetime(2023, 1, 1, 0, 0, 0)
        self.assertEqual(adjusters.advance(dt_obj), expected_adv)
        self.assertEqual(adjusters.retreat(dt_obj), expected_ret)

    def test_of_offsets_zero(self):
        """ Test DateTimeAdjusters.of_offsets with zero month and microsecond offsets.
        """
        with self.assertRaises(AssertionError):
            DateTimeAdjusters.of_offsets(0, 0)


class TestDateTimeAdjustersPostInit(unittest.TestCase):
    """ Unit tests for __post_init__ in the the DateTimeAdjusters class.
    """

    def test_post_init_no_error(self):
        """ Test DateTimeAdjusters.__post_init__ does not raise an error when advance and retreat are set
        """
        DateTimeAdjusters(retreat=lambda x: x, advance=lambda x: x).__post_init__()

    @parameterized.expand([
        ("Both none", None, None), 
        ("Retreat None", None, lambda x: x), 
        ("Advance None", lambda x: x, None)
    ])
    def test_post_init_raises_with_none(self, name, retreat, advance):
        """ Test DateTimeAdjusters.__post_init__ to ensure it raises AssertionError when retreat or advance is None.
        """
        with self.assertRaises(AssertionError):
            DateTimeAdjusters(retreat=retreat, advance=advance).__post_init__()


class TestSecondString(unittest.TestCase):
    """ Unit tests for the _second_string function.
    """

    @parameterized.expand([
        ("whole seconds", 10, 0, "10"),
        ("seconds with microseconds", 10, 500000, "10.5"),
        ("seconds with trailing zero microseconds", 10, 5000000, "10.5"),
        ("seconds with no trailing zero microseconds", 10, 500001, "10.500001"),
        ("zero seconds", 0, 0, "0"),
        ("zero seconds with microseconds", 0, 123456, "0.123456")
    ])
    def test_second_string(self, name, seconds, microseconds, expected):
        """ Test _second_string function with various seconds and microseconds.
        """
        result = _second_string(seconds, microseconds)
        self.assertEqual(result, expected)


class TestAppendSecondElems(unittest.TestCase):
    """ Unit tests for the _append_second_elems function.
    """

    @parameterized.expand([
        ("no days, hours, minutes, or microseconds", [], 10, 0, ["T", "10", "S"]),
        ("with microseconds", [], 10, 500000, ["T", "10.5", "S"]),
        ("with minutes", [], 70, 0, ["T", "1M", "10", "S"]),
        ("with hours", [], 3660, 0, ["T", "1H", "1M"]),
        ("with days", [], 90000, 0, ["1D", "T", "1H"]),
        ("complex case", ["P"], 90061, 500000, ["P", "1D", "T", "1H", "1M", "1.5", "S"])
    ])
    def test_append_second_elems(self, name, elems, seconds, microseconds, expected):
        """ Test _append_second_elems function with various inputs.
        """
        result = _append_second_elems(elems, seconds, microseconds)
        self.assertEqual(result, expected)


class TestAppendMonthElems(unittest.TestCase):
    """ Unit tests for the _append_month_elems function.
    """

    @parameterized.expand([
        ("no months", [], 0, []),
        ("only months", [], 5, ["5M"]),
        ("only years", [], 24, ["2Y"]),
        ("years and months", [], 30, ["2Y", "6M"]),
        ("complex case", ["P"], 18, ["P", "1Y", "6M"])
    ])
    def test_append_month_elems(self, name, elems, months, expected):
        """ Test _append_month_elems function with various inputs.
        """
        result = _append_month_elems(elems, months)
        self.assertEqual(result, expected)


class TestGetMicrosecondPeriodName(unittest.TestCase):
    """ Unit tests for the _get_microsecond_period_name function.
    """

    @parameterized.expand([
        ("zero microseconds", 0, "P"),
        ("only seconds", 1_000_000, "PT1S"),
        ("seconds and microseconds", 1_500_000, "PT1.5S"),
        ("only microseconds", 500_000, "PT0.5S"),
        ("complex case", 3_645_034_555, "PT1H45.034555S")
    ])
    def test_get_microsecond_period_name(self, name, total_microseconds, expected):
        """ Test _get_microsecond_period_name function with various inputs.
        """
        result = _get_microsecond_period_name(total_microseconds)
        self.assertEqual(result, expected)


class TestGetSecondPeriodName(unittest.TestCase):
    """ Unit tests for the _get_second_period_name function.
    """

    @parameterized.expand([
        ("zero seconds", 0, "P"),
        ("only seconds", 10, "PT10S"),
        ("seconds forming minutes", 70, "PT1M10S"),
        ("seconds forming hours", 3660, "PT1H1M"),
        ("complex case", 90061, "P1DT1H1M1S")
    ])
    def test_get_second_period_name(self, name, seconds, expected):
        """ Test _get_second_period_name function with various inputs.
        """
        result = _get_second_period_name(seconds)
        self.assertEqual(result, expected)


class TestGetMonthPeriodName(unittest.TestCase):
    """ Unit tests for the _get_month_period_name function.
    """

    @parameterized.expand([
        ("zero months", 0, "P"),
        ("only months", 5, "P5M"),
        ("only years", 24, "P2Y"),
        ("years and months", 30, "P2Y6M")
    ])
    def test_get_month_period_name(self, name, months, expected):
        """ Test _get_month_period_name function with various inputs.
        """
        result = _get_month_period_name(months)
        self.assertEqual(result, expected)


class TestFmtNaiveMicrosecond(unittest.TestCase):
    """ Unit tests for the _fmt_naive_microsecond function.
    """

    @parameterized.expand([
        ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456), "-", "2023-10-10-10:00:00.123456"),
        ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456), "T", "2023-10-10T10:00:00.123456"),
        ("datetime with zero microseconds", dt.datetime(2023, 10, 10, 10, 0, 0, 0), "-", "2023-10-10-10:00:00.000000")
    ])
    def test_fmt_naive_microsecond(self, name, obj, separator, expected):
        """ Test _fmt_naive_microsecond function with various datetime objects and separators.
        """
        result = _fmt_naive_microsecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveMillisecond(unittest.TestCase):
    """ Unit tests for the _fmt_naive_millisecond function.
    """

    @parameterized.expand([
        ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456), "-", "2023-10-10-10:00:00.123"),
        ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456), "T", "2023-10-10T10:00:00.123"),
        ("datetime with zero microseconds", dt.datetime(2023, 10, 10, 10, 0, 0, 0), "-", "2023-10-10-10:00:00.000")
    ])
    def test_fmt_naive_millisecond(self, name, obj, separator, expected):
        """ Test _fmt_naive_millisecond function with various datetime objects and separators.
        """
        result = _fmt_naive_millisecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveSecond(unittest.TestCase):
    """ Unit tests for the _fmt_naive_second function.
    """

    @parameterized.expand([
        ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0), "-", "2023-10-10-10:00:00"),
        ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0), "T", "2023-10-10T10:00:00")
    ])
    def test_fmt_naive_second(self, name, obj, separator, expected):
        """ Test _fmt_naive_second function with various datetime objects and separators.
        """
        result = _fmt_naive_second(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveMinute(unittest.TestCase):
    """ Unit tests for the _fmt_naive_minute function.
    """

    @parameterized.expand([
        ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0), "-", "2023-10-10-10:00"),
        ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0), "T", "2023-10-10T10:00")
    ])
    def test_fmt_naive_minute(self, name, obj, separator, expected):
        """ Test _fmt_naive_minute function with various datetime objects and separators.
        """
        result = _fmt_naive_minute(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveHour(unittest.TestCase):
    """ Unit tests for the _fmt_naive_hour function.
    """

    @parameterized.expand([
        ("standard datetime with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0), "-", "2023-10-10-10"),
        ("standard datetime with T separator", dt.datetime(2023, 10, 10, 10, 0, 0), "T", "2023-10-10T10")
    ])
    def test_fmt_naive_hour(self, name, obj, separator, expected):
        """ Test _fmt_naive_hour function with various datetime objects and separators.
        """
        result = _fmt_naive_hour(obj, separator)
        self.assertEqual(result, expected)


class TestFmtNaiveDay(unittest.TestCase):
    """ Unit tests for the _fmt_naive_day function.
    """

    def test_fmt_naive_day(self):
        """ Test _fmt_naive_day function with various datetime objects.
        """
        expected = "2023-10-10"
        result = _fmt_naive_day(dt.datetime(2023, 10, 10, 10, 0, 0))
        self.assertEqual(result, expected)


class TestFmtNaiveMonth(unittest.TestCase):
    """ Unit tests for the _fmt_naive_month function.
    """

    def test_fmt_naive_month(self):
        """ Test _fmt_naive_month function with various datetime objects.
        """
        expected = "2023-10"
        result = _fmt_naive_month(dt.datetime(2023, 10, 10, 10, 0, 0))
        self.assertEqual(result, expected)


class TestFmtNaiveYear(unittest.TestCase):
    """ Unit tests for the _fmt_naive_year function.
    """

    def test_fmt_naive_year(self):
        """ Test _fmt_naive_year function with various datetime objects.
        """
        expected = "2023"
        result = _fmt_naive_year(dt.datetime(2023, 10, 10, 10, 0, 0))
        self.assertEqual(result, expected)


class TestFmtTzdelta(unittest.TestCase):
    """ Unit tests for the _fmt_tzdelta function.
    """

    @parameterized.expand([
        ("zero delta", dt.timedelta(0), "Z"),
        ("positive delta", dt.timedelta(hours=5, minutes=30), "+05:30"),
        ("negative delta", dt.timedelta(hours=-5, minutes=-30), "-05:30"),
        ("positive delta with seconds", dt.timedelta(hours=1, minutes=45, seconds=30), "+01:45"),
        ("negative delta with seconds", dt.timedelta(hours=-1, minutes=-45, seconds=-30), "-01:45")
    ])
    def test_fmt_tzdelta(self, name, delta, expected):
        """ Test _fmt_tzdelta function with various timedelta objects.
        """
        result = _fmt_tzdelta(delta)
        self.assertEqual(result, expected)

    @parameterized.expand([
        ("delta exceeding one day", dt.timedelta(days=1, hours=1)),
        ("negative delta exceeding one day", dt.timedelta(days=-1, hours=-1))
    ])
    def test_fmt_tzdelta_invalid(self, name, delta):
        """ Test _fmt_tzdelta function with invalid timedelta objects.
        """
        with self.assertRaises(ValueError):
            _fmt_tzdelta(delta)


class TestFmtTzinfo(unittest.TestCase):
    """ Unit tests for the _fmt_tzinfo function.
    """

    @parameterized.expand([
        ("no tzinfo", None, ""),
        ("UTC tzinfo", dt.timezone.utc, "Z"),
        ("positive tzinfo", dt.timezone(dt.timedelta(hours=5, minutes=30)), "+05:30"),
        ("negative tzinfo", dt.timezone(dt.timedelta(hours=-5, minutes=-30)), "-05:30")
    ])
    def test_fmt_tzinfo(self, name, tz, expected):
        """ Test _fmt_tzinfo function with various tzinfo objects.
        """
        result = _fmt_tzinfo(tz)
        self.assertEqual(result, expected)


class TestFmtAwareMicrosecond(unittest.TestCase):
    """ Unit tests for the _fmt_aware_microsecond function.
    """

    @parameterized.expand([
        ("UTC timezone with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc), "-", "2023-10-10-10:00:00.123456Z"),
        ("UTC timezone with T separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc), "T", "2023-10-10T10:00:00.123456Z"),
        ("positive timezone", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))), "-", "2023-10-10-10:00:00.123456+05:30"),
        ("negative timezone", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))), "-", "2023-10-10-10:00:00.123456-05:30")
    ])
    def test_fmt_aware_microsecond(self, name, obj, separator, expected):
        """ Test _fmt_aware_microsecond function with various datetime objects and separators.
        """
        result = _fmt_aware_microsecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareMillisecond(unittest.TestCase):
    """ Unit tests for the _fmt_aware_millisecond function.
    """

    @parameterized.expand([
        ("UTC timezone with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc), "-", "2023-10-10-10:00:00.123Z"),
        ("UTC timezone with T separator", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone.utc), "T", "2023-10-10T10:00:00.123Z"),
        ("positive timezone", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))), "-", "2023-10-10-10:00:00.123+05:30"),
        ("negative timezone", dt.datetime(2023, 10, 10, 10, 0, 0, 123456, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))), "-", "2023-10-10-10:00:00.123-05:30")
    ])
    def test_fmt_aware_millisecond(self, name, obj, separator, expected):
        """ Test _fmt_aware_millisecond function with various datetime objects and separators.
        """
        result = _fmt_aware_millisecond(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareSecond(unittest.TestCase):
    """ Unit tests for the _fmt_aware_second function.
    """

    @parameterized.expand([
        ("UTC timezone with dash separator", dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone.utc), "-", "2023-10-10-10:00:00Z"),
        ("UTC timezone with T separator", dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone.utc), "T", "2023-10-10T10:00:00Z"),
        ("positive timezone", dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))), "-", "2023-10-10-10:00:00+05:30"),
        ("negative timezone", dt.datetime(2023, 10, 10, 10, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))), "-", "2023-10-10-10:00:00-05:30")
    ])
    def test_fmt_aware_second(self, name, obj, separator, expected):
        """ Test _fmt_aware_second function with various datetime objects and separators.
        """
        result = _fmt_aware_second(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareMinute(unittest.TestCase):
    """ Unit tests for the _fmt_aware_minute function.
    """

    @parameterized.expand([
        ("UTC timezone with dash separator", dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc), "-", "2023-10-10-10:00Z"),
        ("UTC timezone with T separator", dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone.utc), "T", "2023-10-10T10:00Z"),
        ("positive timezone", dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))), "-", "2023-10-10-10:00+05:30"),
        ("negative timezone", dt.datetime(2023, 10, 10, 10, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))), "-", "2023-10-10-10:00-05:30")
    ])
    def test_fmt_aware_minute(self, name, obj, separator, expected):
        """ Test _fmt_aware_minute function with various datetime objects and separators.
        """
        result = _fmt_aware_minute(obj, separator)
        self.assertEqual(result, expected)


class TestFmtAwareHour(unittest.TestCase):
    """ Unit tests for the _fmt_aware_hour function.
    """

    @parameterized.expand([
        ("UTC timezone with dash separator", dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone.utc), "-", "2023-10-10-10Z"),
        ("UTC timezone with T separator", dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone.utc), "T", "2023-10-10T10Z"),
        ("positive timezone", dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))), "-", "2023-10-10-10+05:30"),
        ("negative timezone", dt.datetime(2023, 10, 10, 10, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30))), "-", "2023-10-10-10-05:30")
    ])
    def test_fmt_aware_hour(self, name, obj, separator, expected):
        """ Test _fmt_aware_hour function with various datetime objects and separators.
        """
        result = _fmt_aware_hour(obj, separator)
        self.assertEqual(result, expected)


class TestOfYears(unittest.TestCase):
    """ Unit tests for the Properties.of_years method.
    """

    @parameterized.expand([
        ("one year", 1),
        ("two years", 2),
    ])
    def test_valid_years(self, name, no_of_years):
        """ Test Properties.of_years method with various year inputs.
        """
        expected = Properties(step=_STEP_MONTHS, multiplier=no_of_years*12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_years(no_of_years)
        self.assertEqual(result, expected)

    def test_zero_years(self):
        """ Test Properties.of_years method with various year inputs.
        """
        with self.assertRaises(AssertionError):
            Properties.of_years(0)


class TestOfMonths(unittest.TestCase):
    """ Unit tests for the Properties.of_months method.
    """

    @parameterized.expand([
        ("one month", 1),
        ("twelve months", 12),
    ])
    def test_valid_months(self, name, no_of_months):
        """ Test Properties.of_months method with various month inputs.
        """
        expected = Properties(step=_STEP_MONTHS, multiplier=no_of_months, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_months(no_of_months)
        self.assertEqual(result, expected)

    def test_zero_months(self):
        """ Test Properties.of_months method with zero month input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_months(0)


class TestOfDays(unittest.TestCase):
    """ Unit tests for the Properties.of_days method.
    """

    @parameterized.expand([
        ("one day", 1),
        ("seven days", 7),
    ])
    def test_valid_days(self, name, no_of_days):
        """ Test Properties.of_days method with various day inputs.
        """
        expected = Properties(step=_STEP_SECONDS, multiplier=no_of_days * 86_400, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_days(no_of_days)
        self.assertEqual(result, expected)

    def test_zero_days(self):
        """ Test Properties.of_days method with zero day input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_days(0)


class TestOfHours(unittest.TestCase):
    """ Unit tests for the Properties.of_hours method.
    """

    @parameterized.expand([
        ("one hour", 1),
        ("twenty-four hours", 24),
    ])
    def test_valid_hours(self, name, no_of_hours):
        """ Test Properties.of_hours method with various hour inputs.
        """
        expected = Properties(step=_STEP_SECONDS, multiplier=no_of_hours * 3_600, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_hours(no_of_hours)
        self.assertEqual(result, expected)

    def test_zero_hours(self):
        """ Test Properties.of_hours method with zero hour input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_hours(0)


class TestOfMinutes(unittest.TestCase):
    """ Unit tests for the Properties.of_minutes method.
    """

    @parameterized.expand([
        ("one minute", 1),
        ("sixty minutes", 60),
    ])
    def test_valid_minutes(self, name, no_of_minutes):
        """ Test Properties.of_minutes method with various minute inputs.
        """
        expected = Properties(step=_STEP_SECONDS, multiplier=no_of_minutes * 60, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_minutes(no_of_minutes)
        self.assertEqual(result, expected)

    def test_zero_minutes(self):
        """ Test Properties.of_minutes method with zero minute input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_minutes(0)


class TestOfSeconds(unittest.TestCase):
    """ Unit tests for the Properties.of_seconds method.
    """

    @parameterized.expand([
        ("one second", 1),
        ("sixty seconds", 60),
    ])
    def test_valid_seconds(self, name, no_of_seconds):
        """ Test Properties.of_seconds method with various second inputs.
        """
        expected = Properties(step=_STEP_SECONDS, multiplier=no_of_seconds, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_seconds(no_of_seconds)
        self.assertEqual(result, expected)

    def test_zero_seconds(self):
        """ Test Properties.of_seconds method with zero second input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_seconds(0)


class TestOfMicroseconds(unittest.TestCase):
    """ Unit tests for the Properties.of_microseconds method.
    """

    @parameterized.expand([
        ("one microsecond", 1),
        ("one million microseconds", 1_000_000),
    ])
    def test_valid_microseconds(self, name, no_of_microseconds):
        """ Test Properties.of_microseconds method with various microsecond inputs.
        """
        seconds, microseconds = divmod(no_of_microseconds, 1_000_000)
        if microseconds == 0:
            expected = Properties(step=_STEP_SECONDS, multiplier=seconds, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        else:
            expected = Properties(step=_STEP_MICROSECONDS, multiplier=no_of_microseconds, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_microseconds(no_of_microseconds)
        self.assertEqual(result, expected)

    def test_zero_microseconds(self):
        """ Test Properties.of_microseconds method with zero microsecond input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_microseconds(0)


class TestOfStepAndMultiplier(unittest.TestCase):
    """ Unit tests for the Properties.of_step_and_multiplier method.
    """

    @parameterized.expand([
        ("microseconds step", _STEP_MICROSECONDS, 1_000_000),
        ("seconds step", _STEP_SECONDS, 60),
        ("months step", _STEP_MONTHS, 12),
    ])
    def test_valid_step_and_multiplier(self, name, step, multiplier):
        """ Test Properties.of_step_and_multiplier method with various step and multiplier inputs.
        """
        expected = Properties(step=step, multiplier=multiplier, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = Properties.of_step_and_multiplier(step, multiplier)
        self.assertEqual(result, expected)

    def test_invalid_step(self):
        """ Test Properties.of_step_and_multiplier method with invalid step input.
        """
        with self.assertRaises(AssertionError):
            Properties.of_step_and_multiplier(999, 1)


class TestNormaliseOffsets(unittest.TestCase):
    """ Unit tests for the Properties.normalise_offsets method.
    """

    @parameterized.expand([
        ("normalised months", Properties(step=_STEP_MONTHS, multiplier=12, month_offset=13, microsecond_offset=0, tzinfo=None, ordinal_shift=0), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=1, microsecond_offset=0, tzinfo=None, ordinal_shift=0)),
        ("normalised seconds", Properties(step=_STEP_SECONDS, multiplier=60, month_offset=0, microsecond_offset=61_000_000, tzinfo=None, ordinal_shift=0), Properties(step=_STEP_SECONDS, multiplier=60, month_offset=0, microsecond_offset=1_000_000, tzinfo=None, ordinal_shift=0)),
        ("normalised microseconds", Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=1_000_001, tzinfo=None, ordinal_shift=0), Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=1, tzinfo=None, ordinal_shift=0)),
        ("already normalised", Properties(step=_STEP_MONTHS, multiplier=12, month_offset=1, microsecond_offset=0, tzinfo=None, ordinal_shift=0), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=1, microsecond_offset=0, tzinfo=None, ordinal_shift=0))
    ])
    def test_normalise_offsets(self, name, props, expected):
        """ Test Properties.normalise_offsets method with various inputs.
        """
        result = props.normalise_offsets()
        self.assertEqual(result, expected)


class TestWithMonthOffset(unittest.TestCase):
    """ Unit tests for the Properties.with_month_offset method.
    """

    @parameterized.expand([
        ("add one month", 1, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=1, microsecond_offset=0, tzinfo=None, ordinal_shift=0)),
        ("add thirteen months", 13, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=1, microsecond_offset=0, tzinfo=None, ordinal_shift=0)),
        ("no change", 0, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0))
    ])
    def test_with_month_offset(self, name, month_amount, expected):
        """ Test Properties.with_month_offset method with various month amounts.
        """
        props = Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = props.with_month_offset(month_amount)
        self.assertEqual(result, expected)

    def test_invalid_offset(self):
        """ Test Properties.of_step_and_multiplier method with invalid step input.
        """
        props = Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        with self.assertRaises(AssertionError):
            props.with_month_offset(-1)


class TestWithMicrosecondOffset(unittest.TestCase):
    """ Unit tests for the Properties.with_microsecond_offset method.
    """

    @parameterized.expand([
        ("add one microsecond", 1, Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=1, tzinfo=None, ordinal_shift=0)),
        ("add one million microseconds", 1_000_000, Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)),
        ("no change", 0, Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0))
    ])
    def test_with_microsecond_offset(self, name, microsecond_amount, expected):
        """ Test Properties.with_microsecond_offset method with various microsecond amounts.
        """
        props = Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = props.with_microsecond_offset(microsecond_amount)
        self.assertEqual(result, expected)

    def test_invalid_offset(self):
        """ Test Properties.with_microsecond_offset method with invalid microsecond input.
        """
        props = Properties(step=_STEP_MICROSECONDS, multiplier=1_000_000, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        with self.assertRaises(AssertionError):
            props.with_microsecond_offset(-1)


class TestWithTzinfo(unittest.TestCase):
    """ Unit tests for the Properties.with_tzinfo method.
    """

    @parameterized.expand([
        ("UTC timezone", dt.timezone.utc, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=dt.timezone.utc, ordinal_shift=0)),
        ("None timezone", None, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)),
        ("positive timezone", dt.timezone(dt.timedelta(hours=5, minutes=30)), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)), ordinal_shift=0)),
        ("negative timezone", dt.timezone(dt.timedelta(hours=-5, minutes=-30)), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=dt.timezone(dt.timedelta(hours=-5, minutes=-30)), ordinal_shift=0))
    ])
    def test_with_tzinfo(self, name, tzinfo, expected):
        """ Test Properties.with_tzinfo method with various tzinfo objects.
        """
        props = Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = props.with_tzinfo(tzinfo)
        self.assertEqual(result, expected)


class TestWithOrdinalShift(unittest.TestCase):
    """ Unit tests for the Properties.with_ordinal_shift method.
    """

    @parameterized.expand([
        ("positive shift", 1, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=1)),
        ("negative shift", -1, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=-1)),
        ("zero shift", 0, Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0))
    ])
    def test_with_ordinal_shift(self, name, ordinal_shift, expected):
        """ Test Properties.with_ordinal_shift method with various ordinal shift values.
        """
        props = Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = props.with_ordinal_shift(ordinal_shift)
        self.assertEqual(result, expected)


class TestWithOffsetPeriodFields(unittest.TestCase):
    """ Unit tests for the Properties.with_offset_period_fields method.
    """

    @parameterized.expand([
        ("offset period fields with months and seconds", (6, 1_000_000), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=6, microsecond_offset=1_000_000, tzinfo=None, ordinal_shift=0)),
        ("offset period fields with normalised months and seconds", (12, 1_000_000), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=1_000_000, tzinfo=None, ordinal_shift=0)),
        ("offset period fields with zero months and seconds", (0, 0), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0))
    ])
    @patch('time_series.period.PeriodFields')
    def test_with_offset_period_fields(self, name, months_seconds, expected, mock_period_fields):
        """ Test Properties.with_offset_period_fields method with various PeriodFields inputs.
        """
        mock_months_seconds = Mock()
        mock_months_seconds.months = months_seconds[0]
        mock_months_seconds.total_microseconds.return_value = months_seconds[1]
        mock_period_fields.get_months_seconds.return_value = mock_months_seconds

        props = Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = props.with_offset_period_fields(mock_period_fields)
        self.assertEqual(result, expected)


class TestWithOffsetMonthsSeconds(unittest.TestCase):
    """ Unit tests for the Properties.with_offset_months_seconds method.
    """

    @parameterized.expand([
        ("offset period fields with months and seconds", (6, 1_000_000), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=6, microsecond_offset=1_000_000, tzinfo=None, ordinal_shift=0)),
        ("offset period fields with normalised months and seconds", (12, 1_000_000), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=1_000_000, tzinfo=None, ordinal_shift=0)),
        ("offset period fields with zero months and seconds", (0, 0), Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0))
    ])
    @patch('time_series.period.MonthsSeconds')
    def test_with_offset_months_seconds(self, name, months_seconds, expected, mock_months_seconds):
        """ Test Properties.with_offset_months_seconds method with various MonthsSeconds inputs.
        """
        mock_months_seconds = Mock()
        mock_months_seconds.months = months_seconds[0]
        mock_months_seconds.total_microseconds.return_value = months_seconds[1]

        props = Properties(step=_STEP_MONTHS, multiplier=12, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0)
        result = props.with_offset_months_seconds(mock_months_seconds)
        self.assertEqual(result, expected)





if __name__ == '__main__':
    unittest.main()