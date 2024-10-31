"""
Unit tests for the period module
"""

from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
)

import collections
import datetime
import unittest
import zoneinfo

from typing import (
    Any,
    Optional,
)
from parameterized import parameterized
import re

import period as p
from period import Period


TZ_UTC = datetime.timezone(datetime.timedelta(hours=0))


# ----------------------------------------------------------------------
_GOOD_ISO_DURATION = set(
    [
        "p0.00001s",
        "p0.0001s",
        "p0.001s",
        "p0.01s",
        "p0.1s",
        "p1000d",
        "p100d",
        "p10d",
        "p1d",
        "p1d1h1m",
        "p1d1h1m1s",
        "p1d1h1s",
        "p1d1m",
        "p1d1s",
        "p1h1m",
        "p1h1m1s",
        "p1h1s",
        "p1m",
        "p1s",
        "p1y",
        "p1y1m",
        "p9m",
        "p9y",
    ]
)

_BAD_ISO_DURATION = set(
    [
        "",
        "1",
        "1y",
        "99999",
        "p-1y",
        "p0.0000001s",
        "p0.0s",
        "p0.5y",
        "p0s",
        "p0y",
        "p1",
        "p1d1y",
        "p1h1y",
        "p1ht1s",
        "p1ht1y",
        "p1m1m",
        "p1m1s",
        "p1mt1m",
        "p1t2",
        "p1y 1m",
        "p1y!",
        "p1y0",
        "p1y0.1s",
        "p1y1d",
        "p1y1h",
        "p1y1m1d",
        "p1y1m1d1h1m1s",
        "p1y1s",
        "p1y1y",
        "p1yt1m",
        "pp1y",
        "pt",
        "pt1y",
        "wibblesticks",
    ]
)

_GOOD_DURATION = set(
    [
        "p1y+1d",
        "p1y+1h",
        "p1y+1m",
        "p1y+1s",
        "p1y+9y",
        "p1y+t1s",
        "p9y+1y",
        "p9y9m+1y1m1d1h1m1s",
    ]
)

_BAD_DURATION = set(
    [
        "",
        "1+1m",
        "p1y +1m",
        "p1y+",
        "p1y+1",
        "p1y+1d+1h",
        "p1y+m",
        "p1y+m2h",
    ]
    + [f"{bad_iso}+0.001s" for bad_iso in _BAD_ISO_DURATION]
)

_GOOD_DATE = set(
    [
        "1066",
        "1883",
        "1984",
        "2050",
        "1066-01",
        "1883-01",
        "1984-01",
        "2050-01",
        "1984-01-01",
        "1984-01-02 00",
        "1984-01-03 23",
        "1984-01-04T00",
        "1984-01-05T23",
        "1984-01-06 00:00",
        "1984-01-07 23:59",
        "1984-01-08T00:00",
        "1984-02-09T23:59",
        "1984-03-10 00:00:00",
        "1984-04-11 23:59:59",
        "1984-05-12T00:00:00",
        "1984-06-13T23:59:59",
        "1984-07-14T01:01:51.1",
        "1984-08-15T01:13:49.23",
        "1984-09-16T01:25:37.456",
        "1984-10-17T01:37:25.7809",
        "1984-11-18T01:49:13.12345",
        "1984-12-19T01:51:01.109876",
    ]
)

_BAD_DATE = set(
    [
        "",
        "0",
        "12345",
        "99999",
        "1984-00",
        "1984-13",
        "198401",
        "19840101",
        "01/1984",
        "01/01/1984",
        "1984-13-01T00:00:00",
        "1985-02-29T00:00:00",
        "1984-01-32T00:00:00",
        "1984-01-01T24:00:00",
        "1984-01-01T00:60:00",
        "1984-01-01T00:00:60",
        "1984-01-01T01:01:01.1111111",
        "1984-01-01_34:56:78",
    ]
)

_GOOD_DATE_AND_DURATION = set(
    [f"{d}/{p}" for d in _GOOD_DATE for p in _GOOD_ISO_DURATION]
)

_BAD_DATE_AND_DURATION = set(
    [""]
    + [f"{d}/{p}" for d in _BAD_DATE for p in _BAD_ISO_DURATION]
    + [f"{d}/{p}" for d in _GOOD_DATE for p in _BAD_ISO_DURATION]
    + [f"{d}/{p}" for d in _BAD_DATE for p in _GOOD_ISO_DURATION]
    + [f"{d}{p}" for d in _GOOD_DATE for p in _GOOD_ISO_DURATION]
    + [f"{d}//{p}" for d in _GOOD_DATE for p in _GOOD_ISO_DURATION]
    + []
)

_GOOD_REPR = set(
    [f"{p}[]0" for p in _GOOD_ISO_DURATION]
    + [f"{p}[]-1" for p in _GOOD_ISO_DURATION]
    + [f"{p}[]1" for p in _GOOD_ISO_DURATION]
    + [f"{p}[+00:00]0" for p in _GOOD_ISO_DURATION]
    + [f"{p}[-00:00]-1" for p in _GOOD_ISO_DURATION]
    + [f"{p}[+00:00]1" for p in _GOOD_ISO_DURATION]
    + [f"{p}[]" for p in _GOOD_ISO_DURATION]
    + []
)

_BAD_REPR = set(
    [""]
    + [f"{p}" for p in _GOOD_ISO_DURATION]
    + [f"{p}[]+1" for p in _GOOD_ISO_DURATION]
    + [f"{p}[+00:00]+1" for p in _GOOD_ISO_DURATION]
    + [f"{p}[00:00]0" for p in _GOOD_ISO_DURATION]
    + []
)


class TestPeriodOf(unittest.TestCase):
    """Test Period.of"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_good(self, text: Any) -> None:
        """Test Period.of against a set of known good arguments"""
        period: Period = Period.of(text)
        self.assertIsInstance(period, Period)

    @parameterized.expand(sorted(_BAD_ISO_DURATION | _BAD_DURATION))
    def test_bad(self, text: Any) -> None:
        """Test Period.of against a set of known bad arguments"""
        with self.assertRaises(ValueError):
            Period.of(text)


# ----------------------------------------------------------------------
class TestPeriodOfIsoDuration(unittest.TestCase):
    """Test Period.of_iso_duration static method"""

    @parameterized.expand(sorted(_GOOD_ISO_DURATION))
    def test_good(self, text: Any) -> None:
        """Test Period.of_iso_duration against a set of known good arguments"""
        period: Period = Period.of_iso_duration(text)
        self.assertIsInstance(period, Period)

    @parameterized.expand(sorted(_BAD_ISO_DURATION))
    def test_bad(self, text: Any) -> None:
        """Test Period.of_iso_duration against a set of known bad arguments"""
        with self.assertRaises(ValueError):
            Period.of_iso_duration(text)


# ----------------------------------------------------------------------
class TestPeriodOfDuration(unittest.TestCase):
    """Test Period.of_duration static method"""

    @parameterized.expand(sorted(_GOOD_ISO_DURATION | _GOOD_DURATION))
    def test_good(self, text: Any) -> None:
        """Test Period.of_duration against a set of known good arguments"""
        period: Period = Period.of_duration(text)
        self.assertIsInstance(period, Period)

    @parameterized.expand(sorted(_BAD_ISO_DURATION | _BAD_DURATION))
    def test_bad(self, text: Any) -> None:
        """Test Period.of_duration against a set of known bad arguments"""
        with self.assertRaises(ValueError):
            Period.of_duration(text)

    @parameterized.expand(
        [
            ("P1Y", set(["P12M"])),
            ("P2Y", set(["P24M"])),
            ("P1D", set(["P24H", "PT1440M", "P86400S"])),
            ("P2D", set(["P48H", "PT2880M", "P172800S"])),
            ("P1H", set(["PT60M", "P3600S"])),
            ("PT1M", set(["P60S"])),
        ]
    )
    def test_equivalent(self, name: Any, other_names: set[str]) -> None:
        """Test Period.with_month_offset method"""
        period: Period = Period.of_duration(name)
        for other_name in sorted(other_names):
            other: Period = Period.of_duration(other_name)
            self.assertEqual(period, other)


class TestPeriodOfDateAndDuration(unittest.TestCase):
    """Test Period.of_date_and_duration static method"""

    @parameterized.expand(sorted(_GOOD_DATE_AND_DURATION))
    def test_good(self, text: Any) -> None:
        """Test Period.of_date_and_duration against a set of known good arguments"""
        period: Period = Period.of_date_and_duration(text)
        self.assertIsInstance(period, Period)

    @parameterized.expand(sorted(_BAD_DATE_AND_DURATION))
    def test_bad(self, text: Any) -> None:
        """Test Period.of_date_and_duration against a set of known bad arguments"""
        with self.assertRaises(ValueError):
            Period.of_date_and_duration(text)


class TestPeriodOfRepr(unittest.TestCase):
    """Test Period.of_repr static method"""

    @parameterized.expand(sorted(_GOOD_REPR))
    def test_good(self, text: Any) -> None:
        """Test Period.of_repr against a set of known good arguments"""
        period: Period = Period.of_repr(text)
        self.assertIsInstance(period, Period)

    @parameterized.expand(sorted(_BAD_REPR))
    def test_bad(self, text: Any) -> None:
        """Test Period.of_repr against a set of known bad arguments"""
        with self.assertRaises(ValueError):
            Period.of_repr(text)


class TestPeriodOfYears(unittest.TestCase):
    """Test Period.of_years static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, years: Any) -> None:
        """Test Period.of_years against a set of known good years values"""
        period: Period = Period.of_years(years)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-999, -1, 0])
    def test_bad(self, years: Any) -> None:
        """Test Period.of_years against a set of known bad years values"""
        with self.assertRaises(AssertionError):
            Period.of_years(years)

    @parameterized.expand(
        [
            ("P1Y", 1),
            ("P2Y", 2),
            ("P3Y", 3),
            ("P4Y", 4),
            ("P5Y", 5),
            ("P10Y", 10),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_years: int) -> None:
        """Test Period.of_years method"""
        period: Period = Period.of_iso_duration(name)
        year_period: Period = Period.of_years(no_of_years)
        self.assertEqual(period, year_period)


class TestPeriodOfMonths(unittest.TestCase):
    """Test Period.of_months static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, months: Any) -> None:
        """Test Period.of_months against a set of known good months values"""
        period: Period = Period.of_months(months)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-1, 0])
    def test_bad(self, months: Any) -> None:
        """Test Period.of_months against a set of known bad months values"""
        with self.assertRaises(AssertionError):
            Period.of_months(months)

    @parameterized.expand(
        [
            ("P1M", 1),
            ("P2M", 2),
            ("P3M", 3),
            ("P4M", 4),
            ("P5M", 5),
            ("P6M", 6),
            ("P1Y", 12),
            ("P2Y", 24),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_months: int) -> None:
        """Test Period.of_months method"""
        period: Period = Period.of_iso_duration(name)
        month_period: Period = Period.of_months(no_of_months)
        self.assertEqual(period, month_period)


class TestPeriodOfDays(unittest.TestCase):
    """Test Period.of_days static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, days: Any) -> None:
        """Test Period.of_days against a set of known good days values"""
        period: Period = Period.of_days(days)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-1, 0])
    def test_bad(self, days: Any) -> None:
        """Test Period.of_days against a set of known bad days values"""
        with self.assertRaises(AssertionError):
            Period.of_days(days)

    @parameterized.expand(
        [
            ("P1D", 1),
            ("P2D", 2),
            ("P3D", 3),
            ("P4D", 4),
            ("P5D", 5),
            ("P6D", 6),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_days: int) -> None:
        """Test Period.of_days method"""
        period: Period = Period.of_iso_duration(name)
        day_period: Period = Period.of_days(no_of_days)
        self.assertEqual(period, day_period)


class TestPeriodOfHours(unittest.TestCase):
    """Test Period.of_hours static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, hours: Any) -> None:
        """Test Period.of_hours against a set of known good hours values"""
        period: Period = Period.of_hours(hours)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-1, 0])
    def test_bad(self, hours: Any) -> None:
        """Test Period.of_hours against a set of known bad hours values"""
        with self.assertRaises(AssertionError):
            Period.of_hours(hours)

    @parameterized.expand(
        [
            ("P1H", 1),
            ("P2H", 2),
            ("P3H", 3),
            ("P4H", 4),
            ("P5H", 5),
            ("P6H", 6),
            ("P1D", 24),
            ("P2D", 48),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_hours: int) -> None:
        """Test Period.of_hours method"""
        period: Period = Period.of_iso_duration(name)
        hour_period: Period = Period.of_hours(no_of_hours)
        self.assertEqual(period, hour_period)


class TestPeriodOfMinutes(unittest.TestCase):
    """Test Period.of_minutes static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, minutes: Any) -> None:
        """Test Period.of_minutes against a set of known good minutes values"""
        period: Period = Period.of_minutes(minutes)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-1, 0])
    def test_bad(self, minutes: Any) -> None:
        """Test Period.of_minutes against a set of known bad minutes values"""
        with self.assertRaises(AssertionError):
            Period.of_minutes(minutes)

    @parameterized.expand(
        [
            ("PT1M", 1),
            ("PT2M", 2),
            ("PT3M", 3),
            ("PT4M", 4),
            ("PT5M", 5),
            ("PT6M", 6),
            ("P1H", 60),
            ("P1D", 1_440),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_minutes: int) -> None:
        """Test Period.of_minutes method"""
        period: Period = Period.of_iso_duration(name)
        minute_period: Period = Period.of_minutes(no_of_minutes)
        self.assertEqual(period, minute_period)


class TestPeriodOfSeconds(unittest.TestCase):
    """Test Period.of_seconds static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, seconds: Any) -> None:
        """Test Period.of_seconds against a set of known good seconds values"""
        period: Period = Period.of_seconds(seconds)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-1, 0])
    def test_bad(self, seconds: Any) -> None:
        """Test Period.of_seconds against a set of known bad seconds values"""
        with self.assertRaises(AssertionError):
            Period.of_seconds(seconds)

    @parameterized.expand(
        [
            ("P1S", 1),
            ("P2S", 2),
            ("P3S", 3),
            ("P4S", 4),
            ("P5S", 5),
            ("P6S", 6),
            ("PT1M", 60),
            ("P1H", 3_600),
            ("P1D", 86_400),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_seconds: int) -> None:
        """Test Period.of_seconds method"""
        period: Period = Period.of_iso_duration(name)
        second_period: Period = Period.of_seconds(no_of_seconds)
        self.assertEqual(period, second_period)


class TestPeriodOfMicroseconds(unittest.TestCase):
    """Test Period.of_microseconds static method"""

    @parameterized.expand([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100])
    def test_good(self, microseconds: Any) -> None:
        """Test Period.of_microseconds against a set of known good microseconds values"""
        period: Period = Period.of_microseconds(microseconds)
        self.assertIsInstance(period, Period)

    @parameterized.expand([-1, 0])
    def test_bad(self, microseconds: Any) -> None:
        """Test Period.of_microseconds against a set of known bad microseconds values"""
        with self.assertRaises(AssertionError):
            Period.of_microseconds(microseconds)

    @parameterized.expand(
        [
            ("P0.000001S", 1),
            ("P0.00001S", 10),
            ("P0.0001S", 100),
            ("P0.001S", 1_000),
            ("P0.01S", 10_000),
            ("P0.1S", 100_000),
            ("P1S", 1_000_000),
            ("PT1M", 60_000_000),
            ("P1H", 3_600_000_000),
            ("P1D", 86_400_000_000),
        ]
    )
    def test_iso_duration(self, name: Any, no_of_microseconds: int) -> None:
        """Test Period.of_microseconds method"""
        period: Period = Period.of_iso_duration(name)
        microsecond_period: Period = Period.of_microseconds(no_of_microseconds)
        self.assertEqual(period, microsecond_period)


class TestPeriodOfTimedelta(unittest.TestCase):
    """Test Period.of_timedelta static method"""

    @parameterized.expand(
        [
            datetime.timedelta(days=1),
            datetime.timedelta(hours=1),
            datetime.timedelta(minutes=1),
            datetime.timedelta(microseconds=1),
        ]
    )
    def test_good(self, timedelta: Any) -> None:
        """Test Period.of_timedelta against a set of known good timedelta values"""
        period: Period = Period.of_timedelta(timedelta)
        self.assertIsInstance(period, Period)

    @parameterized.expand(
        [
            datetime.timedelta(days=0),
            datetime.timedelta(days=-1),
            datetime.timedelta(microseconds=0),
            datetime.timedelta(microseconds=-1),
        ]
    )
    def test_bad(self, timedelta: Any) -> None:
        """Test Period.of_timedelta against a set of known bad timedelta values"""
        with self.assertRaises(AssertionError):
            Period.of_timedelta(timedelta)

    @parameterized.expand(
        [
            ("P1D", datetime.timedelta(days=1)),
            ("P1H", datetime.timedelta(hours=1)),
            ("PT1M", datetime.timedelta(minutes=1)),
            ("P1S", datetime.timedelta(seconds=1)),
            ("P0.000001S", datetime.timedelta(microseconds=1)),
            ("P2D", datetime.timedelta(days=2)),
            ("P2H", datetime.timedelta(hours=2)),
            ("PT2M", datetime.timedelta(minutes=2)),
            ("P2S", datetime.timedelta(seconds=2)),
            ("P0.000002S", datetime.timedelta(microseconds=2)),
        ]
    )
    def test_iso_duration(self, name: Any, delta: datetime.timedelta) -> None:
        """Test Period.of_timedelta method"""
        period: Period = Period.of_iso_duration(name)
        delta_period: Period = Period.of_timedelta(delta)
        self.assertEqual(period, delta_period)


# After applying this with the str.translate() method
# a valid period string should end up empty.
_PERIOD_CHARS_TRANSLATE = str.maketrans("", "", "0123456789PYMDTHS.")


class TestPeriodIsoDuration(unittest.TestCase):
    """Test Period.iso_duration property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.iso_duration property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        iso_duration: str = period.iso_duration
        self.assertIsInstance(iso_duration, str)
        self.assertTrue(iso_duration == iso_duration.upper())
        self.assertTrue(iso_duration.startswith("P"))
        self.assertEqual(iso_duration.translate(_PERIOD_CHARS_TRANSLATE), "")


class TestPeriodTzinfo(unittest.TestCase):
    """Test Period.tzinfo property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.iso_duration property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        period_no_tz: Period = period.with_tzinfo(None)
        self.assertEqual(period_no_tz.tzinfo, None)
        period_utc: Period = period.with_tzinfo(TZ_UTC)
        self.assertEqual(period_utc.tzinfo, TZ_UTC)
        period_no_tz2: Period = period_utc.with_tzinfo(None)
        self.assertEqual(period_no_tz2.tzinfo, None)


class TestPeriodMinOrdinal(unittest.TestCase):
    """Test Period.min_ordinal property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.min_ordinal property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        min_ordinal: int = period.min_ordinal
        self.assertIsInstance(min_ordinal, int)
        self.assertIsInstance(period.datetime(min_ordinal), datetime.datetime)


class TestPeriodMaxOrdinal(unittest.TestCase):
    """Test Period.max_ordinal property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.max_ordinal property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        max_ordinal: int = period.max_ordinal
        self.assertIsInstance(max_ordinal, int)
        self.assertIsInstance(period.datetime(max_ordinal), datetime.datetime)


class TestPeriodTimedelta(unittest.TestCase):
    """Test Period.timedelta property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.timedelta property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        delta: Optional[datetime.timedelta] = period.timedelta
        if delta is None:
            self.assertEqual(period._properties.step, p._STEP_MONTHS)
        else:
            min_ordinal: int = period.min_ordinal
            d0: datetime.datetime = period.datetime(min_ordinal)
            d1: datetime.datetime = period.datetime(min_ordinal + 1)
            d_delta = d1 - d0
            self.assertEqual(delta, d_delta)


# After applying this with the str.translate() method
# a valid Polars interval/offset string should end up empty.
_PL_INTERVAL_CHARS_TRANSLATE = str.maketrans("", "", "0123456789mous")


class TestPeriodPlInterval(unittest.TestCase):
    """Test Period.pl_interval property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.pl_interval property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        pl_interval: str = period.pl_interval
        self.assertIsInstance(pl_interval, str)
        self.assertTrue(pl_interval == pl_interval.lower())
        self.assertEqual(pl_interval.translate(_PL_INTERVAL_CHARS_TRANSLATE), "")


class TestPeriodPlOffset(unittest.TestCase):
    """Test Period.pl_offset property"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.pl_offset property is valid for a set of Period instances"""
        period: Period = Period.of(text)
        pl_offset: str = period.pl_offset
        self.assertIsInstance(pl_offset, str)
        self.assertTrue(pl_offset == pl_offset.lower())
        self.assertEqual(pl_offset.translate(_PL_INTERVAL_CHARS_TRANSLATE), "")


class TestPeriodIsEpochAgnostic(unittest.TestCase):
    """Test Period.is_epoch_agnostic method"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.is_epoch_agnostic method is valid for a set of Period instances"""
        period: Period = Period.of(text)
        epoch_agnostic: bool = period.is_epoch_agnostic()
        self.assertIsInstance(epoch_agnostic, bool)

    @parameterized.expand(
        [
            ("P1Y", True),
            ("P2Y", False),
            ("P1M", True),
            ("P2M", True),
            ("P3M", True),
            ("P4M", True),
            ("P5M", False),
            ("P6M", True),
            ("P7M", False),
            ("P1D", True),
            ("P2D", False),
            ("P3D", False),
            ("P4D", False),
            ("P5D", False),
            ("P6D", False),
            ("P7D", False),
            ("P1H", True),
            ("P2H", True),
            ("P3H", True),
            ("P4H", True),
            ("P5H", False),
            ("P6H", True),
            ("P7H", False),
            ("P8H", True),
            ("P9H", False),
            ("P10H", False),
            ("P11H", False),
            ("P12H", True),
            ("P13H", False),
            ("PT1M", True),
            ("PT2M", True),
            ("PT3M", True),
            ("PT4M", True),
            ("PT5M", True),
            ("PT6M", True),
            ("PT7M", False),
            ("PT8M", True),
            ("PT9M", True),
            ("PT10M", True),
            ("PT11M", False),
            ("PT12M", True),
            ("PT13M", False),
            ("PT14M", False),
            ("PT15M", True),
            ("P1S", True),
            ("P2S", True),
            ("P3S", True),
            ("P4S", True),
            ("P5S", True),
            ("P6S", True),
            ("P7S", False),
            ("P8S", True),
            ("P9S", True),
            ("P10S", True),
            ("P11S", False),
            ("P12S", True),
            ("P13S", False),
            ("P14S", False),
            ("P15S", True),
            ("P0.001S", True),
            ("P0.002S", True),
            ("P0.003S", True),
            ("P0.004S", True),
            ("P0.005S", True),
            ("P0.006S", True),
            ("P0.007S", False),
            ("P0.000001S", True),
            ("P0.000002S", True),
            ("P0.000003S", True),
            ("P0.000004S", True),
            ("P0.000005S", True),
            ("P0.000006S", True),
            ("P0.000007S", False),
            ("P0.000008S", True),
            ("P0.000009S", True),
            ("P0.000010S", True),
            ("P0.000011S", False),
            ("P0.000012S", True),
            ("P0.000013S", False),
            ("P0.000014S", False),
            ("P0.000015S", True),
            ("P0.000016S", True),
            ("P0.000017S", False),
            ("P0.000018S", True),
            ("P0.000019S", False),
            ("P0.000020S", True),
        ]
    )
    def test_iso_durations(self, name: Any, expected: bool) -> None:
        """Test Period.is_epoch_agnostic method"""
        period: Period = Period.of_iso_duration(name)
        self.assertEqual(period.is_epoch_agnostic(), expected)


class TestPeriodNaiveFormatter(unittest.TestCase):
    """Test Period.naive_formatter method"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.naive_formatter method is valid for a set of Period instances"""
        period: Period = Period.of(text)
        naive_formatter: Callable[[dt.datetime], str] = period.naive_formatter()
        self.assertTrue(callable(naive_formatter))

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_good(self, text: Any) -> None:
        """Test Period.naive_formatter method with good arguments"""
        period: Period = Period.of(text)
        period.naive_formatter(" ")
        period.naive_formatter("T")
        period.naive_formatter("t")

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_bad(self, text: Any) -> None:
        """Test Period.naive_formatter method with bad arguments"""
        period: Period = Period.of(text)
        with self.assertRaises(ValueError):
            period.naive_formatter("")
        with self.assertRaises(ValueError):
            period.naive_formatter("x")
        with self.assertRaises(ValueError):
            period.naive_formatter("  ")
        with self.assertRaises(ValueError):
            period.naive_formatter("t ")

    @parameterized.expand(
        [
            ("P1Y", " ", datetime.datetime(2023, 10, 20, 1, 2, 3, 456789), "2023"),
            ("P1Y", "T", datetime.datetime(2023, 10, 20, 1, 2, 3, 456789), "2023"),
            ("P5Y", " ", datetime.datetime(2023, 10, 20, 1, 2, 3, 456789), "2023"),
            ("P1M", " ", datetime.datetime(2023, 10, 20, 1, 2, 3, 456789), "2023-10"),
            (
                "P1D",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20",
            ),
            (
                "P1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1H",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01",
            ),
            (
                "PT1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "PT1M",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02",
            ),
            (
                "P1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1S",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03",
            ),
            (
                "P0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P0.001S",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03.456",
            ),
            (
                "P0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P0.000001S",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03.456789",
            ),
        ]
    )
    def test_iso_durations(
        self, name: str, sep: str, dt: datetime.datetime, expected: str
    ) -> None:
        """Test Period.naive_formatter"""
        period: Period = Period.of_iso_duration(name)
        fmt = period.naive_formatter(sep)
        result = fmt(dt)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            (
                "P1Y+1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10",
            ),
            (
                "P1Y+1D",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20",
            ),
            (
                "P1Y+1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1Y+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1Y+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1Y+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1Y+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1M+1D",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20",
            ),
            (
                "P1M+1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1M+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1M+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1M+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1M+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1D+1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1D+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1D+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1D+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1D+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1H+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1H+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1H+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1H+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "PT1M+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "PT1M+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "PT1M+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1S+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1S+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
        ]
    )
    def test_offset_durations(
        self, name: str, sep: str, dt: datetime.datetime, expected: str
    ) -> None:
        """Test Period.naive_formatter"""
        period: Period = Period.of_duration(name)
        fmt = period.naive_formatter(sep)
        result = fmt(dt)
        self.assertEqual(result, expected)


class TestPeriodAwareFormatter(unittest.TestCase):
    """Test Period.aware_formatter method"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.aware_formatter method is valid for a set of Period instances"""
        period: Period = Period.of(text)
        aware_formatter: Callable[[dt.datetime], str] = period.aware_formatter()
        self.assertTrue(callable(aware_formatter))

    @parameterized.expand(
        [
            (
                "P1Y",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01Z",
            ),
            (
                "P5Y",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1D",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1H",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01Z",
            ),
            (
                "PT1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "PT1M",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02Z",
            ),
            (
                "P1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1S",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03Z",
            ),
            (
                "P0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P0.001S",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03.456Z",
            ),
            (
                "P0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P0.000001S",
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03.456789Z",
            ),
        ]
    )
    def test_iso_durations_utc(
        self, name: str, sep: str, dt: datetime.datetime, expected: str
    ) -> None:
        """Test Period.aware_formatter"""
        period: Period = Period.of_iso_duration(name)
        fmt = period.aware_formatter(sep)
        result = fmt(dt)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            (
                "P1Y+1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y+1D",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y+1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1Y+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1Y+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1Y+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1M+1D",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1M+1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1M+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1M+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1M+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1M+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1D+1H",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1D+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1D+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1D+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1D+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1H+T1M",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1H+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1H+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1H+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "PT1M+1S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "PT1M+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "PT1M+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1S+0.001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1S+0.000001S",
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
        ]
    )
    def test_offset_durations_utc(
        self, name: str, sep: str, dt: datetime.datetime, expected: str
    ) -> None:
        """Test Period.aware_formatter"""
        period: Period = Period.of_duration(name)
        fmt = period.aware_formatter(sep)
        result = fmt(dt)
        self.assertEqual(result, expected)


class TestPeriodFormatter(unittest.TestCase):
    """Test Period.formatter method"""

    @parameterized.expand(
        sorted(
            _GOOD_ISO_DURATION | _GOOD_DURATION | _GOOD_DATE_AND_DURATION | _GOOD_REPR
        )
    )
    def test_is_valid(self, text: Any) -> None:
        """Test Period.formatter method is valid for a set of Period instances"""
        period: Period = Period.of(text)
        formatter: Callable[[dt.datetime], str] = period.formatter()
        self.assertTrue(callable(formatter))

    @parameterized.expand(
        [
            (
                "P1Y",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023",
            ),
            (
                "P1Y",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023",
            ),
            (
                "P5Y",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023",
            ),
            (
                "P1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10",
            ),
            (
                "P1D",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20",
            ),
            (
                "P1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1H",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01",
            ),
            (
                "PT1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "PT1M",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02",
            ),
            (
                "P1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1S",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03",
            ),
            (
                "P0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P0.001S",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03.456",
            ),
            (
                "P0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P0.000001S",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03.456789",
            ),
            (
                "P1Y",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023",
            ),
            (
                "P1Y",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023",
            ),
            (
                "P5Y",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023",
            ),
            (
                "P1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10",
            ),
            (
                "P1D",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20",
            ),
            (
                "P1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01",
            ),
            (
                "P1H",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01",
            ),
            (
                "PT1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02",
            ),
            (
                "PT1M",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02",
            ),
            (
                "P1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03",
            ),
            (
                "P1S",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03",
            ),
            (
                "P0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P0.001S",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03.456",
            ),
            (
                "P0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P0.000001S",
                None,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03.456789",
            ),
            (
                "P1Y",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1Y",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01",
            ),
            (
                "P5Y",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1D",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1H",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01",
            ),
            (
                "PT1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "PT1M",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02",
            ),
            (
                "P1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1S",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03",
            ),
            (
                "P0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P0.001S",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03.456",
            ),
            (
                "P0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P0.000001S",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20T01:02:03.456789",
            ),
            (
                "P1Y",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01Z",
            ),
            (
                "P5Y",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1D",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1H",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01Z",
            ),
            (
                "PT1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "PT1M",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02Z",
            ),
            (
                "P1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1S",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03Z",
            ),
            (
                "P0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P0.001S",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03.456Z",
            ),
            (
                "P0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P0.000001S",
                TZ_UTC,
                "T",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20T01:02:03.456789Z",
            ),
        ]
    )
    def test_iso_durations_utc(
        self,
        name: str,
        tz: datetime.tzinfo,
        sep: str,
        dt: datetime.datetime,
        expected: str,
    ) -> None:
        """Test Period.aware_formatter arguments"""
        period: Period = Period.of_iso_duration(name).with_tzinfo(tz)
        fmt = period.formatter(sep)
        result = fmt(dt)
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            (
                "P1Y+1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10",
            ),
            (
                "P1Y+1D",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20",
            ),
            (
                "P1Y+1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1Y+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1Y+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1Y+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1Y+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1M+1D",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20",
            ),
            (
                "P1M+1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1M+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1M+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1M+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1M+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1D+1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1D+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1D+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1D+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1D+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1H+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1H+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1H+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1H+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "PT1M+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "PT1M+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "PT1M+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1S+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1S+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1Y+1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10",
            ),
            (
                "P1Y+1D",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20",
            ),
            (
                "P1Y+1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01",
            ),
            (
                "P1Y+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02",
            ),
            (
                "P1Y+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03",
            ),
            (
                "P1Y+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1Y+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1M+1D",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20",
            ),
            (
                "P1M+1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01",
            ),
            (
                "P1M+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02",
            ),
            (
                "P1M+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03",
            ),
            (
                "P1M+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1M+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1D+1H",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01",
            ),
            (
                "P1D+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02",
            ),
            (
                "P1D+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03",
            ),
            (
                "P1D+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1D+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1H+T1M",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02",
            ),
            (
                "P1H+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03",
            ),
            (
                "P1H+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1H+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "PT1M+1S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03",
            ),
            (
                "PT1M+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "PT1M+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1S+0.001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1S+0.000001S",
                None,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1Y+1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1Y+1D",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1Y+1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1Y+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1Y+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1Y+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1Y+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1M+1D",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1M+1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1M+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1M+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1M+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1M+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1D+1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01",
            ),
            (
                "P1D+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1D+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1D+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1D+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1H+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02",
            ),
            (
                "P1H+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "P1H+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1H+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "PT1M+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03",
            ),
            (
                "PT1M+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "PT1M+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1S+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456",
            ),
            (
                "P1S+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789),
                "2023-10-20 01:02:03.456789",
            ),
            (
                "P1Y+1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y+1D",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y+1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1Y+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1Y+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1Y+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1Y+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1M+1D",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1M+1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1M+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1M+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1M+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1M+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1D+1H",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01Z",
            ),
            (
                "P1D+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1D+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1D+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1D+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1H+T1M",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02Z",
            ),
            (
                "P1H+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "P1H+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1H+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "PT1M+1S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03Z",
            ),
            (
                "PT1M+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "PT1M+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
            (
                "P1S+0.001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456Z",
            ),
            (
                "P1S+0.000001S",
                TZ_UTC,
                " ",
                datetime.datetime(2023, 10, 20, 1, 2, 3, 456789, TZ_UTC),
                "2023-10-20 01:02:03.456789Z",
            ),
        ]
    )
    def test_offset_durations_utc(
        self,
        name: str,
        tz: datetime.tzinfo,
        sep: str,
        dt: datetime.datetime,
        expected: str,
    ) -> None:
        """Test Period.aware_formatter arguments"""
        period: Period = Period.of_duration(name).with_tzinfo(tz)
        fmt = period.formatter(sep)
        result = fmt(dt)
        self.assertEqual(result, expected)


class TestPeriodIsAligned(unittest.TestCase):
    """Test Period.is_aligned method"""

    @parameterized.expand(
        [
            ("P1Y", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P1Y", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), False),
            ("P1Y", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), False),
            ("P1Y", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), False),
            ("P1Y", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("P1Y", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("P1Y", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), False),
            ("P1M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P1M", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), True),
            ("P1M", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), False),
            ("P1M", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), False),
            ("P1M", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("P1M", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("P1M", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), False),
            ("P1D", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P1D", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), True),
            ("P1D", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), True),
            ("P1D", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), False),
            ("P1D", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("P1D", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("P1D", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), False),
            ("P1H", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P1H", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), True),
            ("P1H", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), True),
            ("P1H", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), True),
            ("P1H", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("P1H", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("P1H", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), False),
            ("PT1M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("PT1M", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), True),
            ("PT1M", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), True),
            ("PT1M", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), True),
            ("PT1M", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), True),
            ("PT1M", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("PT1M", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), False),
            ("P1S", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P1S", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), True),
            ("P1S", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), True),
            ("P1S", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), True),
            ("P1S", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), True),
            ("P1S", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), True),
            ("P1S", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), False),
            ("P0.000001S", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P0.000001S", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), True),
            ("P0.000001S", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), True),
            ("P0.000001S", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), True),
            ("P0.000001S", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), True),
            ("P0.000001S", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), True),
            ("P0.000001S", datetime.datetime(2024, 1, 1, 0, 0, 0, 1), True),
            ("P2M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P2M", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), False),
            ("P2M", datetime.datetime(2024, 3, 1, 0, 0, 0, 0), True),
            ("P2M", datetime.datetime(2024, 4, 1, 0, 0, 0, 0), False),
            ("P3M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P3M", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), False),
            ("P3M", datetime.datetime(2024, 3, 1, 0, 0, 0, 0), False),
            ("P3M", datetime.datetime(2024, 4, 1, 0, 0, 0, 0), True),
            ("P6M", datetime.datetime(2024, 7, 1, 0, 0, 0, 0), True),
            ("P6M", datetime.datetime(2024, 8, 1, 0, 0, 0, 0), False),
            ("P6M", datetime.datetime(2024, 9, 1, 0, 0, 0, 0), False),
            ("P6M", datetime.datetime(2024, 10, 1, 0, 0, 0, 0), False),
            ("P2H", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P2H", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), False),
            ("P2H", datetime.datetime(2024, 1, 1, 2, 0, 0, 0), True),
            ("P2H", datetime.datetime(2024, 1, 1, 3, 0, 0, 0), False),
            ("P3H", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P3H", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), False),
            ("P3H", datetime.datetime(2024, 1, 1, 2, 0, 0, 0), False),
            ("P3H", datetime.datetime(2024, 1, 1, 3, 0, 0, 0), True),
            ("PT2M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("PT2M", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("PT2M", datetime.datetime(2024, 1, 1, 0, 2, 0, 0), True),
            ("PT2M", datetime.datetime(2024, 1, 1, 0, 3, 0, 0), False),
            ("PT3M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("PT3M", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("PT3M", datetime.datetime(2024, 1, 1, 0, 2, 0, 0), False),
            ("PT3M", datetime.datetime(2024, 1, 1, 0, 3, 0, 0), True),
            ("P2S", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P2S", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("P2S", datetime.datetime(2024, 1, 1, 0, 0, 2, 0), True),
            ("P2S", datetime.datetime(2024, 1, 1, 0, 0, 3, 0), False),
            ("P3S", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), True),
            ("P3S", datetime.datetime(2024, 1, 1, 0, 0, 1, 0), False),
            ("P3S", datetime.datetime(2024, 1, 1, 0, 0, 2, 0), False),
            ("P3S", datetime.datetime(2024, 1, 1, 0, 0, 3, 0), True),
        ]
    )
    def test_iso_durations(
        self, name: Any, dt: datetime.datetime, expected: bool
    ) -> None:
        """Test Period.is_aligned method"""
        period: Period = Period.of_iso_duration(name)
        self.assertEqual(period.is_aligned(dt), expected)

    @parameterized.expand(
        [
            ("P1M+1D", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), False),
            ("P1M+1D", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), True),
            ("P1M+1D", datetime.datetime(2024, 1, 3, 0, 0, 0, 0), False),
            ("P1M+1D", datetime.datetime(2024, 2, 1, 0, 0, 0, 0), False),
            ("P1M+1D", datetime.datetime(2024, 2, 2, 0, 0, 0, 0), True),
            ("P1M+1D", datetime.datetime(2024, 2, 3, 0, 0, 0, 0), False),
            ("P1M+1D1H", datetime.datetime(2024, 1, 2, 0, 0, 0, 0), False),
            ("P1M+1D1H", datetime.datetime(2024, 1, 2, 1, 0, 0, 0), True),
            ("P1M+1D1H", datetime.datetime(2024, 1, 2, 2, 0, 0, 0), False),
            ("P1D+1H", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), False),
            ("P1D+1H", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), True),
            ("P1D+1H", datetime.datetime(2024, 1, 1, 2, 0, 0, 0), False),
            ("P1D+1H1M", datetime.datetime(2024, 1, 1, 1, 0, 0, 0), False),
            ("P1D+1H1M", datetime.datetime(2024, 1, 1, 1, 1, 0, 0), True),
            ("P1D+1H1M", datetime.datetime(2024, 1, 1, 1, 2, 0, 0), False),
            ("P1H+T1M", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), False),
            ("P1H+T1M", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), True),
            ("P1H+T1M", datetime.datetime(2024, 1, 1, 0, 2, 0, 0), False),
            ("P1H+T1M1S", datetime.datetime(2024, 1, 1, 0, 1, 0, 0), False),
            ("P1H+T1M1S", datetime.datetime(2024, 1, 1, 0, 1, 1, 0), True),
            ("P1H+T1M1S", datetime.datetime(2024, 1, 1, 0, 1, 2, 0), False),
            ("P1Y+9M9H", datetime.datetime(2024, 1, 1, 0, 0, 0, 0), False),
            ("P1Y+9M9H", datetime.datetime(2024, 10, 1, 0, 0, 0, 0), False),
            ("P1Y+9M9H", datetime.datetime(2024, 10, 1, 9, 0, 0, 0), True),
        ]
    )
    def test_offset_durations(
        self, name: Any, dt: datetime.datetime, expected: bool
    ) -> None:
        """Test Period.is_aligned method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(period.is_aligned(dt), expected)


class TestPeriodBasePeriod(unittest.TestCase):
    """Test Period.base_period method"""

    @parameterized.expand(sorted(_GOOD_ISO_DURATION))
    def test_already_base(self, text: Any) -> None:
        """Test Period.base_period method returns same Period instance"""
        period: Period = Period.of(text)
        self.assertEqual(period, period.base_period())

    @parameterized.expand(
        [
            ("P1Y+1M", "P1Y"),
            ("P5Y+1Y", "P5Y"),
            ("P5Y+1M", "P5Y"),
            ("P1Y+1D", "P1Y"),
            ("P5Y+1D", "P5Y"),
            ("P1M+1D", "P1M"),
            ("P5M+1D", "P5M"),
            ("P1D+1H", "P1D"),
            ("P5D+1D", "P5D"),
            ("P5D+1H", "P5D"),
            ("P1H+T1M", "P1H"),
            ("P5H+1H", "P5H"),
            ("P5H+T1M", "P5H"),
            ("PT1M+1S", "PT1M"),
            ("PT5M+T1M", "PT5M"),
            ("PT5M+1S", "PT5M"),
            ("P1S+0.001S", "P1S"),
            ("P5S+1S", "P5S"),
            ("P5S+0.001S", "P5S"),
            ("P0.001S+0.000001S", "P0.001S"),
            ("P0.005S+0.001S", "P0.005S"),
            ("P0.005S+0.000001S", "P0.005S"),
        ]
    )
    def test_offset_durations(self, name: Any, base_name: str) -> None:
        """Test Period.base_period method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(period.base_period(), Period.of_iso_duration(base_name))


class TestPeriodWithMultiplier(unittest.TestCase):
    """Test Period.with_multiplier method"""

    @parameterized.expand(
        [
            ("P1Y", 2, "P2Y"),
            ("P1M", 12, "P1Y"),
            ("P1Y+1M", 2, "P2Y+1M"),
            ("P1D", 7, "P7D"),
            ("P7D", 7, "P49D"),
            ("P1S", 60, "PT1M"),
            ("PT1M", 60, "P1H"),
            ("P1H", 24, "P1D"),
            ("P2H", 12, "P1D"),
            ("P3H", 8, "P1D"),
            ("P4H", 6, "P1D"),
            ("P5H", 5, "P25H"),
            ("P6H", 4, "P1D"),
            ("P7H", 3, "P21H"),
            ("P0.01S", 10, "P0.1S"),
            ("P0.001S", 100, "P0.1S"),
            ("P0.0001S", 1_000, "P0.1S"),
            ("P0.00001S", 10_000, "P0.1S"),
            ("P0.000001S", 100_000, "P0.1S"),
            ("P0.1S", 10, "P1S"),
            ("P0.01S", 100, "P1S"),
            ("P0.001S", 1_000, "P1S"),
            ("P0.0001S", 10_000, "P1S"),
            ("P0.00001S", 100_000, "P1S"),
            ("P0.000001S", 100_0000, "P1S"),
            ("P0.2S", 5, "P1S"),
            ("P0.02S", 50, "P1S"),
            ("P0.002S", 500, "P1S"),
            ("P0.0002S", 5_000, "P1S"),
            ("P0.00002S", 50_000, "P1S"),
            ("P0.000002S", 500_000, "P1S"),
            ("P0.000001S", 60_000_000, "PT1M"),
            ("P0.000001S", 3_600_000_000, "P1H"),
            ("P0.000001S", 86_400_000_000, "P1D"),
        ]
    )
    def test_offset_durations(
        self, name: Any, multiplier: int, result_name: str
    ) -> None:
        """Test Period.with_multiplier method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_multiplier(multiplier), Period.of_duration(result_name)
        )
        self.assertEqual(period.with_multiplier(1), period)


class TestPeriodWithYearOffset(unittest.TestCase):
    """Test Period.with_year_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y"),
            ("P2Y", 1, "P2Y+1Y"),
            ("P3Y", 1, "P3Y+1Y"),
            ("P4Y", 1, "P4Y+1Y"),
            ("P5Y", 1, "P5Y+1Y"),
            ("P1Y", 2, "P1Y"),
            ("P1Y", 3, "P1Y"),
            ("P1Y", 4, "P1Y"),
            ("P1Y", 5, "P1Y"),
            ("P10Y", 1, "P10Y+1Y"),
            ("P2Y", 2, "P2Y"),
            ("P3Y", 2, "P3Y+2Y"),
            ("P4Y", 2, "P4Y+2Y"),
            ("P5Y", 2, "P5Y+2Y"),
            ("P10Y", 2, "P10Y+2Y"),
        ]
    )
    def test_offset_durations(
        self, name: Any, year_amount: int, result_name: str
    ) -> None:
        """Test Period.with_year_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_year_offset(year_amount), Period.of_duration(result_name)
        )


class TestPeriodWithMonthOffset(unittest.TestCase):
    """Test Period.with_month_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y+1M"),
            ("P1Y", 2, "P1Y+2M"),
            ("P1Y", 3, "P1Y+3M"),
            ("P1M", 1, "P1M"),
            ("P1M", 2, "P1M"),
            ("P1M", 3, "P1M"),
            ("P1M", 4, "P1M"),
            ("P1M", 5, "P1M"),
            ("P3M", 1, "P3M+1M"),
            ("P4M", 1, "P4M+1M"),
            ("P5M", 1, "P5M+1M"),
            ("P6M", 1, "P6M+1M"),
            ("P2M", 2, "P2M"),
            ("P3M", 2, "P3M+2M"),
            ("P4M", 2, "P4M+2M"),
            ("P5M", 2, "P5M+2M"),
            ("P6M", 2, "P6M+2M"),
        ]
    )
    def test_offset_durations(
        self, name: Any, month_amount: int, result_name: str
    ) -> None:
        """Test Period.with_month_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_month_offset(month_amount), Period.of_duration(result_name)
        )


class TestPeriodWithDayOffset(unittest.TestCase):
    """Test Period.with_day_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y+1D"),
            ("P1Y", 2, "P1Y+2D"),
            ("P1Y", 3, "P1Y+3D"),
            ("P1M", 1, "P1M+1D"),
            ("P1M", 2, "P1M+2D"),
            ("P1M", 3, "P1M+3D"),
            ("P1D", 1, "P1D"),
            ("P1D", 2, "P1D"),
            ("P1D", 3, "P1D"),
            ("P1D", 4, "P1D"),
            ("P1D", 5, "P1D"),
            ("P3D", 1, "P3D+1D"),
            ("P4D", 1, "P4D+1D"),
            ("P5D", 1, "P5D+1D"),
            ("P6D", 1, "P6D+1D"),
            ("P2D", 2, "P2D"),
            ("P3D", 2, "P3D+2D"),
            ("P4D", 2, "P4D+2D"),
            ("P5D", 2, "P5D+2D"),
            ("P6D", 2, "P6D+2D"),
        ]
    )
    def test_offset_durations(
        self, name: Any, day_amount: int, result_name: str
    ) -> None:
        """Test Period.with_day_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_day_offset(day_amount), Period.of_duration(result_name)
        )


class TestPeriodWithHourOffset(unittest.TestCase):
    """Test Period.with_hour_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y+1H"),
            ("P1Y", 2, "P1Y+2H"),
            ("P1Y", 3, "P1Y+3H"),
            ("P1M", 1, "P1M+1H"),
            ("P1M", 2, "P1M+2H"),
            ("P1M", 3, "P1M+3H"),
            ("P1D", 1, "P1D+1H"),
            ("P1D", 2, "P1D+2H"),
            ("P1D", 3, "P1D+3H"),
            ("P1H", 1, "P1H"),
            ("P1H", 2, "P1H"),
            ("P1H", 3, "P1H"),
            ("P1H", 4, "P1H"),
            ("P1H", 5, "P1H"),
            ("P3H", 1, "P3H+1H"),
            ("P4H", 1, "P4H+1H"),
            ("P5H", 1, "P5H+1H"),
            ("P6H", 1, "P6H+1H"),
            ("P2H", 2, "P2H"),
            ("P3H", 2, "P3H+2H"),
            ("P4H", 2, "P4H+2H"),
            ("P5H", 2, "P5H+2H"),
            ("P6H", 2, "P6H+2H"),
        ]
    )
    def test_offset_durations(
        self, name: Any, hour_amount: int, result_name: str
    ) -> None:
        """Test Period.with_hour_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_hour_offset(hour_amount), Period.of_duration(result_name)
        )


class TestPeriodWithMinuteOffset(unittest.TestCase):
    """Test Period.with_minute_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y+T1M"),
            ("P1Y", 2, "P1Y+T2M"),
            ("P1Y", 3, "P1Y+T3M"),
            ("P1M", 1, "P1M+T1M"),
            ("P1M", 2, "P1M+T2M"),
            ("P1M", 3, "P1M+T3M"),
            ("P1D", 1, "P1D+T1M"),
            ("P1D", 2, "P1D+T2M"),
            ("P1D", 3, "P1D+T3M"),
            ("P1H", 1, "P1H+T1M"),
            ("P1H", 2, "P1H+T2M"),
            ("P1H", 3, "P1H+T3M"),
            ("PT1M", 1, "PT1M"),
            ("PT1M", 2, "PT1M"),
            ("PT1M", 3, "PT1M"),
            ("PT1M", 4, "PT1M"),
            ("PT1M", 5, "PT1M"),
            ("PT3M", 1, "PT3M+T1M"),
            ("PT4M", 1, "PT4M+T1M"),
            ("PT5M", 1, "PT5M+T1M"),
            ("PT6M", 1, "PT6M+T1M"),
            ("PT2M", 2, "PT2M"),
            ("PT3M", 2, "PT3M+T2M"),
            ("PT4M", 2, "PT4M+T2M"),
            ("PT5M", 2, "PT5M+T2M"),
            ("PT6M", 2, "PT6M+T2M"),
        ]
    )
    def test_offset_durations(
        self, name: Any, minute_amount: int, result_name: str
    ) -> None:
        """Test Period.with_minute_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_minute_offset(minute_amount), Period.of_duration(result_name)
        )


class TestPeriodWithSecondOffset(unittest.TestCase):
    """Test Period.with_second_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y+1S"),
            ("P1Y", 2, "P1Y+2S"),
            ("P1Y", 3, "P1Y+3S"),
            ("P1M", 1, "P1M+1S"),
            ("P1M", 2, "P1M+2S"),
            ("P1M", 3, "P1M+3S"),
            ("P1D", 1, "P1D+1S"),
            ("P1D", 2, "P1D+2S"),
            ("P1D", 3, "P1D+3S"),
            ("P1H", 1, "P1H+1S"),
            ("P1H", 2, "P1H+2S"),
            ("P1H", 3, "P1H+3S"),
            ("PT1M", 1, "PT1M+1S"),
            ("PT1M", 2, "PT1M+2S"),
            ("PT1M", 3, "PT1M+3S"),
            ("P1S", 1, "P1S"),
            ("P1S", 2, "P1S"),
            ("P1S", 3, "P1S"),
            ("P1S", 4, "P1S"),
            ("P1S", 5, "P1S"),
            ("P3S", 1, "P3S+1S"),
            ("P4S", 1, "P4S+1S"),
            ("P5S", 1, "P5S+1S"),
            ("P6S", 1, "P6S+1S"),
            ("P2S", 2, "P2S"),
            ("P3S", 2, "P3S+2S"),
            ("P4S", 2, "P4S+2S"),
            ("P5S", 2, "P5S+2S"),
            ("P6S", 2, "P6S+2S"),
        ]
    )
    def test_offset_durations(
        self, name: Any, second_amount: int, result_name: str
    ) -> None:
        """Test Period.with_second_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_second_offset(second_amount), Period.of_duration(result_name)
        )


class TestPeriodWithMicrosecondOffset(unittest.TestCase):
    """Test Period.with_microsecond_offset method"""

    @parameterized.expand(
        [
            ("P1Y", 1, "P1Y+0.000001S"),
            ("P1Y", 2, "P1Y+0.000002S"),
            ("P1Y", 3, "P1Y+0.000003S"),
            ("P1M", 1, "P1M+0.000001S"),
            ("P1M", 2, "P1M+0.000002S"),
            ("P1M", 3, "P1M+0.000003S"),
            ("P1D", 1, "P1D+0.000001S"),
            ("P1D", 2, "P1D+0.000002S"),
            ("P1D", 3, "P1D+0.000003S"),
            ("P1H", 1, "P1H+0.000001S"),
            ("P1H", 2, "P1H+0.000002S"),
            ("P1H", 3, "P1H+0.000003S"),
            ("PT1M", 1, "PT1M+0.000001S"),
            ("PT1M", 2, "PT1M+0.000002S"),
            ("PT1M", 3, "PT1M+0.000003S"),
            ("P1S", 1, "P1S+0.000001S"),
            ("P1S", 2, "P1S+0.000002S"),
            ("P1S", 3, "P1S+0.000003S"),
            ("P0.000001S", 1, "P0.000001S"),
            ("P0.000001S", 2, "P0.000001S"),
            ("P0.000001S", 3, "P0.000001S"),
            ("P0.000001S", 4, "P0.000001S"),
            ("P0.000001S", 5, "P0.000001S"),
            ("P0.000003S", 1, "P0.000003S+0.000001S"),
            ("P0.000004S", 1, "P0.000004S+0.000001S"),
            ("P0.000005S", 1, "P0.000005S+0.000001S"),
            ("P0.000006S", 1, "P0.000006S+0.000001S"),
            ("P0.000002S", 2, "P0.000002S"),
            ("P0.000003S", 2, "P0.000003S+0.000002S"),
            ("P0.000004S", 2, "P0.000004S+0.000002S"),
            ("P0.000005S", 2, "P0.000005S+0.000002S"),
            ("P0.000006S", 2, "P0.000006S+0.000002S"),
            ("P1S", 1_000_000, "P1S"),
            ("P1S", 500_000, "P1S+0.5S"),
        ]
    )
    def test_offset_durations(
        self, name: Any, microsecond_amount: int, result_name: str
    ) -> None:
        """Test Period.with_microsecond_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(
            period.with_microsecond_offset(microsecond_amount),
            Period.of_duration(result_name),
        )


class TestPeriodWithTzinfo(unittest.TestCase):
    """Test Period.with_tzinfo method"""

    @parameterized.expand(
        [
            ("P1Y", [TZ_UTC, None]),
            ("P1M", [TZ_UTC, None]),
            ("P1D", [TZ_UTC, None]),
        ]
    )
    def test_offset_durations(
        self, name: Any, tz_info_list: list[Optional[datetime.tzinfo]]
    ) -> None:
        """Test Period.with_microsecond_offset method"""
        period: Period = Period.of_duration(name)
        old_tz: datetime.tzinfo = None
        for new_tz in tz_info_list:
            self.assertEqual(period.tzinfo, old_tz)
            period = period.with_tzinfo(new_tz)
            self.assertEqual(period.tzinfo, new_tz)
            old_tz = new_tz


class TestPeriodWithoutOffset(unittest.TestCase):
    """Test Period.without_offset method"""

    @parameterized.expand(
        [
            ("P1Y", "P1Y"),
            ("P1M", "P1M"),
            ("P1D", "P1D"),
            ("P1H", "P1H"),
            ("PT1M", "PT1M"),
            ("P1S", "P1S"),
            ("P0.000001S", "P0.000001S"),
            ("P2Y+1Y", "P2Y"),
            ("P2M+1M", "P2M"),
            ("P2D+1D", "P2D"),
            ("P2H+1H", "P2H"),
            ("PT2M+T1M", "PT2M"),
            ("P2S+1S", "P2S"),
            ("P0.000002S+0.000001S", "P0.000002S"),
            ("P1Y+1M", "P1Y"),
            ("P1M+1D", "P1M"),
            ("P1D+1H", "P1D"),
            ("P1H+T1M", "P1H"),
            ("PT1M+1S", "PT1M"),
            ("P1S+0.000001S", "P1S"),
        ]
    )
    def test_offset_durations(self, name: Any, result_name: str) -> None:
        """Test Period.without_offset method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(period.without_offset(), Period.of_iso_duration(result_name))


class TestPeriodWithoutOrdinalShift(unittest.TestCase):
    """Test Period.without_ordinal_shift method"""

    @parameterized.expand(
        [
            ("P1Y", datetime.datetime(1066, 1, 1)),
            ("P1Y", datetime.datetime(1980, 1, 1)),
            ("P1Y", datetime.datetime(4781, 1, 1)),
            ("P1M", datetime.datetime(1066, 1, 1)),
            ("P1M", datetime.datetime(1980, 1, 1)),
            ("P1M", datetime.datetime(4781, 1, 1)),
            ("P1D", datetime.datetime(1066, 1, 1)),
            ("P1D", datetime.datetime(1980, 1, 1)),
            ("P1D", datetime.datetime(4781, 1, 1)),
        ]
    )
    def test_offset_durations(self, name: Any, origin_dt: datetime.datetime) -> None:
        """Test Period.without_ordinal_shift method"""
        period: Period = Period.of_duration(name)
        self.assertEqual(period._properties.ordinal_shift, 0)
        period_with_origin: Period = period.with_origin(origin_dt)
        self.assertNotEqual(period_with_origin._properties.ordinal_shift, 0)
        self.assertNotEqual(period, period_with_origin)
        period_without_shift: Period = period_with_origin.without_ordinal_shift()
        self.assertNotEqual(period_with_origin, period_without_shift)
        self.assertEqual(period_without_shift._properties.ordinal_shift, 0)
        self.assertEqual(period_without_shift, period)


class TestPeriodWithOrigin(unittest.TestCase):
    """Test Period.with_origin method"""

    @parameterized.expand(
        [
            ("P1Y", datetime.datetime(1066, 1, 1), "P1Y"),
            ("P1Y", datetime.datetime(1980, 1, 1), "P1Y"),
            ("P1Y", datetime.datetime(4781, 1, 1), "P1Y"),
            ("P1M", datetime.datetime(1066, 1, 1), "P1M"),
            ("P1M", datetime.datetime(1980, 1, 1), "P1M"),
            ("P1M", datetime.datetime(4781, 1, 1), "P1M"),
            ("P1D", datetime.datetime(1066, 1, 1), "P1D"),
            ("P1D", datetime.datetime(1980, 1, 1), "P1D"),
            ("P1D", datetime.datetime(4781, 1, 1), "P1D"),
            ("P1Y", datetime.datetime(1980, 7, 1), "P1Y+6M"),
            ("P1Y", datetime.datetime(1980, 1, 2), "P1Y+1D"),
            ("P1Y", datetime.datetime(4873, 3, 6), "P1Y+2M5D"),
            ("P1Y", datetime.datetime(1980, 1, 1, 1, 0, 0), "P1Y+1H"),
            ("P1Y", datetime.datetime(1980, 1, 1, 0, 1, 0), "P1Y+T1M"),
            ("P1Y", datetime.datetime(1980, 1, 1, 0, 0, 1), "P1Y+1S"),
            ("P1M", datetime.datetime(1980, 1, 5), "P1M+4D"),
            ("P1M", datetime.datetime(1980, 1, 1, 5, 0, 0), "P1M+5H"),
            ("P1M", datetime.datetime(1980, 1, 1, 0, 5, 0), "P1M+T5M"),
            ("P1M", datetime.datetime(1980, 1, 1, 0, 0, 5), "P1M+5S"),
            ("P1D", datetime.datetime(1872, 7, 12, 10, 0, 0), "P1D+10H"),
            ("P1D", datetime.datetime(1872, 7, 12, 0, 10, 0), "P1D+T10M"),
            ("P1D", datetime.datetime(1872, 7, 12, 0, 0, 10), "P1D+10S"),
            ("P6H", datetime.datetime(2540, 9, 15, 3, 2, 1), "P6H+3H2M1S"),
            (
                "PT15M",
                datetime.datetime(1457, 3, 6, 9, 8, 1, 123456),
                "PT15M+T8M1.123456S",
            ),
        ]
    )
    def test_offset_durations(
        self, name: Any, origin_dt: datetime.datetime, result: str
    ) -> None:
        """Test Period.with_origin method"""
        period: Period = Period.of_duration(name)
        period_with_origin: Period = period.with_origin(origin_dt)
        self.assertNotEqual(period, period_with_origin)
        self.assertEqual(period_with_origin.ordinal(origin_dt), 0)
        self.assertTrue(period_with_origin.is_aligned(origin_dt))
        period_without_shift: Period = period_with_origin.without_ordinal_shift()
        result_period: Period = Period.of_duration(result)
        self.assertNotEqual(period_with_origin, period_without_shift)
        self.assertEqual(period_without_shift, result_period)


class TestYearPeriod(unittest.TestCase):
    """Test YearPeriod class"""

    @parameterized.expand(
        [
            ("P1Y", datetime.datetime(1066, 1, 1), 1066),
            ("P1Y", datetime.datetime(1984, 1, 1), 1984),
            ("P1Y", datetime.datetime(5432, 1, 1), 5432),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal YearPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.YearPeriod),
            f"Class mismatch: {period} is not of class {p.YearPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y+0.000001S",),
            ("P1Y+1D",),
            ("P2Y",),
            ("P3Y",),
            ("P4Y",),
            ("P6M",),
            ("P1D",),
            ("P1H",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal YearPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.YearPeriod),
            f"Class mismatch: {period} is of class {p.YearPeriod}",
        )


class TestMultiYearPeriod(unittest.TestCase):
    """Test MultiYearPeriod class"""

    @parameterized.expand(
        [
            ("P2Y", datetime.datetime(1066, 1, 1), 533),
            ("P3Y", datetime.datetime(1983, 1, 1), 661),
            ("P4Y", datetime.datetime(5432, 1, 1), 1358),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal MultiYearPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.MultiYearPeriod),
            f"Class mismatch: {period} is not of class {p.MultiYearPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y+0.000001S",),
            ("P3Y+1D",),
            ("P4Y+1Y",),
            ("P6M",),
            ("P1D",),
            ("P1H",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal MultiYearPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.MultiYearPeriod),
            f"Class mismatch: {period} is of class {p.MultiYearPeriod}",
        )


class TestMonthPeriod(unittest.TestCase):
    """Test MonthPeriod class"""

    @parameterized.expand(
        [
            ("P1M", datetime.datetime(1066, 1, 1), 12792),
            ("P1M", datetime.datetime(1983, 1, 1), 23796),
            ("P1M", datetime.datetime(5432, 1, 1), 65184),
            ("P1M", datetime.datetime(2024, 10, 1), 24297),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal MonthPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.MonthPeriod),
            f"Class mismatch: {period} is not of class {p.MonthPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P12M",),
            ("P2M",),
            ("P1M+0.000001S",),
            ("P1M+1D",),
            ("P4Y+1Y",),
            ("P6M",),
            ("P1D",),
            ("P1H",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal MonthPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.MonthPeriod),
            f"Class mismatch: {period} is of class {p.MonthPeriod}",
        )


class TestMultiMonthPeriod(unittest.TestCase):
    """Test MultiMonthPeriod class"""

    @parameterized.expand(
        [
            ("P2M", datetime.datetime(1066, 1, 1), 6396),
            ("P3M", datetime.datetime(1983, 1, 1), 7932),
            ("P4M", datetime.datetime(5432, 1, 1), 16296),
            ("P6M", datetime.datetime(2024, 7, 1), 4049),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal MultiMonthPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.MultiMonthPeriod),
            f"Class mismatch: {period} is not of class {p.MultiMonthPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1M",),
            ("P1Y",),
            ("P12M",),
            ("P2M+0.000001S",),
            ("P2M+1D",),
            ("P2M+1H",),
            ("P1D",),
            ("P1H",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal MultiMonthPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.MultiMonthPeriod),
            f"Class mismatch: {period} is of class {p.MultiMonthPeriod}",
        )


class TestNaiveDayPeriod(unittest.TestCase):
    """Test NaiveDayPeriod class"""

    @parameterized.expand(
        [
            ("P1D", datetime.datetime(1066, 1, 1), 388984),
            ("P1D", datetime.datetime(1983, 1, 1), 723911),
            ("P1D", datetime.datetime(2024, 10, 10), 739169),
            ("P1D", datetime.datetime(5432, 2, 29), 1983691),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal NaiveDayPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.NaiveDayPeriod),
            f"Class mismatch: {period} is not of class {p.NaiveDayPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D+0.000001S",),
            ("P1D+1H",),
            ("P1D+T1M",),
            ("P2D",),
            ("P1H",),
            ("PT1M",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal NaiveDayPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.NaiveDayPeriod),
            f"Class mismatch: {period} is of class {p.NaiveDayPeriod}",
        )


class TestAwareDayPeriod(unittest.TestCase):
    """Test AwareDayPeriod class"""

    @parameterized.expand(
        [
            ("P1D", datetime.datetime(1066, 1, 1), 388984),
            ("P1D", datetime.datetime(1983, 1, 1), 723911),
            ("P1D", datetime.datetime(2024, 10, 10), 739169),
            ("P1D", datetime.datetime(5432, 2, 29), 1983691),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal AwareDayPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.AwareDayPeriod),
            f"Class mismatch: {period} is of class {p.AwareDayPeriod}",
        )
        period = period.with_tzinfo(TZ_UTC)
        self.assertTrue(
            isinstance(period, p.AwareDayPeriod),
            f"Class mismatch: {period} is not of class {p.AwareDayPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        dt = dt.replace(tzinfo=None)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D+0.000001S",),
            ("P1D+1H",),
            ("P1D+T1M",),
            ("P2D",),
            ("P1H",),
            ("PT1M",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal AwareDayPeriod instances"""
        period: Period = Period.of_duration(name).with_tzinfo(TZ_UTC)
        self.assertFalse(
            isinstance(period, p.AwareDayPeriod),
            f"Class mismatch: {period} is of class {p.AwareDayPeriod}",
        )


class TestNaiveMultiDayPeriod(unittest.TestCase):
    """Test NaiveMultiDayPeriod class"""

    @parameterized.expand(
        [
            ("P2D", datetime.datetime(1066, 1, 1), 194492),
            ("P3D", datetime.datetime(1983, 1, 2), 241304),
            ("P4D", datetime.datetime(2024, 10, 9), 184792),
            ("P5D", datetime.datetime(5432, 2, 28), 396738),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal NaiveMultiDayPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.NaiveMultiDayPeriod),
            f"Class mismatch: {period} is not of class {p.NaiveMultiDayPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("P2D+0.000001S",),
            ("P2D+1H",),
            ("P2D+T1M",),
            ("P1H",),
            ("PT1M",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal NaiveMultiDayPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.NaiveMultiDayPeriod),
            f"Class mismatch: {period} is of class {p.NaiveMultiDayPeriod}",
        )


class TestAwareMultiDayPeriod(unittest.TestCase):
    """Test AwareMultiDayPeriod class"""

    @parameterized.expand(
        [
            ("P2D", datetime.datetime(1066, 1, 1), 194492),
            ("P3D", datetime.datetime(1983, 1, 2), 241304),
            ("P4D", datetime.datetime(2024, 10, 9), 184792),
            ("P5D", datetime.datetime(5432, 2, 28), 396738),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal AwareMultiDayPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.AwareMultiDayPeriod),
            f"Class mismatch: {period} is of class {p.AwareMultiDayPeriod}",
        )
        period = period.with_tzinfo(TZ_UTC)
        self.assertTrue(
            isinstance(period, p.AwareMultiDayPeriod),
            f"Class mismatch: {period} is not of class {p.AwareMultiDayPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        dt = dt.replace(tzinfo=None)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("P2D+0.000001S",),
            ("P2D+1H",),
            ("P2D+T1M",),
            ("P1H",),
            ("PT1M",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal AwareMultiDayPeriod instances"""
        period: Period = Period.of_duration(name).with_tzinfo(TZ_UTC)
        self.assertFalse(
            isinstance(period, p.AwareMultiDayPeriod),
            f"Class mismatch: {period} is of class {p.AwareMultiDayPeriod}",
        )


class TestMultiHourPeriod(unittest.TestCase):
    """Test MultiHourPeriod class"""

    @parameterized.expand(
        [
            ("P1H", datetime.datetime(1066, 1, 1), 9335616),
            ("P2H", datetime.datetime(1983, 1, 1), 8686932),
            ("P3H", datetime.datetime(5432, 1, 1), 15869056),
            ("P4H", datetime.datetime(2024, 7, 1), 4434408),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal MultiHourPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.MultiHourPeriod),
            f"Class mismatch: {period} is not of class {p.MultiHourPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1D",),
            ("P1M",),
            ("P1Y",),
            ("P2D",),
            ("P2M",),
            ("P2Y",),
            ("P12M",),
            ("P1H+0.000001S",),
            ("P1H+1S",),
            ("P1H+T1M",),
            ("P1H+T30M",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal MultiHourPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.MultiHourPeriod),
            f"Class mismatch: {period} is of class {p.MultiHourPeriod}",
        )


class TestNaiveMultiMinutePeriod(unittest.TestCase):
    """Test NaiveMultiMinutePeriod class"""

    @parameterized.expand(
        [
            ("PT1M", datetime.datetime(1066, 1, 1), 560136960),
            ("PT2M", datetime.datetime(1983, 1, 2), 521216640),
            ("PT3M", datetime.datetime(2024, 10, 9), 354800640),
            ("PT4M", datetime.datetime(5432, 2, 28), 714128400),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal NaiveMultiMinutePeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.NaiveMultiMinutePeriod),
            f"Class mismatch: {period} is not of class {p.NaiveMultiMinutePeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("PT1M+0.000001S",),
            ("PT1M+30S",),
            ("PT1M+1S",),
            ("P1H",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal NaiveMultiMinutePeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.NaiveMultiMinutePeriod),
            f"Class mismatch: {period} is of class {p.NaiveMultiMinutePeriod}",
        )


class TestAwareMultiMinutePeriod(unittest.TestCase):
    """Test AwareMultiMinutePeriod class"""

    @parameterized.expand(
        [
            ("PT1M", datetime.datetime(1066, 1, 1), 560136960),
            ("PT2M", datetime.datetime(1983, 1, 2), 521216640),
            ("PT3M", datetime.datetime(2024, 10, 9), 354800640),
            ("PT4M", datetime.datetime(5432, 2, 28), 714128400),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal AwareMultiMinutePeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.AwareMultiMinutePeriod),
            f"Class mismatch: {period} is of class {p.AwareMultiMinutePeriod}",
        )
        period = period.with_tzinfo(TZ_UTC)
        self.assertTrue(
            isinstance(period, p.AwareMultiMinutePeriod),
            f"Class mismatch: {period} is not of class {p.AwareMultiMinutePeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        dt = dt.replace(tzinfo=None)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("PT1M+0.000001S",),
            ("PT1M+30S",),
            ("PT1M+1S",),
            ("P1H",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal AwareMultiMinutePeriod instances"""
        period: Period = Period.of_duration(name).with_tzinfo(TZ_UTC)
        self.assertFalse(
            isinstance(period, p.AwareMultiMinutePeriod),
            f"Class mismatch: {period} is of class {p.AwareMultiMinutePeriod}",
        )


class TestNaiveMultiSecondPeriod(unittest.TestCase):
    """Test NaiveMultiSecondPeriod class"""

    @parameterized.expand(
        [
            ("P1S", datetime.datetime(1066, 1, 1), 33608217600),
            ("P2S", datetime.datetime(1983, 1, 2), 31272998400),
            ("P3S", datetime.datetime(2024, 10, 9), 21288038400),
            ("P4S", datetime.datetime(5432, 2, 28), 42847704000),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal NaiveMultiSecondPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.NaiveMultiSecondPeriod),
            f"Class mismatch: {period} is not of class {p.NaiveMultiSecondPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("P1H",),
            ("PT1M",),
            ("P1S+0.000001S",),
            ("P2S+0.5S",),
            ("P3S+1S",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal NaiveMultiSecondPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.NaiveMultiSecondPeriod),
            f"Class mismatch: {period} is of class {p.NaiveMultiSecondPeriod}",
        )


class TestAwareMultiSecondPeriod(unittest.TestCase):
    """Test AwareMultiSecondPeriod class"""

    @parameterized.expand(
        [
            ("P1S", datetime.datetime(1066, 1, 1), 33608217600),
            ("P2S", datetime.datetime(1983, 1, 2), 31272998400),
            ("P3S", datetime.datetime(2024, 10, 9), 21288038400),
            ("P4S", datetime.datetime(5432, 2, 28), 42847704000),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal AwareMultiSecondPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.AwareMultiSecondPeriod),
            f"Class mismatch: {period} is of class {p.AwareMultiSecondPeriod}",
        )
        period = period.with_tzinfo(TZ_UTC)
        self.assertTrue(
            isinstance(period, p.AwareMultiSecondPeriod),
            f"Class mismatch: {period} is not of class {p.AwareMultiSecondPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        dt = dt.replace(tzinfo=None)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("P1H",),
            ("PT1M",),
            ("P1S+0.000001S",),
            ("P2S+0.5S",),
            ("P3S+1S",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal AwareMultiSecondPeriod instances"""
        period: Period = Period.of_duration(name).with_tzinfo(TZ_UTC)
        self.assertFalse(
            isinstance(period, p.AwareMultiSecondPeriod),
            f"Class mismatch: {period} is of class {p.AwareMultiSecondPeriod}",
        )


class TestNaiveMicroSecondPeriod(unittest.TestCase):
    """Test NaiveMicroSecondPeriod class"""

    @parameterized.expand(
        [
            ("P0.000001S", datetime.datetime(1066, 1, 1), 33608217600000000),
            ("P0.001S", datetime.datetime(1983, 1, 2), 62545996800000),
            ("P0.04S", datetime.datetime(2024, 10, 9), 1596602880000),
            ("P0.5S", datetime.datetime(5432, 2, 28), 342781632000),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal NaiveMicroSecondPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertTrue(
            isinstance(period, p.NaiveMicroSecondPeriod),
            f"Class mismatch: {period} is not of class {p.NaiveMicroSecondPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("P1H",),
            ("PT1M",),
            ("P1S",),
            ("P0.5S+0.000001S",),
            ("P0.04S+0.001S",),
            ("P0.000002S+0.000001S",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal NaiveMicroSecondPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.NaiveMicroSecondPeriod),
            f"Class mismatch: {period} is of class {p.NaiveMicroSecondPeriod}",
        )


class TestAwareMicroSecondPeriod(unittest.TestCase):
    """Test AwareMicroSecondPeriod class"""

    @parameterized.expand(
        [
            ("P0.000001S", datetime.datetime(1066, 1, 1), 33608217600000000),
            ("P0.001S", datetime.datetime(1983, 1, 2), 62545996800000),
            ("P0.04S", datetime.datetime(2024, 10, 9), 1596602880000),
            ("P0.5S", datetime.datetime(5432, 2, 28), 342781632000),
        ]
    )
    def test_good(
        self, name: Any, test_dt: datetime.datetime, test_ordinal: int
    ) -> None:
        """Test legal AwareMicroSecondPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.AwareMicroSecondPeriod),
            f"Class mismatch: {period} is of class {p.AwareMicroSecondPeriod}",
        )
        period = period.with_tzinfo(TZ_UTC)
        self.assertTrue(
            isinstance(period, p.AwareMicroSecondPeriod),
            f"Class mismatch: {period} is not of class {p.AwareMicroSecondPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        dt = dt.replace(tzinfo=None)
        self.assertEqual(dt, test_dt)

    @parameterized.expand(
        [
            ("P1Y",),
            ("P2Y",),
            ("P1M",),
            ("P2M",),
            ("P12M",),
            ("P1D",),
            ("P1H",),
            ("PT1M",),
            ("P1S",),
            ("P0.5S+0.000001S",),
            ("P0.04S+0.001S",),
            ("P0.000002S+0.000001S",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal AwareMicroSecondPeriod instances"""
        period: Period = Period.of_duration(name).with_tzinfo(TZ_UTC)
        self.assertFalse(
            isinstance(period, p.AwareMicroSecondPeriod),
            f"Class mismatch: {period} is of class {p.AwareMicroSecondPeriod}",
        )


class TestGetPeriodTzinfo(unittest.TestCase):
    """Test _get_period_tzinfo function"""

    @parameterized.expand(
        [
            ("P1D", None, p.NaiveDayPeriod, p.AwareDayPeriod, p.NaiveDayPeriod),
            ("P1D", TZ_UTC, p.NaiveDayPeriod, p.AwareDayPeriod, p.AwareDayPeriod),
            (
                "P2D",
                None,
                p.NaiveMultiDayPeriod,
                p.AwareMultiDayPeriod,
                p.NaiveMultiDayPeriod,
            ),
            (
                "P2D",
                TZ_UTC,
                p.NaiveMultiDayPeriod,
                p.AwareMultiDayPeriod,
                p.AwareMultiDayPeriod,
            ),
            (
                "PT1M",
                None,
                p.NaiveMultiMinutePeriod,
                p.AwareMultiMinutePeriod,
                p.NaiveMultiMinutePeriod,
            ),
            (
                "PT1M",
                TZ_UTC,
                p.NaiveMultiMinutePeriod,
                p.AwareMultiMinutePeriod,
                p.AwareMultiMinutePeriod,
            ),
            (
                "PT2M",
                None,
                p.NaiveMultiMinutePeriod,
                p.AwareMultiMinutePeriod,
                p.NaiveMultiMinutePeriod,
            ),
            (
                "PT2M",
                TZ_UTC,
                p.NaiveMultiMinutePeriod,
                p.AwareMultiMinutePeriod,
                p.AwareMultiMinutePeriod,
            ),
            (
                "P1S",
                None,
                p.NaiveMultiSecondPeriod,
                p.AwareMultiSecondPeriod,
                p.NaiveMultiSecondPeriod,
            ),
            (
                "P1S",
                TZ_UTC,
                p.NaiveMultiSecondPeriod,
                p.AwareMultiSecondPeriod,
                p.AwareMultiSecondPeriod,
            ),
            (
                "P2S",
                None,
                p.NaiveMultiSecondPeriod,
                p.AwareMultiSecondPeriod,
                p.NaiveMultiSecondPeriod,
            ),
            (
                "P2S",
                TZ_UTC,
                p.NaiveMultiSecondPeriod,
                p.AwareMultiSecondPeriod,
                p.AwareMultiSecondPeriod,
            ),
            (
                "P0.1S",
                None,
                p.NaiveMicroSecondPeriod,
                p.AwareMicroSecondPeriod,
                p.NaiveMicroSecondPeriod,
            ),
            (
                "P0.1S",
                TZ_UTC,
                p.NaiveMicroSecondPeriod,
                p.AwareMicroSecondPeriod,
                p.AwareMicroSecondPeriod,
            ),
        ]
    )
    def test_calls(
        self,
        name: Any,
        tz: datetime.tzinfo,
        naive_class: Any,
        aware_class: Any,
        result_class: Any,
    ) -> None:
        """Test _get_period_tzinfo calls"""
        properties: p.Properties = Period.of_duration(name).with_tzinfo(tz)._properties
        result: Period = p._get_period_tzinfo(tz, properties, naive_class, aware_class)
        self.assertTrue(
            isinstance(result, result_class),
            f"Class mismatch: {result} is not of class {result_class}",
        )


class TestGetShiftedPeriod(unittest.TestCase):
    """Test _get_shifted_period function"""

    @parameterized.expand(sorted(_GOOD_ISO_DURATION))
    def test_iso(self, name: Any) -> None:
        base_period: Period = Period.of_iso_duration(name)
        base_properties: p.Properties = base_period._properties
        self.assertEqual(base_period, p._get_shifted_period(base_properties))
        shifted_properties: p.Properties = base_properties.with_ordinal_shift(1)
        shifted_period: Period = p._get_shifted_period(shifted_properties)
        self.assertTrue(isinstance(shifted_period, p.ShiftedPeriod))
        self.assertEqual(base_period, shifted_period.base_period())


class TestGetOffsetPeriod(unittest.TestCase):
    """Test _get_offset_period function"""

    @parameterized.expand(sorted(_GOOD_ISO_DURATION))
    def test_iso(self, name: Any) -> None:
        base_period: Period = Period.of_iso_duration(name)
        base_properties: p.Properties = base_period._properties
        self.assertEqual(base_period, p._get_offset_period(base_properties))
        shifted_properties: p.Properties = base_properties.with_ordinal_shift(1)
        with self.assertRaises(AssertionError):
            p._get_offset_period(shifted_properties)
        offset_properties: p.Properties = base_properties.with_microsecond_offset(1)
        offset_period: Period = p._get_offset_period(offset_properties)
        self.assertTrue(isinstance(offset_period, p.OffsetPeriod))
        self.assertEqual(base_period, offset_period.base_period())


class TestGetBasePeriod(unittest.TestCase):
    """Test _get_base_period function"""

    @parameterized.expand(sorted(_GOOD_ISO_DURATION))
    def test_iso(self, name: Any) -> None:
        base_period: Period = Period.of_iso_duration(name)
        base_properties: p.Properties = base_period._properties
        self.assertEqual(base_period, p._get_base_period(base_properties))
        shifted_properties: p.Properties = base_properties.with_ordinal_shift(1)
        with self.assertRaises(AssertionError):
            p._get_base_period(shifted_properties)
        offset_properties: p.Properties = base_properties.with_microsecond_offset(1)
        with self.assertRaises(AssertionError):
            p._get_base_period(offset_properties)


_OFFSET_PARAMS = [
    (
        "P1Y+1M",
        None,
        datetime.datetime(1066, 2, 1, 0, 0, 0, 0, None),
        1066,
        p.YearPeriod,
    ),
    (
        "P2Y+1M",
        None,
        datetime.datetime(1066, 2, 1, 0, 0, 0, 0, None),
        533,
        p.MultiYearPeriod,
    ),
    (
        "P1M+1D",
        None,
        datetime.datetime(1883, 1, 2, 0, 0, 0, 0, None),
        22596,
        p.MonthPeriod,
    ),
    (
        "P2M+1D",
        None,
        datetime.datetime(1883, 1, 2, 0, 0, 0, 0, None),
        11298,
        p.MultiMonthPeriod,
    ),
    (
        "P1D+1H",
        None,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, None),
        724276,
        p.NaiveDayPeriod,
    ),
    (
        "P1D+1H",
        TZ_UTC,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, TZ_UTC),
        724276,
        p.AwareDayPeriod,
    ),
    (
        "P2D+1H",
        None,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, None),
        362138,
        p.NaiveMultiDayPeriod,
    ),
    (
        "P2D+1H",
        TZ_UTC,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, TZ_UTC),
        362138,
        p.AwareMultiDayPeriod,
    ),
    (
        "P1H+T1M",
        None,
        datetime.datetime(5432, 1, 1, 0, 1, 0, 0, None),
        47607168,
        p.MultiHourPeriod,
    ),
    (
        "P2H+T1M",
        None,
        datetime.datetime(5432, 1, 1, 0, 1, 0, 0, None),
        23803584,
        p.MultiHourPeriod,
    ),
    (
        "PT1M+1S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, None),
        2856430080,
        p.NaiveMultiMinutePeriod,
    ),
    (
        "PT1M+1S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, TZ_UTC),
        2856430080,
        p.AwareMultiMinutePeriod,
    ),
    (
        "PT2M+1S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, None),
        1428215040,
        p.NaiveMultiMinutePeriod,
    ),
    (
        "PT2M+1S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, TZ_UTC),
        1428215040,
        p.AwareMultiMinutePeriod,
    ),
    (
        "P1S+0.001S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 1000, None),
        171385804801,
        p.NaiveMultiSecondPeriod,
    ),
    (
        "P1S+0.001S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 1000, TZ_UTC),
        171385804801,
        p.AwareMultiSecondPeriod,
    ),
    (
        "P2S+0.001S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 2, 1000, None),
        85692902401,
        p.NaiveMultiSecondPeriod,
    ),
    (
        "P2S+0.001S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 2, 1000, TZ_UTC),
        85692902401,
        p.AwareMultiSecondPeriod,
    ),
    (
        "P0.1S+0.001S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 0, 1000, None),
        1713858048000,
        p.NaiveMicroSecondPeriod,
    ),
    (
        "P0.1S+0.001S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 0, 1000, TZ_UTC),
        1713858048000,
        p.AwareMicroSecondPeriod,
    ),
]


class TestOffsetPeriod(unittest.TestCase):
    """Test OffsetPeriod class"""

    @parameterized.expand(_OFFSET_PARAMS)
    def test_good(
        self,
        name: Any,
        tz: datetime.tzinfo,
        test_dt: datetime.datetime,
        test_ordinal: int,
        base_p_class: Any,
    ) -> None:
        """Test legal OffsetPeriod instances"""
        period: Period = Period.of_duration(name).with_tzinfo(tz)
        self.assertTrue(
            isinstance(period, p.OffsetPeriod),
            f"Class mismatch: {period} is not of class {p.OffsetPeriod}",
        )
        ordinal: int = period.ordinal(test_dt)
        self.assertEqual(ordinal, test_ordinal)
        dt: datetime.datetime = period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, period.tzinfo)
        self.assertEqual(dt, test_dt)
        base_p: Period = period.base_period()
        self.assertTrue(
            isinstance(base_p, base_p_class),
            f"Class mismatch: Base period {period} is not of class {base_p_class}",
        )

    @parameterized.expand(
        [
            ("P2Y",),
            ("P1Y",),
            ("P2M",),
            ("P1M",),
            ("P2D",),
            ("P1D",),
            ("P2H",),
            ("P1H",),
            ("PT2M",),
            ("PT1M",),
            ("P2S",),
            ("P1S",),
            ("P0.002S",),
            ("P0.001S",),
            ("P0.000002S",),
            ("P0.000001S",),
        ]
    )
    def test_bad(self, name: Any) -> None:
        """Test illegal OffsetPeriod instances"""
        period: Period = Period.of_duration(name)
        self.assertFalse(
            isinstance(period, p.OffsetPeriod),
            f"Class mismatch: {period} is of class {p.OffsetPeriod}",
        )


_BASE_PARAMS = [
    ("P1Y", None, datetime.datetime(1066, 2, 1, 0, 0, 0, 0, None), 1066, p.YearPeriod),
    (
        "P2Y",
        None,
        datetime.datetime(1066, 2, 1, 0, 0, 0, 0, None),
        533,
        p.MultiYearPeriod,
    ),
    (
        "P1M",
        None,
        datetime.datetime(1883, 1, 2, 0, 0, 0, 0, None),
        22596,
        p.MonthPeriod,
    ),
    (
        "P2M",
        None,
        datetime.datetime(1883, 1, 2, 0, 0, 0, 0, None),
        11298,
        p.MultiMonthPeriod,
    ),
    (
        "P1D",
        None,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, None),
        724276,
        p.NaiveDayPeriod,
    ),
    (
        "P1D",
        TZ_UTC,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, TZ_UTC),
        724276,
        p.AwareDayPeriod,
    ),
    (
        "P2D",
        None,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, None),
        362138,
        p.NaiveMultiDayPeriod,
    ),
    (
        "P2D",
        TZ_UTC,
        datetime.datetime(1984, 1, 1, 1, 0, 0, 0, TZ_UTC),
        362138,
        p.AwareMultiDayPeriod,
    ),
    (
        "P1H",
        None,
        datetime.datetime(5432, 1, 1, 0, 1, 0, 0, None),
        47607168,
        p.MultiHourPeriod,
    ),
    (
        "P2H",
        None,
        datetime.datetime(5432, 1, 1, 0, 1, 0, 0, None),
        23803584,
        p.MultiHourPeriod,
    ),
    (
        "PT1M",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, None),
        2856430080,
        p.NaiveMultiMinutePeriod,
    ),
    (
        "PT1M",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, TZ_UTC),
        2856430080,
        p.AwareMultiMinutePeriod,
    ),
    (
        "PT2M",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, None),
        1428215040,
        p.NaiveMultiMinutePeriod,
    ),
    (
        "PT2M",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 0, TZ_UTC),
        1428215040,
        p.AwareMultiMinutePeriod,
    ),
    (
        "P1S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 1000, None),
        171385804801,
        p.NaiveMultiSecondPeriod,
    ),
    (
        "P1S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 1, 1000, TZ_UTC),
        171385804801,
        p.AwareMultiSecondPeriod,
    ),
    (
        "P2S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 2, 1000, None),
        85692902401,
        p.NaiveMultiSecondPeriod,
    ),
    (
        "P2S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 2, 1000, TZ_UTC),
        85692902401,
        p.AwareMultiSecondPeriod,
    ),
    (
        "P0.1S",
        None,
        datetime.datetime(5432, 1, 1, 0, 0, 0, 1000, None),
        1713858048000,
        p.NaiveMicroSecondPeriod,
    ),
    (
        "P0.1S",
        TZ_UTC,
        datetime.datetime(5432, 1, 1, 0, 0, 0, 1000, TZ_UTC),
        1713858048000,
        p.AwareMicroSecondPeriod,
    ),
]


class TestShiftedPeriod(unittest.TestCase):
    """Test ShiftedPeriod class"""

    @parameterized.expand(_OFFSET_PARAMS)
    def test_good_offset(
        self,
        name: Any,
        tz: datetime.tzinfo,
        test_dt: datetime.datetime,
        test_ordinal: int,
        base_p_class: Any,
    ) -> None:
        """Test ShiftedPeriod instances"""
        offset_period: Period = Period.of_duration(name).with_tzinfo(tz)
        self.assertTrue(
            isinstance(offset_period, p.OffsetPeriod),
            f"Class mismatch: {offset_period} is not of class {p.OffsetPeriod}",
        )
        shifted_period: Period = offset_period.with_origin(test_dt)
        ordinal: int = shifted_period.ordinal(test_dt)
        self.assertEqual(ordinal, 0)
        dt: datetime.datetime = shifted_period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, shifted_period.tzinfo)
        self.assertEqual(dt, test_dt)
        base_p_offset: Period = offset_period.base_period()
        base_p_shifted: Period = shifted_period.base_period()
        self.assertEqual(
            base_p_offset,
            base_p_shifted,
            f"ShiftedPeriod: Base period mismatch: {base_p_offset} != {base_p_shifted}",
        )
        self.assertTrue(
            isinstance(base_p_shifted, base_p_class),
            f"Class mismatch: {base_p_shifted} is not of class {base_p_class}",
        )

    @parameterized.expand(_BASE_PARAMS)
    def test_good_base(
        self,
        name: Any,
        tz: datetime.tzinfo,
        test_dt: datetime.datetime,
        test_ordinal: int,
        base_p_class: Any,
    ) -> None:
        """Test ShiftedPeriod instances"""
        base_period: Period = Period.of_duration(name).with_tzinfo(tz)
        shifted_period: Period = base_period.with_origin(test_dt)
        ordinal: int = shifted_period.ordinal(test_dt)
        self.assertEqual(ordinal, 0)
        dt: datetime.datetime = shifted_period.datetime(ordinal)
        self.assertEqual(dt.tzinfo, shifted_period.tzinfo)
        self.assertEqual(dt, test_dt)
        base_p_shifted: Period = shifted_period.base_period()
        self.assertEqual(
            base_p_shifted,
            base_period,
            f"ShiftedPeriod: Base period mismatch: {base_p_shifted} != {base_period}",
        )
        self.assertTrue(
            isinstance(base_p_shifted, base_p_class),
            f"Class mismatch: {base_p_shifted} is not of class {base_p_class}",
        )


_DATE_TIME_PARSE = [
    ("1980", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-01", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-1", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-01-01", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-1-01", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-01-1", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-1-1", datetime.date(1980, 1, 1), datetime.time(0, 0, 0, 0, None)),
    ("1980-01-01 09", datetime.date(1980, 1, 1), datetime.time(9, 0, 0, 0, None)),
    ("1980-01-01T09", datetime.date(1980, 1, 1), datetime.time(9, 0, 0, 0, None)),
    ("1980-01-01 08:00", datetime.date(1980, 1, 1), datetime.time(8, 0, 0, 0, None)),
    ("1980-01-01T08:00", datetime.date(1980, 1, 1), datetime.time(8, 0, 0, 0, None)),
    (
        "1980-01-01 15:00:00",
        datetime.date(1980, 1, 1),
        datetime.time(15, 0, 0, 0, None),
    ),
    (
        "1980-01-01T15:00:00",
        datetime.date(1980, 1, 1),
        datetime.time(15, 0, 0, 0, None),
    ),
    ("1980-01-01 09:00:00", datetime.date(1980, 1, 1), datetime.time(9, 0, 0, 0, None)),
    ("1980-01-01T09:00:00", datetime.date(1980, 1, 1), datetime.time(9, 0, 0, 0, None)),
    (
        "1980-01-01 09:00:00Z",
        datetime.date(1980, 1, 1),
        datetime.time(9, 0, 0, 0, TZ_UTC),
    ),
    (
        "1980-01-01T09:00:00Z",
        datetime.date(1980, 1, 1),
        datetime.time(9, 0, 0, 0, TZ_UTC),
    ),
    (
        "1980-01-01 09:00:00+00:00",
        datetime.date(1980, 1, 1),
        datetime.time(9, 0, 0, 0, TZ_UTC),
    ),
    (
        "1980-01-01T09:00:00+00:00",
        datetime.date(1980, 1, 1),
        datetime.time(9, 0, 0, 0, TZ_UTC),
    ),
    (
        "2024-02-29T08:29:59",
        datetime.date(2024, 2, 29),
        datetime.time(8, 29, 59, 0, None),
    ),
    (
        "2024-02-29T08:29:59.5",
        datetime.date(2024, 2, 29),
        datetime.time(8, 29, 59, 500000, None),
    ),
    (
        "2024-02-29T08:29:59.543Z",
        datetime.date(2024, 2, 29),
        datetime.time(8, 29, 59, 543000, TZ_UTC),
    ),
]


class TestDateMatch(unittest.TestCase):
    """Test _date_match function"""

    @parameterized.expand(_DATE_TIME_PARSE)
    def test_good(self, text: Any, d: datetime.date, t: datetime.time) -> None:
        prefix: str = "d"
        regex = re.compile(p._datetime_regex(prefix))
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        dt: datetime.date = p._date_match(prefix, matcher)
        self.assertEqual(dt, d)


class TestTimeMatch(unittest.TestCase):
    """Test _time_match function"""

    @parameterized.expand(_DATE_TIME_PARSE)
    def test_good(self, text: Any, d: datetime.date, t: datetime.time) -> None:
        prefix: str = "d"
        regex = re.compile(p._datetime_regex(prefix))
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        tm: datetime.date = p._time_match(prefix, matcher)
        self.assertEqual(tm, t)


class TestTimezone(unittest.TestCase):
    """Test _timezone function"""

    @parameterized.expand(
        [
            (None, None),
            ("Z", datetime.timezone(datetime.timedelta(hours=0, minutes=0))),
            ("+00:00", datetime.timezone(datetime.timedelta(hours=0, minutes=0))),
            ("+0:00", datetime.timezone(datetime.timedelta(hours=0, minutes=0))),
            ("+00:0", datetime.timezone(datetime.timedelta(hours=0, minutes=0))),
            ("+0:0", datetime.timezone(datetime.timedelta(hours=0, minutes=0))),
            ("+01:00", datetime.timezone(datetime.timedelta(hours=1, minutes=0))),
            ("+01:30", datetime.timezone(datetime.timedelta(hours=1, minutes=30))),
            ("-01:30", datetime.timezone(datetime.timedelta(hours=-1, minutes=-30))),
        ]
    )
    def test_good(self, text: Any, tz: Optional[datetime.tzinfo]) -> None:
        self.assertEqual(tz, p._timezone(text))


class TestTzinfoMatch(unittest.TestCase):
    """Test _tzinfo_match function"""

    @parameterized.expand(_DATE_TIME_PARSE)
    def test_good(self, text: Any, d: datetime.date, t: datetime.time) -> None:
        prefix: str = "d"
        regex = re.compile(p._datetime_regex(prefix))
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        tz: datetime.date = p._tzinfo_match(prefix, matcher)
        self.assertEqual(tz, t.tzinfo)


class TestDatetimeMatch(unittest.TestCase):
    """Test _datetime_match function"""

    @parameterized.expand(_DATE_TIME_PARSE)
    def test_good(self, text: Any, d: datetime.date, t: datetime.time) -> None:
        prefix: str = "d"
        regex = re.compile(p._datetime_regex(prefix))
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        dt: datetime.date = p._datetime_match(prefix, matcher)
        answer: datetime.datetime = datetime.datetime.combine(date=d, time=t)
        self.assertEqual(dt, answer)


class TestMatchPeriod(unittest.TestCase):
    """Test _match_period function"""

    @parameterized.expand(
        [
            ("1Y", "P1Y"),
            ("1y", "P1Y"),
            ("2Y", "P2Y"),
            ("2y", "P2Y"),
            ("1M", "P1M"),
            ("1m", "P1M"),
            ("2M", "P2M"),
            ("2m", "P2M"),
        ]
    )
    def test_good(self, text: Any, iso: str) -> None:
        regex = re.compile(p._period_regex("period"))
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        period: Period = p._match_period(matcher)
        self.assertEqual(period, Period.of_iso_duration(iso))


class TestMatchPeriodOffset(unittest.TestCase):
    """Test _match_period_offset function"""

    @parameterized.expand(
        [
            ("1Y -- 1D", "P1Y+1D"),
            ("2Y -- 2M", "P2Y+2M"),
            ("1M -- 2D", "P1M+2D"),
            ("2M -- 5D", "P2M+5D"),
            ("2D -- 1D", "P2D+1D"),
            ("1D -- 12H", "P1D+12H"),
            ("1H -- T5M", "P1H+T5M"),
            ("T1M -- 5S", "PT1M+5S"),
            ("T1S -- 0.005S", "P1S+0.005S"),
        ]
    )
    def test_good(self, text: Any, duration: str) -> None:
        regex = re.compile(
            p._period_regex("period") + " -- " + p._period_regex("offset")
        )
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        period: Period = p._match_period_offset(matcher)
        self.assertEqual(period, Period.of_duration(duration))


class TestMatchDatetimePeriod(unittest.TestCase):
    """Test _match_datetime_period function"""

    @parameterized.expand(
        [
            (
                "1980 -- 1Y",
                Period.of_years(1).with_origin(datetime.datetime(1980, 1, 1)),
            ),
            (
                "1984-03-03 -- 1M",
                Period.of_months(1).with_origin(datetime.datetime(1984, 3, 3)),
            ),
            (
                "1980-04-04 -- 1D",
                Period.of_days(1).with_origin(datetime.datetime(1980, 4, 4)),
            ),
            (
                "1980-05-08 -- 12H",
                Period.of_hours(12).with_origin(datetime.datetime(1980, 5, 8)),
            ),
            (
                "1980-06-07 -- T5M",
                Period.of_minutes(5).with_origin(datetime.datetime(1980, 6, 7)),
            ),
            (
                "1980-07-06 -- 5S",
                Period.of_seconds(5).with_origin(datetime.datetime(1980, 7, 6)),
            ),
            (
                "1980-08-05 -- 0.005S",
                Period.of_microseconds(5000).with_origin(datetime.datetime(1980, 8, 5)),
            ),
        ]
    )
    def test_good(self, text: Any, match_period: Period) -> None:
        regex = re.compile(p._datetime_regex("d") + r" -- " + p._period_regex("period"))
        matcher: Optional[re.Match[str]] = regex.match(text)
        self.assertNotEqual(matcher, None)
        period: Period = p._match_datetime_period(matcher)
        self.assertEqual(period, match_period)


_REPR_RE_STR: str = (
    p._period_regex("period")
    + r","
    + r"(?:(?P<offset>\+)"
    + p._period_regex("offset")
    + r")?"
    + r","
    + r"(?:(?P<zone>[Zz]|(?:[+-]\d{1,2}(?::\d{1,2})?)))?"
    + r","
    + r"(?:(?P<shift>-?\d+))?"
)
_REPR_REGEX = re.compile(_REPR_RE_STR)


class TestMatchRepr(unittest.TestCase):
    """Test _match_repr function"""

    @parameterized.expand(
        [
            ("1Y,,,", Period.of_years(1)),
            ("5Y,,,", Period.of_years(5)),
            ("1Y,,Z,", Period.of_years(1).with_tzinfo(TZ_UTC)),
            ("5Y,,Z,", Period.of_years(5).with_tzinfo(TZ_UTC)),
            ("5Y,,+00:00,", Period.of_years(5).with_tzinfo(TZ_UTC)),
            ("1Y,+1M,,", Period.of_years(1).with_month_offset(1)),
            ("1Y,+1M,Z,", Period.of_years(1).with_month_offset(1).with_tzinfo(TZ_UTC)),
            ("1M,,,", Period.of_months(1)),
            ("6M,,,", Period.of_months(6)),
            ("1M,+1D,,", Period.of_months(1).with_day_offset(1)),
            ("6M,+5D,,", Period.of_months(6).with_day_offset(5)),
            ("1D,,,", Period.of_days(1)),
            ("9D,,,", Period.of_days(9)),
            ("1D,+3H,,", Period.of_days(1).with_hour_offset(3)),
            ("9D,+6H,,", Period.of_days(9).with_hour_offset(6)),
            ("1D,+3H,Z,", Period.of_days(1).with_hour_offset(3).with_tzinfo(TZ_UTC)),
            ("9D,+6H,Z,", Period.of_days(9).with_hour_offset(6).with_tzinfo(TZ_UTC)),
            ("1H,,,", Period.of_hours(1)),
            ("12H,,,", Period.of_hours(12)),
            ("1H,+T30M,,", Period.of_hours(1).with_minute_offset(30)),
            ("12H,+T10M,,", Period.of_hours(12).with_minute_offset(10)),
        ]
    )
    def test_good(self, text: Any, match_period: Period) -> None:
        matcher: Optional[re.Match[str]] = _REPR_REGEX.match(text)
        self.assertNotEqual(matcher, None)
        period: Period = p._match_repr(matcher)
        self.assertEqual(period, match_period)
        shift_date: datetime.datetime = datetime.datetime(1980, 1, 1)
        ordinal: int = period.ordinal(shift_date)
        origin_date: datetime.datetime = period.datetime(ordinal)
        shift_amount: int = 0 - ordinal
        matcher2: Optional[re.Match[str]] = _REPR_REGEX.match(f"{text}{shift_amount}")
        self.assertNotEqual(matcher2, None)
        shifted_period: Period = p._match_repr(matcher2)
        ordinal2: int = shifted_period.ordinal(origin_date)
        self.assertEqual(ordinal2, 0)


# ----------------------------------------------------------------------
S_YYYY: str = r"\d{4}"
S_YYYY_MM: str = r"\d{4}-\d{2}"
S_YYYY_MM_DD: str = r"\d{4}-\d{2}-\d{2}"
S_YYYY_MM_DD_HH: str = r"\d{4}-\d{2}-\d{2} \d{2}"
S_YYYY_MM_DD_HH_MM: str = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"
S_YYYY_MM_DD_HH_MM_SS: str = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
S_YYYY_MM_DD_HH_MM_SS_SSS: str = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}"
S_YYYY_MM_DD_HH_MM_SS_SSSSSS: str = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}"
S_TZ: str = r"([+-]\d{2}:\d{2}|Z)"

# Naive datetime string formats
RE_N_YYYY: re.Pattern = re.compile(r"^" + S_YYYY + r"$")
RE_N_YYYY_MM: re.Pattern = re.compile(r"^" + S_YYYY_MM + r"$")
RE_N_YYYY_MM_DD: re.Pattern = re.compile(r"^" + S_YYYY_MM_DD + r"$")
RE_N_YYYY_MM_DD_HH: re.Pattern = re.compile(r"^" + S_YYYY_MM_DD_HH + r"$")
RE_N_YYYY_MM_DD_HH_MM: re.Pattern = re.compile(r"^" + S_YYYY_MM_DD_HH_MM + r"$")
RE_N_YYYY_MM_DD_HH_MM_SS: re.Pattern = re.compile(r"^" + S_YYYY_MM_DD_HH_MM_SS + r"$")
RE_N_YYYY_MM_DD_HH_MM_SS_SSS: re.Pattern = re.compile(
    r"^" + S_YYYY_MM_DD_HH_MM_SS_SSS + r"$"
)
RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS: re.Pattern = re.compile(
    r"^" + S_YYYY_MM_DD_HH_MM_SS_SSSSSS + r"$"
)

# Aware datetime string formats
RE_A_YYYY_MM_DD_HH: re.Pattern = re.compile(r"^" + S_YYYY_MM_DD_HH + S_TZ + r"$")
RE_A_YYYY_MM_DD_HH_MM: re.Pattern = re.compile(r"^" + S_YYYY_MM_DD_HH_MM + S_TZ + r"$")
RE_A_YYYY_MM_DD_HH_MM_SS: re.Pattern = re.compile(
    r"^" + S_YYYY_MM_DD_HH_MM_SS + S_TZ + r"$"
)
RE_A_YYYY_MM_DD_HH_MM_SS_SSS: re.Pattern = re.compile(
    r"^" + S_YYYY_MM_DD_HH_MM_SS_SSS + S_TZ + r"$"
)
RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS: re.Pattern = re.compile(
    r"^" + S_YYYY_MM_DD_HH_MM_SS_SSSSSS + S_TZ + r"$"
)


@dataclass(frozen=True)
class Item:
    iso_duration: str
    step: int
    multiplier: int
    naive_class: type
    aware_class: type
    naive_regex: re.Pattern
    aware_regex: re.Pattern
    epoch_agnostic: bool


item_list: list[Item] = [
    Item(
        iso_duration="P1Y",
        step=p._STEP_MONTHS,
        multiplier=12,
        naive_class=p.YearPeriod,
        aware_class=p.YearPeriod,
        naive_regex=RE_N_YYYY,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P2Y",
        step=p._STEP_MONTHS,
        multiplier=2 * 12,
        naive_class=p.MultiYearPeriod,
        aware_class=p.MultiYearPeriod,
        naive_regex=RE_N_YYYY,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P3Y",
        step=p._STEP_MONTHS,
        multiplier=3 * 12,
        naive_class=p.MultiYearPeriod,
        aware_class=p.MultiYearPeriod,
        naive_regex=RE_N_YYYY,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P5Y",
        step=p._STEP_MONTHS,
        multiplier=5 * 12,
        naive_class=p.MultiYearPeriod,
        aware_class=p.MultiYearPeriod,
        naive_regex=RE_N_YYYY,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P10Y",
        step=p._STEP_MONTHS,
        multiplier=10 * 12,
        naive_class=p.MultiYearPeriod,
        aware_class=p.MultiYearPeriod,
        naive_regex=RE_N_YYYY,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P1M",
        step=p._STEP_MONTHS,
        multiplier=1,
        naive_class=p.MonthPeriod,
        aware_class=p.MonthPeriod,
        naive_regex=RE_N_YYYY_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P2M",
        step=p._STEP_MONTHS,
        multiplier=2,
        naive_class=p.MultiMonthPeriod,
        aware_class=p.MultiMonthPeriod,
        naive_regex=RE_N_YYYY_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P3M",
        step=p._STEP_MONTHS,
        multiplier=3,
        naive_class=p.MultiMonthPeriod,
        aware_class=p.MultiMonthPeriod,
        naive_regex=RE_N_YYYY_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P4M",
        step=p._STEP_MONTHS,
        multiplier=4,
        naive_class=p.MultiMonthPeriod,
        aware_class=p.MultiMonthPeriod,
        naive_regex=RE_N_YYYY_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P5M",
        step=p._STEP_MONTHS,
        multiplier=5,
        naive_class=p.MultiMonthPeriod,
        aware_class=p.MultiMonthPeriod,
        naive_regex=RE_N_YYYY_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P100M",
        step=p._STEP_MONTHS,
        multiplier=100,
        naive_class=p.MultiMonthPeriod,
        aware_class=p.MultiMonthPeriod,
        naive_regex=RE_N_YYYY_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P1D",
        step=p._STEP_SECONDS,
        multiplier=86_400,
        naive_class=p.NaiveDayPeriod,
        aware_class=p.AwareDayPeriod,
        naive_regex=RE_N_YYYY_MM_DD,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P2D",
        step=p._STEP_SECONDS,
        multiplier=2 * 86_400,
        naive_class=p.NaiveMultiDayPeriod,
        aware_class=p.AwareMultiDayPeriod,
        naive_regex=RE_N_YYYY_MM_DD,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P1H",
        step=p._STEP_SECONDS,
        multiplier=60 * 60,
        naive_class=p.MultiHourPeriod,
        aware_class=p.MultiHourPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P2H",
        step=p._STEP_SECONDS,
        multiplier=2 * 60 * 60,
        naive_class=p.MultiHourPeriod,
        aware_class=p.MultiHourPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P3H",
        step=p._STEP_SECONDS,
        multiplier=3 * 60 * 60,
        naive_class=p.MultiHourPeriod,
        aware_class=p.MultiHourPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P4H",
        step=p._STEP_SECONDS,
        multiplier=4 * 60 * 60,
        naive_class=p.MultiHourPeriod,
        aware_class=p.MultiHourPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P5H",
        step=p._STEP_SECONDS,
        multiplier=5 * 60 * 60,
        naive_class=p.MultiHourPeriod,
        aware_class=p.MultiHourPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH,
        aware_regex=RE_A_YYYY_MM_DD_HH,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="PT1M",
        step=p._STEP_SECONDS,
        multiplier=60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="PT2M",
        step=p._STEP_SECONDS,
        multiplier=2 * 60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="PT3M",
        step=p._STEP_SECONDS,
        multiplier=3 * 60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="PT4M",
        step=p._STEP_SECONDS,
        multiplier=4 * 60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="PT5M",
        step=p._STEP_SECONDS,
        multiplier=5 * 60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="PT6M",
        step=p._STEP_SECONDS,
        multiplier=6 * 60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="PT7M",
        step=p._STEP_SECONDS,
        multiplier=7 * 60,
        naive_class=p.NaiveMultiMinutePeriod,
        aware_class=p.AwareMultiMinutePeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P1S",
        step=p._STEP_SECONDS,
        multiplier=1,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P2S",
        step=p._STEP_SECONDS,
        multiplier=2,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P3S",
        step=p._STEP_SECONDS,
        multiplier=3,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P4S",
        step=p._STEP_SECONDS,
        multiplier=4,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P5S",
        step=p._STEP_SECONDS,
        multiplier=5,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P6S",
        step=p._STEP_SECONDS,
        multiplier=6,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P7S",
        step=p._STEP_SECONDS,
        multiplier=7,
        naive_class=p.NaiveMultiSecondPeriod,
        aware_class=p.AwareMultiSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P0.1S",
        step=p._STEP_MICROSECONDS,
        multiplier=100_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.2S",
        step=p._STEP_MICROSECONDS,
        multiplier=200_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.3S",
        step=p._STEP_MICROSECONDS,
        multiplier=300_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.7S",
        step=p._STEP_MICROSECONDS,
        multiplier=700_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P0.01S",
        step=p._STEP_MICROSECONDS,
        multiplier=10_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.02S",
        step=p._STEP_MICROSECONDS,
        multiplier=20_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.03S",
        step=p._STEP_MICROSECONDS,
        multiplier=30_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.07S",
        step=p._STEP_MICROSECONDS,
        multiplier=70_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P0.001S",
        step=p._STEP_MICROSECONDS,
        multiplier=1_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.002S",
        step=p._STEP_MICROSECONDS,
        multiplier=2_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.003S",
        step=p._STEP_MICROSECONDS,
        multiplier=3_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.007S",
        step=p._STEP_MICROSECONDS,
        multiplier=7_000,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSS,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P0.0001S",
        step=p._STEP_MICROSECONDS,
        multiplier=100,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.0002S",
        step=p._STEP_MICROSECONDS,
        multiplier=200,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.0003S",
        step=p._STEP_MICROSECONDS,
        multiplier=300,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.0007S",
        step=p._STEP_MICROSECONDS,
        multiplier=700,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P0.00001S",
        step=p._STEP_MICROSECONDS,
        multiplier=10,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.00002S",
        step=p._STEP_MICROSECONDS,
        multiplier=20,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.00003S",
        step=p._STEP_MICROSECONDS,
        multiplier=30,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.00007S",
        step=p._STEP_MICROSECONDS,
        multiplier=70,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=False,
    ),
    Item(
        iso_duration="P0.000001S",
        step=p._STEP_MICROSECONDS,
        multiplier=1,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.000002S",
        step=p._STEP_MICROSECONDS,
        multiplier=2,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.000003S",
        step=p._STEP_MICROSECONDS,
        multiplier=3,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=True,
    ),
    Item(
        iso_duration="P0.000007S",
        step=p._STEP_MICROSECONDS,
        multiplier=7,
        naive_class=p.NaiveMicroSecondPeriod,
        aware_class=p.AwareMicroSecondPeriod,
        naive_regex=RE_N_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        aware_regex=RE_A_YYYY_MM_DD_HH_MM_SS_SSSSSS,
        epoch_agnostic=False,
    ),
]

PARAMS_ITEM: list[tuple[str, Item]] = [
    (f"{item.iso_duration}", item) for item in item_list
]


def _create_by_iso_duration(iso_duration: str) -> list[Period]:
    return [
        Period.of_iso_duration(iso_duration),
        Period.of_iso_duration(iso_duration.upper()),
        Period.of_iso_duration(iso_duration.lower()),
    ]


def _create_by_duration(duration: str) -> list[Period]:
    return [
        Period.of_duration(duration),
        Period.of_duration(duration.upper()),
        Period.of_duration(duration.lower()),
    ]


def _create_by_date_duration(duration: str) -> list[Period]:
    base_period: Period = Period.of_iso_duration(duration)
    base_ordinal: int = base_period.ordinal(datetime.datetime(1900, 1, 1, 0, 0, 0))
    base_date: datetime.datetime = base_period.datetime(base_ordinal)
    date_duration: str = base_date.isoformat() + "/" + duration
    return [
        Period.of_date_and_duration(date_duration).without_ordinal_shift(),
        Period.of_date_and_duration(date_duration.upper()).without_ordinal_shift(),
        Period.of_date_and_duration(date_duration.lower()).without_ordinal_shift(),
    ]


def _create_by_step_and_multiplier(item: Item) -> list[Period]:
    if item.step == p._STEP_MONTHS:
        return Period.of_months(p.multiplier)
    elif item.step == p._STEP_SECONDS:
        return Period.of_seconds(p.multiplier)
    elif item.step == p._STEP_MICROSECONDS:
        return Period.of_microseconds(p.multiplier)
    else:
        raise AssertionError()


def _create_all_no_tz(item: Item) -> list[Period]:
    result: list[Period] = []
    result.extend(_create_by_iso_duration(item.iso_duration))
    result.extend(_create_by_duration(item.iso_duration))
    result.extend(_create_by_date_duration(item.iso_duration))
    result.append(
        p._get_shifted_period(
            p.Properties.of_step_and_multiplier(item.step, item.multiplier)
        )
    )
    return result


def _create_all_with_tz(item: Item, tz: datetime.tzinfo) -> list[Period]:
    return [period.with_tzinfo(tz) for period in _create_all_no_tz(item)]


def _get_test_ordinal_list(period: Period) -> list[int]:
    min_ordinal: int = period.min_ordinal
    max_ordinal: int = period.max_ordinal
    ordinal_step: int = (1 + (max_ordinal - min_ordinal)) // 100
    ordinal_set: set[int] = set(
        [o for o in range(min_ordinal, max_ordinal, ordinal_step)]
    )
    ordinal_set.add(max_ordinal)
    return sorted(ordinal_set)


class TestPeriodItems(unittest.TestCase):
    @parameterized.expand(PARAMS_ITEM)
    def test_create(self, _: Any, item: Item) -> None:
        """Check all creation methods produce the same Period object"""
        naive_list: list[Period] = _create_all_no_tz(item)
        naive_set: set[Period] = set(naive_list)
        self.assertEqual(len(naive_set), 1, f"Multiple periods in set: {naive_set}")

        utc_list: list[Period] = _create_all_with_tz(item, TZ_UTC)
        utc_set: set[Period] = set(utc_list)
        self.assertEqual(len(utc_set), 1, f"Multiple periods in set: {utc_set}")

    @parameterized.expand(PARAMS_ITEM)
    def test_repr_create(self, _: Any, item: Item) -> None:
        """Check repr creation method"""
        p1: Period = Period.of_iso_duration(item.iso_duration)
        repr_string: str = repr(p1)
        p2: Period = Period.of_repr(repr_string)
        self.assertEqual(p1, p2, f"Repr period mismatch: {p1} {p2}")

    @parameterized.expand(PARAMS_ITEM)
    def test_step(self, _: Any, item: Item) -> None:
        """Check the step property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        self.assertEqual(
            period._properties.step, item.step, f"Step mismatch: {period} != {item}"
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_multiplier(self, _: Any, item: Item) -> None:
        """Check the multiplier property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        self.assertEqual(
            period._properties.multiplier,
            item.multiplier,
            f"Multiplier mismatch: {period} != {item}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_naive_class(self, _: Any, item: Item) -> None:
        """Check the naive class"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        self.assertTrue(
            isinstance(period, item.naive_class),
            f"Naive class mismatch: {period} != {item}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_aware_class(self, _: Any, item: Item) -> None:
        """Check the aware class"""
        period: Period = Period.of_iso_duration(item.iso_duration).with_tzinfo(TZ_UTC)
        self.assertTrue(
            isinstance(period, item.aware_class),
            f"Aware class mismatch: {period} != {item}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_min_ordinal(self, _: Any, item: Item) -> None:
        """Check the min_ordinal property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        min_ordinal: int = period.min_ordinal
        dt: datetime.datetime = period.datetime(min_ordinal)
        self.assertEqual(
            min_ordinal,
            period.ordinal(dt),
            f"min_ordinal error: {min_ordinal} != {period.ordinal( dt )}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_max_ordinal(self, _: Any, item: Item) -> None:
        """Check the max_ordinal property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        max_ordinal: int = period.max_ordinal
        dt: datetime.datetime = period.datetime(max_ordinal)
        self.assertEqual(
            max_ordinal,
            period.ordinal(dt),
            f"max_ordinal error: {max_ordinal} != {period.ordinal( dt )}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_sample_ordinals(self, _: Any, item: Item) -> None:
        """Check some sample ordinals"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        for ordinal in _get_test_ordinal_list(period):
            for tzinfo in [None, TZ_UTC]:
                dt: datetime.datetime = period.datetime(ordinal)
                naive_ordinal: int = period.ordinal(dt)
                self.assertEqual(
                    ordinal,
                    naive_ordinal,
                    f"ordinal error: {ordinal} != {naive_ordinal}",
                )
                tz_dt: datetime.datetime = dt.replace(tzinfo=tzinfo)
                tz_ordinal: int = period.ordinal(tz_dt)
                self.assertEqual(
                    ordinal,
                    tz_ordinal,
                    f"ordinal error: {tzinfo}: {ordinal} != {tz_ordinal}",
                )

    @parameterized.expand(PARAMS_ITEM)
    def test_naive_format(self, _: Any, item: Item) -> None:
        """Check naive date format"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        fmt: Callable[[datetime.datetime], str] = period.naive_formatter(" ")
        for ordinal in _get_test_ordinal_list(period):
            dt: datetime.datetime = period.datetime(ordinal)
            dt_str: str = fmt(dt)
            self.assertTrue(
                isinstance(item.naive_regex.fullmatch(dt_str), re.Match),
                f"Format error: {period}: {dt_str}",
            )

    @parameterized.expand(PARAMS_ITEM)
    def test_utc_format(self, _: Any, item: Item) -> None:
        """Check UTC date format"""
        period: Period = Period.of_iso_duration(item.iso_duration).with_tzinfo(TZ_UTC)
        fmt: Callable[[datetime.datetime], str] = period.aware_formatter(" ")
        for ordinal in _get_test_ordinal_list(period):
            dt: datetime.datetime = period.datetime(ordinal)
            dt_str: str = fmt(dt)
            self.assertTrue(
                isinstance(item.aware_regex.fullmatch(dt_str), re.Match),
                f"Format error: {period}: {dt_str}",
            )
            self.assertTrue(dt_str.endswith("Z"), f"Format error: {period}: {dt_str}")

    @parameterized.expand(PARAMS_ITEM)
    def test_tzinfo_property(self, _: Any, item: Item) -> None:
        """Test tzinfo property"""
        for tzinfo in [None, TZ_UTC]:
            period: Period = Period.of_iso_duration(item.iso_duration).with_tzinfo(
                tzinfo
            )
            self.assertEqual(
                tzinfo,
                period.tzinfo,
                f"tzinfo property error: {tzinfo}: {period.tzinfo}",
            )

    @parameterized.expand(PARAMS_ITEM)
    def test_timedelta_property(self, _: Any, item: Item) -> None:
        """Test timedelta property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        if item.step == p._STEP_MONTHS:
            self.assertEqual(
                period.timedelta,
                None,
                f"Month step timedelta property error: {period.timedelta}",
            )
        else:
            self.assertTrue(
                isinstance(period.timedelta, datetime.timedelta),
                f"Second/microsecond step timedelta property error: {period.timedelta}",
            )
            period2: Period = Period.of_timedelta(period.timedelta)
            self.assertEqual(
                period,
                period2,
                f"Second/microsecond step timedelta property error: {period.timedelta}",
            )

    @parameterized.expand(PARAMS_ITEM)
    def test_pl_interval_property(self, _: Any, item: Item) -> None:
        """Test pl_interval property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        self.assertTrue(
            isinstance(period.pl_interval, str),
            f"pl_interval property error: {period.pl_interval}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_pl_offset_property(self, _: Any, item: Item) -> None:
        """Test pl_offset property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        self.assertTrue(
            isinstance(period.pl_offset, str),
            f"pl_offset property error: {period.pl_offset}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_epoch_agnostic1(self, _: Any, item: Item) -> None:
        """Test is_epoch_agnostic property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        self.assertEqual(
            period.is_epoch_agnostic(),
            item.epoch_agnostic,
            f"epoch_agnostic error: {period}",
        )

    @parameterized.expand(PARAMS_ITEM)
    def test_epoch_agnostic2(self, _: Any, item: Item) -> None:
        """Test is_epoch_agnostic property"""
        period: Period = Period.of_iso_duration(item.iso_duration)
        years = range(50, 9950)
        match_years: int = 0
        nomatch_years: int = 0
        for year in years:
            epoch_datetime = datetime.datetime(year, 1, 1, 0, 0, 0)
            ordinal: int = period.ordinal(epoch_datetime)
            epoch_datetime1 = period.datetime(ordinal)
            period2: Period = period.with_origin(epoch_datetime).without_ordinal_shift()
            epoch_datetime2 = period.datetime(ordinal)
            epoch_datetime3 = period2.datetime(ordinal)
            if period == period2:
                match_years += 1
            else:
                nomatch_years += 1
        if period.is_epoch_agnostic():
            self.assertTrue(
                match_years == len(years), f"Period {period} is not epoch agnostic"
            )
        else:
            self.assertTrue(
                match_years < len(years), f"Period {period} is epoch agnostic"
            )


# ----------------------------------------------------------------------
_EQUIVALENT_PERIODS = [
    (
        "P5Y",
        [
            Period.of_iso_duration("P5Y"),
            Period.of_duration("P5Y"),
            Period.of_repr("P5Y[]"),
            Period.of_years(5),
            Period.of_months(60),
            Period.of_years(1).with_multiplier(5),
            Period.of_date_and_duration("1980-01-01/P5Y").without_ordinal_shift(),
            Period.of_date_and_duration("2024-10-31 10:37:33.059404+00:00/P5Y")
            .with_tzinfo(None)
            .without_offset()
            .without_ordinal_shift(),
        ],
    ),
    (
        "P1Y",
        [
            Period.of_iso_duration("P1Y"),
            Period.of_duration("P1Y"),
            Period.of_repr("P1Y[]"),
            Period.of_years(1),
            Period.of_months(12),
            Period.of_months(1).with_multiplier(12),
            Period.of_date_and_duration("1981-01-01/P1Y").without_ordinal_shift(),
        ],
    ),
    (
        "P6M",
        [
            Period.of_iso_duration("P6M"),
            Period.of_duration("P6M"),
            Period.of_repr("P6M[]"),
            Period.of_months(6),
            Period.of_months(1).with_multiplier(6),
            Period.of_date_and_duration("1981-07-01/P6M").without_ordinal_shift(),
        ],
    ),
    (
        "P1M",
        [
            Period.of_iso_duration("P1M"),
            Period.of_duration("P1M"),
            Period.of_repr("P1M[]"),
            Period.of_months(1),
            Period.of_date_and_duration("1981-12-01/P1M").without_ordinal_shift(),
        ],
    ),
    (
        "P5D",
        [
            Period.of_iso_duration("P5D"),
            Period.of_duration("P5D"),
            Period.of_repr("P5D[]"),
            Period.of_days(5),
            Period.of_hours(5 * 24),
            Period.of_minutes(5 * 24 * 60),
            Period.of_seconds(5 * 24 * 60 * 60),
            Period.of_microseconds(5 * 24 * 60 * 60 * 1_000_000),
            Period.of_timedelta(datetime.timedelta(days=5)),
            Period.of_days(1).with_multiplier(5),
            Period.of_hours(1)
            .with_multiplier(2)
            .with_multiplier(3)
            .with_multiplier(4)
            .with_multiplier(5),
            Period.of_hours(1)
            .with_multiplier(5)
            .with_multiplier(4)
            .with_multiplier(3)
            .with_multiplier(2),
        ],
    ),
    (
        "P6H",
        [
            Period.of_iso_duration("P6H"),
            Period.of_duration("P6H"),
            Period.of_repr("P6H[]"),
            Period.of_hours(6),
            Period.of_minutes(6 * 60),
            Period.of_seconds(6 * 60 * 60),
            Period.of_microseconds(6 * 60 * 60 * 1_000_000),
            Period.of_timedelta(datetime.timedelta(hours=6)),
            Period.of_hours(1).with_multiplier(6),
            Period.of_date_and_duration(
                "1982-04-17T00:00:00/P6H"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1982-04-17T06:00:00/P6H"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1982-04-17T12:00:00/P6H"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1982-04-17T18:00:00/P6H"
            ).without_ordinal_shift(),
        ],
    ),
    (
        "PT5M",
        [
            Period.of_iso_duration("PT5M"),
            Period.of_duration("PT5M"),
            Period.of_repr("PT5M[]"),
            Period.of_minutes(5),
            Period.of_seconds(5 * 60),
            Period.of_microseconds(5 * 60 * 1_000_000),
            Period.of_timedelta(datetime.timedelta(minutes=5)),
            Period.of_minutes(1).with_multiplier(5),
            Period.of_date_and_duration(
                "1985-11-23T17:40:00/PT5M"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1985-11-23T17:45:00/PT5M"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1985-11-23T17:50:00/PT5M"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1985-11-23T17:55:00/PT5M"
            ).without_ordinal_shift(),
        ],
    ),
    (
        "P5S",
        [
            Period.of_iso_duration("P5S"),
            Period.of_duration("P5S"),
            Period.of_repr("P5S[]"),
            Period.of_seconds(5),
            Period.of_microseconds(5 * 1_000_000),
            Period.of_timedelta(datetime.timedelta(seconds=5)),
            Period.of_seconds(1).with_multiplier(5),
            Period.of_date_and_duration(
                "1987-09-05T17:23:25/P5S"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1987-09-05T17:23:30/P5S"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1987-09-05T17:23:35/P5S"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1987-09-05T17:23:40/P5S"
            ).without_ordinal_shift(),
        ],
    ),
    (
        "P0.000005S",
        [
            Period.of_iso_duration("P0.000005S"),
            Period.of_duration("P0.000005S"),
            Period.of_repr("P0.000005S[]"),
            Period.of_microseconds(5),
            Period.of_timedelta(datetime.timedelta(microseconds=5)),
            Period.of_microseconds(1).with_multiplier(5),
            Period.of_date_and_duration(
                "1989-08-07T16:53:45.975310/P0.000005S"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1989-08-07T16:53:45.975315/P0.000005S"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1989-08-07T16:53:45.975320/P0.000005S"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1989-08-07T16:53:45.975325/P0.000005S"
            ).without_ordinal_shift(),
        ],
    ),
    (
        "P1D+9H",
        [
            Period.of_iso_duration("P1D").with_hour_offset(9),
            Period.of_duration("P1D+9H"),
            Period.of_repr("P1D+9H[]"),
            Period.of_days(1).with_hour_offset(9),
            Period.of_days(1).with_minute_offset(9 * 60),
            Period.of_days(1).with_second_offset(9 * 60 * 60),
            Period.of_days(1).with_microsecond_offset(9 * 60 * 60 * 1_000_000),
            Period.of_hours(1).with_multiplier(24).with_hour_offset(9),
            Period.of_days(1)
            .with_origin(datetime.datetime(1984, 5, 6, 9, 0, 0, 0))
            .without_ordinal_shift(),
            Period.of_date_and_duration(
                "1991-05-06T09:00:00.000000/P1D"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1991-05-07T09:00:00.000000/P1D"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1991-05-08T09:00:00.000000/P1D"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1991-05-09T09:00:00.000000/P1D"
            ).without_ordinal_shift(),
        ],
    ),
    (
        "P1Y+9M9H",
        [
            Period.of_iso_duration("P1Y").with_hour_offset(9).with_month_offset(9),
            Period.of_iso_duration("P1Y").with_month_offset(9).with_hour_offset(9),
            Period.of_duration("P1Y+9M9H"),
            Period.of_repr("P1Y+9M9H[]"),
            Period.of_years(1).with_hour_offset(9).with_month_offset(9),
            Period.of_years(1).with_month_offset(9).with_hour_offset(9),
            Period.of_years(1)
            .with_origin(datetime.datetime(1984, 10, 1, 9, 0, 0, 0))
            .without_ordinal_shift(),
            Period.of_date_and_duration(
                "1991-10-01T09:00:00.000000/P1Y"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1992-10-01T09:00:00.000000/P1Y"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1993-10-01T09:00:00.000000/P1Y"
            ).without_ordinal_shift(),
            Period.of_date_and_duration(
                "1994-10-01T09:00:00.000000/P1Y"
            ).without_ordinal_shift(),
            Period.of_date_and_duration("2024-10-31 10:37:33.059404+00:00/P1M")
            .with_tzinfo(None)
            .without_offset()
            .without_ordinal_shift()
            .with_multiplier(12)
            .with_month_offset(9)
            .with_hour_offset(9),
        ],
    ),
    (
        "P1Y (UTC)",
        [
            Period.of_iso_duration("P1Y").with_tzinfo(TZ_UTC),
            Period.of_duration("P1Y").with_tzinfo(TZ_UTC),
            Period.of_repr("P1Y[]").with_tzinfo(TZ_UTC),
            Period.of_repr("P1Y[+00:00]"),
            Period.of_repr("P1Y[-00:00]"),
            Period.of_years(1).with_tzinfo(TZ_UTC),
            Period.of_years(1)
            .with_origin(datetime.datetime(1984, 1, 1, 0, 0, 0, 0, TZ_UTC))
            .without_ordinal_shift(),
            Period.of_date_and_duration(
                "1981-01-01T00:00:00+00:00/P1Y"
            ).without_ordinal_shift(),
            Period.of_date_and_duration("2024-10-31 10:37:33.059404+00:00/P1M")
            .without_offset()
            .without_ordinal_shift()
            .with_multiplier(12),
        ],
    ),
]


class TestEquivalentPeriods(unittest.TestCase):
    """Test equivalent periods"""

    @parameterized.expand(_EQUIVALENT_PERIODS)
    def test_equivalent(self, name: Any, period_list: list[Period]) -> None:
        """Test all periods in list are the same period"""
        period_set: set[Period] = set(period_list)
        self.assertEqual(
            len(period_set), 1, f"Multiple periods in set: {name}: {period_set}"
        )


# ----------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
