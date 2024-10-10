"""
Period: A period of time.

More specifically, a period splits the Gregorian calendar timeline
into sequential intervals, each interval being identified by an
integer "ordinal" value.

The Period class provides two methods that provide a mapping
between datetime objects and period ordinals:

    def ordinal( self , datetime_obj: datetime.datetime ) -> int:

    def datetime( self , ordinal: int ) -> datetime.datetime:

The ordinal() method returns the ordinal of the interval within
which the datetime lies.
The datetime() method returns the datetime of the start of the
interval.

The datetime() method will return the ordinal value "n" for all
datetime objects where:
    datetime >= period.datetime( n ) and
    datetime < period.datetime( n+1 )

The Period abstract class is the public face of this module. Ideally
all external code will only ever use this class and the methods and
properties that it exposes.  Everything else in this module should
be considered private and subject to change.

Example usage:

    To create a Period object use one of the static methods of the
    Period class:

       p1d = Period.of_days( 1 )
       p1d = Period.of_iso_duration( "P1D" )
       p1m = Period.of_months( 1 )
       pt1m = Period.of_minutes( 1 )
       pt15m = Period.of_minutes( 15 )
       pt15m = Period.of_iso_duration( "PT15M" )
       WATER_YEAR: Period = ( Period.of_years( 1 )
           .with_month_offset( 9 )
           .with_hour_offset( 9 ) )

"""

import calendar
import datetime as dt
import re
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Optional,
    override,
)


def _naive(datetime_obj: dt.datetime) -> dt.datetime:
    """Return a datetime object without a tzinfo

    Args:
        datetime_obj: The input datetime object

    Returns:
        A datetime object where tzinfo is None
    """
    if datetime_obj.tzinfo is None:
        return datetime_obj
    return datetime_obj.replace(tzinfo=None)


def _gregorian_seconds(datetime_obj: dt.datetime) -> int:
    """Calculate number of seconds since the "day epoch"

    The "day epoch" is midnight on day "0" of the proleptic
    Gregorian ordinal (i.e. midnight at the start of the day before
    January 1 of year 1).  See the toordinal() and fromordinal()
    methods of the datetime.date and datetime.datetime classes.

    Args:
        datetime_obj: The input datetime object

    Returns:
        The number of seconds between the supplied datetime and the "day epoch"
    """
    return (datetime_obj.toordinal() * 86_400) + (
        (datetime_obj.hour * 60 + datetime_obj.minute) * 60 + datetime_obj.second
    )


def _period_regex(prefix: str) -> str:
    """Return a regular expression string for matching an ISO 8601 duration
    (but without the initial "P" character)

    The _period_match function can be used to extract data from such
    a regular expression.

    Args:
        prefix: The name prefix used for Python regex group names

    Returns:
        A string containing a regular expression that can be used to
        parse an ISO 8601 duration
    """
    return (
        rf"(?:(?P<{prefix}_years>\d+)[Yy])?"
        rf"(?:(?P<{prefix}_months>\d+)[Mm])?"
        rf"(?:(?P<{prefix}_days>\d+)[Dd])?"
        r"(?:[Tt]?"
        rf"(?:(?P<{prefix}_hours>\d+)[Hh])?"
        rf"(?:(?P<{prefix}_minutes>\d+)[Mm])?"
        rf"(?:(?P<{prefix}_seconds>\d+)"
        rf"(?:\.(?P<{prefix}_microseconds>\d{{1,6}}))?"
        r"[Ss])?"
        r")?"
    )


# ------------------------------------------------------------------------------
# DateTimeAdjusters
# ------------------------------------------------------------------------------
def year_shift(date_time: dt.datetime, shift_amount: int) -> dt.datetime:
    """Shift a datetime object by a given number of years (+ve or -ve)

    Args:
        date_time: The date_time object to be shifted
        shift_amount: The number of years by which to shift date_time

    Returns:
        A datetime object
    """
    if shift_amount == 0:
        return date_time
    year = date_time.year
    new_year = year + shift_amount
    day = date_time.day
    if day <= 28:
        return date_time.replace(year=new_year)
    month = date_time.month
    days_in_month = calendar.monthrange(new_year, month)[1]
    return date_time.replace(year=new_year, day=min(day, days_in_month))


def month_shift(date_time: dt.datetime, shift_amount: int) -> dt.datetime:
    """Shift a datetime object by a given number of months (+ve or -ve)

    Args:
        date_time: The date_time object to be shifted
        shift_amount: The number of months by which to shift date_time

    Returns:
        A datetime object
    """
    if shift_amount == 0:
        return date_time
    year = date_time.year
    month = date_time.month
    y_m = year * 12 + month - 1
    new_y_m = y_m + shift_amount
    new_year, new_month0 = divmod(new_y_m, 12)
    new_month = new_month0 + 1
    day = date_time.day
    if day <= 28:
        return date_time.replace(year=new_year, month=new_month)
    days_in_month = calendar.monthrange(new_year, new_month)[1]
    return date_time.replace(year=new_year, month=new_month, day=min(day, days_in_month))


@dataclass(frozen=True)
class DateTimeAdjusters:
    """Two functions that can be used to adjust a datetime
    object backwards or forwards in time.
    """

    retreat: Callable[[dt.datetime], dt.datetime]
    advance: Callable[[dt.datetime], dt.datetime]

    @staticmethod
    def of_month_offset(month_offset: int) -> Optional["DateTimeAdjusters"]:
        """Return a DateTimeAdjusters object for a given month offset,
        or return None if the month offset is 0

        Args:
            month_offset: The number of months the DateTimeAdjusters object
                          is to shift datetime objects

        Returns:
            The required DateTimeAdjusters object, or None if there is no shift
        """
        if month_offset == 0:
            return None
        years, months_in_year = divmod(month_offset, 12)
        if months_in_year == 0:
            return DateTimeAdjusters(
                retreat=lambda datetime_obj: year_shift(datetime_obj, 0 - years),
                advance=lambda datetime_obj: year_shift(datetime_obj, years),
            )
        return DateTimeAdjusters(
            retreat=lambda datetime_obj: month_shift(datetime_obj, 0 - month_offset),
            advance=lambda datetime_obj: month_shift(datetime_obj, month_offset),
        )

    @staticmethod
    def of_microsecond_offset(microsecond_offset: int) -> Optional["DateTimeAdjusters"]:
        """Return a DateTimeAdjusters object for a given microsecond offset,
        or return None if the microsecond offset is 0

        Args:
            microsecond_offset: The number of microseconds the DateTimeAdjusters object
                                is to shift datetime objects

        Returns:
            The required DateTimeAdjusters object, or None if there is no shift
        """
        if microsecond_offset == 0:
            return None
        timedelta = dt.timedelta(microseconds=microsecond_offset)
        return DateTimeAdjusters(
            retreat=lambda datetime_obj: datetime_obj - timedelta, advance=lambda datetime_obj: datetime_obj + timedelta
        )

    @staticmethod
    def of_offsets(month_offset: int, microsecond_offset: int) -> "DateTimeAdjusters":
        """Return a DateTimeAdjusters object for a given month offset
        and microsecond offset.
        At least one of the arguments must be non-zero

        Args:
            month_offset: The number of months the DateTimeAdjusters object
                          is to shift datetime objects
            microsecond_offset: The number of microseconds the DateTimeAdjusters object
                                is to shift datetime objects

        Returns:
            The required DateTimeAdjusters object
        """
        month_adjusters = DateTimeAdjusters.of_month_offset(month_offset)
        microsecond_adjusters = DateTimeAdjusters.of_microsecond_offset(microsecond_offset)
        if (microsecond_adjusters is not None) and (month_adjusters is None):
            return microsecond_adjusters
        if (month_adjusters is not None) and (microsecond_adjusters is None):
            return month_adjusters
        if (month_adjusters is None) or (microsecond_adjusters is None):
            raise AssertionError()
        m_retreat = month_adjusters.retreat
        m_advance = month_adjusters.advance
        s_retreat = microsecond_adjusters.retreat
        s_advance = microsecond_adjusters.advance
        # It's an arbitrary choice as to whether the month shift is done
        # before or after the microsecond shift, but a retreat should
        # 'undo' an advance, and vice-versa, so the order needs to be
        # swapped.  Shifting by a month and shifting by a microsecond
        # are not commutative operations. The order matters.
        # i.e. An advance of 1 month followed by 1 day:
        #     Apr 30 (+1 month) -> May 30 (+1 day) -> May 31
        # The retreat must be day followed by month to get back to where we started:
        #     May 31 (-1 day) -> May 30 (-1 month) -> Apr 30
        # A retreat of 1 month followed by one days yields a different date:
        #     May 31 (-1 month) -> Apr 30 (-1 day) -> Apr 29
        return DateTimeAdjusters(
            retreat=lambda datetime_obj: m_retreat(s_retreat(datetime_obj)),
            advance=lambda datetime_obj: s_advance(m_advance(datetime_obj)),
        )

    def __post_init__(self) -> None:
        if (self.retreat is None) or (self.advance is None):
            raise AssertionError()


# ------------------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------------------
def _second_string(seconds: int, microseconds: int) -> str:
    """Convert seconds and microseconds (0-999_999)
    to a string that can be used to format an ISO 8601
    duration string.

    Args:
        seconds: The number of seconds
        microseconds: The number of microseconds

    Returns:
        A string representing seconds and microseconds that
        can be used in an ISO 8601 duration string
    """
    if microseconds == 0:
        return str(seconds)
    return f"{seconds}.{microseconds:06}".rstrip("0")


def _append_second_elems(elems: list[str], seconds: int, microseconds: int) -> list[str]:
    """Append total seconds and microseconds
    to a list of strings that, when joined, produce an ISO 8601
    duration string.

    Args:
        elems: The list of strings
        seconds: The number of seconds
        microseconds: The number of microseconds

    Returns:
        The amended list of strings
    """
    days, seconds_in_day = divmod(seconds, 86_400)
    if days > 0:
        elems.append(f"{days}D")
    if (seconds_in_day > 0) or (microseconds > 0):
        elems.append("T")
        hours, seconds_in_hour = divmod(seconds_in_day, 3_600)
        if hours > 0:
            elems.append(f"{hours}H")
        minutes, seconds_in_minute = divmod(seconds_in_hour, 60)
        if minutes > 0:
            elems.append(f"{minutes}M")
        if (seconds_in_minute > 0) or (microseconds > 0):
            elems.append(_second_string(seconds_in_minute, microseconds))
            elems.append("S")
    return elems


def _append_month_elems(elems: list[str], months: int) -> list[str]:
    """Append total number of months
    to a list of strings that, when joined, produce an ISO 8601
    duration string.

    Args:
        elems: The list of strings
        months: The total number of months

    Returns:
        The amended list of strings
    """
    years, months_in_year = divmod(months, 12)
    if years > 0:
        elems.append(f"{years}Y")
    if months_in_year > 0:
        elems.append(f"{months_in_year}M")
    return elems


def _get_microsecond_period_name(total_microseconds: int) -> str:
    """Return an ISO 8601 duration string
    from a total number of microseconds in a period

    Args:
        total_microseconds: The total number of microseconds

    Returns:
        The ISO 8601 duration string representing a period
        of n-microseconds
    """
    seconds, microseconds = divmod(total_microseconds, 1_000_000)
    return "".join(_append_second_elems(["P"], seconds, microseconds))


def _get_second_period_name(seconds: int) -> str:
    """Return an ISO 8601 duration string
    from a total number of seconds in a period

    Args:
        seconds: The total number of seconds

    Returns:
        The ISO 8601 duration string representing a period
        of n-seconds
    """
    return "".join(_append_second_elems(["P"], seconds, 0))


def _get_month_period_name(months: int) -> str:
    """Return an ISO 8601 duration string
    from a total number of months in a period

    Args:
        months: The total number of months

    Returns:
        The ISO 8601 duration string representing a period
        of n-months
    """
    return "".join(_append_month_elems(["P"], months))


_STEP_MICROSECONDS = 1
_STEP_SECONDS = 2
_STEP_MONTHS = 3
_VALID_STEPS = frozenset([_STEP_MICROSECONDS, _STEP_SECONDS, _STEP_MONTHS])


def _fmt_naive_microsecond(obj: dt.datetime, separator: str) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm:ss.ssssss
    """
    return (
        f"{obj.year:04}-{obj.month:02}-{obj.day:02}"
        f"{separator}"
        f"{obj.hour:02}:{obj.minute:02}:{obj.second:02}"
        f".{obj.microsecond:06}"
    )


def _fmt_naive_millisecond(obj: dt.datetime, separator: str) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm:ss.sss
    """
    millisecond: int = obj.microsecond // 1_000
    return (
        f"{obj.year:04}-{obj.month:02}-{obj.day:02}"
        f"{separator}"
        f"{obj.hour:02}:{obj.minute:02}:{obj.second:02}"
        f".{millisecond:03}"
    )


def _fmt_naive_second(obj: dt.datetime, separator: str) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm:ss
    """
    return f"{obj.year:04}-{obj.month:02}-{obj.day:02}" f"{separator}" f"{obj.hour:02}:{obj.minute:02}:{obj.second:02}"


def _fmt_naive_minute(obj: dt.datetime, separator: str) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm
    """
    return f"{obj.year:04}-{obj.month:02}-{obj.day:02}" f"{separator}" f"{obj.hour:02}:{obj.minute:02}"


def _fmt_naive_hour(obj: dt.datetime, separator: str) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh
    """
    return f"{obj.year:04}-{obj.month:02}-{obj.day:02}" f"{separator}" f"{obj.hour:02}"


def _fmt_naive_day(obj: dt.datetime) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd
    """
    return f"{obj.year:04}-{obj.month:02}-{obj.day:02}"


def _fmt_naive_month(obj: dt.datetime) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm
    """
    return f"{obj.year:04}-{obj.month:02}"


def _fmt_naive_year(obj: dt.datetime) -> str:
    """Convert a naive datetime to an ISO 8601 format string

    Args:
        obj: The datetime object to be formatted.

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy
    """
    return f"{obj.year:04}"


def _fmt_tzdelta(delta: dt.timedelta) -> str:
    """Convert a timedelta which represents a timezone
    to a string that respresents that timezone

    Args:
        delta: The timedelta

    Returns:
        A string that can be used to represent a timezone
        in an ISO 8601 format string
    """
    seconds = int(delta.total_seconds())
    if seconds == 0:
        return "Z"
    sign = "-" if seconds < 0 else "+"
    days, seconds_in_day = divmod(abs(seconds), 86_400)
    if days > 0:
        raise ValueError(f"Illegal tz delta: {delta}")
    hours, seconds_in_hour = divmod(seconds_in_day, 3_600)
    minutes = seconds_in_hour // 60
    return f"{sign}{hours:02}:{minutes:02}"


def _fmt_tzinfo(tz: Optional[dt.tzinfo]) -> str:
    """Convert an optional tzinfo object into a string that
    can be used to represent the timezone in an ISO 8601 format

    Args:
        tz: The tzinfo object, or None

    Returns:
        A string that can be used to represent a timezone
        in an ISO 8601 format string
    """
    if tz is not None:
        delta = tz.utcoffset(dt.datetime.min)
        if delta is not None:
            return _fmt_tzdelta(delta)
    return ""


def _fmt_aware_microsecond(obj: dt.datetime, separator: str) -> str:
    """Convert a datetime which includes a timezone to an ISO 8601
    format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm:ss.ssssss{tz}
    """
    tz_str: str = _fmt_tzinfo(obj.tzinfo)
    return (
        f"{obj.year:04}-{obj.month:02}-{obj.day:02}"
        f"{separator}"
        f"{obj.hour:02}:{obj.minute:02}:{obj.second:02}"
        f".{obj.microsecond:06}"
        f"{tz_str}"
    )


def _fmt_aware_millisecond(obj: dt.datetime, separator: str) -> str:
    """Convert a datetime which includes a timezone to an ISO 8601
    format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm:ss.sss{tz}
    """
    millisecond: int = obj.microsecond // 1_000
    tz_str: str = _fmt_tzinfo(obj.tzinfo)
    return (
        f"{obj.year:04}-{obj.month:02}-{obj.day:02}"
        f"{separator}"
        f"{obj.hour:02}:{obj.minute:02}:{obj.second:02}"
        f".{millisecond:03}"
        f"{tz_str}"
    )


def _fmt_aware_second(obj: dt.datetime, separator: str) -> str:
    """Convert a datetime which includes a timezone to an ISO 8601
    format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm:ss{tz}
    """
    tz_str: str = _fmt_tzinfo(obj.tzinfo)
    return (
        f"{obj.year:04}-{obj.month:02}-{obj.day:02}"
        f"{separator}"
        f"{obj.hour:02}:{obj.minute:02}:{obj.second:02}"
        f"{tz_str}"
    )


def _fmt_aware_minute(obj: dt.datetime, separator: str) -> str:
    """Convert a datetime which includes a timezone to an ISO 8601
    format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh:mm{tz}
    """
    tz_str: str = _fmt_tzinfo(obj.tzinfo)
    return f"{obj.year:04}-{obj.month:02}-{obj.day:02}" f"{separator}" f"{obj.hour:02}:{obj.minute:02}" f"{tz_str}"


def _fmt_aware_hour(obj: dt.datetime, separator: str) -> str:
    """Convert a datetime which includes a timezone to an ISO 8601
    format string

    Args:
        obj: The datetime object to be formatted.
        separator: The character used to separate the date and time
                   parts of the formatted output

    Returns:
        The ISO 8601 string representation of the datetime:
            yyyy-mm-dd{separator}hh{tz}
    """
    tz_str: str = _fmt_tzinfo(obj.tzinfo)
    return f"{obj.year:04}-{obj.month:02}-{obj.day:02}" f"{separator}" f"{obj.hour:02}" f"{tz_str}"


@dataclass(eq=True, order=True, frozen=True)
class Properties:
    """The basic properties of a period. Each specific Period
    subclass uses the parts that are relevant to it to implement
    the Period abstract base class.

    Properties objects are:
    - Immutable
    - Sortable
    - Hashable (can be used in sets and as keys in dictionaries, etc.)

    Each Period subclass contains a Properties instance and forwards
    the relevant methods to it, which means that Period objects
    are also sortable and hashable (they are also immutable).
    """

    step: int
    multiplier: int
    month_offset: int
    microsecond_offset: int
    tzinfo: Optional[dt.tzinfo]
    ordinal_shift: int

    @staticmethod
    def of_years(no_of_years: int) -> "Properties":
        """Return Properties object for an "n"-year period

        Args:
            no_of_years: The number of years in the period

        Returns:
            A Properties object
        """
        return Properties.of_months(no_of_years * 12)

    @staticmethod
    def of_months(no_of_months: int) -> "Properties":
        """Return Properties object for an "n"-month period

        Args:
            no_of_months: The number of months in the period

        Returns:
            A Properties object
        """
        return Properties.of_step_and_multiplier(_STEP_MONTHS, no_of_months)

    @staticmethod
    def of_days(no_of_days: int) -> "Properties":
        """Return Properties object for an "n"-day period

        Args:
            no_of_days: The number of days in the period

        Returns:
            A Properties object
        """
        return Properties.of_seconds(no_of_days * 86_400)

    @staticmethod
    def of_hours(no_of_hours: int) -> "Properties":
        """Return Properties object for an "n"-hour period

        Args:
            no_of_hours: The number of hours in the period

        Returns:
            A Properties object
        """
        return Properties.of_seconds(no_of_hours * 3_600)

    @staticmethod
    def of_minutes(no_of_minutes: int) -> "Properties":
        """Return Properties object for an "n"-minute period

        Args:
            no_of_minutes: The number of minutes in the period

        Returns:
            A Properties object
        """
        return Properties.of_seconds(no_of_minutes * 60)

    @staticmethod
    def of_seconds(no_of_seconds: int) -> "Properties":
        """Return Properties object for an "n"-second period

        Args:
            no_of_seconds: The number of seconds in the period

        Returns:
            A Properties object
        """
        return Properties.of_step_and_multiplier(_STEP_SECONDS, no_of_seconds)

    @staticmethod
    def of_microseconds(no_of_microseconds: int) -> "Properties":
        """Return Properties object for an "n"-microsecond period

        Args:
            no_of_microseconds: The number of microseconds in the period

        Returns:
            A Properties object
        """
        seconds, microseconds = divmod(no_of_microseconds, 1_000_000)
        if microseconds == 0:
            return Properties.of_seconds(seconds)
        return Properties.of_step_and_multiplier(_STEP_MICROSECONDS, no_of_microseconds)

    @staticmethod
    def of_step_and_multiplier(step: int, multiplier: int) -> "Properties":
        """Return a Properties object that represents a period of the given
        step and multiplier

        The step represents the basic "unit" of the period and can be one of:
            _STEP_MICROSECONDS
            _STEP_SECONDS
            _STEP_MONTHS

        The multiplier is the number of 'steps' in the period.

        Args:
            step: The step of the period:
            multiplier: The multiplier, or the number of steps
                        in the period

        Examples:
            A period of one year (P1Y) has:
                step = _STEP_MONTHS and multiplier = 12
            A period of one day (P1D) has:
                step = _STEP_SECONDS and multiplier = 24*60*60
            A period of fifteen minutes (PT15M) has:
                step = _STEP_SECONDS and multiplier = 15*60
            A period of 25Hz (PT0.04S) has:
                step = _STEP_MICROSECONDS and multiplier = 1_000_000/25

        Returns:
            A Properties object
        """
        return Properties(
            step=step, multiplier=multiplier, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )

    def __post_init__(self) -> None:
        if self.step not in _VALID_STEPS:
            raise AssertionError(f"Illegal step: {self.step}")
        if self.multiplier <= 0:
            raise AssertionError(f"Illegal multiplier: {self.multiplier}")
        if self.month_offset < 0:
            raise AssertionError(f"Illegal month offset: {self.month_offset}")
        if self.microsecond_offset < 0:
            raise AssertionError(f"Illegal microsecond offset: {self.microsecond_offset}")
        if (self.step != _STEP_MONTHS) and (self.month_offset != 0):
            raise AssertionError("Illegal month offset for non-month step")

    def normalise_offsets(self) -> "Properties":
        """Return an equivalent Properties object where
        month_offset and microsecond_offset are within the
        bounds of the step and multiplier

        For a period of n-months the normalised month_offset
        is month_offset%n

        For a period of n-seconds the normalised microsecond_offset
        is microsecond_offset%(n*1_000_000)

        For a period of n-microseconds the normalised microsecond_offset
        is microsecond_offset%n

        Example: A one year period with an offset of 13 months is
        normalised to a one year period with an offset of 1 month,
        as the net effect of the offsets is the same regarding how they
        split the timeline into intervals.

        Returns:
            A Properties object
        """
        new_month_offset: int = self.month_offset
        new_microsecond_offset: int = self.microsecond_offset
        if self.step == _STEP_MONTHS:
            new_month_offset = self.month_offset % self.multiplier
        elif self.step == _STEP_SECONDS:
            assert self.month_offset == 0
            new_microsecond_offset = self.microsecond_offset % (self.multiplier * 1_000_000)
        elif self.step == _STEP_MICROSECONDS:
            assert self.month_offset == 0
            new_microsecond_offset = self.microsecond_offset % self.multiplier
        else:
            raise AssertionError("Oops")

        if (
            (new_month_offset == self.month_offset)
            and (new_microsecond_offset == self.microsecond_offset)
            and (self.ordinal_shift == 0)
        ):
            return self

        return Properties(
            step=self.step,
            multiplier=self.multiplier,
            month_offset=new_month_offset,
            microsecond_offset=new_microsecond_offset,
            tzinfo=self.tzinfo,
            ordinal_shift=0,
        )

    def with_month_offset(self, month_amount: int) -> "Properties":
        """Return a Properties object derived from this one with
        the given month_offset

        The ordinal_shift for the new Properties object will be
        set to 0 as the new month_offset will render the
        previous value meaningless.

        Args:
            month_amount: The amount to add to the month_offset

        Returns:
            A Properties object
        """
        return Properties(
            step=self.step,
            multiplier=self.multiplier,
            month_offset=self.month_offset + month_amount,
            microsecond_offset=self.microsecond_offset,
            tzinfo=self.tzinfo,
            ordinal_shift=0,
        ).normalise_offsets()

    def with_microsecond_offset(self, microsecond_amount: int) -> "Properties":
        """Return a Properties object derived from this one with
        the given microsecond_offset

        The ordinal_shift for the new Properties object will be
        set to 0 as the new microsecond_offset will render the
        previous value meaningless.

        Args:
            microsecond_amount: The amount to add to the microsecond_offset

        Returns:
            A Properties object
        """
        return Properties(
            step=self.step,
            multiplier=self.multiplier,
            month_offset=self.month_offset,
            microsecond_offset=self.microsecond_offset + microsecond_amount,
            tzinfo=self.tzinfo,
            ordinal_shift=0,
        ).normalise_offsets()

    def with_tzinfo(self, tzinfo: Optional[dt.tzinfo]) -> "Properties":
        """Return a Properties object derived from this one with
        the given datetime tzinfo object (time zone)

        Args:
            tzinfo: The tzinfo object (or None) of the new Properties

        Returns:
            A Properties object
        """
        return Properties(
            step=self.step,
            multiplier=self.multiplier,
            month_offset=self.month_offset,
            microsecond_offset=self.microsecond_offset,
            tzinfo=tzinfo,
            ordinal_shift=self.ordinal_shift,
        )

    def with_ordinal_shift(self, ordinal_shift: int) -> "Properties":
        """Return a Properties object derived from this one with
        the given ordinal shift value.

        Args:
            ordinal_shift: The amount by which ordinal values are shifted
                           in a Period object created from the new
                           Properties object

        Returns:
            A Properties object
        """
        return Properties(
            step=self.step,
            multiplier=self.multiplier,
            month_offset=self.month_offset,
            microsecond_offset=self.microsecond_offset,
            tzinfo=self.tzinfo,
            ordinal_shift=ordinal_shift,
        )

    def with_offset_period_fields(self, offset_period_fields: "PeriodFields") -> "Properties":
        """Return a Properties object derived from this one with
        new month and microsecond offsets

        Args:
            offset_period_fields: A PeriodFields object that is used
                        to calculate the month and microsecond offsets

        Returns:
            A Properties object
        """
        return self.with_offset_months_seconds(offset_period_fields.get_months_seconds())

    def with_offset_months_seconds(self, offset_months_seconds: "MonthsSeconds") -> "Properties":
        """Return a Properties object derived from this one with
        new month and microsecond offsets

        Args:
            offset_months_seconds: A MonthsSeconds object that is used
                        to calculate the month and microsecond offsets

        Returns:
            A Properties object
        """
        return Properties(
            step=self.step,
            multiplier=self.multiplier,
            month_offset=offset_months_seconds.months,
            microsecond_offset=offset_months_seconds.total_microseconds(),
            tzinfo=self.tzinfo,
            ordinal_shift=0,
        ).normalise_offsets()

    def get_iso8601(self) -> str:
        """Return the ISO 8601 duration string of the period
        defined by this Properties object

        Returns:
            The ISO 8601 duration string of this Period
        """
        if self.step == _STEP_MICROSECONDS:
            return _get_microsecond_period_name(self.multiplier)
        if self.step == _STEP_SECONDS:
            return _get_second_period_name(self.multiplier)
        if self.step == _STEP_MONTHS:
            return _get_month_period_name(self.multiplier)
        raise AssertionError(f"Illegal step: {self.step}")

    def get_timedelta(self) -> Optional[dt.timedelta]:
        """Return a timedelta object that matches the duration
        of this period, or None if no such timedelta exists

        There is no timedelta for monthly or yearly periods, as
        these are not of a fixed length.

        Returns:
            A timedelta object, or None
        """
        if self.step == _STEP_MICROSECONDS:
            seconds, microseconds = divmod(self.multiplier, 1_000_000)
            return dt.timedelta(seconds=seconds, microseconds=microseconds)
        if self.step == _STEP_SECONDS:
            return dt.timedelta(seconds=self.multiplier)
        if self.step == _STEP_MONTHS:
            return None
        raise AssertionError(f"Illegal step: {self.step}")

    def _append_step_elems(self, elems: list[str]) -> None:
        """Add elements to a list of string that describe the
        step and multiplier and can be joined to form an ISO 8601
        duration string

        The list of strings is used to calculate the repr string
        """
        if self.step == _STEP_MICROSECONDS:
            seconds, microseconds = divmod(self.multiplier, 1_000_000)
            _append_second_elems(elems, seconds, microseconds)
        elif self.step == _STEP_SECONDS:
            _append_second_elems(elems, self.multiplier, 0)
        elif self.step == _STEP_MONTHS:
            _append_month_elems(elems, self.multiplier)
        else:
            raise AssertionError(f"Illegal step: {self.step}")

    def _append_offset_elems(self, elems: list[str]) -> None:
        """Add elements to a list of string that describe the
        month and microsecond offsets in a style similar to an
        ISO 8601 duration string

        The list of strings is used to calculate the repr string
        """
        if (self.month_offset == 0) and (self.microsecond_offset == 0):
            return
        elems.append("+")
        if self.month_offset > 0:
            _append_month_elems(elems, self.month_offset)
        if self.microsecond_offset > 0:
            seconds, microseconds = divmod(self.microsecond_offset, 1_000_000)
            _append_second_elems(elems, seconds, microseconds)
        return

    def _append_tz_elems(self, elems: list[str]) -> None:
        """Add elements to a list of string that describe the
        tzinfo object

        The list of strings is used to calculate the repr string
        """
        elems.append("[")
        if self.tzinfo is not None:
            delta = self.tzinfo.utcoffset(dt.datetime.min)
            if delta is not None:
                elems.append(_fmt_tzdelta(delta))
        elems.append("]")

    def _append_shift_elems(self, elems: list[str]) -> None:
        """Add elements to a list of string that describe the
        ordinal shift

        The list of strings is used to calculate the repr string
        """
        if self.ordinal_shift != 0:
            elems.append(str(self.ordinal_shift))

    def get_naive_formatter(self, separator: str = "T") -> Callable[[dt.datetime], str]:
        """Return a datetime formatter function suitable for formatting
        naive datetime objects of this period

        Args:
            separator: The character used to separate the ISO 8601
                       date and time parts

        Returns:
            A function that takes a single datetime argument and
            returns a string
        """
        o_total_milliseconds, o_microseconds_nnn = divmod(self.microsecond_offset, 1_000)
        if o_microseconds_nnn != 0:
            return lambda dt: _fmt_naive_microsecond(dt, separator)
        o_total_seconds, o_milliseconds_nnn = divmod(o_total_milliseconds, 1_000)
        if self.step == _STEP_MICROSECONDS:
            s_total_milliseconds, s_microseconds_nnn = divmod(self.multiplier, 1_000)
            if s_microseconds_nnn != 0:
                return lambda dt: _fmt_naive_microsecond(dt, separator)
            s_milliseconds_nnn = s_total_milliseconds % 1_000
            if (s_milliseconds_nnn != 0) or (o_milliseconds_nnn != 0):
                return lambda dt: _fmt_naive_millisecond(dt, separator)
            return lambda dt: _fmt_naive_second(dt, separator)
        if o_milliseconds_nnn != 0:
            return lambda dt: _fmt_naive_millisecond(dt, separator)

        o_total_minutes, o_seconds_nn = divmod(o_total_seconds, 60)
        if o_seconds_nn != 0:
            return lambda dt: _fmt_naive_second(dt, separator)
        o_total_hours, o_minutes_nn = divmod(o_total_minutes, 60)
        o_total_days, o_hours_nn = divmod(o_total_hours, 24)
        if self.step == _STEP_SECONDS:
            s_total_minutes, s_seconds_nn = divmod(self.multiplier, 60)
            if s_seconds_nn != 0:
                return lambda dt: _fmt_naive_second(dt, separator)
            s_total_hours, s_minutes_nn = divmod(s_total_minutes, 60)
            if (s_minutes_nn != 0) or (o_minutes_nn != 0):
                return lambda dt: _fmt_naive_minute(dt, separator)
            s_hours_nn = s_total_hours % 24
            if (s_hours_nn != 0) or (o_hours_nn != 0):
                return lambda dt: _fmt_naive_hour(dt, separator)
            return _fmt_naive_day
        if o_minutes_nn != 0:
            return lambda dt: _fmt_naive_minute(dt, separator)
        if o_hours_nn != 0:
            return lambda dt: _fmt_naive_hour(dt, separator)
        if o_total_days > 0:
            return _fmt_naive_day

        assert self.step == _STEP_MONTHS
        o_months_nn = self.month_offset % 12
        s_months_nn = self.multiplier % 12
        if (s_months_nn != 0) or (o_months_nn != 0):
            return _fmt_naive_month
        return _fmt_naive_year

    def get_aware_formatter(self, separator: str = "T") -> Callable[[dt.datetime], str]:
        """Return a datetime formatter function suitable for formatting
        timezone aware datetime objects of this period

        Args:
            separator: The character used to separate the ISO 8601
                       date and time parts

        Returns:
            A function that takes a single datetime argument and
            returns a string
        """
        o_total_milliseconds, o_microseconds_nnn = divmod(self.microsecond_offset, 1_000)
        if o_microseconds_nnn != 0:
            return lambda dt: _fmt_aware_microsecond(dt, separator)

        o_total_seconds, o_milliseconds_nnn = divmod(o_total_milliseconds, 1_000)
        if self.step == _STEP_MICROSECONDS:
            s_total_milliseconds, s_microseconds_nnn = divmod(self.multiplier, 1_000)
            if s_microseconds_nnn != 0:
                return lambda dt: _fmt_aware_microsecond(dt, separator)
            s_milliseconds_nnn = s_total_milliseconds % 1_000
            if (s_milliseconds_nnn != 0) or (o_milliseconds_nnn != 0):
                return lambda dt: _fmt_aware_millisecond(dt, separator)
            return lambda dt: _fmt_aware_second(dt, separator)
        if o_milliseconds_nnn != 0:
            return lambda dt: _fmt_aware_millisecond(dt, separator)

        o_total_minutes, o_seconds_nn = divmod(o_total_seconds, 60)
        if o_seconds_nn != 0:
            return lambda dt: _fmt_aware_second(dt, separator)
        o_total_hours, o_minutes_nn = divmod(o_total_minutes, 60)
        o_hours_nn = o_total_hours % 24
        if self.step == _STEP_SECONDS:
            s_total_minutes, s_seconds_nn = divmod(self.multiplier, 60)
            if s_seconds_nn != 0:
                return lambda dt: _fmt_aware_second(dt, separator)
            s_minutes_nn = s_total_minutes % 60
            if (s_minutes_nn != 0) or (o_minutes_nn != 0):
                return lambda dt: _fmt_aware_minute(dt, separator)
            return lambda dt: _fmt_aware_hour(dt, separator)
        if o_minutes_nn != 0:
            return lambda dt: _fmt_aware_minute(dt, separator)
        if o_hours_nn != 0:
            return lambda dt: _fmt_aware_hour(dt, separator)

        assert self.step == _STEP_MONTHS
        return lambda dt: _fmt_aware_hour(dt, separator)

    def pl_interval(self) -> str:
        """Return a string that captures the step and multiplier
        of this period and which is suitable for use with
        Polars DataFrames

        The returned string is defined using the Polars duration
        string language, and can be used in method calls such as:

            polars.DataFrame.group_by_dynamic(..., every=string,...)

            polars.datetime_ranges(...,interval=string,...)

        Returns:
            A string suitable for use with Polars DataFrames
        """
        if self.step == _STEP_MICROSECONDS:
            return f"{self.multiplier}us"
        elif self.step == _STEP_SECONDS:
            return f"{self.multiplier}s"
        elif self.step == _STEP_MONTHS:
            return f"{self.multiplier}mo"
        else:
            raise AssertionError(f"Illegal step: {self.step}")

    def pl_offset(self) -> str:
        """Return a string that captures the month and microsecond
        offsets of this period and which is suitable for use with
        Polars DataFrames

        The returned string is defined using the Polars duration
        string language, and can be used in method calls such as:

            polars.Expr.dt.offset_by( by=string )

        Returns:
            A string suitable for use with Polars DataFrames
        """
        return f"{self.month_offset}mo{self.microsecond_offset}us"

    def is_epoch_agnostic(self) -> bool:
        """Return True if the way that this period splits the
        timeline does not depend on the epoch used to perform
        calculations

        This method assumes that the epoch will always be midnight
        at the start of the first day of a year.

        Most commonly used periods such as P1Y, P1M, P1D, PT15M and
        so on are epoch agnostic.

        A period such as P7D is not epoch agnostic however. The
        calculation will typically be done using modulus arithmetic
        on the number of days since the epoch, and this will split
        the timeline into different 7-day intervals depending on the
        epoch.

        Returns:
            True if this period plits the timeline into the same
            intervals regardless of the epoch, False otherwise.
        """
        if self.step == _STEP_MICROSECONDS:
            if self.multiplier > 1_000_000:
                return False
            num_per_second: int = 1_000_000 // self.multiplier
            return (self.multiplier * num_per_second) == 1_000_000
        elif self.step == _STEP_SECONDS:
            if self.multiplier > 86_400:
                return False
            num_per_day: int = 86_400 // self.multiplier
            return (self.multiplier * num_per_day) == 86_400
        elif self.step == _STEP_MONTHS:
            if self.multiplier > 12:
                return False
            num_per_year: int = 12 // self.multiplier
            return (self.multiplier * num_per_year) == 12
        else:
            raise AssertionError(f"Illegal step: {self.step}")

    def __str__(self) -> str:
        elems: list[str] = ["P"]
        self._append_step_elems(elems)
        self._append_offset_elems(elems)
        return "".join(elems)

    def __repr__(self) -> str:
        elems: list[str] = ["P"]
        self._append_step_elems(elems)
        self._append_offset_elems(elems)
        self._append_tz_elems(elems)
        self._append_shift_elems(elems)
        return "".join(elems)


# ------------------------------------------------------------------------------
# MonthsSeconds
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class MonthsSeconds:
    """A basic specification for either a period or a period offset
    where the period/offset amount is a number of months and/or seconds.
    """

    string: str
    months: int
    seconds: int
    microseconds: int

    def __post_init__(self) -> None:
        if (self.months < 0) or (self.seconds < 0) or (self.microseconds < 0) or (self.microseconds > 999_999):
            raise ValueError(f"Illegal period: {self.string}")
        if (self.months == 0) and (self.seconds == 0) and (self.microseconds == 0):
            raise ValueError(f"Illegal period: {self.string}")

    def get_step_and_multiplier(self) -> tuple[int, int]:
        """Return a tuple of two integers containing the
        calculated step and multiplier

        Returns:
            A tuple of (step,multiplier)

        Raises:
            ValueError if a valid step and multiplier
            cannot be created.  This happens, for example,
            if boths months and seconds are >0.
        """
        if (self.months > 0) and (self.seconds == 0) and (self.microseconds == 0):
            return _STEP_MONTHS, self.months
        if (self.months == 0) and (self.seconds > 0) and (self.microseconds == 0):
            return _STEP_SECONDS, self.seconds
        if (self.months == 0) and (self.microseconds > 0):
            return _STEP_MICROSECONDS, self.total_microseconds()
        raise ValueError(f"Illegal period: {self.string}")

    def total_microseconds(self) -> int:
        """Return total microseconds from the seconds
        and microseconds fields

        Returns:
            The total number of microseconds
        """
        return self.seconds * 1_000_000 + self.microseconds

    def get_base_properties(self) -> Properties:
        """Return a basic Properties object from the months
        and (seconds,microseconds) fields

        Raise an error if it is not possible to create
        a valid Properties object.  This happens if,
        for example, both months and seconds are >0.

        Returns:
            A Properties object
        """
        step, multiplier = self.get_step_and_multiplier()
        return Properties(
            step=step, multiplier=multiplier, month_offset=0, microsecond_offset=0, tzinfo=None, ordinal_shift=0
        )


# ------------------------------------------------------------------------------
# PeriodFields
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class PeriodFields:
    """A set of fields that can be read from an ISO 8601 format
    duration string.
    """

    string: str
    years: int
    months: int
    days: int
    hours: int
    minutes: int
    seconds: int
    microseconds: int

    def __post_init__(self) -> None:
        if (
            (self.years < 0)
            or (self.months < 0)
            or (self.days < 0)
            or (self.hours < 0)
            or (self.minutes < 0)
            or (self.seconds < 0)
            or (self.microseconds < 0)
            or (self.microseconds > 999_999)
        ):
            raise ValueError(f"Illegal period: {self.string}")

    def get_months_seconds(self) -> MonthsSeconds:
        """Return a MonthsSeconds object from the fields
        of an ISO 8601 duration

        Returns:
            A MonthsSeconds object
        """
        months = self.years * 12 + self.months
        seconds = (self.days * 86_400) + (self.hours * 3_600) + (self.minutes * 60) + self.seconds
        return MonthsSeconds(string=self.string, months=months, seconds=seconds, microseconds=self.microseconds)

    def get_base_properties(self) -> Properties:
        """Return a basic Properties object from the fields
        of an ISO 8601 duration

        Raise an error if it is not possible to create
        a valid Properties object.

        Returns:
            A Properties object
        """
        return self.get_months_seconds().get_base_properties()


def _str2int(num: Optional[str], default: int = 0) -> int:
    """Convert string to int, None becomes 0 (or default if specified)

    Args:
        num: The string to be converted, or None
        default: Default int value to use if num is None
                 (defaults to 0)

    Converting the string to an int may raise the usual Python error(s)

    Returns:
        The int value read from the input string
    """
    return default if num is None else int(num)


def _str2microseconds(num: Optional[str], default: int = 0) -> int:
    """Convert string to microseconds, None becomes 0 (or default if specified)

    A seconds + microseconds string looks like "nn.uuu". This function
    converts the "uuu" part (from 1 to 6 digits) to an int value
    between 0 and 999_999.

    Converting the string to an int may raise the usual Python error(s)

    Args:
        num: The string to be converted, or None
        default: Default int value to use if num is None
                 (defaults to 0)

    Returns:
        The int value read from the input string
    """
    return default if num is None else int((num + "00000")[:6])


def _period_match(prefix: str, matcher: re.Match[str]) -> PeriodFields:
    """Return a PeriodFields object from a regex Matcher object

    The regex Pattern that create the matcher is assumed to have been
    created from a string returned by the _period_regex function.

    Args:
        prefix: The name prefix used when creating the regex (
        matcher: The regex Matcher object

    Returns:
        The PeriodFields object
    """
    return PeriodFields(
        string=matcher.string,
        years=_str2int(matcher.group(f"{prefix}_years")),
        months=_str2int(matcher.group(f"{prefix}_months")),
        days=_str2int(matcher.group(f"{prefix}_days")),
        hours=_str2int(matcher.group(f"{prefix}_hours")),
        minutes=_str2int(matcher.group(f"{prefix}_minutes")),
        seconds=_str2int(matcher.group(f"{prefix}_seconds")),
        microseconds=_str2microseconds(matcher.group(f"{prefix}_microseconds")),
    )


def _total_microseconds(delta: dt.timedelta) -> int:
    """Return total number of microseconds in a timedelta

    Args:
        delta: A datetime timedelta object

    Returns:
        The total number of microseconds in a timedelta
    """
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


# ------------------------------------------------------------------------------
# Period
# ------------------------------------------------------------------------------
class Period(ABC):
    """A period in time that can be used to split the gregorian timeline
    into intervals.

    This is an abstract class and it is not possible to create Period
    instances directly.  Use one of the static methods in this class to
    create a Period instance.

    Period instances are immutable.

    Period instances are hashable and can be used in sets and as keys
    in dictionaries.

    Period instances are sortable, though the sort order may not make
    sense in all cases.  Periods contain optional tzinfo objects, and
    the same caveats that apply to sorting datetime objects also apply
    to Period objects; depending on the tzinfo values you may get a
    Python error rather than a sorted collection of Periods.

    Each subclass implements the ordinal() and datetime() methods.
    Each subclass also contains a Properties object which holds all
    the basic information about the Period.  Between them, the
    ordinal() and datetime() methods, and the Properties object,
    contain all the low-level operations and data upon which all
    other functionality is built.

    This class contains the public interface of the period module.
    """

    @staticmethod
    def of(period_string: str) -> "Period":
        """Return a Period from the supplied string.

        Args:
            period_string: A string containing a period
                           definition

        Returns:
            A Period object defined by the supplied string

        Raises:
            ValueError if there is no such Period.
        """
        return _of(period_string)

    @staticmethod
    def of_iso_duration(iso_8601_duration: str) -> "Period":
        """Return a Period from an ISO 8601 duration string

        Args:
            iso_8601_duration: An ISO 8601 duration string such
                               as "P1Y" or "PT15M"

        Returns:
            A Period object defined by the supplied ISO 8601
            duration string

        Raises:
            ValueError if the string does not contain a valid
            ISO 8601 duration value
        """
        return _of_iso_duration(iso_8601_duration)

    @staticmethod
    def of_duration(duration: str) -> "Period":
        """Return a Period from an (extended) ISO 8601 duration string

        Args:
            duration: An ISO 8601 duration string such as "P1Y" or
                      "PT15M"

        Returns:
            A Period object defined by the supplied (extended)
            ISO 8601 duration string

        Raises:
            ValueError if the string does not contain a valid
            (extended) ISO 8601 duration value
        """
        return _of_duration(duration)

    @staticmethod
    def of_date_and_duration(date_duration: str) -> "Period":
        """Return a Period from date/duration string

        Args:
            date_duration: An ISO 8601 duration string of the form
                           <start>/<duration>

        Returns:
            A Period object defined by the supplied ISO 8601
            duration string

        Raises:
            ValueError if the string does not contain a valid
            <start>/<duration> value
        """
        return _of_date_and_duration(date_duration)

    @staticmethod
    def of_repr(repr_string: str) -> "Period":
        """Return a Period from a __repr__ string

        Args:
            repr_string: The repr string of a Period object

        Returns:
            A Period object that is identical (as much as possible
            anyway) to the Period object that produced the repr string.

        Raises:
            ValueError if the string does not contain a valid
            __repr__ string
        """
        return _of_repr(repr_string)

    @staticmethod
    def of_years(no_of_years: int) -> "Period":
        """Return an "n"-year Period

        Args:
            no_of_years: The number of years in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_years(no_of_years))

    @staticmethod
    def of_months(no_of_months: int) -> "Period":
        """Return an "n"-month Period

        Args:
            no_of_months: The number of months in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_months(no_of_months))

    @staticmethod
    def of_days(no_of_days: int) -> "Period":
        """Return an "n"-day Period

        Args:
            no_of_days: The number of days in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_days(no_of_days))

    @staticmethod
    def of_hours(no_of_hours: int) -> "Period":
        """Return an "n"-hour Period

        Args:
            no_of_hours: The number of hours in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_hours(no_of_hours))

    @staticmethod
    def of_minutes(no_of_minutes: int) -> "Period":
        """Return an "n"-minute Period

        Args:
            no_of_minutes: The number of minutes in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_minutes(no_of_minutes))

    @staticmethod
    def of_seconds(no_of_seconds: int) -> "Period":
        """Return an "n"-second Period

        Args:
            no_of_seconds: The number of seconds in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_seconds(no_of_seconds))

    @staticmethod
    def of_microseconds(no_of_microseconds: int) -> "Period":
        """Return an "n"-microsecond Period

        Args:
            no_of_microseconds: The number of microseconds in the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_microseconds(no_of_microseconds))

    @staticmethod
    def of_timedelta(timedelta: dt.timedelta) -> "Period":
        """Return a Period that matches a timedelta

        Args:
            timedelta: The timedelta of the period

        Returns:
            A Period object
        """
        return _get_base_period(Properties.of_microseconds(_total_microseconds(timedelta)))

    def __init__(self, properties: Properties) -> None:
        if properties is None:
            raise ValueError()
        assert isinstance(properties.step, int)
        assert properties.step in _VALID_STEPS
        assert isinstance(properties.multiplier, int)
        assert properties.multiplier > 0
        assert isinstance(properties.month_offset, int)
        assert properties.month_offset >= 0
        assert isinstance(properties.microsecond_offset, int)
        assert properties.microsecond_offset >= 0
        assert isinstance(properties.tzinfo, (dt.tzinfo, type(None)))
        assert isinstance(properties.ordinal_shift, int)
        self._properties = properties

    @property
    def iso_duration(self) -> str:
        """The standard ISO 8601 duration string of this period"""
        return self._properties.get_iso8601()

    @property
    def tzinfo(self) -> Optional[dt.tzinfo]:
        """Get the tzinfo property"""
        return self._properties.tzinfo

    @property
    def min_ordinal(self) -> int:
        """Get the minimum valid ordinal

        Due to the complexities of datetime handling and how
        this class works the returned value may not be the
        actual minimum, but it will be close.

        Returns:
            The minimum ordinal value that you can pass to the
            datetime method and have Python not throw an error
            because the datetime is out of bounds
        """
        min_ordinal: int
        try:
            min_ordinal = self.ordinal(dt.datetime.min)
        except Exception:
            min_ordinal = self.without_offset().ordinal(dt.datetime.min)
        try:
            self.datetime(min_ordinal)
        except Exception:
            min_ordinal += 1
        #           self.datetime( min_ordinal )
        return min_ordinal

    @property
    def max_ordinal(self) -> int:
        """Get the maximum valid ordinal

        Returns:
            The maximum ordinal value that you can pass to the
            datetime method and have Python not throw an error
            because the datetime is out of bounds
        """
        return self.ordinal(dt.datetime.max)

    @property
    def timedelta(self) -> Optional[dt.timedelta]:
        """A timedelta object that matches the duration
        of this period, or None if no such timedelta exists

        There is no timedelta for monthly or yearly periods, as
        these are not of a fixed length.

        Returns:
            A timedelta object, or None
        """
        return self._properties.get_timedelta()

    @property
    def pl_interval(self) -> str:
        """A string that captures the step and multiplier
        of this period and which is suitable for use with
        Polars DataFrames

        The returned string is defined using the Polars duration
        string language, and can be used in method calls such as:

            polars.DataFrame.group_by_dynamic(..., every=string,...)

            polars.datetime_ranges(...,interval=string,...)

        Returns:
            A string suitable for use with Polars DataFrames
        """
        return self._properties.pl_interval()

    @property
    def pl_offset(self) -> str:
        """A string that captures the month and microsecond
        offsets of this period and which is suitable for use with
        Polars DataFrames

        The returned string is defined using the Polars duration
        string language, and can be used in method calls such as:

            polars.Expr.dt.offset_by( by=string )

        Returns:
            A string suitable for use with Polars DataFrames
        """
        return self._properties.pl_offset()

    def is_epoch_agnostic(self) -> bool:
        """Return True if the way that this period splits the
        timeline does not depend on the epoch used to perform
        calculations

        This method assumes that the epoch will always be midnight
        at the start of the first day of a year.

        Most commonly used periods such as P1Y, P1M, P1D, PT15M and
        so on are epoch agnostic.

        A period such as P7D is not epoch agnostic however. The
        calculation will typically be done using modulus arithmetic
        on the number of days since the epoch, and this will split
        the timeline into different 7-day intervals depending on the
        epoch.

        Returns:
            True if this period plits the timeline into the same
            intervals regardless of the epoch, False otherwise.
        """
        return self._properties.is_epoch_agnostic()

    def naive_formatter(self, separator: str = "T") -> Callable[[dt.datetime], str]:
        """Return a datetime formatter suitable for formatting
        naive datetime objects of this period

        Args:
            separator: The character used to separate the ISO 8601
                       date and time parts

        Returns:
            A function that takes a single datetime argument and
            returns a string
        """
        if separator not in (" ", "T", "t"):
            raise ValueError(f"Illegal separator: {separator}")
        return self._properties.get_naive_formatter(separator)

    def aware_formatter(self, separator: str = "T") -> Callable[[dt.datetime], str]:
        """Return a datetime formatter suitable for formatting
        timezone aware datetime objects of this period

        Args:
            separator: The character used to separate the ISO 8601
                       date and time parts

        Returns:
            A function that takes a single datetime argument and
            returns a string
        """
        if separator not in (" ", "T", "t"):
            raise ValueError(f"Illegal separator: {separator}")
        return self._properties.get_aware_formatter(separator)

    def formatter(self, separator: str = "T") -> Callable[[dt.datetime], str]:
        """Return a datetime formatter suitable for formatting
        datetime objects produced by this Period

        Args:
            separator: The character used to separate the ISO 8601
                       date and time parts

        Returns:
            A function that takes a single datetime argument and
            returns a string
        """
        if separator not in (" ", "T", "t"):
            raise ValueError(f"Illegal separator: {separator}")
        return self.naive_formatter(separator) if self._properties.tzinfo is None else self.aware_formatter(separator)

    @abstractmethod
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        """Return an integer ordinal value from the supplied
        datetime argument

        The ordinal() method returns the ordinal of the interval
        within which the datetime lies (>= datetime of start of
        interval, < datetime of start of next interval)

        Args:
            datetime_obj: The datetime object

        Example:
            Calendar month starting at midnight

            The timeline from year 0001 to 9999 is split into
            monthly intervals, with each interval starting
            at midnight on the first day of the month. Each
            interval in the timeline is assigned an integer
            ordinal value, which are sequential.

            The ordinal() method returns the ordinal within
            which the specified datetime lies.

            The datetime() method returns the start of the
            interval identified by the ordinal.

        Notes:

        Will raise an error if an out-of-bounds datetime object
        is involved in any calculation (a datetime outside of
        year range 0001-9999)

        The tzinfo property of the supplied datetime object is
        ignored.

        The ordinal values returned by this method are only
        meaningful to the datetime() method of the same
        Period instance. Passing an ordinal created from
        one Period instance into the datetime() method of another
        Period instance will not yield a meaningful result.

        Returns:
            An integer ordinal value
        """
        raise NotImplementedError("ordinal method must be overridden")

    @abstractmethod
    def datetime(self, ordinal: int) -> dt.datetime:
        """Return a datetime object from the supplied
        integer ordinal

        The datetime() method returns the datetime of the start
        of the interval identified by the ordinal.

        Args:
            ordinal: The integer ordinal

        Notes:

        Will raise an error if an out-of-bounds datetime object
        is involved in any calculation (a datetime outside of
        year range 0001-9999)

        See the ordinal() method for an example.

        The returned datetime object will have the same tzinfo
        property as this Period.

        Returns:
            A datetime object
        """
        raise NotImplementedError("datetime method must be overridden")

    def is_aligned(self, datetime_obj: dt.datetime) -> bool:
        """Return True if datetime is aligned to the start of an
        interval

        The tzinfo of the Period and the datetime are both ignored.

        Returns:
            True if the supplied datetime lies at the start of an
            interval, False otherwise
        """
        ordinal = self.ordinal(datetime_obj)
        datetime_obj2 = self.datetime(ordinal)
        return _naive(datetime_obj) == _naive(datetime_obj2)

    def base_period(self) -> "Period":
        """Return an equivalent Period with no date offset or
        ordinal shift

        Returns:
            A Period with the same step as this Period, but with
            no date offset or ordinal shift
        """
        return self

    def with_multiplier(self, multiplier: int) -> "Period":
        """Return a Period derived from this one but with the
        specified multiplier

        Args:
            multiplier: The multiplier to be applied to the
                        new Period

        Examples:
            p1 = Period.of_years( 10 )
            p2 = Period.of_years( 1 ).with_multiplier( 10 )
            p1 == p2 # True

        The ordinal shift is reset to 0 and date/time offset
        are normalised.

        Returns:
            A Period object
        """
        properties = self._properties
        return _get_offset_period(
            Properties(
                step=properties.step,
                multiplier=properties.multiplier * multiplier,
                month_offset=properties.month_offset,
                microsecond_offset=properties.microsecond_offset,
                tzinfo=properties.tzinfo,
                ordinal_shift=0,
            ).normalise_offsets()
        )

    def with_year_offset(self, year_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified year offset

        Args:
            year_amount: The year offset of the new Period

        Returns:
            A Period object
        """
        return self.with_month_offset(year_amount * 12)

    def with_month_offset(self, month_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified month offset

        Args:
            month_amount: The month offset of the new Period

        Returns:
            A Period object
        """
        return _get_offset_period(self._properties.with_month_offset(month_amount))

    def with_day_offset(self, day_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified day offset

        Args:
            day_amount: The day offset of the new Period

        Returns:
            A Period object
        """
        return self.with_second_offset(day_amount * 86_400)

    def with_hour_offset(self, hour_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified hour offset

        Args:
            hour_amount: The hour offset of the new Period

        Returns:
            A Period object
        """
        return self.with_second_offset(hour_amount * 3_600)

    def with_minute_offset(self, minute_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified minute offset

        Args:
            minute_amount: The minute offset of the new Period

        Returns:
            A Period object
        """
        return self.with_second_offset(minute_amount * 60)

    def with_second_offset(self, second_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified second offset

        Args:
            second_amount: The second offset of the new Period

        Returns:
            A Period object
        """
        return self.with_microsecond_offset(second_amount * 1_000_000)

    def with_microsecond_offset(self, microsecond_amount: int) -> "Period":
        """Return a Period derived from this one but with the
        specified microsecond offset

        Args:
            microsecond_amount: The microsecond offset of the new Period

        Returns:
            A Period object
        """
        return _get_offset_period(self._properties.with_microsecond_offset(microsecond_amount))

    def with_tzinfo(self, tzinfo: Optional[dt.tzinfo]) -> "Period":
        """Return a Period derived from this one but with the
        specified tzinfo

        Args:
            tzinfo: The tzinfo to apply to the new Period

        Returns:
            A Period object
        """
        properties = self._properties
        if properties.tzinfo == tzinfo:
            return self
        return _get_shifted_period(properties.with_tzinfo(tzinfo))

    def without_offset(self) -> "Period":
        """Return a Period derived from this one but with no
        date/time offset

        Returns:
            A Period object
        """
        properties = self._properties
        month_offset = properties.month_offset
        microsecond_offset = properties.microsecond_offset
        if (month_offset == 0) and (microsecond_offset == 0):
            return self
        return _get_shifted_period(
            Properties(
                step=properties.step,
                multiplier=properties.multiplier,
                month_offset=0,
                microsecond_offset=0,
                tzinfo=properties.tzinfo,
                ordinal_shift=properties.ordinal_shift,
            )
        )

    def without_ordinal_shift(self) -> "Period":
        """Return a Period derived from this one but with no
        ordinal shift.

        The returned Period object will split the timeline in
        exactly the same way as this Period, but the ordinal
        values will differ.

        Returns:
            A Period object
        """
        properties = self._properties
        if properties.ordinal_shift == 0:
            return self
        return _get_offset_period(
            Properties(
                step=properties.step,
                multiplier=properties.multiplier,
                month_offset=properties.month_offset,
                microsecond_offset=properties.microsecond_offset,
                tzinfo=properties.tzinfo,
                ordinal_shift=0,
            )
        )

    def with_origin(self, origin_date_time: dt.datetime) -> "Period":
        """Return a Period derived from this one but with the
        origin set to the specified datetime.

        The date/time offset and ordinal shift of this Period are
        discarded and recalculated such that for the resulting
        Period object the following are True:

            period.ordinal( origin_date_time ) == 0
            period.is_aligned( origin_date_time ) == True

        Returns:
            A Period object
        """
        base_period = self.base_period()
        properties = base_period._properties
        origin_ordinal = base_period.ordinal(origin_date_time)
        floor_date_time = base_period.datetime(origin_ordinal)
        if properties.step in (_STEP_MICROSECONDS, _STEP_SECONDS):
            timedelta = _naive(origin_date_time) - _naive(floor_date_time)
            offset_months = 0
            offset_microseconds = _total_microseconds(timedelta)
        else:
            offset_months = ((origin_date_time.year - floor_date_time.year) * 12) + (
                origin_date_time.month - floor_date_time.month
            )
            origin2_date_time = month_shift(origin_date_time, 0 - offset_months)
            timedelta = _naive(origin2_date_time) - _naive(floor_date_time)
            offset_microseconds = _total_microseconds(timedelta)
        return _get_shifted_period(
            Properties(
                step=properties.step,
                multiplier=properties.multiplier,
                month_offset=offset_months,
                microsecond_offset=offset_microseconds,
                tzinfo=origin_date_time.tzinfo,
                ordinal_shift=0,
            )
            .normalise_offsets()
            .with_ordinal_shift(0 - origin_ordinal)
        )

    def __str__(self) -> str:
        return self._properties.__str__()

    def __repr__(self) -> str:
        return self._properties.__repr__()

    def __hash__(self) -> int:
        return self._properties.__hash__()

    def __eq__(self, other: Any) -> bool:
        return self._properties.__eq__(other._properties)

    def __lt__(self, other: Any) -> bool:
        return self._properties.__lt__(other._properties)

    def __le__(self, other: Any) -> bool:
        return self._properties.__le__(other._properties)

    def __gt__(self, other: Any) -> bool:
        return self._properties.__gt__(other._properties)

    def __ge__(self, other: Any) -> bool:
        return self._properties.__ge__(other._properties)


class YearPeriod(Period):
    """A period of a single year, starting at midnight
    on January 1st.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_MONTHS
        assert properties.multiplier == 12
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.ordinal_shift == 0

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.year

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return dt.datetime(ordinal, 1, 1, hour=0, minute=0, second=0, tzinfo=self.tzinfo)


class MultiYearPeriod(Period):
    """A period of "n" years, with each "n" year period
    starting at midnight on January 1st.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_MONTHS
        assert properties.multiplier > 12
        assert (properties.multiplier % 12) == 0
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier // 12

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.year // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return dt.datetime(ordinal * self._n, 1, 1, hour=0, minute=0, second=0, tzinfo=self.tzinfo)


class MonthPeriod(Period):
    """A period of one month, starting at midnight on the
    1st day of the month.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_MONTHS
        assert properties.multiplier == 1
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.ordinal_shift == 0

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.year * 12 + datetime_obj.month - 1

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        year, month0 = divmod(ordinal, 12)
        return dt.datetime(year, month0 + 1, 1, hour=0, minute=0, second=0, tzinfo=self.tzinfo)


class MultiMonthPeriod(Period):
    """A period of "n" months, with each "n" month period
    starting at midnight on the first day of the
    first month of the period.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_MONTHS
        assert properties.multiplier > 1
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return (datetime_obj.year * 12 + datetime_obj.month - 1) // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        year, month0 = divmod(ordinal * self._n, 12)
        return dt.datetime(year, month0 + 1, 1, hour=0, minute=0, second=0, tzinfo=self.tzinfo)


class NaiveDayPeriod(Period):
    """A period of one day, starting at midnight,
    with no tzinfo.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier == 86_400
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.tzinfo is None
        assert properties.ordinal_shift == 0

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.toordinal()

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return dt.datetime.fromordinal(ordinal)


class AwareDayPeriod(Period):
    """A period of one day, starting at midnight,
    with a tzinfo.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier == 86_400
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert isinstance(properties.tzinfo, dt.tzinfo)
        assert properties.ordinal_shift == 0

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.toordinal()

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return dt.datetime.fromordinal(ordinal).replace(tzinfo=self.tzinfo)


class NaiveMultiDayPeriod(Period):
    """A period of "n" days, starting at midnight,
    with no tzinfo.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier >= 86_400
        assert (properties.multiplier % 86_400) == 0
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.tzinfo is None
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier // 86_400

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.toordinal() // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return dt.datetime.fromordinal(ordinal * self._n)


class AwareMultiDayPeriod(Period):
    """A period of "n" days, starting at midnight,
    with a tzinfo.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier >= 86_400
        assert (properties.multiplier % 86_400) == 0
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert isinstance(properties.tzinfo, dt.tzinfo)
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier // 86_400

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return datetime_obj.toordinal() // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return dt.datetime.fromordinal(ordinal * self._n).replace(tzinfo=self.tzinfo)


class MultiHourPeriod(Period):
    """A period of "n" hours, starting at the
    start of the hour.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier >= 3_600
        assert (properties.multiplier % 3_600) == 0
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier // 3_600

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return (datetime_obj.toordinal() * 24 + datetime_obj.hour) // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        gregorian_ordinal, hour = divmod(ordinal * self._n, 24)
        return dt.datetime.fromordinal(gregorian_ordinal).replace(hour=hour, tzinfo=self.tzinfo)


class NaiveMultiMinutePeriod(Period):
    """A period of "n" minutes, starting at second 0,
    with no tzinfo.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier >= 60
        assert (properties.multiplier % 60) == 0
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.tzinfo is None
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier // 60

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return ((datetime_obj.toordinal() * 1_440) + (datetime_obj.hour * 60 + datetime_obj.minute)) // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        gregorian_ordinal, minute_of_day = divmod(ordinal * self._n, 1_440)
        return dt.datetime.fromordinal(gregorian_ordinal) + dt.timedelta(minutes=minute_of_day)


class AwareMultiMinutePeriod(Period):
    """A period of "n" minutes, starting at second 0,
    with a tzinfo.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.multiplier >= 60
        assert (properties.multiplier % 60) == 0
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert isinstance(properties.tzinfo, dt.tzinfo)
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier // 60

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return ((datetime_obj.toordinal() * 1_440) + (datetime_obj.hour * 60 + datetime_obj.minute)) // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        gregorian_ordinal, minute_of_day = divmod(ordinal * self._n, 1_440)
        return (dt.datetime.fromordinal(gregorian_ordinal) + dt.timedelta(minutes=minute_of_day)).replace(
            tzinfo=self.tzinfo
        )


class NaiveMultiSecondPeriod(Period):
    """A period of "n" seconds, with no tzinfo."""

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.tzinfo is None
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return _gregorian_seconds(datetime_obj) // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        gregorian_ordinal, second_of_day = divmod(ordinal * self._n, 86_400)
        return dt.datetime.fromordinal(gregorian_ordinal) + dt.timedelta(seconds=second_of_day)


class AwareMultiSecondPeriod(Period):
    """A period of "n" seconds, with a tzinfo."""

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_SECONDS
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert isinstance(properties.tzinfo, dt.tzinfo)
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return _gregorian_seconds(datetime_obj) // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        gregorian_ordinal, second_of_day = divmod(ordinal * self._n, 86_400)
        return (dt.datetime.fromordinal(gregorian_ordinal) + dt.timedelta(seconds=second_of_day)).replace(
            tzinfo=self.tzinfo
        )


class NaiveMicroSecondPeriod(Period):
    """A period of "n" microseconds, with no tzinfo."""

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_MICROSECONDS
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert properties.tzinfo is None
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        seconds = _gregorian_seconds(datetime_obj)
        total_microseconds = seconds * 1_000_000 + datetime_obj.microsecond
        return total_microseconds // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        total_seconds, microseconds = divmod(ordinal * self._n, 1_000_000)
        gregorian_ordinal, second_of_day = divmod(total_seconds, 86_400)
        return dt.datetime.fromordinal(gregorian_ordinal) + dt.timedelta(
            seconds=second_of_day, microseconds=microseconds
        )


class AwareMicroSecondPeriod(Period):
    """A period of "n" microseconds, with a tzinfo."""

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.step == _STEP_MICROSECONDS
        assert properties.month_offset == 0
        assert properties.microsecond_offset == 0
        assert isinstance(properties.tzinfo, dt.tzinfo)
        assert properties.ordinal_shift == 0
        self._n = properties.multiplier

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        seconds = _gregorian_seconds(datetime_obj)
        total_microseconds = seconds * 1_000_000 + datetime_obj.microsecond
        return total_microseconds // self._n

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        total_seconds, microseconds = divmod(ordinal * self._n, 1_000_000)
        gregorian_ordinal, second_of_day = divmod(total_seconds, 86_400)
        return (
            dt.datetime.fromordinal(gregorian_ordinal) + dt.timedelta(seconds=second_of_day, microseconds=microseconds)
        ).replace(tzinfo=self.tzinfo)


def _get_period_tzinfo(
    tzinfo: Optional[dt.tzinfo],
    properties: Properties,
    naive: Callable[[Properties], Period],
    aware: Callable[[Properties], Period],
) -> Period:
    """Helper for _get_base_period function"""
    return naive(properties) if tzinfo is None else aware(properties)


def _get_shifted_period(properties: Properties) -> Period:
    """Return a Period with a possible month/second
    offset and also a possible ordinal_shift.

    Returns:
        A Period object
    """
    if properties.ordinal_shift != 0:
        return ShiftedPeriod(properties)
    return _get_offset_period(properties)


def _get_offset_period(properties: Properties) -> Period:
    """Return a Period with a possible month or second
    offset but no ordinal_shift.

    Returns:
        A Period object
    """
    assert properties.ordinal_shift == 0
    if (properties.month_offset != 0) or (properties.microsecond_offset != 0):
        return OffsetPeriod(properties)
    return _get_base_period(properties)


def _get_base_period(properties: Properties) -> Period:
    """Return a Period with no month or second
    offset and no ordinal_shift.

    Examines the supplied Properties object and chooses a
    subclass of Period that can perform the required interval
    calculations.

    Returns:
        A Period object
    """
    step = properties.step
    multiplier = properties.multiplier
    month_offset = properties.month_offset
    microsecond_offset = properties.microsecond_offset
    tzinfo = properties.tzinfo
    assert month_offset == 0
    assert microsecond_offset == 0

    if step == _STEP_MONTHS:
        if multiplier == 1:
            return MonthPeriod(properties)
        years, months_in_year = divmod(multiplier, 12)
        if months_in_year == 0:
            if years == 1:
                return YearPeriod(properties)
            return MultiYearPeriod(properties)
        return MultiMonthPeriod(properties)

    if step == _STEP_SECONDS:
        days, seconds_in_day = divmod(multiplier, 86_400)
        naive: Callable[[Properties], Period]
        aware: Callable[[Properties], Period]
        if seconds_in_day == 0:
            if days == 1:
                naive = NaiveDayPeriod
                aware = AwareDayPeriod
            else:
                naive = NaiveMultiDayPeriod
                aware = AwareMultiDayPeriod
            return _get_period_tzinfo(tzinfo, properties, naive, aware)
        hours, seconds_in_hour = divmod(seconds_in_day, 3_600)
        hours += days * 24
        if seconds_in_hour == 0:
            return MultiHourPeriod(properties)
        minutes, seconds_in_minute = divmod(seconds_in_hour, 60)
        minutes += hours * 60
        if seconds_in_minute == 0:
            naive = NaiveMultiMinutePeriod
            aware = AwareMultiMinutePeriod
        else:
            naive = NaiveMultiSecondPeriod
            aware = AwareMultiSecondPeriod
        return _get_period_tzinfo(tzinfo, properties, naive, aware)

    if step == _STEP_MICROSECONDS:
        return _get_period_tzinfo(tzinfo, properties, NaiveMicroSecondPeriod, AwareMicroSecondPeriod)

    raise ValueError("Oops")


class OffsetPeriod(Period):
    """A period that wraps another period but allows
    the period to start at a different point in time.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert (properties.month_offset > 0) or (properties.microsecond_offset > 0)
        assert properties.ordinal_shift == 0
        date_time_adjusters = DateTimeAdjusters.of_offsets(properties.month_offset, properties.microsecond_offset)
        base_period = _get_base_period(
            Properties(
                step=properties.step,
                multiplier=properties.multiplier,
                month_offset=0,
                microsecond_offset=0,
                tzinfo=properties.tzinfo,
                ordinal_shift=0,
            )
        )
        self._base_period = base_period
        self._retreat = date_time_adjusters.retreat
        self._advance = date_time_adjusters.advance

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return self._base_period.ordinal(self._retreat(datetime_obj))

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return self._advance(self._base_period.datetime(ordinal))

    @override
    def base_period(self) -> "Period":
        return self._base_period


class ShiftedPeriod(Period):
    """A period that wraps another period but adjusts
    the ordinal by a specified amount.

    This allows the 'origin' of a Period to be defined, where
    the origin is the datetime that has an ordinal of 0.
    """

    def __init__(self, properties: Properties) -> None:
        super().__init__(properties)
        assert properties.ordinal_shift != 0
        offset_period = _get_offset_period(
            Properties(
                step=properties.step,
                multiplier=properties.multiplier,
                month_offset=properties.month_offset,
                microsecond_offset=properties.microsecond_offset,
                tzinfo=properties.tzinfo,
                ordinal_shift=0,
            )
        )
        self._offset_period = offset_period
        self._ordinal_shift = properties.ordinal_shift

    @override
    def ordinal(self, datetime_obj: dt.datetime) -> int:
        return self._offset_period.ordinal(datetime_obj) + self._ordinal_shift

    @override
    def datetime(self, ordinal: int) -> dt.datetime:
        return self._offset_period.datetime(ordinal - self._ordinal_shift)

    @override
    def base_period(self) -> "Period":
        return self._offset_period.base_period()


# ------------------------------------------------------------------------------
def _datetime_regex(prefix: str) -> str:
    """Return a regular expression that can parse an ISO 8601
    format datetime

    All the regular expression groups are given a name which
    is prefix by the string argument.

    Args:
        prefix: The prefix to apply to the regular expression
                group names

    Returns:
        A regular expression string that can parse an
        ISO 8601 format datetime
    """
    return (
        rf"(?P<{prefix}_yyyy>\d{{4}})"
        rf"(?:-(?P<{prefix}_mm>\d{{1,2}})"
        rf"(?:-(?P<{prefix}_dd>\d{{1,2}})"
        rf"(?:(?:[Tt]|\s++)(?P<{prefix}_HH>\d{{1,2}})"
        rf"(?::(?P<{prefix}_MM>\d{{1,2}})"
        rf"(?::(?P<{prefix}_SS>\d{{1,2}})"
        rf"(?:\.(?P<{prefix}_MS>\d{{1,6}}))?"
        rf"(?:(?P<{prefix}_Z>[Zz]|(?:[+-]\d{{1,2}}(?::\d{{1,2}})?))"
        r")?)?)?)?)?)?"
    )


def _date_match(prefix: str, matcher: re.Match[str]) -> dt.date:
    """Extract a date object out of a regular expression matcher that
    has parsed an ISO 8601 format datetime

    The matcher argument is assumed to come from a regular expression
    created by the _datetime_regex() function.

    Args:
        prefix: The prefix to apply to the regular expression
                group names
        matcher: A regular expression Match object from which
                 to extract the various date fields

    Returns:
        A date object
    """
    return dt.date(
        year=int(matcher.group(f"{prefix}_yyyy")),
        month=_str2int(matcher.group(f"{prefix}_mm"), 1),
        day=_str2int(matcher.group(f"{prefix}_dd"), 1),
    )


def _time_match(prefix: str, matcher: re.Match[str]) -> dt.time:
    """Extract a time object out of a regular expression matcher that
    has parsed an ISO 8601 format datetime

    The matcher argument is assumed to come from a regular expression
    created by the _datetime_regex() function.

    Args:
        prefix: The prefix to apply to the regular expression
                group names
        matcher: A regular expression Match object from which
                 to extract the various time fields

    Returns:
        A time object
    """
    return dt.time(
        hour=_str2int(matcher.group(f"{prefix}_HH")),
        minute=_str2int(matcher.group(f"{prefix}_MM")),
        second=_str2int(matcher.group(f"{prefix}_SS")),
        microsecond=_str2microseconds(matcher.group(f"{prefix}_MS")),
        tzinfo=_tzinfo_match(prefix, matcher),
    )


def _timezone(utc_offset: Optional[str]) -> Optional[dt.tzinfo]:
    """Convert an optional string into an optional tzinfo
    object

    Args:
        utc_offset: A string containing the timezone offset

    Returns:
        A tzinfo object, or None
    """
    if utc_offset is None:
        return None
    seconds = 0
    if utc_offset.upper() != "Z":
        plus_minus = utc_offset[0]
        offset = utc_offset[1:]
        if offset.isdigit():
            seconds = 3_600 * int(offset)
        else:
            arr = offset.split(":")
            seconds = 60 * ((60 * int(arr[0])) + int(arr[1]))
        if plus_minus == "-":
            seconds = 0 - seconds
    return dt.timezone(dt.timedelta(seconds=seconds))


def _tzinfo_match(prefix: str, matcher: re.Match[str]) -> Optional[dt.tzinfo]:
    """Extract a tzinfo object out of a regular expression matcher that
    has parsed an ISO 8601 format datetime

    The matcher argument is assumed to come from a regular expression
    created by the _datetime_regex() function.

    Args:
        prefix: The prefix to apply to the regular expression
                group names
        matcher: A regular expression Match object from which
                 to extract the various tzinfo fields

    Returns:
        A tzinfo object, or None
    """
    return _timezone(matcher.group(f"{prefix}_Z"))


def _datetime_match(prefix: str, matcher: re.Match[str]) -> dt.datetime:
    """Extract a datetime object out of a regular expression matcher that
    has parsed an ISO 8601 format datetime

    The matcher argument is assumed to come from a regular expression
    created by the _datetime_regex() function.

    Args:
        prefix: The prefix to apply to the regular expression
                group names
        matcher: A regular expression Match object from which
                 to extract the various date and time fields

    Returns:
        A datetime object
    """
    return dt.datetime.combine(date=_date_match(prefix, matcher), time=_time_match(prefix, matcher))


def _match_period(matcher: re.Match[str]) -> Period:
    """Return a Period object from a regular expression matcher that
    has parsed an ISO 8601 duration format string

    The matcher argument is assumed to come from a regular expression
    created using the _period_regex() function.

    Args:
        matcher: A regular expression Match object from which
                 to extract the various date and time fields

    Returns:
        A Period object
    """
    period_fields = _period_match("period", matcher)
    base_properties = period_fields.get_base_properties()
    return _get_base_period(base_properties)


def _match_period_offset(matcher: re.Match[str]) -> Period:
    """Return a Period object from a regular expression matcher that
    has parsed an ISO 8601 duration format string that may include
    a non-standard offset

    The matcher argument is assumed to come from a regular expression
    created using the _period_regex() function.

    Args:
        matcher: A regular expression Match object from which
                 to extract the various date and time fields

    Returns:
        A Period object
    """
    period_fields = _period_match("period", matcher)
    offset_period_fields = _period_match("offset", matcher)
    return _get_offset_period(period_fields.get_base_properties().with_offset_period_fields(offset_period_fields))


def _match_datetime_period(matcher: re.Match[str]) -> Period:
    """Return a Period object from a regular expression matcher that
    has parsed an ISO 8601 duration string of the form <start>/<duration>

    The matcher argument is assumed to come from a regular expression
    created using the _datetime_regex() and _period_regex() functions.

    Args:
        matcher: A regular expression Match object from which
                 to extract the various date, time and period
                 fields

    Returns:
        A Period object
    """
    date_time = _datetime_match("d", matcher)
    period_fields = _period_match("period", matcher)
    properties = period_fields.get_base_properties()
    return _get_shifted_period(properties).with_origin(date_time)


def _match_repr(matcher: re.Match[str]) -> Period:
    """Return a Period object from a regular expression matcher that
    has parsed a repr string from some other Period object

    Args:
        matcher: A regular expression Match object from which
                 to extract the various fields required to
                 recreate a Period from a __repr__ string

    Returns:
        A Period object
    """
    period_fields: PeriodFields = _period_match("period", matcher)
    offset: str = matcher.group("offset")
    offset_period_fields: Optional[PeriodFields] = None
    if offset == "+":
        offset_period_fields = _period_match("offset", matcher)
    zone: str = matcher.group("zone")
    tzinfo: Optional[dt.tzinfo] = None
    if zone is not None and len(zone) > 0:
        tzinfo = _timezone(zone)
    shift: str = matcher.group("shift")
    ordinal_shift: int = 0
    if shift is not None:
        ordinal_shift = int(shift)
    properties = period_fields.get_base_properties()
    if offset_period_fields is not None:
        properties = properties.with_offset_period_fields(offset_period_fields)
    if tzinfo is not None:
        properties = properties.with_tzinfo(tzinfo)
    if ordinal_shift != 0:
        properties = properties.with_ordinal_shift(ordinal_shift)
    return _get_shifted_period(properties)


#
# A regular expression Pattern used to parse an ISO 8601 duration
# string such as P1Y, P7M, P5Y3M, P8D9H1M4S, etc.
#
_RE_PERIOD = re.compile(r"^" r"[Pp]" + _period_regex("period") + r"$")

#
# A regular expression Pattern used to parse an extended
# ISO 8601 duration string such as P2Y+3M, P1Y+9M9H.
#
# This format has been invented by me (os) and is totally
# non-standard, but something like this is needed as
# ISO duration strings have no concept of a date/time offset.
#
_RE_PERIOD_OFFSET = re.compile(r"^" r"[Pp]" + _period_regex("period") + r"\+" + _period_regex("offset") + r"$")

#
# A regular expression Pattern used to parse an ISO 8601 duration
# string of the format <start>/<duration> such as "1883-01-01/P1D".
#
_RE_DATETIME_PERIOD = re.compile(r"^" + _datetime_regex("d") + r"/" r"[Pp]" + _period_regex("period") + r"$")

#
# A regular expression Pattern used to parse a Period
# __repr__ string.
#
# The intent is that the repr string can be passed into the
# Period.of_repr() static method to recreated the Period
# object.  The current format achives this, up to a point,
# but tzinfo objects cause problems as there is no standard
# way to convert tzinfo objects to/from strings.
#
# It might be best to remove the Period.of_repr() method
# completely.
#
_RE_REPR = re.compile(
    r"^"
    r"[Pp]" + _period_regex("period") + r"(?:"  # start optional offset
    r"(?P<offset>\+)" + _period_regex("offset") + r")?"  # end optional offset
    r"\["  # start timezone
    r"(?:"
    r"(?P<zone>[Zz]|(?:[+-]\d{1,2}(?::\d{1,2})?))"
    r")?"
    r"\]"  # end timezone
    r"(?:"  # start optional ordinal shift
    r"(?P<shift>-?\d+)"
    r")?"  # end optional ordinal shift
    r"$"
)

#
# A list of tuples used to parse a string into a Period
#   Each tuple contains two elements:
#      [0] - a regular expression Pattern for matching a specific
#            format of string
#      [1] - a function that converts a regular expression Match
#            object created from the Pattern into an optional
#            Period object.
#
_PARSERS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], Optional[Period]]]] = [
    (_RE_PERIOD, _match_period),
    (_RE_PERIOD_OFFSET, _match_period_offset),
    (_RE_DATETIME_PERIOD, _match_datetime_period),
    (_RE_REPR, _match_repr),
]


def _get_period(period_string: str) -> Optional[Period]:
    """Return an optional Period object from a string

    Tries a number of possible string formats and returns
    the first one that gives a match.  If none do then returns
    None.

    Returns:
        A Period object, or None if the supplied string could
        not be converted to a Period
    """
    regex: re.Pattern[str]
    match_fn: Callable[[re.Match[str]], Optional[Period]]
    for regex, match_fn in _PARSERS:
        matcher: Optional[re.Match[str]] = regex.match(period_string)
        if matcher:
            period = match_fn(matcher)
            if period is not None:
                return period
    return None


def _of(period_string: str) -> Period:
    """Return a Period object from a string

    Tries a number of possible string formats and returns
    the first one that gives a match.  If none do then raises
    an error.

    Returns:
        A Period object
    """
    period: Optional[Period] = _get_period(period_string)
    if period is None:
        raise ValueError(f"Illegal period: {period_string}")
    return period


def _of_iso_duration(iso_8601_duration: str) -> Period:
    """Return a Period object from an ISO 8601 duration string

    Returns:
        A Period object

    Raises:
        ValueError if the string does not contain a valid
        ISO 8601 duration.
        ValueError if the string contains a valid ISO 8601
        duration that cannot be converted into a Period object
    """
    matcher: Optional[re.Match[str]] = _RE_PERIOD.match(iso_8601_duration)
    if matcher:
        return _match_period(matcher)
    raise ValueError(f"Illegal ISO 8601 duration: {iso_8601_duration}")


def _of_duration(duration: str) -> Period:
    """Return a Period from an (extended) ISO 8601 duration string

    Both plain and extended ISO 8601 duration strings are supported.
    Not all legal ISO 8601 duration strings can be converted to
    a Period object.

    Returns:
        A Period object

    Raises:
        ValueError if duration does not contains a valid duration
        string
    """
    matcher: Optional[re.Match[str]] = _RE_PERIOD_OFFSET.match(duration)
    if matcher:
        return _match_period_offset(matcher)
    matcher = _RE_PERIOD.match(duration)
    if matcher:
        return _match_period(matcher)
    raise ValueError(f"Illegal duration: {duration}")


def _of_date_and_duration(date_duration: str) -> Period:
    """Return a Period object from an ISO 8601 duration string
    of the form <start>/<duration>

    Not all legal ISO 8601 duration strings can be converted to
    a Period object.

    Returns:
        A Period object

    Raises:
        ValueError if date_duration does not contains a valid value
    """
    matcher: Optional[re.Match[str]] = _RE_DATETIME_PERIOD.match(date_duration)
    if matcher:
        return _match_datetime_period(matcher)
    raise ValueError(f"Illegal date/duration string: {date_duration}")


def _of_repr(repr_string: str) -> Period:
    """Return a Period from a Period __repr__ string

    Returns:
        A Period object

    Raises:
        ValueError if the string does not contain a valid repr string
    """
    matcher: Optional[re.Match[str]] = _RE_REPR.match(repr_string)
    if matcher:
        return _match_repr(matcher)
    raise ValueError(f"Illegal repr string: {repr_string}")
