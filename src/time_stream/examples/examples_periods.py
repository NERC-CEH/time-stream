from time_stream import Period


def simple_factory_methods() -> None:
    # [start_block_1]
    from time_stream import Period

    # Create periods using specific methods
    Period.of_years(1)
    Period.of_months(3)
    Period.of_months(1)
    Period.of_days(1)
    Period.of_hours(1)
    Period.of_minutes(15)
    Period.of_seconds(1)
    Period.of_microseconds(1)
    # [end_block_1]


def iso_factory_methods() -> None:
    # [start_block_2]
    # Using ISO 8601 duration strings
    Period.of_iso_duration("P1Y")
    Period.of_iso_duration("P3M")
    Period.of_iso_duration("P1M")
    Period.of_iso_duration("P1D")
    Period.of_iso_duration("PT1H")
    Period.of_iso_duration("PT15M")
    Period.of_iso_duration("PT1S")
    Period.of_iso_duration("PT0.000001S")
    # [end_block_2]


def timedelta_factory_methods() -> None:
    # [start_block_3]
    from datetime import timedelta

    # Using timedelta objects
    Period.of_timedelta(timedelta(days=1))
    Period.of_timedelta(timedelta(hours=2))
    Period.of_timedelta(timedelta(minutes=30))
    Period.of_timedelta(timedelta(seconds=1))
    Period.of_timedelta(timedelta(microseconds=1))
    # [end_block_3]


def offset_periods() -> None:
    # [start_block_4]
    # Water year (Starting on 9am Oct 1)
    Period.of_years(1).with_month_offset(10).with_hour_offset(9)

    # Water day starting at 9am
    Period.of_days(1).with_hour_offset(9)

    # Custom hour with 30-minute offset, e.g. for data with timestamps (00:30, 01:30, 02:30 ...)
    Period.of_hours(1).with_minute_offset(30)
    # [end_block_4]
