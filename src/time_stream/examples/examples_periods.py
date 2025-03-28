from time_series import Period


def simple_factory_methods():
    # [start_block_1]
    from time_series import Period

    # Create periods using specific methods
    one_year = Period.of_years(1)
    quarterly = Period.of_months(3)
    one_month = Period.of_months(1)
    one_day = Period.of_days(1)
    one_hour = Period.of_hours(1)
    fifteen_minutes = Period.of_minutes(15)
    one_second = Period.of_seconds(1)
    one_microsecond = Period.of_microseconds(1)
    # [end_block_1]


def iso_factory_methods():
    # [start_block_2]
    # Using ISO 8601 duration strings
    one_year = Period.of_iso_duration("P1Y")
    quarterly = Period.of_iso_duration("P3M")
    one_month = Period.of_iso_duration("P1M")
    one_day = Period.of_iso_duration("P1D")
    one_hour = Period.of_iso_duration("PT1H")
    fifteen_minutes = Period.of_iso_duration("PT15M")
    one_second = Period.of_iso_duration("PT1S")
    one_microsecond = Period.of_iso_duration("PT0.000001S")
    # [end_block_2]

def timedelta_factory_methods():
    # [start_block_3]
    from datetime import timedelta

    # Using timedelta objects
    one_day = Period.of_timedelta(timedelta(days=1))
    two_hours = Period.of_timedelta(timedelta(hours=2))
    thirty_min = Period.of_timedelta(timedelta(minutes=30))
    one_second = Period.of_timedelta(timedelta(seconds=1))
    one_microsecond = Period.of_timedelta(timedelta(microseconds=1))
    # [end_block_3]


def offset_periods():
    # [start_block_4]
    # Water year (Starting on 9am Oct 1)
    water_year = Period.of_years(1).with_month_offset(10).with_hour_offset(9)

    # Water day starting at 9am
    water_day = Period.of_days(1).with_hour_offset(9)

    # Custom hour with 30-minute offset, e.g. for data with timestamps (00:30, 01:30, 02:30 ...)
    hour_half = Period.of_hours(1).with_minute_offset(30)
    # [end_block_4]