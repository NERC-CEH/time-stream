# [start_block_1]
from datetime import datetime, timedelta

import polars as pl

from time_series import TimeSeries, Period
# [end_block_1]

from utils import suppress_output


def create_simple_time_series():
    # [start_block_2]
    # Create sample data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20, 21, 19, 26, 24, 26, 28, 30, 31, 29]
    precipitation = [0, 0, 5, 10, 2, 0, 0, 3, 1, 0]

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation
    })

    # Create a simple TimeSeries
    ts = TimeSeries(
        df=df,
        time_name="timestamp"  # Specify which column contains the primary datetime values
    )
    # [end_block_2]
    return ts


def show_default_resolution():
    ts = create_simple_time_series()
    # [start_block_3]
    print(ts.resolution)
    print(ts.periodicity)
    # [end_block_3]


def create_simple_time_series_with_periods():
    # Create sample data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20, 21, 19, 26, 24, 26, 28, 30, 31, 29]
    precipitation = [0, 0, 5, 10, 2, 0, 0, 3, 1, 0]

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation
    })

    # [start_block_4]
    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution = Period.of_days(1),  # Each timestamp is at day precision
        periodicity = Period.of_days(1)  # Data points are spaced 1 day apart
    )

    print(ts.resolution)
    print(ts.periodicity)
    # [end_block_4]
    return ts


def create_simple_time_series_with_metadata():
    # Create sample data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20, 21, 19, 26, 24, 26, 28, 30, 31, 29]
    precipitation = [0, 0, 5, 10, 2, 0, 0, 3, 1, 0]

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation
    })

    # [start_block_5]
    # TimeSeries metadata
    ts_metadata = {
        "location": "River Thames",
        "elevation": 100,
        "station_id": "ABC123"
    }

    # Column metadata
    col_metadata = {
        "temperature": {
            "units": "°C",
            "description": "Average temperature"
        },
        "precipitation": {
            "units": "mm",
            "description": "Precipitation amount",
            "instrument_type": "Tipping bucket"
            # Note that metadata keys are not required to be the same for all columns
        }
    }

    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=Period.of_days(1),
        periodicity=Period.of_days(1),
        metadata=ts_metadata,
        column_metadata=col_metadata
    )
    # [end_block_5]
    return ts


def show_time_series_metadata():
    ts = create_simple_time_series_with_metadata()
    # [start_block_6]
    print("All metadata:", ts.metadata())
    print("Specific keys:", ts.metadata(["location", "elevation"]))
    # [end_block_6]


def show_column_metadata():
    ts = create_simple_time_series_with_metadata()
    # [start_block_7]
    # Access via attributes:
    print(ts.temperature.units)
    print(ts.precipitation.description)

    # Or via metadata method:
    print(ts.precipitation.metadata(["units", "instrument_type"]))
    # [end_block_7]