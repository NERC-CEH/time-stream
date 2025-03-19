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
    print("All metadata: ", ts.metadata())
    print("Specific keys: ", ts.metadata(["location", "elevation"]))
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


def resolution_check_fail():
    # [start_block_8]
    # This will raise a warning because some timestamps don't align to midnight (00:00:00),
    # as required by daily resolution
    timestamps = [
        datetime(2023, 1, 1, 0, 0, 0),  # Aligned to midnight
        datetime(2023, 1, 2, 0, 0, 0),  # Aligned to midnight
        datetime(2023, 1, 3, 12, 0, 0),  # Not aligned (noon)
    ]

    df = pl.DataFrame({"timestamp": timestamps, "value": [1, 2, 3]})

    # This will raise a UserWarning about resolution alignment
    try:
        ts = TimeSeries(
            df=df,
            time_name="timestamp",
            resolution=Period.of_days(1)
        )
    except UserWarning as w:
        print(f"Warning: {w}")
    # [end_block_8]


def periodicity_check_fail():
    # [start_block_9]
    # This will raise a warning because we have two points in the same day
    timestamps = [
        datetime(2023, 1, 1, 0, 0, 0),
        datetime(2023, 1, 1, 0, 0, 0),  # Same day as above
        datetime(2023, 1, 2, 0, 0, 0),
    ]

    df = pl.DataFrame({"timestamp": timestamps, "value": [1, 2, 3]})

    # This will raise a UserWarning about periodicity
    try:
        ts = TimeSeries(
            df=df,
            time_name="timestamp",
            periodicity=Period.of_days(1)
        )
    except UserWarning as w:
        print(f"Warning: {w}")
    # [end_block_9]


def accessing_data():
    ts = create_simple_time_series_with_metadata()
    # [start_block_10]
    # Get the full DataFrame
    df = ts.df
    # [end_block_10]
    # [start_block_11]
    # Select specific columns from the DataFrame
    temp_precip_df = ts.df.select(["timestamp", "temperature", "precipitation"])

    # Filter the DataFrame
    rainy_days_df = ts.df.filter(pl.col("precipitation") > 0)
    # [end_block_11]
    # [start_block_12]
    # Access column as a TimeSeriesColumn object
    temperature_col = ts.temperature
    print("Type temperature_col: ", type(temperature_col))

    # Get the underlying data from a column
    temperature_data = ts.temperature.data
    print("Type temperature_data: ", type(temperature_data))

    # Access column properties
    temperature_units = ts.temperature.units
    print("Temperature units: ", temperature_units)

    # Get column as a TimeSeries
    temperature_ts = ts["temperature"]
    print("Type temperature_ts: ", type(temperature_ts))
    print(temperature_ts)

    # Select multiple columns as a TimeSeries
    selected_ts = ts.select(["temperature", "precipitation"])
    # or
    selected_ts = ts[["temperature", "precipitation"]]
    print("Type selected_ts: ", type(selected_ts))
    print(selected_ts)
    # [end_block_12]


def create_simple_time_series_with_supp_cols():
    # [start_block_13]
    # Create sample data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20, 21, 1, 26, 24, 26, 28, 41, 51, None]
    precipitation = [0, 0, 5, 10, 2, 0, 0, 3, 1, 0]
    observer_comments = ["", "", "Power cut between 8am and 1pm", "", "", "",
                         "Agricultural work in adjacent field", "", "", "Tree felling"]

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation,
        "observer_comments": observer_comments
    })

    # Create a TimeSeries
    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=Period.of_days(1),
        periodicity=Period.of_days(1),
        supplementary_columns=["observer_comments"]
    )

    print("Data columns: ", ts.data_columns)
    print("Supplementary columns: ", ts.supplementary_columns)
    print("Observer comments column: ", ts.observer_comments)
    # [end_block_13]
    return ts


def convert_existing_col_to_supp_col():
    with suppress_output():
        ts = create_simple_time_series_with_supp_cols()
    # [start_block_14]
    # Convert an existing column to supplementary
    ts.set_supplementary_column("precipitation")

    print("Data columns: ", ts.data_columns)
    print("Supplementary columns: ", ts.supplementary_columns)
    # [end_block_14]
    return ts


def add_new_col_as_supp_col():
    with suppress_output():
        ts = convert_existing_col_to_supp_col()
    # [start_block_15]
    # Add a new supplementary column, with new data
    new_data = [12, 15, 6, 12, 10, 14, 19, 17, 16, 13]
    ts.init_supplementary_column("battery_voltage", new_data)

    print("Data columns: ", ts.data_columns)
    print("Supplementary columns: ", ts.supplementary_columns)
    print("Battery voltage column: ", ts.battery_voltage)
    # [end_block_15]


def create_simple_time_series_with_flag_cols():
    # [start_block_16]
    # Create sample data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20, 21, 1, 26, 24, 26, 28, 41, 51, None]
    precipitation = [0, 0, 5, 10, 2, 0, 0, 3, 1, 0]

    temperature_qc_flags = [0, 0, 2, 0, 0, 0, 0, 0, 3, 8]
    flag_systems = {"quality_control_checks": {"OUT_OF_RANGE": 1, "SPIKE": 2, "LOW_VOLTAGE": 4, "MISSING": 8}}

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation,
        "temperature_qc_flags": temperature_qc_flags,
    })

    # Create a TimeSeries
    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=Period.of_days(1),
        periodicity=Period.of_days(1),
        flag_systems=flag_systems,
        flag_columns={"temperature_qc_flags": "quality_control_checks"}
    )

    print("Data columns: ", ts.data_columns)
    print("Flag columns: ", ts.flag_columns)
    print("Flag systems: ", ts.flag_systems)
    print("Temperature flag column: ", ts.temperature_qc_flags)
    # [end_block_16]
    return ts


def set_flag_cols_after_init():
    # [start_block_17]
    # Create sample data
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20, 21, 1, 26, 24, 26, 28, 41, 51, None]
    precipitation = [0, 0, 5, 10, 2, 0, 0, 3, 1, 0]

    flag_systems = {"quality_control_checks": {"OUT_OF_RANGE": 1, "SPIKE": 2, "LOW_VOLTAGE": 4, "MISSING": 8}}

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation
    })

    # Create a TimeSeries
    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=Period.of_days(1),
        periodicity=Period.of_days(1),
        flag_systems=flag_systems
    )

    # Add a flag column for temperature data, which will use the quality_control_checks flag system
    ts.init_flag_column("quality_control_checks", "temperature_qc_flags")
    # [end_block_17]
    return ts

def set_and_remove_flags():
    with suppress_output():
        ts = set_flag_cols_after_init()

    # [start_block_18]
    # Add flags
    ts.add_flag("temperature_qc_flags", "OUT_OF_RANGE", pl.col("temperature") > 40)
    ts.add_flag("temperature_qc_flags", "MISSING", pl.col("temperature").is_null())

    print(ts.temperature_qc_flags)

    # Remove a flag
    ts.remove_flag("temperature_qc_flags", "OUT_OF_RANGE", pl.col("temperature") <= 45)

    print(ts.temperature_qc_flags)
    # [end_block_18]

def add_new_column_to_df():
    with suppress_output():
        ts = create_simple_time_series_with_periods()

    # [start_block_19]
    # Update the DataFrame by adding a new column
    ts.df = ts.df.with_columns(
        (pl.col("temperature") * 1.8 + 32).alias("temperature_f")
    )

    # The new column will be available as a DataColumn
    print("New temperature column in fahrenheit: ", ts[["temperature", "temperature_f"]])
    # [end_block_19]

def update_time_of_df_error():
    with suppress_output():
        ts = create_simple_time_series_with_periods()

    # [start_block_20]
    # Try and update the DataFrame by filtering columns
    # (which inherently removes some of the time series)
    try:
        ts.df = ts.df.filter(pl.col("precipitation") > 0)
    except ValueError as w:
        print(f"Warning: {w}")
    # [end_block_20]

def add_column_relationships():
    with suppress_output():
        ts = create_simple_time_series_with_supp_cols()
        new_data = [12, 15, 6, 12, 10, 14, 19, 17, 16, 13]
        ts.init_supplementary_column("battery_voltage", new_data)

        flag_system = {"OUT_OF_RANGE": 1, "SPIKE": 2, "LOW_VOLTAGE": 4, "MISSING": 8}
        ts.add_flag_system("quality_control_checks", flag_system)
        ts.init_flag_column("quality_control_checks", "temperature_qc_flags")
        ts.add_flag("temperature_qc_flags", "OUT_OF_RANGE", pl.col("temperature") > 40)

    # [start_block_21]
    # Starting with an example TimeSeries, with supplementary and flag columns:
    print(ts)

    print("Data columns: ", ts.data_columns)
    print("Supplementary columns: ", ts.supplementary_columns)
    print("Flag columns: ", ts.flag_columns)

    # Create a relationship between the supplementary column and data columns (using method on the TimeSeries object)
    ts.add_column_relationship("battery_voltage", ["temperature", "precipitation"])

    # Create a relationship between temperature and its flags (using method on the Column object)
    ts.temperature.add_relationship("temperature_qc_flags")

    print("Temperature column relationships: ", ts.temperature.get_relationships())
    print("Battery voltage column relationships: ", ts.battery_voltage.get_relationships())
    # [end_block_21]

add_column_relationships()