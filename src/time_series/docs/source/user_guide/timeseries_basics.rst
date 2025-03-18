TimeSeries Basics
================

This guide covers the fundamentals of working with the ``TimeSeries`` class, the core data structure in the
Time Series Package.

Creating a TimeSeries
--------------------

A ``TimeSeries`` wraps a Polars DataFrame and adds specialized functionality for time series operations.

Basic Creation
~~~~~~~~~~~~~

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series()

Without specifying resolution and periodicity, the default initialisation sets these properties to "1 microsecond", to
account for any set of datetime values:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_3]
   :end-before: [end_block_3]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.show_default_resolution()

With Resolution and Periodicity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Although the default of 1 microsecond will account for any datetime values, for more control over certain
time series functionality it is important to specify the actual resolution and periodicity if known:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_periods()

With Metadata
~~~~~~~~~~~~

``TimeSeries`` can be initialised with metadata to describe your data. This can be metadata about the time series as
a whole, or about the individual columns.

Keeping the metadata and the data together in one object like this
can help simplify downstream processes, such as derivation functions, running infilling routines, plotting data, etc.

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_metadata()

Time series level metadata can be accessed via the ``.metadata()`` method:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_6]
   :end-before: [end_block_6]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.show_time_series_metadata()

Column level metadata can be accessed via attributes on the column itself, or via the ``column.metadata()`` method:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_7]
   :end-before: [end_block_7]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.show_column_metadata()

Time Validation
--------------

The ``TimeSeries`` performs validation on timestamps:

Resolution Validation
~~~~~~~~~~~~~~~~~~~

The resolution defines how precise the timestamps should be:

.. code-block:: python

    # This will raise a warning because some timestamps might not align
    # to midnight (00:00:00) as required by daily resolution
    timestamps = [
        datetime(2023, 1, 1, 0, 0, 0),   # Aligned to midnight
        datetime(2023, 1, 2, 0, 0, 0),   # Aligned to midnight
        datetime(2023, 1, 3, 12, 0, 0),  # Not aligned (noon)
    ]

    df = pl.DataFrame({"timestamp": timestamps, "value": [1, 2, 3]})

    # This will raise a UserWarning about resolution alignment
    try:
        ts = TimeSeries(
            df=df,
            time_name="timestamp",
            resolution=Period.of_days(1)  # Requires timestamps at midnight
        )
    except UserWarning as w:
        print(f"Warning: {w}")

Periodicity Validation
~~~~~~~~~~~~~~~~~~~~~

The periodicity defines how frequently data points should appear:

.. code-block:: python

    # This will raise a warning because we have two points in the same day
    timestamps = [
        datetime(2023, 1, 1, 0, 0, 0),
        datetime(2023, 1, 1, 12, 0, 0),  # Same day as above
        datetime(2023, 1, 2, 0, 0, 0),
    ]

    df = pl.DataFrame({"timestamp": timestamps, "value": [1, 2, 3]})

    # This will raise a UserWarning about periodicity
    try:
        ts = TimeSeries(
            df=df,
            time_name="timestamp",
            periodicity=Period.of_days(1)  # Requires one point per day
        )
    except UserWarning as w:
        print(f"Warning: {w}")

Accessing Data
-------------

There are multiple ways to access data from a ``TimeSeries``:

Accessing the DataFrame
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get the full DataFrame
    df = ts.df

    # Select specific columns from the DataFrame
    temp_precip_df = ts.df.select(["timestamp", "temperature", "precipitation"])

    # Filter the DataFrame
    rainy_days_df = ts.df.filter(pl.col("precipitation") > 0)

Accessing Columns
~~~~~~~~~~~~~~~

.. code-block:: python

    # Access column as a TimeSeriesColumn object
    temp_col = ts.temperature

    # Get the underlying data from a column
    temp_data = ts.temperature.data

    # Access column properties
    temp_units = ts.temperature.units          # Using metadata

    # Get column as a TimeSeries
    temp_ts = ts["temperature"]

    # Select multiple columns as a TimeSeries
    selected_ts = ts.select(["temperature", "precipitation"])
    # or
    selected_ts = ts[["temperature", "precipitation"]]

Accessing Metadata
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get column metadata
    temp_metadata = ts.columns["temperature"].metadata()
    # or
    temp_metadata = ts.column_metadata("temperature")

    # Get a specific metadata value
    temp_units = ts.temperature.units

    # Get TimeSeries metadata
    ts_metadata = ts.metadata()

    # Get a specific TimeSeries metadata value
    location = ts.location  # Assuming "location" exists in metadata

Working with Columns
-------------------

The ``TimeSeries`` class provides methods for working with different column types:

Column Types
~~~~~~~~~~~

There are four column types:

1. **Primary Time Column**: The datetime column
2. **Data Columns**: Regular data columns (default type)
3. **Supplementary Columns**: Metadata or contextual information
4. **Flag Columns**: Quality control markers

Creating Supplementary Columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Add a new supplementary column with station information
    ts.init_supplementary_column("station_name", "Central Station")

    # Convert an existing column to supplementary
    ts.set_supplementary_column("precipitation")

    # Access supplementary columns
    supp_columns = ts.supplementary_columns

Creating Flag Columns
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create a flag system
    flag_systems = {"quality": {"GOOD": 1, "SUSPICIOUS": 2, "MISSING": 4}}

    # Create a new TimeSeries with the flag system
    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        flag_systems=flag_systems
    )

    # Add a flag column for temperature data
    ts.init_flag_column("quality", "temp_flags")

    # Convert an existing column to a flag column
    if "existing_flags" in ts.df.columns:
        ts.set_flag_column("quality", "existing_flags")

    # Add flags to data
    ts.add_flag("temp_flags", "SUSPICIOUS", pl.col("temperature") > 25)
    ts.add_flag("temp_flags", "MISSING", pl.col("temperature").is_null())

    # Remove a flag
    ts.remove_flag("temp_flags", "SUSPICIOUS", pl.col("temperature") <= 22)

Updating a TimeSeries
--------------------

You can update the underlying DataFrame while preserving column settings:

.. code-block:: python

    # Update the entire DataFrame
    new_df = ts.df.with_columns(
        (pl.col("temperature") * 1.8 + 32).alias("temperature_f")
    )
    ts.df = new_df

    # The new column will be available as a DataColumn
    print(ts.temperature_f.data)

    # Filter data
    ts.df = ts.df.filter(pl.col("precipitation") > 0)

    # Add a calculated column
    ts.df = ts.df.with_columns(
        (pl.col("temperature") / 10).alias("temperature_normalized")
    )

Column Relationships
------------------

You can define relationships between columns:

.. code-block:: python

    # Create a flag column for temperature
    ts.init_flag_column("quality", "temp_flags")

    # Create a relationship between temperature and its flags
    ts.add_column_relationship("temperature", "temp_flags")

    # Create supplementary info for temperature
    ts.init_supplementary_column("temp_sensor_id", "TEMP123")

    # Create a relationship
    ts.add_column_relationship("temperature", "temp_sensor_id")

    # Get related flag column
    flag_col = ts.get_flag_system_column("temperature", "quality")

    # Remove a relationship
    ts.remove_column_relationship("temperature", "temp_sensor_id")

Aggregating Data
--------------

The ``TimeSeries`` class provides powerful aggregation capabilities:

.. code-block:: python

    from time_series.aggregation import Mean, Min, Max

    # Create a monthly aggregation of daily data
    monthly_period = Period.of_months(1)

    # Calculate monthly mean temperature
    monthly_mean_temp = ts.aggregate(monthly_period, Mean, "temperature")

    # Calculate monthly minimum temperature
    monthly_min_temp = ts.aggregate(monthly_period, Min, "temperature")

    # Calculate monthly maximum temperature
    monthly_max_temp = ts.aggregate(monthly_period, Max, "temperature")

    # Use it with other periods
    weekly_period = Period.of_days(7)
    weekly_mean_temp = ts.aggregate(weekly_period, Mean, "temperature")

For more details on aggregation, see the dedicated :doc:`aggregation` guide.

Best Practices
------------

1. **Always specify resolution and periodicity** when creating a TimeSeries to ensure proper validation
2. **Use appropriate column types**:
   - Use data columns for core measurements
   - Use supplementary columns for metadata or context
   - Use flag columns for quality control
3. **Define relationships** between related columns
4. **Add metadata** to enhance understanding of your data
5. **Regularly validate** your time series against expected resolution and periodicity

Next Steps
---------

Now that you understand the basics of the ``TimeSeries`` class, explore:

- :doc:`periods` - Learn more about working with time periods
- :doc:`aggregation` - Dive deeper into aggregation capabilities
- :doc:`flagging` - Master the quality control system
- :doc:`column_relationships` - Understand column relationships in detail