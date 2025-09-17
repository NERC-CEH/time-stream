TimeSeries Basics
================

This guide covers the fundamentals of working with the ``TimeSeries`` class, the core data structure in the
Time Series Package.

Creating a TimeSeries
--------------------

A ``TimeSeries`` wraps a Polars DataFrame and adds specialized functionality for time series operations.

Basic Creation
~~~~~~~~~~~~~

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_metadata()

Time series level metadata can be accessed via the ``.metadata()`` method:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_6]
   :end-before: [end_block_6]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.show_time_series_metadata()

Column level metadata can be accessed via attributes on the column itself, or via the ``column.metadata()`` method:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_8]
   :end-before: [end_block_8]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.resolution_check_fail()

Periodicity Validation
~~~~~~~~~~~~~~~~~~~~~

The periodicity defines how frequently data points should appear:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_9]
   :end-before: [end_block_9]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.periodicity_check_fail()

Duplicate Detection
~~~~~~~~~~~~~~~~~~~

The ``TimeSeries`` automatically checks for rows with duplicates in the specified time column. You have control over
what the model should do when it detects rows with duplicate time values.  Consider this DataFrame with duplicate
time values:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_27]
   :end-before: [end_block_27]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_df_with_duplicate_rows()

The following strategies are available to use with the ``on_duplicate`` argument:

1. Error Strategy (Default): ``on_duplicate="error"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Raises an error when duplicate rows are found. This is the default behavior to ensure data integrity.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_28]
   :end-before: [end_block_28]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_duplicate_row_example_error()

2. Keep First Strategy: ``on_duplicate="keep_first"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a given group of rows with the same time value, keeps only the first row and discards the others.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_29]
   :end-before: [end_block_29]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_duplicate_row_example_keep_first()

3. Keep Last Strategy: ``on_duplicate="keep_last"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a given group of rows with the same time value, keeps only the last row and discards the others.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_30]
   :end-before: [end_block_30]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_duplicate_row_example_keep_last()

4. Drop Strategy: ``on_duplicate="drop"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Removes all rows that have duplicate timestamps. This strategy is appropriate when you are unsure of the integrity of
duplicate rows and only want unique, unambiguous data.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_31]
   :end-before: [end_block_31]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_duplicate_row_example_drop()

5. Merge Strategy: ``on_duplicate="merge"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For a given group of rows with the same time value, performs a merge of all rows. This combines values with a top-down
approach that preserves the first non-null value for each column.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_32]
   :end-before: [end_block_32]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_duplicate_row_example_merge()

Missing Rows
------------
The ``TimeSeries`` class provides functionality to automatically pad missing time points within a time series,
ensuring complete temporal coverage without gaps. Consider daily data with some missing days:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_33]
   :end-before: [end_block_33]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_df_with_missing_rows()

The padding is controlled by the pad parameter during ``TimeSeries`` initialisation:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_34]
   :end-before: [end_block_34]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.pad_timeseries()

The padding functionality respects the resolution and periodicity of your data. The above example was simple, with
missing daily data being filled in the datetime of the missing days. It gets more complex when you are dealing with
time series that may have different resolution to periodicities. For example, consider a time series that is the
"annual maximum of 15-minute river flow data in a given UK water-year". The resolution would be 15-minutes,
however the periodicity would be ``P1Y+9MT9H``, because a water-year starts at 9am on the 1st October:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_35]
   :end-before: [end_block_35]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_df_with_missing_rows_water_year()

The padding takes this resolution and periodicity into account, and sets missing rows to the **start** of the period:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_36]
   :end-before: [end_block_36]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.pad_timeseries_water_year()

.. warning::
    It is very important to set the ``periodicity`` and ``resolution`` parameters if you want to pad your data.
    Otherwise, the padding process will use the default periods of 1 microsecond, and try to pad your entire dataset
    with microsecond data, which will almost definitely result in a memory error!


Accessing Data
-------------

There are multiple ways to access data from a ``TimeSeries``:

Accessing the DataFrame
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_10]
   :end-before: [end_block_10]
   :dedent:

This gives the underlying Polars DataFrame, with which you can carry out normal Polars functionality:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_11]
   :end-before: [end_block_11]
   :dedent:

Accessing Columns
~~~~~~~~~~~~~~~

The ``TimeSeries`` class gives other ways to access data within the time series, whilst maintaining the core link to
primary datetime column.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_12]
   :end-before: [end_block_12]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.accessing_data()


Updating a TimeSeries
--------------------

You can update the underlying DataFrame (while preserving column settings),
**as long as the primary datetime column remains unchanged**.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_19]
   :end-before: [end_block_19]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.add_new_column_to_df()

If an update to the DataFrame results in a change to the primary datetime values, resolution or periodicity, then
an error will be raised.  A new TimeSeries object should be created.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_20]
   :end-before: [end_block_20]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.update_time_of_df_error()

Working with Columns
-------------------

The ``TimeSeries`` class provides methods for working with different column types:

Column Types
~~~~~~~~~~~

There are four column types:

1. **Primary Time Column**: The datetime column
2. **Data Columns**: Regular data columns (default type)
3. **Supplementary Columns**: Metadata or contextual information
4. **Flag Columns**: Flag markers giving specific information about data points

Creating Supplementary Columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supplementary columns can be specified on initialisation of the ``TimeSeries`` object:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_13]
   :end-before: [end_block_13]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_supp_cols()

Existing data columns can be converted to be a supplementary column:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_14]
   :end-before: [end_block_14]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.convert_existing_col_to_supp_col()

Or a completely new column can be initialised as a supplementary column:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_15]
   :end-before: [end_block_15]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.add_new_col_as_supp_col()

Creating Flag Columns
~~~~~~~~~~~~~~~~~~~

Flag columns are inherently linked to a **flag system**.  The flag system sets out the meanings of values that
can be added to the flag column.

If they already exist, flag columns and their associated flag systems can be specified on initialisation of the
TimeSeries object:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_16]
   :end-before: [end_block_16]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_flag_cols()

Otherwise, flag columns can be initialised dynamically on the TimeSeries object:


.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_17]
   :end-before: [end_block_17]
   :dedent:

Methods are available to add (or remove) flags to a Flag Column:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_18]
   :end-before: [end_block_18]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.set_and_remove_flags()

Aggregating Data
--------------

The ``TimeSeries`` class provides powerful aggregation capabilities.

Given a year's worth of minute data:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_24]
   :end-before: [end_block_24]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_set_up()

We can aggregate this data to various new resolutions.

This example shows an aggregation to monthly mean temperatures. The aggregation function can also be specified by a string (
upper or lower case). Note that this returns a new TimeSeries object, as the primary time attributes have changed.

The returned TimeSeries provides additional context columns:

- Expected count of the number of data points expected if the aggregation period was full
- Actual count of the number of data points found in the data for the given aggregation period.
- For Max and Min, the datetime of the Max/Min data point within the given aggregation period.
- A boolean column that indicates where the individual aggregated data point is valid or not (see below).

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_25]
   :end-before: [end_block_25]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_mean_monthly_temperature()


Missing criteria
~~~~~~~~~~~~~~~~~~~

By default, the aggregation method will provide data regardless of how many missing data points there are in the period.
For example, if we have two 1 minute data points on a given day, doing a mean aggregation would return the
mean of those 2 values, even though we'd expect 1440 values for a full day.

To have more control, you can specify criteria for a valid aggregation using the ``missing_criteria`` argument.

- ``("missing", 30)`` Aggregation is valid if there are no more than 30 values missing in the period.
- ``("available", 30)`` Aggregation is valid if there are at least 30 input values in the period.
- ``("percent", 30)`` Aggregation is valid if the data in the period is at least 30 percent complete (accepts integers or floats).

The resulting aggregation TimeSeries will contain a ``valid_<column name>`` column - a boolean series that indicates
whether the individual aggregated data points are valid or not, based on the missing criteria specified.
If no ``missing_criteria`` are specified, the ``valid`` column will be set to ``True``.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_37]
   :end-before: [end_block_37]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_missing_criteria_examples()

Multiple columns
~~~~~~~~~~~~~~~~~~~
The aggregation method can accept multiple columns to aggregate. They will all be aggregated using the same
method and criteria. The context columns explained above will be generated for each column individually, and be
clearly labelled in the column name so you are aware which data they refer to.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_38]
   :end-before: [end_block_38]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_multiple_columns()

Some more aggregation examples:
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_26]
   :end-before: [end_block_26]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_more_examples()

For more details on aggregation, see the dedicated :doc:`aggregation` guide.

Best Practices
------------

1. **Always specify resolution and periodicity** when creating a TimeSeries to ensure proper validation
2. **Use appropriate column types**:
   - Use data columns for core measurements
   - Use supplementary columns for metadata or context
   - Use flag columns for quality control
3. **Add metadata** to enhance understanding of your data

Next Steps
---------

Now that you understand the basics of the ``TimeSeries`` class, explore:

- :doc:`periods` - Learn more about working with time periods
- :doc:`aggregation` - Dive deeper into aggregation capabilities
- :doc:`flagging` - Master the flagging control system
