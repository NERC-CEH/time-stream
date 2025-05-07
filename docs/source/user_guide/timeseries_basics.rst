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

Column Relationships
------------------

You can define relationships between columns that are linked together in some way. Data columns can be given a
relationship to both supplementary and flag columns, though supplementary and flag columns cannot be given a
relationship to each other.


.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_21]
   :end-before: [end_block_21]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.add_column_relationships()

Relationships can be removed:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_22]
   :end-before: [end_block_22]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.remove_column_relationships()

The relationship also defines what happens when a column is removed.  For example, if a Data Column is dropped, then
this will cascade to any linked Flag Columns.  Any linked Supplementary Columns are not dropped, but the relationship
removed:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_23]
   :end-before: [end_block_23]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.drop_column_with_relationships()

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

This example shows an aggregation to monthly mean temperatures.
Note that this returns a new TimeSeries object, as the primary time attributes have changed.

The returned TimeSeries provides additional context columns:

- Expected count of the number of data points expected if the aggregation period was full
- Actual count of the number of data points found in the data for the given aggregation period.
- For Max and Min, the datetime of the Max/Min data point within the given aggregation period.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_25]
   :end-before: [end_block_25]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.aggregation_mean_monthly_temperature()


By default, this will aggregate the data regardless of how many missing data points there are in the period.
For example, if we have two 1 minute data points on a given day, doing a mean aggregation would return the
mean of those 2 values, even though we'd expect 1440 values for a full day.

You can specify criteria for a valid aggregation using the ``missing_criteria`` argument.

- ``{"missing": 30}`` Aggregate only if there are no more than 30 values missing in the period.
- ``{"available": 30}`` Aggregate only if there are at least 30 input values in the period.
- ``{"percent": 30}`` Aggregate only if the data in the period is at least 30 percent complete (accepts integers or floats).
   
Some more aggregation examples:

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
3. **Define relationships** between related columns
4. **Add metadata** to enhance understanding of your data

Next Steps
---------

Now that you understand the basics of the ``TimeSeries`` class, explore:

- :doc:`periods` - Learn more about working with time periods
- :doc:`aggregation` - Dive deeper into aggregation capabilities
- :doc:`flagging` - Master the flagging control system
- :doc:`column_relationships` - Understand column relationships in detail
