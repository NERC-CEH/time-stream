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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_10]
   :end-before: [end_block_10]
   :dedent:

This gives the underlying Polars DataFrame, with which you can carry out normal Polars functionality:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_11]
   :end-before: [end_block_11]
   :dedent:

Accessing Columns
~~~~~~~~~~~~~~~

The ``TimeSeries`` class gives other ways to access data within the time series, whilst maintaining the core link to
primary datetime column.

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_13]
   :end-before: [end_block_13]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_supp_cols()

Existing data columns can be converted to be a supplementary column:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_14]
   :end-before: [end_block_14]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.convert_existing_col_to_supp_col()

Or a completely new column can be initialised as a supplementary column:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_16]
   :end-before: [end_block_16]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_flag_cols()

Otherwise, flag columns can be initialised dynamically on the TimeSeries object:


.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_17]
   :end-before: [end_block_17]
   :dedent:

Methods are available to add (or remove) flags to a Flag Column:

.. literalinclude:: ../../../examples/examples_timeseries_basics.py
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


.. literalinclude:: ../../../examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_21]
   :end-before: [end_block_21]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.add_column_relationships()


.. code-block:: python
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

Next Steps
---------

Now that you understand the basics of the ``TimeSeries`` class, explore:

- :doc:`periods` - Learn more about working with time periods
- :doc:`aggregation` - Dive deeper into aggregation capabilities
- :doc:`flagging` - Master the flagging control system
- :doc:`column_relationships` - Understand column relationships in detail
