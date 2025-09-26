.. _quick-start:

============
Quick start
============

This quick start guide will walk you through creating and working with timeseries data using the Time-Stream package.

We follow the import pattern:

.. code-block:: python

   import time_stream as ts


Create a TimeFrame
==================

Create sample data in a Polars DataFrame:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

Now wrap the Polars DataFrame in a :class:`~time_stream.TimeFrame`, which adds specialized functionality
for time series operations:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

With Time Properties
====================

The :class:`~time_stream.TimeFrame` object can configure important properties about the time aspect of your data.
More information about these properties and concepts can be found on the :doc:`concepts page <concepts>` page.

Here, we will show some basic usage of these time properties.

Periodicity, Resolution and Time Anchor
---------------------------------------

Without specifying resolution and periodicity, the default initialisation sets these properties to **1 microsecond**, to
account for any set of datetime values.  The time anchor property is set to **start**:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_3]
   :end-before: [end_block_3]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.show_default_resolution()


Although the default of 1 microsecond will account for any datetime values, for more control over certain
time series functionality it is important to specify the actual resolution and periodicity if known.
These properties can be provided as an ISO 8601 duration string like **P1D** (1 day) or **PT15M** (15 minutes).

The time anchor property can be set to **start**, **end**, or **point**.

Again, more detail can be found on the :doc:`concepts page <concepts>` page about all these properties.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_simple_time_series_with_periods()

Duplicate Detection
-------------------

:class:`~time_stream.TimeFrame` automatically checks for rows with duplicates in the specified time column.
You have control over what the model should do when it detects rows with duplicate time values.
Consider this DataFrame with duplicate time values:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.create_df_with_duplicate_rows()

The following strategies are available to use with the ``on_duplicate`` argument:

1. **Error (Default):** ``on_duplicate="error"``

Raises an error when duplicate rows are found. This is the default behavior to ensure data integrity.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_28]
   :end-before: [end_block_28]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.duplicate_row_example_error()

2. **Keep First:** ``on_duplicate="keep_first"``

For a given group of rows with the same time value, keeps only the first row and discards the others.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_29]
   :end-before: [end_block_29]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.duplicate_row_example_keep_first()

3. **Keep Last:** ``on_duplicate="keep_last"``

For a given group of rows with the same time value, keeps only the last row and discards the others.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_30]
   :end-before: [end_block_30]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.duplicate_row_example_keep_last()

4. **Drop**: ``on_duplicate="drop"``

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
   ts = examples_timeseries_basics.duplicate_row_example_drop()

5. **Merge**: ``on_duplicate="merge"``

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
   ts = examples_timeseries_basics.duplicate_row_example_merge()

With Metadata
=============

The :class:`~time_stream.TimeFrame` object can hold metadata to describe your data.
This can be metadata about the time series dataset as a whole, or about the individual columns. Keeping the metadata
and the data together in one object like this can help simplify downstream processes,
such as derivation functions, running infilling routines, plotting data, etc.

Dataset-level metadata can be set with the :meth:`~time_stream.TimeFrame.with_metadata` method:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

Column-level metadata can be set with the :meth:`~time_stream.TimeFrame.with_column_metadata` method:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_6]
   :end-before: [end_block_6]
   :dedent:

Metadata can be accessed via the :attr:`~time_stream.TimeFrame.metadata` (dataset-level)
and :attr:`~time_stream.TimeFrame.column_metadata` (column-level) attributes:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
    :language: python
    :start-after: [start_block_7]
    :end-before: [end_block_7]
    :dedent:

.. jupyter-execute::
    :hide-code:

    import examples_timeseries_basics
    ts = examples_timeseries_basics.show_time_series_metadata()

Data Access and Update
======================

Data Selection
--------------

The underlying Polars DataFrame is accessed via the :attr:`~time_stream.TimeFrame.df` property

.. code-block:: python

   tf.df

You can create new :class:`~time_stream.TimeFrame` objects as a selection, using the
:meth:`~time_stream.TimeFrame.select` method, or via indexing syntax:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_12]
   :end-before: [end_block_12]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.accessing_data()


.. note::
   The primary time column is automatically maintained in any selection.


Data Update
-----------

If you need to make changes to the underlying Polars DataFrame, use the :meth:`~time_stream.TimeFrame.with_df` method.
This performs some checks on the new DataFrame to check the integrity of the time data has been maintained, and
returns a new :class:`~time_stream.TimeFrame` object with the updated data.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_19]
   :end-before: [end_block_19]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   ts = examples_timeseries_basics.add_new_column_to_df()
