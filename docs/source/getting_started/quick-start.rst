.. _quick-start:

============
Quick start
============

.. rst-class:: lead

    Create and work with timeseries data using the **Time-Stream** package.

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

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   tf = examples_timeseries_basics.create_simple_time_series()

With Time Properties
====================

The :class:`~time_stream.TimeFrame` object can configure important properties about the time aspect of your data.
More information about these properties and concepts can be found on the :doc:`concepts page <concepts>`.

Here, we will show some basic usage of these time properties.

Resolution, Offset, Periodicity and Time Anchor
-----------------------------------------------

Without specifying ``resolution`` or ``periodicity``, the default initialisation sets these properties to
**1 microsecond**, to account for any set of datetime values.
The default is for no ``offset``. The ``time_anchor`` is set to **start**:

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_3]
   :end-before: [end_block_3]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   examples_timeseries_basics.show_default_resolution()


Although the default of 1 microsecond will account for any datetime values, for more control over certain
time series functionality it is important to specify the actual ``resolution``, ``offset`` and ``periodicity`` if known.
These properties can be provided as an ISO 8601 duration string, e.g. **P1D** (1 day) or **PT15M** (15 minutes).

The ``time_anchor`` property can be set to **start**, **end**, or **point**.

Again, more detail can be found on the :doc:`concepts page <concepts>` about all these properties.

Resolution
~~~~~~~~~~

For most cases, it is sufficient to just specify the ``resolution``. The ``offset`` will default to "no offset", and the
``periodicity`` will be set to the same as the resolution.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   tf = examples_timeseries_basics.create_simple_time_series_with_periods()

Offset
~~~~~~

The next most common modification might be to specify an ``offset``. This is where your data is measured at a
point in time offset from the "natural boundary" of the ``resolution`` (more info here:
:doc:`concepts page <concepts>`). The ``periodicity`` is automatically built from the ``resolution + offset``, to
specify that we only expect 1 value within those points in time.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_8]
   :end-before: [end_block_8]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   tf = examples_timeseries_basics.create_simple_time_series_with_periods2()

Periodicity
~~~~~~~~~~~

Finally, you may have data that is measured at a given resolution, but you only expect 1 value in a different period of
time. This is when you would specify a specific ``periodicity``. The classic hydrological example would be where you
have an annual-maximum (AMAX) timeseries, where the measured data is a daily resolution, but we only expect 1 value
per year.

.. literalinclude:: ../../../src/time_stream/examples/examples_timeseries_basics.py
   :language: python
   :start-after: [start_block_9]
   :end-before: [end_block_9]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   tf = examples_timeseries_basics.create_simple_time_series_with_periods3()

Duplicate Detection
-------------------

:class:`~time_stream.TimeFrame` automatically checks for rows with duplicates in the specified time column.
You have control over what the model should do when it detects rows with duplicate time values.
Consider this DataFrame with duplicate time values:

.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   df = examples_timeseries_basics.create_df_with_duplicate_rows()

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
   examples_timeseries_basics.duplicate_row_example_error()

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
   examples_timeseries_basics.duplicate_row_example_keep_first()

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
   examples_timeseries_basics.duplicate_row_example_keep_last()

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
   examples_timeseries_basics.duplicate_row_example_drop()

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
   examples_timeseries_basics.duplicate_row_example_merge()


Misaligned Row Detection
------------------------

As part of the resolution checks, misaligned rows (rows with a different resolution to that expected) are detected 
automatically. By default an error is raised, however if ``on_misaligned_rows="resolve"`` is set on creation of the 
TimeFrame, then any misaligned rows will be automatically removed. 

.. note::
   A resolution must be provided to the :class:`~time_stream.TimeFrame` initialisation for the misaligned row detection
   to work. If not provided, a default resolution of 1 microsecond is assigned, which will pass any resolution checks.

The following strategies are available to use with the ``on_misaligned_rows`` argument:

1. **Error (Default):** ``on_misaligned_rows="error"``

Raises an error when misaligned rows are found within the time series. 

.. code-block:: python

   tf = ts.TimeFrame(df, "time", "PT30M", on_misaligned_rows="error")

The following error will be shown if misaligned rows are found. Note that this example assumes PT30M data: 

.. code-block:: python

   Time values are not aligned to resolution[+offset]: PT30M

1. **Resolve:** ``on_misaligned_rows="resolve"``

Any rows identified as being misaligned are removed and the TimeFrame data is updated.


.. code-block:: python

   tf = ts.TimeFrame(df, "time", "PT30M", on_misaligned_rows="resolve")


.. jupyter-execute::
   :hide-code:

   import examples_timeseries_basics
   examples_timeseries_basics.create_misaligned_row_example()


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
    examples_timeseries_basics.show_time_series_metadata()

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
   examples_timeseries_basics.accessing_data()


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
    examples_timeseries_basics.add_new_column_to_df()
