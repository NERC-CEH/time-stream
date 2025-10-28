.. _aggregation:

===========
Aggregation
===========

.. rst-class:: lead

    Simple code, robust results.

Why use Time-Stream?
====================

Aggregating time series data sounds simple - until you hit real-world edge cases, particularly when working with
hydrological and environmental datasets. There are considerations such as:
handling leap years, anchor points, working with offset-periods (like water-years), or
tracking completeness of data going into each aggregation.

Forget the pain of writing your own custom code; with **Time-Stream**, you declare the aggregation period you want
(daily, monthly, yearly, or any other custom period) and the library handles the rest. Simple code, robust results.

One-liner
---------

Rolling your own aggregation functionality can get complex.
With **Time-Stream** you express the intent directly:

.. code-block:: python

   tf.aggregate("P1D", "sum", "precipitation")

That's it, a single line with clear intent: "I want a *daily* *sum* of my *precipitation* data."

Complex example
---------------

UK hydrologists often use a "UK water-year" that runs from **1 October at 09:00** through to the
following year. Writing code to aggregate data (whether that's 15-minute river level, or daily flow values)
into water-year totals can be painful.

Take, for example, generating a water-year "annual maximum" (AMAX)
series. You might want to:

- Generate an output time series, with the datetime values of the start of each water-year
- Keep hold of the exact timestamp that the maximum occurred on
- Know exactly how many data points were available in each water year

Here's how you might have to do this in `Pandas <https://pandas.pydata.org/>`_, `Polars <https://pola.rs/>`_
and then in Time-Stream:

**Input:**

.. _example_input_data:

15-minute river flow timeseries, including some missing data.

.. jupyter-execute::
   :hide-code:

   import examples_aggregation
   ts = examples_aggregation.get_example_df("polars")

**Code:**

.. tab-set::
    :class: outline padded-tabs

    .. tab-item:: :iconify:`devicon:pandas` Pandas
        :sync: pandas

        .. literalinclude:: ../../../src/time_stream/examples/examples_aggregation.py
           :language: python
           :start-after: [start_block_1]
           :end-before: [end_block_1]
           :dedent:

    .. tab-item:: :iconify:`simple-icons:polars` Polars
        :sync: polars

        .. literalinclude:: ../../../src/time_stream/examples/examples_aggregation.py
           :language: python
           :start-after: [start_block_2]
           :end-before: [end_block_2]
           :dedent:

    .. tab-item:: :iconify:`tdesign:time` Time-Stream
        :sync: time_stream

        .. literalinclude:: ../../../src/time_stream/examples/examples_aggregation.py
           :language: python
           :start-after: [start_block_3]
           :end-before: [end_block_3]
           :dedent:

**Output:**

.. tab-set::
    :class: outline padded-tabs

    .. tab-item:: :iconify:`devicon:pandas` Pandas
        :sync: pandas

        .. jupyter-execute::
           :hide-code:

           import examples_aggregation
           ts = examples_aggregation.pandas_example()

    .. tab-item:: :iconify:`simple-icons:polars` Polars
        :sync: polars

        .. jupyter-execute::
           :hide-code:

           import examples_aggregation
           ts = examples_aggregation.polars_example()

    .. tab-item:: :iconify:`tdesign:time` Time-Stream
        :sync: time_stream

        .. jupyter-execute::
           :hide-code:

           import examples_aggregation
           ts = examples_aggregation.time_stream_example()

Key benefits
------------

- **Less boilerplate**: No need to wrangle custom datetime columns or write manual offset logic.
- **Fewer mistakes**: Periodicity, alignment, and anchor semantics are enforced for you.
- **Domain-ready**: Express hydrological conventions directly: daily at 09:00, or water year from October.
- **Readable & reproducible**: Your code is self-explanatory to collaborators and reviewers.

In more detail
==============

The :meth:`~time_stream.TimeFrame.aggregate` method is the entry point for performing aggregations with
timeseries data in **Time-Stream**. It combines **Polars performance** with **TimeFrame semantics**
(resolution, periodicity, anchor).

Aggregation period
------------------

The time window you want to aggregate **into**. This can be specified as an ISO-8601 duration string, with optional
modification to specify a custom *offset* to the period.

Common examples:

- ``"P1D"`` – calendar day
- ``"P1M"`` – calendar month
- ``"P1Y"`` – calendar year
- ``"P1Y+9MT9H"`` – **water year** starting **1 Oct 09:00**
- ``"PT15M"`` – 15-minute buckets

.. note::
   The resulting :class:`~time_stream.TimeFrame` will have its resolution and periodicity set to this value.

Aggregation methods
-------------------

Choose how values inside each window are summarised. Pass a **string** corresponding to one of the built-in functions:

- ``"sum"`` - **Add up all values in each period.**

  Use this for quantities that accumulate over time, such as precipitation.

- ``"mean"`` - **Average of all values in each period.**

  Useful for variables like temperature or concentration, where the average represents the period well.

- ``"min"`` - **Smallest value observed in the period.**

  Often used to track minimum daily temperature, or low flows in rivers.

- ``"max"`` - **Largest value observed in the period.**

  Common in hydrology for annual maxima (AMAX) or flood frequency analysis.

- ``"percentile"`` - **The 'nth' percentile value for the period.**

  Useful for capturing extremes within a given period, such as the 5th or 95th percentile of streamflow.
  The percentile value to be calculated is provided as an integer parameter (p) from 0 to 100 (inclusive).


Column selection
----------------

Specify which columns to aggregate; only these will be used by the aggregation function. This can be a single
column name, a list of columns, or if not provided - the method will use *all* columns in the timeseries.


Missing criteria
----------------

Control whether a window is considered **"complete enough"** to produce a value by specifying a specific
*missing criteria policy* and associated *threshold*.

The policies you can specify are:

- ``available``:
  Requires at least "n" points within the aggregation window.

- ``percent``:
  Requires at least "n"% of points within the aggregation window.

- ``missing``:
  No more than "n" points can be missing within the aggregation window.

**Examples**

Using the :ref:`15-minute flow example data <example_input_data>`:

.. code-block:: python

    # Require at least 2,400 values present (25 days of 15 minute data)
    tf_agg = tf.aggregate("P1M", "mean", "flow", missing_criteria=("available", 25 * 96))

    # Require at least 75% of data to be present
    tf_agg = tf.aggregate("P1M", "mean", "flow", missing_criteria=("percent", 75))

    # Allow at most 150 missing values
    tf_agg = tf.aggregate("P1M", "mean", "flow", missing_criteria=("missing", 150))

The resulting :class:`~time_stream.TimeFrame` object will contain metadata columns that provide detail about
the completeness of the aggregation windows, and whether an aggregated data point is considered valid:

- ``count_<column>``: The number of points found in each aggregation window
- ``expected_count_<time>``: The number of points expected if the aggregation window was full
- ``valid_<column>``: Whether the individual aggregated data points are valid or not,
  based on the missing criteria specified.

.. jupyter-execute::
   :hide-code:

   import examples_aggregation
   examples_aggregation.aggregation_missing_criteria_example()


Time anchoring
--------------

Choose the time anchor of the aggregated TimeFrame, which you may want to be different than the input TimeFrame.
For example, meteorological observations are often considered as **end** anchored - where the value is considered valid
*up to* the given timestamp. When producing a daily mean from this data, it may make more sense for the result
to use a **start** anchor - indicating the value is valid *from* the start of the day to the end of the day.

See the :doc:`concepts page </getting_started/concepts>` page for more information about time anchors.

.. note::

    If omitted, the aggregation uses the input TimeFrame's time anchor
