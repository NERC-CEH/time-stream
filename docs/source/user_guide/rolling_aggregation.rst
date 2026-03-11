.. _rolling_aggregation:

===================
Rolling Aggregation
===================

.. rst-class:: lead

    Sliding windows over your time series, with the same robust completeness tracking.

What is rolling aggregation?
=============================

Rolling (or sliding window) aggregation computes a summary statistic for each timestamp using the
observations in a fixed-size window around that timestamp. Unlike :doc:`aggregation`, which reduces the
resolution of the data (e.g., 15-minute values aggregated to daily values), rolling aggregation
**preserves the original timestamps and resolution** - the output has the same number of rows as the input.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Property
     - :meth:`~time_stream.TimeFrame.aggregate`
     - :meth:`~time_stream.TimeFrame.rolling_aggregate`
   * - Output rows
     - One per aggregation period
     - Same as input (one per observation)
   * - Output resolution
     - Aggregation period
     - Original resolution
   * - Timestamps
     - Period labels (e.g., midnight each day)
     - Original timestamps preserved
   * - Typical use
     - Daily/monthly summaries
     - Smoothing, rolling statistics


One-liner
---------

To calculate a 3-hour rolling mean of flow data:

.. code-block:: python

   tf.rolling_aggregate("PT3H", "mean", "flow")

That's it, a single line with clear intent: "I want a *3 hourly* *rolling mean* of my *flow* data."

All :ref:`aggregation functions <aggregation_functions>` supported by :meth:`~time_stream.TimeFrame.aggregate`
are equally supported here.

In more detail
==============

The :meth:`~time_stream.TimeFrame.rolling_aggregate` method is the entry point for performing rolling aggregations with
timeseries data in **Time-Stream**. It works similarly to the standard :meth:`~time_stream.TimeFrame.aggregate` method,
utilising the same **Polars performance** with **TimeFrame semantics**.

**Example**

Using the :ref:`15-minute flow example data <example_input_data_agg>`:

.. literalinclude:: ../../../src/time_stream/examples/examples_rolling_aggregation.py
    :language: python
    :start-after: [start_block_1]
    :end-before: [end_block_1]
    :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_rolling_aggregation
   examples_rolling_aggregation.rolling_mean_example()

There are some specific parameters that are provided to the rolling aggregation method, explained below.

.. note::
   The resulting :class:`~time_stream.TimeFrame` contains the same data completeness information as the standard
   aggregation - this time based on completeness of each rolling window.  See :ref:`data-completeness` below.

Window size
-----------

The size of the time window you want to do a rolling aggregation over. This can be specified as an ISO-8601 duration
string, and can be combined with the window alignment (see below) to fine-tune the rolling window.

Common examples:

- ``"P1D"`` – 1 day window
- ``"PT3H"`` – 3-hour window
- ``"PT15M"`` – 15-minute window

Window alignment
----------------

The ``alignment`` parameter controls where the rolling window is positioned relative to each timestamp.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Alignment
     - Window
     - Edge effects
   * - ``TRAILING`` (default)
     - ``(t - window_size, t]``
     - At the **start** of the series (first rows see partial windows)
   * - ``LEADING``
     - ``[t, t + window_size)``
     - At the **end** of the series (last rows see partial windows)
   * - ``CENTER``
     - ``[t - window_size/2, t + window_size/2]``
     - At **both ends** of the series


Trailing (default)
^^^^^^^^^^^^^^^^^^

The window looks **backward** from each timestamp. Each output value summarises the current observation
and the preceding ones within the window. This is the conventional default for rolling statistics.

.. code-block:: python

   # 3-hour trailing mean: each output reflects the current hour and the two hours before it.
   tf_trailing = tf.rolling_aggregate("PT3H", "mean", "flow")
   # equivalently:
   tf_trailing = tf.rolling_aggregate("PT3H", "mean", "flow", alignment="trailing")

For hourly data with a 3-hour trailing window, the first output row has only 1 observation
in its window, the second has 2, and all subsequent rows have 3 (the full window).

Leading
^^^^^^^

The window looks **forward** from each timestamp. Each output value summarises the current observation
and the following ones within the window.

.. code-block:: python

   # 3-hour leading mean: each output reflects the current hour and the two hours after it.
   tf_leading = tf.rolling_aggregate("PT3H", "mean", "flow", alignment="leading")

For hourly data with a 3-hour leading window, the last two output rows see partial windows.

Center
^^^^^^

The window is **centered** on each timestamp, looking equally backward and forward. Edge effects (where the window
contains partial data) appear at both the start and end of the series.

.. code-block:: python

   # 3-hour centered mean: each output reflects the 1.5 hours before and after the current timestamp.
   tf_center = tf.rolling_aggregate("PT3H", "mean", "flow", alignment="center")

.. note::

   ``CENTER`` alignment is not supported for calendar-based window sizes (months, years) because
   they have variable length and cannot be halved to a fixed offset.

.. _data-completeness:

Data completeness
=================

Rolling aggregation tracks data completeness in the same way as standard aggregation. The output always includes:

- ``expected_count_<time>``: the number of observations expected in a full window.
- ``count_<column>``: the number of observations actually present.
- ``valid_<column>``: whether the result passes the completeness check.

When the window extends beyond the edges of the data (edge effects), the actual count will be less than the expected
count. You can use ``missing_criteria`` to flag or filter these rows.

For example, to mark a result as invalid unless the window contains at least 3 observations:

.. literalinclude:: ../../../src/time_stream/examples/examples_rolling_aggregation.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_rolling_aggregation
   examples_rolling_aggregation.rolling_missing_criteria_example()

Rows where ``count_flow < 3`` have ``valid_flow = false``, as visible at the start of the series where the trailing
window has not yet accumulated enough observations.

See :ref:`Missing data criteria <missing_criteria>` in the aggregation guide for the full list of available criteria.
