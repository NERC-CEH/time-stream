.. _time_of_year_aggregation:

========================
Time of Year Aggregation
========================

.. rst-class:: lead

    The range, mean or percentiles of past values for each time of year.

What is time of year aggregation?
=================================

Time of year aggregation brings together the values at the same time of year across years. For daily data, every
1 January is aggregated together, then every 2 January, and so on. This gives, for example, the highest flow ever
seen on each day of the year, or the long-term mean for each day.

Like :doc:`rolling_aggregation`, the result **keeps the original rows**. The result for each time of year is given
on every row with that time of year, beside the original values. This makes it easy to compare a value with the
values seen before at that time of year.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Property
     - :meth:`~time_stream.TimeFrame.aggregate`
     - :meth:`~time_stream.TimeFrame.rolling_aggregate`
     - :meth:`~time_stream.TimeFrame.time_of_year_aggregate`
   * - Groups values
     - Within each period
     - Within a window around each row
     - At the same time of year, across years
   * - Output rows
     - One per aggregation period
     - Same as input
     - Same as input
   * - Typical use
     - Daily/monthly summaries
     - Smoothing, rolling statistics
     - Historic ranges, long-term means, anomalies

One-liner
---------

.. code-block:: python

   tf.time_of_year_aggregate("max", "flow")

The output adds the result, named as :meth:`~time_stream.TimeFrame.aggregate` names it: ``max_flow``. As with
:meth:`~time_stream.TimeFrame.aggregate`, it also adds a ``count_flow`` (the values found), ``expected_count_time``
(the time steps covered) and ``valid_flow`` (see `Missing data`_ below).

``min`` and ``max`` also give the date-time of when the record was set (``time_of_max_flow``).

In more detail
==============

Any of the :ref:`aggregation functions <aggregation_functions>` can be used, except ``mean_sum`` and ``nth``, which
need whole aggregation periods. Each call adds its results to the TimeFrame, so calls can be chained to build up
various time-of-year aggregations. Pass ``columns`` to each call when chaining, otherwise the results of earlier calls
are aggregated too.

**Example**

Using the :ref:`15-minute flow example data <example_input_data_agg>`, the highest and lowest flow ever seen on
each 15-minute time step of the year, and when it was seen:

.. literalinclude:: ../examples/time_of_year_aggregation.py
    :language: python
    :start-after: [start:min_max_example]
    :end-before: [end:min_max_example]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import time_of_year_aggregation

   time_of_year_aggregation.min_max_example()

The ``count_<column>``, ``expected_count_<time>`` and ``valid_<column>`` and equivalent across all time-of-year
aggregation runs that use the same ``reference_interval`` and ``missing_criteria`. Note that applying the same function
to the same column twice raises a ``DuplicateColumnError``, as it would replace the first results.

Aggregation period
------------------

``aggregation_period`` sets how much of the year each time of year covers. By default it is the periodicity of the
data, so 15-minute data has a time of year for every 15 minutes. A coarser period groups more values together:
``aggregation_period="P1D"`` aggregates every 15-minute value on each day of the year, as in the examples above.

The aggregation period must fit evenly into a **year**, so a day, an hour or a month works, but a week does not. It must
also be a whole number of time steps of the data. An offset is allowed: ``aggregation_period="P1D+T9H"`` groups water
days running from 09:00, for data with a time step that lines up with 09:00, such as hourly data.

The time anchor is respected. An ``end`` anchored value belongs to the period it closes, so a daily value at
00:00 on 1 March belongs to the last day of February.

Reference interval
------------------

``reference_interval`` limits the data that is aggregated, as ``(start, end)`` (both inclusive). The results are still
given on every row, including those outside the reference interval.  This can be useful if you have a particular
baseline reference period over which you want a climatology calculated to compare more recent values against.

Here, a century of daily temperature (1925 to 2024) is compared with the mean for each day of the year over
1990 to 2010. The difference is the anomaly - how much warmer or colder each day was than the climatology:

.. literalinclude:: ../examples/time_of_year_aggregation.py
    :language: python
    :start-after: [start:reference_interval_example]
    :end-before: [end:reference_interval_example]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import time_of_year_aggregation

   time_of_year_aggregation.reference_interval_example()

Missing data
------------

The expected count of a time of year is the number of time steps it covers across the reference interval. For daily
data over 30 years, each day of the year expects 30 values. ``missing_criteria`` works as it does for
:meth:`~time_stream.TimeFrame.aggregate`, setting ``valid_<column>``. Here a day of the year is only valid if at
least 80% of its values exist:

.. code-block:: python

   tf.time_of_year_aggregate("max", "flow", missing_criteria=("percent", 80))

Leap days
---------

Times of year match on month and day, so 1 March lines up in leap and non-leap years. ``leap_day`` sets how
29 February is handled:

- ``"separate"`` (default): its own time of year, from leap years only. Its ``count_<column>`` shows how few values
  it has.
- ``"merge"``: aggregated with 28 February, and given its result.
- ``"drop"``: left out of the aggregation, giving nulls (and a count of 0) on 29 February.

.. code-block:: python

   tf.time_of_year_aggregate("max", "flow", leap_day="merge")
