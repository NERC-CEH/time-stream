.. _irregular:

======================
Irregular Time Series
======================

.. rst-class:: lead

    Time series of ad-hoc measurements, with no expected spacing between values.

What is an irregular time series?
=================================

Some data is measured whenever someone gets the chance, on random days and at random times: manual river
gaugings, ADCP surveys, spot water quality samples. There is no expected spacing between the values, so there is
no :ref:`periodicity <concepts>` to describe.

A :class:`~time_stream.TimeFrame` with ``periodicity="irregular"`` represents this data. ``resolution`` keeps its
usual meaning, the precision of the timestamps:

- ``resolution="PT1M"``: measurement times recorded to the minute.
- ``resolution="P1D"``: only the date of each measurement is known.

Timestamps must still align to the resolution, and duplicate timestamps are still handled by ``on_duplicates``.

Creating an irregular TimeFrame
===============================

.. literalinclude:: ../examples/irregular.py
    :language: python
    :start-after: [start:create_irregular]
    :end-before: [end:create_irregular]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import irregular

   irregular.create_irregular()

For an irregular TimeFrame, :attr:`~time_stream.TimeFrame.periodicity` is ``None`` and
:attr:`~time_stream.TimeFrame.is_irregular` is ``True``.

.. note::

   A TimeFrame created with neither ``resolution`` nor ``periodicity`` is irregular, with a resolution of
   1 microsecond.

What works with irregular data
==============================

Anything that needs an expected time step can't work without a periodicity. Everything else works as normal.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operation
     - Irregular behaviour
   * - :meth:`~time_stream.TimeFrame.aggregate`
     - Works. The result is a **regular** TimeFrame at the aggregation period.
   * - :meth:`~time_stream.TimeFrame.rolling_aggregate`
     - Works. The result keeps the input timestamps and stays **irregular**.
   * - :meth:`~time_stream.TimeFrame.qc_check`
     - Works. Checks that compare neighbouring rows, such as spike and flat-line checks, do so whatever the time
       between them.
   * - :meth:`~time_stream.TimeFrame.pad`
     - Raises ``PeriodicityError``: there are no expected time steps to pad.
   * - :meth:`~time_stream.TimeFrame.infill`
     - Raises ``PeriodicityError``: there are no expected time steps to define a gap.

Aggregation
===========

Values are grouped by the aggregation period they fall in, as for regular data:

.. literalinclude:: ../examples/irregular.py
    :language: python
    :start-after: [start:aggregate_irregular]
    :end-before: [end:aggregate_irregular]
    :dedent:

.. jupyter-execute::
   :hide-code:

   irregular.aggregate_irregular()

There is no way to know how many values to expect in a period, so:

- ``expected_count_<time>`` is null.
- The ``"available"`` and ``"na"`` missing criteria work, as they only count the values found. ``"percent"`` and
  ``"missing"`` raise ``MissingCriteriaError``.
- The ``mean_sum`` and ``nth`` aggregations raise ``AggregationError``, as both need an expected count.
- ``time_window`` works, as it filters values by time of day.

For example, to only trust monthly means made from at least two gaugings:

.. literalinclude:: ../examples/irregular.py
    :language: python
    :start-after: [start:aggregate_available_criteria]
    :end-before: [end:aggregate_available_criteria]
    :dedent:

.. jupyter-execute::
   :hide-code:

   irregular.aggregate_available_criteria()

Rolling aggregation
===================

The rolling window is a span of time, so it is placed around each value whatever the spacing:

.. literalinclude:: ../examples/irregular.py
    :language: python
    :start-after: [start:rolling_irregular]
    :end-before: [end:rolling_irregular]
    :dedent:

.. jupyter-execute::
   :hide-code:

   irregular.rolling_irregular()

The same rules as aggregation apply to the expected count, missing criteria, ``mean_sum`` and ``nth``. ``center``
alignment works with any fixed-length window, as there are no time steps to centre on.

Time properties of the result
=============================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - :meth:`~time_stream.TimeFrame.aggregate`
     - :meth:`~time_stream.TimeFrame.rolling_aggregate`
   * - Resolution
     - Aggregation period
     - Same as the input
   * - Periodicity
     - Aggregation period (regular)
     - Irregular
   * - Timestamps
     - One per aggregation period
     - Same as the input

What raises
===========

.. literalinclude:: ../examples/irregular.py
    :language: python
    :start-after: [start:unsupported_operations]
    :end-before: [end:unsupported_operations]
    :dedent:

.. jupyter-execute::
   :hide-code:

   irregular.unsupported_operations()
