.. _qc:

===============
Quality Control
===============

.. rst-class:: lead

    Build QC your way - flexible checks with a consistent framework.

Why use Time-Stream?
====================

Quality control is essential for any environmental dataset, but QC rules vary between projects, organisations,
and sensor types. **Time-Stream** doesn't make those decisions for you - instead, it provides a **framework** for
applying common types of QC.

One-liner
---------

QC checks are lightweight, configurable, and explicit:

.. code-block:: python

   tf_flagged = tf.qc_check(
      "comparison", "rainfall", compare_to=0, operator="<",
      flag_params=("rainfall_flag", "FLAGGED"),
   )

A single call with rich meaning: "I want to *QC check* when my *rainfall* data is *less than* a value of *0*,
and record the result in the *rainfall_flag* flag column."

Key benefits
------------

- **You stay in control**
  Flexibility to choose your thresholds, operators, and ranges.

- **Reproducible QC**
  The same logic can be applied across datasets.

- **Traceable results**
  Checks can add explicit boolean columns or flag values for later analysis.

- **Flexible**
  Combine multiple checks, apply them in sequence, or restrict them to intervals.

In more detail
==============

The :meth:`~time_stream.TimeFrame.qc_check` method applies a single QC check to one column.
Each QC check is configurable through parameters specific to that check.

Let's look at the method in more detail:

.. automethod:: time_stream.TimeFrame.qc_check
   :no-index:

Quality control methods
-----------------------

``comparison``
^^^^^^^^^^^^^^
:class:`time_stream.qc.ComparisonCheck`

    **What it does:** Compares values against a constant or list using a comparison operator
    (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``, ``is_in``).

    **When to use:** Use for value thresholds (e.g. negative rainfall) or matching against
    lists of known error codes.

    **Additional args:**
        ``compare_to``: The value (or list of values for ``is_in``) to compare against.
        ``operator``: The comparison operator string.
        ``flag_na``: If ``True``, also flag NaN/null values as failing the check (default: ``False``).

    **Example usage:**

    **Temperature greater than or equal to 50:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_2]
       :end-before: [end_block_2]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.comparison_qc_1()

    **Sensor codes within a list:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_4]
       :end-before: [end_block_4]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.comparison_qc_3()

``range``
^^^^^^^^^
:class:`time_stream.qc.RangeCheck`

    **What it does:** Checks whether values fall inside or outside a min-max interval.

    **When to use:** Use for physical plausibility bounds, such as temperature between -30 and 50°C

    **Additional args:**
        ``min_value``: Minimum of the range.
        ``max_value``: Maximum of the range.
        ``closed``: Which sides of the interval are inclusive - ``"both"``, ``"left"``, ``"right"``, or ``"none"`` (default: ``"both"``).
        ``within``: Whether to flag values within the range (``True``, default) or outside it (``False``).

    **Example usage:**

    **Temperatures outside of the range -30 to 50:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_5]
       :end-before: [end_block_5]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.range_qc_1()

``time_range``
^^^^^^^^^^^^^^
:class:`time_stream.qc.TimeRangeCheck`

    **What it does:** Flags rows where the primary time column falls within a given range. Accepts
    ``datetime.time``, ``datetime.date``, or ``datetime.datetime`` bounds.

    **When to use:** Use for known bad periods such as sensor outages or automated calibration
    times.

    **Additional args:**
        ``min_value``: Start of the time range.
        ``max_value``: End of the time range.
        ``closed``: Which sides of the interval are inclusive - ``"both"``, ``"left"``, ``"right"``, or ``"none"`` (default: ``"both"``).
        ``within``: Whether to flag values within the range (``True``, default) or outside it (``False``).

    **Example usage:**

    **Flag rainfall values between the hours of 01:00 and 03:00:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_9]
       :end-before: [end_block_9]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.time_range_qc_1()

    **Flag temperature values between 03:30 and 09:30 on the 1st January:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_10]
       :end-before: [end_block_10]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.time_range_qc_2()

``spike``
^^^^^^^^^
:class:`time_stream.qc.SpikeCheck`

    **What it does:** Detects sudden jumps by assessing differences with neighbouring values.
    A point is flagged when the combined neighbour difference (minus skew) exceeds twice the
    threshold.

    **When to use:** Use for detecting unrealistic single-point spikes - isolated values that
    jump sharply compared to their neighbours.

    **Additional args:**
        ``threshold``: The spike detection threshold.

    **Example usage:**

    **Spike check on temperature data:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_7]
       :end-before: [end_block_7]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.spike_qc_1()

    .. note::
        The result doesn't flag the neighbouring high values of 50 and 52. The spike test detects a sudden
        jump where one value sits between otherwise normal values.

    .. note::
        The result returns ``null`` for the first and last values; the spike test relies on comparisons with
        neighbouring values.

``flat_line``
^^^^^^^^^^^^^
:class:`time_stream.qc.FlatLineCheck`

    **What it does:** Detects consecutive repeated (or near-repeated) values in a column.

    **When to use:** Use when a sensor stuck at a fixed value should be flagged as suspect.

    **Additional args:**
        ``min_count``: Minimum number of consecutive repeated values required for a flat line (must be at least 2).
        ``tolerance``: Optional tolerance for near-equality comparison. When set, consecutive values differing by less than or equal to this amount are considered equal (default: ``None``, exact equality).
        ``ignore_value``: Optional value or list of values that are allowed to repeat without being flagged.

    **Example usage:**

    **Flag temperature values stuck at the same reading for 3 or more consecutive timesteps:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_11]
       :end-before: [end_block_11]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.flat_line_qc_1()

    **Using** ``ignore_value`` **- suppress flagging when the repeated value is 0.0:**

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_12]
       :end-before: [end_block_12]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.flat_line_qc_2()

    .. note::
        More than one ``ignore_value`` can be specified in a list, e.g. [0.0, 20.0]

    **Using** ``tolerance`` **- flag values that barely change (within 0.01) for 3 or more consecutive readings:**

    The data below drifts slightly around 20 °C (varying by less than 0.01 between readings) before jumping
    to a different range. The ``tolerance`` parameter catches these near-flat runs that exact equality would miss.

    .. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
       :language: python
       :start-after: [start_block_13]
       :end-before: [end_block_13]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_quality_control
       examples_quality_control.flat_line_qc_3()

Observation interval
====================

Specify an observation interval to restrict the QC check to a **specific time window**. This is useful when:

- You only want to QC a specific period of observations (e.g. summer 2024).
- You need to re-run checks on recent data without reprocessing the full archive.
- You want to exclude known bad periods (e.g. sensor maintenance) from checks.

Flag Parameters
===============
The result of a QC check can be consumed in one of two ways, selected via the ``flag_params`` argument:

- **Boolean series** (default, ``flag_params`` omitted) - ``qc_check`` returns a Polars boolean
  ``Series`` of the same length as the TimeFrame, with ``True`` marking the rows that failed the
  check. Useful for chaining into custom expressions or feeding into
  :meth:`~time_stream.TimeFrame.add_flag` manually.
- **Flag column update** (``flag_params=(flag_column_name, flag_value)``) - ``qc_check`` adds the
  given flag value to the named flag column on each failing row and returns a new TimeFrame. The
  flag column must already exist (see :doc:`flagging`).

The examples below use the flag-column style. Each sets up a bitwise flag system with a single
``FLAGGED`` flag and calls :meth:`~time_stream.TimeFrame.init_flag_column` before running the
check:

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_14]
   :end-before: [end_block_14]
   :dedent:

API reference
=============

.. autosummary::

    ~time_stream.qc
    ~time_stream.TimeFrame.qc_check
