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
      "comparison", "rainfall", compare_to=0, operator="<", into="rainfall_flag"
   )

A single call with rich meaning: "I want to *QC check* when my *rainfall* data is *less than* a value of *0*,
with results saved to a column named *rainfall_flag*.

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
It can return a boolean mask (for filtering) or update the TimeFrame with a new column containing the results
of the QC check. Each QC check is configurable through parameters specific to that check - see examples below.

Comparison check
----------------

Compare values against a constant or list using operators: ``<, <=, >, >=, ==, !=, is_in``.

Use for value thresholds or lists of error codes.

**Example: Temperature greater than or equal to 50:**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   examples_quality_control.comparison_qc_1()

**Example: Sensor codes within a list:**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   examples_quality_control.comparison_qc_3()

Range check
-----------

Check if values lie inside or outside a min-max interval.

Use for physical plausibility bounds (e.g. temperature between -30 and 50 °C).

**Example: Temperatures outside of the range -30 to 50:**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   examples_quality_control.range_qc_1()

Time range check
----------------

Flag data between specific time ranges.

Use for known bad periods such as sensor outages or calibration times.

**Example: Flag rainfall values between the hours of 01:00 and 03:00:**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_9]
   :end-before: [end_block_9]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   examples_quality_control.time_range_qc_1()

**Example: Flag temperature values between 03:30 and 09:30 on the 1st January:**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_10]
   :end-before: [end_block_10]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   examples_quality_control.time_range_qc_2()

Spike check
-----------

Detect sudden jumps using neighbour differences.

Use for unrealistic single-point spikes.

**Example: Spike check on temperature data:**

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

Flat line check
---------------

Detect consecutive repeated (or near-repeated) values.

Use when a sensor stuck at a fixed value should be flagged as suspect.

**Example: Flag temperature values stuck at the same reading for 3 or more consecutive timesteps:**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_11]
   :end-before: [end_block_11]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   examples_quality_control.flat_line_qc_1()

**Example: Using** ``ignore_value`` **- suppress flagging when the repeated value is 0.0:**

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

**Example: Using** ``tolerance`` **- flag values that barely change (within 0.01) for 3 or more consecutive readings:**

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

Additional parameters
=====================

Observation interval
--------------------

Specify an observation interval to restrict the QC check to a **specific time window**. This is useful when:

- You only want to QC a specific period of observations (e.g. summer 2024).
- You need to re-run checks on recent data without reprocessing the full archive.
- You want to exclude known bad periods (e.g. sensor maintenance) from checks.

Into
----

The ``into`` argument controls what you get back:

- ``into=False`` → return a boolean Series (mask of failed rows).
- ``into=True`` → add a new boolean column with an automatic name.
- ``into="my_column"`` → add a new boolean column with a custom name.

.. note::

   If a column name already exists, **Time-Stream** auto-suffixes it to avoid overwriting.
