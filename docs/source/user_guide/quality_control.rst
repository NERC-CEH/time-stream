Quality control checks
==================

The quality control (QC) system in the Time Stream library provides a flexible framework for flagging
potential issues in time series data. It allows users to define and apply QC checks to individual
columns of a ``TimeSeries`` object, storing the results in flag columns for further inspection or filtering.

Applying a QC Check
-------------------

To apply a QC check, call the ``TimeSeries.qc_check`` method on a ``TimeSeries`` object. This method allows you to:

- Specify the **check type** (see below for available built-in quality control checks)
- Choose the **column** to evaluate
- Assign a **flag column** to store the results
- Provide a **flag value** to mark failing data
- Optionally limit the QC check to a **time window**

Built-in Quality Control Checks
---------------
Several built-in QC checks are available. Each check encapsulates a validation rule and supports configuration
through parameters specific to that check:

The examples given below all use this ``TimeSeries`` object:

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

Comparison Check
~~~~~~~~~~~~~

Compares values in the ``TimeSeries`` with a constant value using a specified operator.

**Name**: ``"comparison"``

.. autoclass:: time_stream.qc.ComparisonCheck

The ``is_in`` operator is a special case, where you must pass a list of values to the check against. The check then
flags results based on whether a value in the ``TimeSeries`` is within this list.

Examples
^^^^^^^^^^^^^

**1. Temperature greater than or equal to 50**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.comparison_qc_1()

**2. Precipitation less than 0**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_3]
   :end-before: [end_block_3]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.comparison_qc_2()

**3. Sensor codes within a list**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.comparison_qc_3()

Range Check
~~~~~~~~~~~~~

Flags values in the ``TimeSeries`` outside or within a specified value range.

**Name**: ``"range"``

.. autoclass:: time_stream.qc.RangeCheck

Examples
^^^^^^^^^^^^^

**1. Temperatures outside of min and max range (below -30 and above 50)**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.range_qc_1()


**2. Precipitation values between -3 and 1**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_6]
   :end-before: [end_block_6]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.range_qc_2()

Time Range Check
~~~~~~~~~~~~~
Flags values in the ``TimeSeries`` outside or within a specified time range in the ``TimeSeries``
primary time column.

This can either be used with min / max values of:

- ``datetime.time`` : Useful for scenarios where there are consistent errors at a certain time of day,
  e.g., during an automated sensor calibration time.

- ``datetime.date`` : Useful for scenarios where a specific date range is known to be bad,
  e.g., during a date range of known sensor malfunction.

- ``datetime.datetime`` : As above, but where there you need to add a time to the date range as well.

**Name**: ``"time_range"``

.. note::
    This is equivalent to using ``RangeCheck`` with ``check_column = ts.time_name``. This is a
    convenience method to be explicit that we are working with the primary time column in the ``TimeSeries`` object.

Examples
^^^^^^^^^^^^^

**1. Flag values between the hours of 01:00 and 03:00**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_9]
   :end-before: [end_block_9]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.time_range_qc_1()

**2. Flag values between 03:30 on the 1st January and 09:30 on the 1st January**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_10]
   :end-before: [end_block_10]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.time_range_qc_2()


**3. Flag values between 1st January and the 2nd January**

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_11]
   :end-before: [end_block_11]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.time_range_qc_3()

Spike Check
~~~~~~~~~~~~~

Flags sudden jumps between values based on their differences with adjacent values (both previous and next).

.. note::
    The first and last values in a time series cannot be assessed by the spike test as it requires neighbouring values.
    The result for the first and last items will be set to NULL.

**Name**: ``"spike"``

.. autoclass:: time_stream.qc.SpikeCheck

Examples
^^^^^^^^^^^^^

**Spike check on temperature**

.. note::
    Note that the result doesn't flag the neighbouring high values of 50, 52. The spike test is really for detecting a
    sudden jump with one value between "normal" values.

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_7]
   :end-before: [end_block_7]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.spike_qc_1()

Applying QC checks during a specific time range
----------------------
The ``observation_interval`` argument can be used to constrain the QC check to a chunk of your time series.

.. literalinclude:: ../../../src/time_stream/examples/examples_quality_control.py
   :language: python
   :start-after: [start_block_8]
   :end-before: [end_block_8]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quality_control
   ts = examples_quality_control.observation_interval_example()
