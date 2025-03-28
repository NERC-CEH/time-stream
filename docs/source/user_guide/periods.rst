Working with Periods
==================

The ``Period`` class is a fundamental component of the Time Series Package, representing time intervals used for
various operations.

Understanding Periods
-------------------

A ``Period`` represents a fixed interval of time, such as:

- One year (``P1Y``)
- One month (``P1M``)
- One day (``P1D``)
- One hour (``PT1H``)
- 15 minutes (``PT15M``)

Periods serve several purposes:

1. Defining the **resolution** of timestamps in a TimeSeries
2. Defining the **periodicity** of data points in a TimeSeries
3. Specifying time intervals for **aggregation**
4. Converting between **datetime objects** and **ordinal values**

Creating Periods
--------------

Basic Factory Methods
~~~~~~~~~~~~~~~~~~~

The ``Period`` class provides various factory methods:

.. literalinclude:: ../../../src/time_stream/examples/examples_periods.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

From ISO 8601 Duration Strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create periods from ISO 8601 duration strings:

.. literalinclude:: ../../../src/time_stream/examples/examples_periods.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

From timedelta Objects
~~~~~~~~~~~~~~~~~~~

Convert Python timedelta objects to Periods (for intervals of time on the scale of days and below):

.. literalinclude:: ../../../src/time_stream/examples/examples_periods.py
   :language: python
   :start-after: [start_block_3]
   :end-before: [end_block_3]
   :dedent:

Periods with Offsets
~~~~~~~~~~~~~~~~~~~~

Offsets allow you to create custom Periods that start at a point in time offset from the default.  For example,
a UK "Water year" starts on 9am October 1st.  This would be defined with a 10 month and 9 hour offset to a 1 year
period. Some more examples below:

.. literalinclude:: ../../../src/time_stream/examples/examples_periods.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:
