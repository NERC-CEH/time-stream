Quick Start
==========

This quick start guide will walk you through creating and working with time series data using the Time Series Package.

Creating a Time Series
---------------------

First, import the necessary modules:

.. literalinclude:: ../../../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

Create a simple dataframe with a datetime column and a value column:

.. literalinclude:: ../../../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

And create a Time Series object:

.. literalinclude:: ../../../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_3]
   :end-before: [end_block_3]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.create_timeseries()

.. note::
   More information about resolution and periodicity can be found in the :doc:`concepts page <concepts>`.

Aggregating Data
---------------

Aggregating time series data is straightforward:

.. literalinclude:: ../../../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.aggregate_data()

By default, this will aggregate the data regardless of how many missing data points there are in the period.
For example, if we have two 1 minute data points on a given day, doing a mean aggregation would return the
mean of those 2 values, even though we'd expect 1440 values for a full day.

You can specify criteria for a valid aggregation using the ``missing_criteria`` argument.

- ``{"missing": 30}`` Aggregate only if there are no more than 30 values missing in the period.
- ``{"available": 30}`` Aggregate only if there are at least 30 input values in the period.
- ``{"percent": 30}`` Aggregate only if the data in the period is at least 30 percent complete.


Adding Flags for Quality Control
-------------------------------

The Time Series object contains functionality for adding data "flags" that provide detail to specific data points.
One example usage of this is to provide information about what quality control has been carried out on the data.

Create a "flagging system" as dictionary and provide it to the Time Series initialisation:

.. literalinclude:: ../../../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.create_flagging_system()


Now we can use this flagging system to add information to our data points:

.. literalinclude:: ../../../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_6]
   :end-before: [end_block_6]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.use_flagging_system()
