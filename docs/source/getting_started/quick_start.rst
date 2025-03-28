Quick Start
==========

This quick start guide will walk you through creating and working with time series data using the Time Series Package.

Creating a Time Series
---------------------

First, import the necessary modules:

.. literalinclude:: ../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

Create a simple dataframe with a datetime column and a value column:

.. literalinclude:: ../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

And create a Time Series object:

.. literalinclude:: ../src/time_stream/examples/examples_quick_start.py
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

.. literalinclude:: ../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_4]
   :end-before: [end_block_4]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.aggregate_data()

Adding Flags for Quality Control
-------------------------------

The Time Series object contains functionality for adding data "flags" that provide detail to specific data points.
One example usage of this is to provide information about what quality control has been carried out on the data.

Create a "flagging system" as dictionary and provide it to the Time Series initialisation:

.. literalinclude:: ../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_5]
   :end-before: [end_block_5]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.create_flagging_system()


Now we can use this flagging system to add information to our data points:

.. literalinclude:: ../src/time_stream/examples/examples_quick_start.py
   :language: python
   :start-after: [start_block_6]
   :end-before: [end_block_6]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_quick_start
   ts = examples_quick_start.use_flagging_system()
