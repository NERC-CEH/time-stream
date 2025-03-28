Welcome to Time Series Documentation
===========================================

A comprehensive package for handling time series data with specialised
focus on time intervals, aggregation, and data flagging.

Key Features
-----------

* **Time series data structure**: Robust time series data model built on Polars DataFrames
* **Period-based time management**: Flexible handling of time resolutions and periodicity (days, months, years, etc.)
* **Aggregation framework**: Easily aggregate time series data over various periods
* **Flagging system**: Built-in flagging system for data point management and provenance
* **Column relationships**: Define and manage relationships between data and metadata

Getting Started
--------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   getting_started/installation
   getting_started/quick_start
   getting_started/concepts

User Guide
---------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/timeseries_basics
   user_guide/periods
   user_guide/aggregation
   user_guide/flagging
   user_guide/column_relationships

Examples
-------

.. toctree::
   :maxdepth: 2
   :caption: Examples
   
   examples/creating_timeseries
   examples/aggregation_examples
   examples/quality_flagging
   examples/custom_periods

API Reference
------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/timeseries
   api/period
   api/aggregation
   api/columns
   api/flag_manager
   api/relationships
   api/bitwise

Developer Guide
-------------

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   
   developer/contributing
   developer/testing

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
