Welcome to Time Stream Documentation
===========================================

**Time S**\ eries **T**\ oolkit for **R**\ apid **E**\ nvironmental **A**\ nalysis and **M**\ onitoring: A comprehensive package
for handling series data with specialised focus on time intervals, aggregation, and data flagging.

Key Features
-----------

* **Time series data structure**: Robust time series data model built on Polars DataFrames
* **Period-based time management**: Flexible handling of time resolutions and periodicity (days, months, years, etc.)
* **Aggregation framework**: Easily aggregate time series data over various periods
* **Flagging system**: Built-in flagging system for data point management and provenance
* **Column relationships**: Define and manage relationships between data and metadata
* **Infilling routines**: Fill gaps in time series data using a variety of infilling methods

Getting Started
--------------

.. toctree::
   :maxdepth: 2
   
   getting_started/installation
   getting_started/quick_start
   getting_started/concepts

User Guide
---------

.. toctree::
   :maxdepth: 2
   
   user_guide/timeseries_basics
   user_guide/periods
   user_guide/quality_control
   user_guide/infilling
   user_guide/aggregation
   user_guide/flagging
   user_guide/column_relationships

Examples
-------

.. toctree::
   :maxdepth: 2
   
   examples/creating_timeseries
   examples/aggregation_examples
   examples/quality_flagging
   examples/custom_periods

API Reference
------------

.. toctree::
   :maxdepth: 2
   
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
   
   developer/contributing
   developer/testing

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

This project is licensed under the GNU General Public License v3.0 (GPLv3).

You may obtain a copy of the license at:

https://www.gnu.org/licenses/gpl-3.0.en.html

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
