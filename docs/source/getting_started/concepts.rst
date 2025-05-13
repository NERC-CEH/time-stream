Basic Concepts
=============

This page explains some of the core concepts of the Time Series Package.

TimeSeries Class
---------------

The ``TimeSeries`` class is the central component of the package. It wraps a Polars DataFrame and provides
specialized functionality for time series operations.

Key features of the ``TimeSeries`` class include:

- Management of temporal data with validation of resolution and periodicity
- Column classification (data, supplementary, flag)
- Relationship management between columns
- Aggregation functionality
- Data point annotations through flagging systems
- TimeSeries-level and column-level metadata

Time Management
--------------

At the core of the ``TimeSeries`` object is the management of the "time" aspect of the data.  Some fundamental concepts
include:

- Uniqueness of time values - there cannot be duplicate datetime values in the time series
- The TimeSeries object is time zone aware - if time zone is not inherent in the data, then a default time zone of UTC
  is set
- A time series has properties of resolution and periodicity:

Resolution
~~~~~~~~~

**Resolution** defines the precision of timestamps in your time series, i.e. to what precision of time unit should each
datetime in the time series match to. For example:

- ``P1S`` - Resolution of 1 second (no sub-second precision)
- ``PT1M`` - Resolution of 1 minute (seconds must be 0)
- ``P1D`` - Resolution of 1 day (time must be 00:00:00)

Periodicity
~~~~~~~~~~
**Periodicity** defines how frequently data points appear, or the "spacing" between points, i.e. how many data
points are allowed within a given period of time. For example:

- ``PT15M`` - At most 1 datetime can occur within any 15-minute duration. Each 15-minute durations starts at
  ("00", "15", "30", "45") minutes past the hour.
- ``P1D`` - At most 1 datetime can occur within any given calendar day (from midnight of first day up to, but
  not including, midnight of the next day)
- ``P1M`` - At most 1 datetime can occur within any given calendar month (from midnight on the 1st of the month
  up to, but not including, midnight on the 1st of the following month).
- ``P3M`` - At most 1 datetime can occur within any given quarterly period.

If resolution and periodicity are not provided on ``TimeSeries`` initialisation, both will be set to a period of
``PT0.000001S`` - effectively allowing any datetime values in the time series (though datetime values must still
be unique).

Period Class
-----------

The ``Period`` class represents a time interval and is used to:

1. Define the resolution and periodicity of a TimeSeries
2. Specify intervals for aggregation
3. Map between datetime objects and ordinal values
4. Support various calendar operations

More information can be found in the :doc:`periods user guide page <../user_guide/periods>`.

Column Types
-----------

The Time Series Package offers four types of columns:

1. **Primary Time Column**: The datetime column that controls the time series
2. **Data Columns**: Contain the primary measurements or values
3. **Supplementary Columns**: Contain metadata or auxiliary information
4. **Flag Columns**: Contain quality flags or markers

Aggregation
----------

The package provides a flexible framework for aggregating time series data:

- Aggregate data over various time periods (daily, monthly, etc.)
- Apply different aggregation functions (mean, min, max, etc.)
- Track data availability with count fields
- Validate aggregation using criteria for data availability
- Preserve relationships between columns during aggregation

Flagging System
--------------

The flagging system supports data annotation management:

- Define flag systems with specific meanings
- Create flag columns linked to data columns
- Use bitwise operations to efficiently store multiple flags
- Query data based on flag status

Relationships
------------

Columns in a time series can have different relationships with other columns:

- **One-to-Many**: Common between data and flag columns, where a data column can link with multiple flag columns,
  but a flag column can only be linked to a single data column.
- **Many-to-Many**: Common between data and supplementary columns, where a supplementary column can link with multiple
  data columns, and vice vera.

These relationships are maintained when selecting, filtering, or aggregating data.


Metadata
--------

The ``TimeSeries`` object supports two levels of metadata:

- **Time series level** - Metadata describing things about the time series as whole.  For example, all the data may relate to a single location or site
- **Column level** - Metadata describing things about individual columns.  For example, units of a particular variable
