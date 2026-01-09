.. __calculations:

============
Calculations
============


Calculate the min-max envelope
===============================

The :meth:`~time_stream.TimeFrame.calculate_min_max_envelope` function calculates the historical minimum and maximum
values for each unique day-time within the time series in accordance to the time series' resolution. This can be 
useful to allow plotting of current data against the historical minimum and maxmum values across the time range.

For example, the below data shows for a monthly time series containing temperature values over a 3 year time period, and
the calculated min-max envelope. The computed minimum and maximum values are merged back onto the original time series
to allow easy plotting.

In the interest of readability, only data for the first 4 months of each year are shown

.. table:: Input time series dataframe

   +------------+-------+
   | timestamp  | value |
   +============+=======+
   | 2020-01-01 | 0.5   |
   +------------+-------+
   | 2020-02-01 | 2.0   |
   +------------+-------+
   | 2020-03-01 | 5.0   |
   +------------+-------+
   | 2020-04-01 | 15.0  |
   +------------+-------+
   | ...        | ...   |
   +------------+-------+
   | 2021-01-01 | -2.0  |
   +------------+-------+
   | 2021-02-01 | 5.0   |
   +------------+-------+
   | 2021-03-01 | 0.0   |
   +------------+-------+
   | 2021-04-01 | 10.0  |
   +------------+-------+
   | ...        | ...   |
   +------------+-------+
   | 2022-01-01 | 2.0   |
   +------------+-------+
   | 2023-02-01 | -1.0  |
   +------------+-------+
   | 2022-03-01 | 7.0   |
   +------------+-------+
   | 2022-04-01 | 8.0   |
   +------------+-------+


.. table:: Output time series dataframe with the min and max historical values added

   +------------+-------+------+-----+
   | timestamp  | value | min  | max |
   +============+=======+======+=====+
   | 2020-01-01 | 0.5   | -2.0 | 2.0 |
   +------------+-------+------+-----+
   | 2020-02-01 | 2.0   | -1.0 | 5.0 |
   +------------+-------+------+-----+
   | 2020-03-01 | 5.0   | 0.0  | 7.0 |
   +------------+-------+------+-----+
   | 2020-04-01 | 15.0  | 15.0 | 8.0 |
   +------------+-------+------+-----+
   | ...        | ...   | ...  | ... |
   +------------+-------+------+-----+
   | 2021-01-01 | -2.0  | -2.0 | 2.0 |
   +------------+-------+------+-----+
   | 2021-02-01 | 5.0   | -1.0 | 5.0 |
   +------------+-------+------+-----+
   | 2021-03-01 | 0.0   | 0.0  | 7.0 |
   +------------+-------+------+-----+
   | 2021-04-01 | 10.0  | 15.0 | 8.0 |
   +------------+-------+------+-----+
   | ...        | ...   | ...  | ... |
   +------------+-------+------+-----+
   | 2022-01-01 | 2.0   | -2.0 | 2.0 |
   +------------+-------+------+-----+
   | 2023-02-01 | -1.0  | -1.0 | 5.0 |
   +------------+-------+------+-----+
   | 2022-03-01 | 7.0   | 0.0  | 7.0 |
   +------------+-------+------+-----+
   | 2022-04-01 | 8.0   | 15.0 | 8.0 |
   +------------+-------+------+-----+


For other TimeFrame resolutions, e.g. daily, hourly etc, the same logic is applied. For example, for a daily time
series the min-max envelope would be calculated on a daily basis (e.g. every instance of 1st Jan, 2nd Jan etc). 
For an hourly time series, the min max-envelope would be calculated on an hourly basis (e.g. every instance of 
1st Jan 00:00, 1st Jan 01:00, 1st Jan 02:00 etc). For minute resolution the min-max envelope would be calculated for
1st Jan 00:00:00, 1st Jan 00:01:00 and so on.