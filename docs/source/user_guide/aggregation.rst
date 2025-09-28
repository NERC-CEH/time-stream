.. _aggregation:

===========
Aggregation
===========

.. rst-class:: lead

    Simple code, robust results.

Why use Time-Stream?
====================

Aggregating time series data sounds simple - until you hit real-world edge cases, particularly when working with
hydrological and environmental datasets. There are considerations such as:
handling leap years, anchor points, working with offset-periods (like water-years), or
tracking completeness of data going into each aggregation.

Forget the pain of writing your own custom code; with **Time-Stream**, you declare the aggregation period you want
(daily, monthly, yearly, or any other custom period) and the library handles the rest. Simple code, robust results.

One-liner
---------

Rolling your own aggregation functionality can get complex.
With *Time-Stream* you express the intent directly:

.. code-block:: python

   tf.aggregate("P1D", "sum", "precipitation")

That's it, a single line with clear intent: "I want a *daily* *sum* of my *precipitation* data."

Complex example
---------------

UK hydrologists often use a "UK water-year" that runs from **1 October at 09:00** through to the
following year. Writing code to aggregate data (whether that's 15-minute river level, or daily flow values)
into water-year totals can be painful.

Take, for example, generating a water-year "annual maximum" (AMAX)
series. You might want to:

- Generate an output time series, with the datetime values of the start of each water-year
- Keep hold of the exact timestamp that the maximum occurred on
- Know exactly how many data points were available in each water year

Here's how you might have to do this in Pandas, Polars and in Time-Stream:

.. tab-set::
    :class: outline padded-tabs

    .. tab-item:: Pandas

        .. code-block:: python

            import pandas as pd

    .. tab-item:: Polars

        .. code-block:: python

            import polars as pl

    .. tab-item:: Time-Stream

        .. code-block:: python

            import time_stream as ts

            tf_amax = tf.aggregate("P1Y+9MT9H", "max", "flow")

Key benefits
------------

- **Less boilerplate**
  No need to wrangle custom datetime columns or write manual offset logic.

- **Fewer mistakes**
  Periodicity, alignment, and anchor semantics are enforced for you.

- **Domain-ready**
  Express hydrological conventions directly: daily at 09:00, or water year from October.

- **Readable & reproducible**
  Your code is self-explanatory to collaborators and reviewers.

In more detail: :meth:`~time_stream.TimeFrame.aggregate`
========================================================

to do

Aggregation methods
-------------------

to do

Column selection
----------------

to do

Missing criteria
----------------

to do

Time anchoring
--------------

to do

