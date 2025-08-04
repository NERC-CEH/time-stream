Infilling
==================

The infill module of the Time Stream library provides various methods for filling missing values in your
time series data. Missing data is a common challenge in time series analysis, whether due to sensor failures,
network outages, data transmission errors, or scheduled maintenance periods.

The infill system in the Time Stream library provides a flexible framework for filling in missing values in your
time series data. It allows users to define and apply infilling to individual columns of a ``TimeSeries`` object.

Applying an Infilling Procedure
-------------------

To apply infilling, call the ``TimeSeries.infill`` method on a ``TimeSeries`` object. This method allows you to:

- Specify the **infill method** (see below for available built-in methods)
- Choose the **column** to infill
- Optionally limit the infilling to a **time observation window**
- Optionally limit the infilling to a maximum **gap window** size, to avoid unrealistic estimates across large missing periods

Built-in Infilling Methods
---------------
Several built-in infilling methods are available. These are built upon well established methods from the `SciPy
data science library <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_:

**Polynomial Interpolation**
   - **Linear**: Simple straight-line interpolation between points
   - **Quadratic**: Smooth curves using second-order polynomials
   - **Cubic**: Natural-looking curves using third-order polynomials
   - **B-Spline**: Flexible piecewise polynomials with configurable order

**Shape-Preserving Methods**
   - **PCHIP**: Preserves monotonicity and avoids overshoots
   - **Akima**: Reduces oscillations in data with rapid changes

Each method supports configuration through parameters specific to that method.

The examples given below all use this ``TimeSeries`` object:

.. literalinclude:: ../../../src/time_stream/examples/examples_infilling.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_infilling
   ts = examples_infilling.create_simple_time_series_with_gaps()

.. plot::

   import polars as pl
   import matplotlib.pyplot as plt

   # Create sample data
   df = pl.DataFrame({
       "x": range(10),
       "y": [i**2 for i in range(10)]
   })

   # Plot using Polars' matplotlib backend
   df.plot.scatter(x="x", y="y")
   plt.title("Polars Scatter Plot")
   plt.show()


Examples
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../../src/time_stream/examples/examples_infilling.py
   :language: python
   :start-after: [start_block_2]
   :end-before: [end_block_2]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_infilling
   ts = examples_infilling.all_infills()

Linear Interpolation
~~~~~~~~~~~~~~~~~~~

**Name**: ``"linear"``

.. autoclass:: time_stream.infill.LinearInterpolation

Quadratic Interpolation
~~~~~~~~~~~~~~~~~~~

**Name**: ``"quadratic"``

.. autoclass:: time_stream.infill.QuadraticInterpolation

Quadratic Interpolation
~~~~~~~~~~~~~~~~~~~

**Name**: ``"cubic"``

.. autoclass:: time_stream.infill.CubicInterpolation

Akima Interpolation
~~~~~~~~~~~~~~~~~~~

**Name**: ``"akima"``

.. autoclass:: time_stream.infill.AkimaInterpolation

PCHIP Interpolation
~~~~~~~~~~~~~~~~~~~

**Name**: ``"pchip"``

.. autoclass:: time_stream.infill.PchipInterpolation
