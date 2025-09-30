.. _infilling:

===========
Infilling
===========

.. rst-class:: lead

    Missing data happens. Fill the gaps with precision and care.

Why use Time-Stream?
====================

It is inevitable that real-world monitoring data has gaps, whether that's from: communications outages,
sensor swaps or power cuts. With **Time-Stream**, you can fill those missing values with a robust
infilling procedure that benefits from deep knowledge of the time properties of your data.

One-liner
---------

With **Time-Stream** you state intent, not mechanics:

.. code-block:: python

    tf.infill("linear", "flow", max_gap=3)

That's it, a single line with clear intent: "I want to use the *linear* infill method on my *flow* data,
but only for *gaps* ≤ 3 steps".

Complex example
---------------

Let's take our example 15-minute river flow data that contains a few short outages. You might want to:

- Fill only gaps up to 3 consecutive steps (≤45 minutes).
- Use `linear interpolation
  <https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#piecewise-linear-interpolation>`_ to infill tiny gaps
  (1 step) and a more complex interpolator (e.g. `PCHIP
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_) for ≥2 step gaps.

**Input:**

.. _example_input_data:

15-minute river flow timeseries, including some missing data.

.. jupyter-execute::
   :hide-code:

   import examples_aggregation
   ts = examples_aggregation.get_example_df("polars")

**Code:**

.. literalinclude:: ../../../src/time_stream/examples/examples_infilling.py
    :language: python
    :start-after: [start_block_1]
    :end-before: [end_block_1]
    :dedent:

**Output:**

.. jupyter-execute::
    :hide-code:

    import examples_infilling
    ts = examples_infilling.time_stream_example()

Key benefits
------------

- **Conservative**: You set the rules; the library enforces them.
- **Time aware**: Honours the resolution and periodicity properties of your data.
- **Simple code**: One call conveys the method, scope, and policy.

In more detail
==============

The :meth:`~time_stream.TimeFrame.infill` method is the entry point for infilling your
timeseries data in **Time-Stream**. It delegates to well established methods from the `SciPy data science library
<https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_, combined with the time-integrity of your **TimeFrame**.

Infill methods
--------------

Choose how missing values are estimated by passing a method name as a string. Each method has its strengths,
depending on your data.

Polynomial interpolation
~~~~~~~~~~~~~~~~~~~~~~~~

- ``"linear"`` - **straight-line interpolation between neighbouring points.**

  Simple and neutral; best for very short gaps (1–2 steps).

- ``"quadratic"`` - **second-order polynomial curve.**

  Captures gentle curvature; suitable when changes aren't linear.

- ``"cubic"`` - **third-order polynomial curve.**

  Smooth transitions; can be useful for variables with cyclical patterns.

- ``"bspline"`` - **B-spline interpolation (configurable order).**

  Flexible piecewise polynomials; user decides.*

Shape-preserving methods
~~~~~~~~~~~~~~~~~~~~~~~~

- ``"pchip"`` - **Piecewise Cubic Hermite Interpolating Polynomial.**

  Preserves monotonicity and avoids overshoot; can help to avoid unrealistic fluctuations between values.

- ``"akima"`` - **Akima spline**.

  A smooth curve fit for data with significant local variations and potential outliers.

.. note::

   All methods honour the maximum gap limit: they will only fill runs of missing values up to your chosen length,
   leaving longer gaps as NaN.

.. note::

    NaN values at the very beginning and very end of a timeseries will remain NaN; there is no pre- or post- data to
    constrain the infilling method.

Column selection
----------------

Specify which column to infill; only this column will be used by the infill function.


Column selection
----------------

Specify which column to infill; only this column will be used by the infill function.


Observation interval
--------------------

Specify an observation interval to restrict infilling to a **specific time window**. This is useful when:

- You only want to work with a subset of data (e.g. one hydrological year).
- You want to fill recent gaps without touching the historical record.
- You need to use different methods for different parts of your timeseries.

Example:

.. code-block:: python

   from datetime import datetime

   tf_recent = tf.infill(
       "linear",
       "flow",
       observation_interval=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
   )

This will only attempt infilling **between January to Decemeber 2024**; gaps outside that interval remain untouched.

Max gap size
------------

Use the maximum gap size to prevent **over-eager interpolation**. Only gaps less than this
(measured in consecutive missing **steps**) will be infilled.

Example:

.. code-block:: python

   # Fill single-step gaps only (≤ 15 minutes at 15-min resolution)
   tf1 = tf.infill("linear", "flow", max_gap_size=1)

   # Fill gaps up to 2 steps (≤ 30 minutes)
   tf2 = tf.infill("akima", "flow", max_gap_size=2)

.. note::

   The definition of "gap size" depends on the **TimeFrame resolution**.
   At 15-minute resolution, ``max_gap_size=2`` = 30 minutes; at daily resolution,
   ``max_gap_size=2`` = 2 days.

Visualisation of methods
========================

A quick visualisation of the results from the different infill methods is sometimes useful. However, bear in mind
that this is a very simplistic example and the correct method to use is dependent on your data.
You should do your research into which is most appropriate.

.. jupyter-execute::
   :hide-code:

   import examples_infilling
   tf = examples_infilling.all_infills()

.. plot:: ../../src/time_stream/examples/examples_infilling.py plot_all_infills
