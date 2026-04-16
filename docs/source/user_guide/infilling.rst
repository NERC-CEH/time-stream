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

    tf.infill("linear", "flow", max_gap_size=3)

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
timeseries data in **Time-Stream**. There are various infill methods available; from using alternative data from
another source, to delegating to well established methods from the `SciPy data science library
<https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_. All methods are combined with the time-integrity
of your **TimeFrame**.

Let's look at the method in more detail:

.. automethod:: time_stream.TimeFrame.infill
   :no-index:

Infill methods
--------------

The ``infill_method`` parameter lets you choose how missing values are estimated by passing a method name as a string.
Each method has its strengths, depending on your data. The currently available methods are:

Simple infilling techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``alt_data``
^^^^^^^^^^^^
:class:`time_stream.infill.AltData`

    **What it does:** Infills using data from an alternative source - either another column in your
    TimeFrame, or data from a different DataFrame entirely.

    **When to use:** When you have a secondary data source that can stand in for missing values,
    such as a nearby gauge or a modelled estimate.

    **Additional args:**
        ``alt_data_column``: The name of the column providing the alternative data.
        ``correction_factor``: An optional multiplier to apply to the alternative data (default: 1.0).
        ``alt_df``: A separate Polars DataFrame containing the alternative data. If omitted, the
        column is taken from the current TimeFrame.

    **Example usage:** ``tf_filled = tf.infill("alt_data", "flow", alt_data_column="flow_model", alt_df=model_df)``

Polynomial interpolation
~~~~~~~~~~~~~~~~~~~~~~~~

``linear``
^^^^^^^^^^
:class:`time_stream.infill.LinearInterpolation`

    **What it does:** Straight-line interpolation between neighbouring known points.

    **When to use:** Simple and neutral; best for short gaps where the underlying signal is
    unlikely to curve significantly.

    **Additional args:** None.

    **Example usage:** ``tf_filled = tf.infill("linear", "flow", max_gap_size=3)``

``quadratic``
^^^^^^^^^^^^^
:class:`time_stream.infill.QuadraticInterpolation`

    **What it does:** Second-order polynomial curve through neighbouring known points.

    **When to use:** Captures gentle curvature; suitable when changes are not strictly linear
    but you do not need a high-order fit.

    **Additional args:** None.

    **Example usage:** ``tf_filled = tf.infill("quadratic", "flow", max_gap_size=3)``

``cubic``
^^^^^^^^^
:class:`time_stream.infill.CubicInterpolation`

    **What it does:** Third-order polynomial curve through neighbouring known points.

    **When to use:** Produces smooth transitions; can be useful for variables with cyclical
    patterns or gradually changing curvature.

    **Additional args:** None.

    **Example usage:** ``tf_filled = tf.infill("cubic", "flow", max_gap_size=3)``

``bspline``
^^^^^^^^^^^
:class:`time_stream.infill.BSplineInterpolation`

    **What it does:** B-spline interpolation with a configurable polynomial order.

    **When to use:** When you want full control over the interpolation order. The other
    polynomial methods (linear, quadratic, cubic) are convenience wrappers around this.

    **Additional args:**
        ``order``: Order of the B-spline (1-5, where 1=linear, 2=quadratic, 3=cubic).

    **Example usage:** ``tf_filled = tf.infill("bspline", "flow", max_gap_size=3, order=4)``

Shape-preserving methods
~~~~~~~~~~~~~~~~~~~~~~~~

``pchip``
^^^^^^^^^
:class:`time_stream.infill.PchipInterpolation`

    **What it does:** Piecewise Cubic Hermite Interpolating Polynomial.

    **When to use:** Preserves monotonicity and avoids overshoot; can help to avoid unrealistic
    fluctuations between known values. A good default when you want a smooth curve that respects
    the shape of your data.

    **Additional args:** None.

    **Example usage:** ``tf_filled = tf.infill("pchip", "flow", max_gap_size=3)``

``akima``
^^^^^^^^^
:class:`time_stream.infill.AkimaInterpolation`

    **What it does:** Akima spline - a smooth curve fit that reduces oscillations near outliers.

    **When to use:** Best for data with significant local variations and potential outliers,
    where standard cubic interpolation might overshoot.

    **Additional args:** None.

    **Example usage:** ``tf_filled = tf.infill("akima", "flow", max_gap_size=5)``

.. note::

   All methods honour the maximum gap limit: they will only fill runs of missing values up to your chosen length,
   leaving longer gaps as NaN.

.. note::

    For infill methods using interpolation techniques, NaN values at the very beginning and very end of a timeseries
    will remain NaN; there is no pre- or post- data to constrain the infilling method.

Column selection
----------------

The ``column_name`` parameter lets you specify which column to infill; only this column will be used by the infill
function.


Observation interval
--------------------

The ``observation_interval`` parameter lets you specify an observation interval to restrict infilling
to a **specific time window**. This is useful when:

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

This will only attempt infilling **between January to December 2024**; gaps outside that interval remain untouched.

Max gap size
------------

Use the ``max_gap_size`` parameter to prevent **over-eager interpolation**. Only gaps less than this
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

Flagging infilled values
------------------------

When :meth:`~time_stream.TimeFrame.infill` replaces a null value with an interpolated one, you
often want a record of which rows were touched. Pass ``flag_params=(flag_column_name, flag_value)``
and **Time-Stream** will add the given flag to every row that went from null to non-null during
the infill. The flag column must already exist - see :doc:`flagging` for how to create one.

**Code:**

.. literalinclude:: ../../../src/time_stream/examples/examples_infilling.py
    :language: python
    :start-after: [start_block_3]
    :end-before: [end_block_3]
    :dedent:

**Output:**

.. jupyter-execute::
    :hide-code:

    import examples_infilling
    examples_infilling.flagged_infill()

Only rows whose value changed from null to non-null are flagged; rows that were already populated,
or that remain null because the gap exceeded ``max_gap_size``, are left untouched.

Examples
========

Alternative data infilling
--------------------------

The ``"alt_data"`` infill method allows you to fill missing values in a column using data from an alternative source.

You can specify the alternative data in two ways:

1.  **From a column within the same TimeFrame**: If the alternative data is already present as a column in your
    current :class:`~time_stream.TimeFrame` object, you can directly reference it.
2.  **From a separate DataFrame**: You can provide an entirely separate
    Polars DataFrame containing the alternative data.

In both cases, you can also apply a ``correction_factor`` to the alternative data before it's used for infilling.

Infilling from a separate DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say you have a primary dataset with missing "flow" values, and a separate ``alt_df`` with "alt_data" that
can be used to infill these gaps.

**Input:**

.. tab-set::
    :class: outline padded-tabs

    .. tab-item:: Main Data

        .. jupyter-execute::
            :hide-code:

            import examples_infilling
            ts = examples_infilling.alt_data_main()

    .. tab-item:: Alternative Data

        .. jupyter-execute::
            :hide-code:

            import examples_infilling
            ts = examples_infilling.alt_data_alt()

**Code:**

.. literalinclude:: ../../../src/time_stream/examples/examples_infilling.py
    :language: python
    :start-after: [start_block_2]
    :end-before: [end_block_2]
    :dedent:

**Output:**

.. jupyter-execute::
    :hide-code:

    import examples_infilling
    ts = examples_infilling.alt_data_infill()

Visualisation of interpolation methods
======================================

A quick visualisation of the results from the different interpolation infill methods is sometimes useful. However,
bear in mind that this is a very simplistic example and the correct method to use is dependent on your data.
You should do your research into which is most appropriate.

.. jupyter-execute::
   :hide-code:

   import examples_infilling
   tf = examples_infilling.all_infills()

.. plot:: ../../src/time_stream/examples/examples_infilling.py plot_all_infills

API reference
=============

.. autosummary::

    ~time_stream.infilling
    ~time_stream.TimeFrame.infill
