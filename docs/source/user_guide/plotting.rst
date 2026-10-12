.. _plotting:

========
Plotting
========

.. rst-class:: lead

    Plot your time series in a way that respects its temporal properties.

We provide the :meth:`~time_stream.TimeFrame.plot` method for quick and easy plots of a TimeFrame, in a style familiar
from UKCEH's hydrological plots. It is a convenience, and not the only way to plot your data: ``tf.df`` is an ordinary
Polars DataFrame, so if you need more control over how you want plots to look you can also plot it yourself with your
favourite plotting library.

The TimeFrame :meth:`~time_stream.TimeFrame.plot` method needs matplotlib, which is an optional extra:

.. code-block:: bash

   pip install "time-stream[plot]"

.. note::

   The examples on this page use real data from the UK Centre for Ecology & Hydrology:

   - Daily air temperature and precipitation from the `COSMOS-UK <https://cosmos.ceh.ac.uk/>`_ site at
     `Chimney Meadows <https://cosmos.ceh.ac.uk/sites/CHIMN>`_.
   - Gauged daily flow from the `UK National River Flow Archive <https://nrfa.ceh.ac.uk/>`_, station
     `39027, Pang at Pangbourne <https://nrfa.ceh.ac.uk/data/station/info/39027>`_.

One-liner
---------

Twelve years of daily mean air temperature at the COSMOS-UK site at Chimney Meadows:

:meth:`~time_stream.TimeFrame.plot` returns a matplotlib ``Figure``, so you can save it with ``fig.savefig(...)``
or adjust it further.

.. literalinclude:: ../examples/plotting.py
    :language: python
    :start-after: [start:quick_plot]
    :end-before: [end:quick_plot]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import plotting

   plotting.quick_plot()

In more detail
--------------

Metadata
========

Plots read these keywords from the TimeFrame metadata. Any keyword can be left out.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Keyword
     - Where
     - Effect
   * - ``label``
     - Column metadata
     - Name of the column in legends and axis labels. Defaults to the column name.
   * - ``units``
     - Column metadata
     - Added to the y axis label, e.g. ``Flow (m³/s)``.
   * - ``color``
     - Column metadata
     - Colour of the column: any matplotlib colour. Defaults to the TSP colours, starting with black.
   * - ``plot_style``
     - Column metadata
     - ``"step"``, ``"line"``, ``"bar"`` or ``"scatter"``. Defaults from the temporal properties, as above.
   * - ``title``
     - TimeFrame metadata
     - Title at the top left of the plot.

Arguments to :meth:`~time_stream.TimeFrame.plot`, such as ``title`` and ``plot_style``, take precedence over the
metadata.

The ``start`` and ``end`` arguments choose the time window to plot. Both ends are inclusive.

.. literalinclude:: ../examples/plotting.py
    :language: python
    :start-after: [start:metadata_plot]
    :end-before: [end:metadata_plot]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import plotting

   plotting.metadata_plot()

Envelope and long-term mean
===========================

``envelope`` draws the highest and lowest value at each time of year. The area above the maximum is shaded blue and
the area below the minimum red, so the white band shows the range seen before. ``long_term_mean`` adds the mean at
each time of year as a dashed line.

Both are calculated with :doc:`time_of_year_aggregation`, so values are matched by month, day and time of day: for
daily data, 1 January in the plot is compared with every 1 January in the data.

The statistics are calculated from all the data in the TimeFrame, not just the window being plotted. So load the full
record and use ``start`` and ``end`` to choose the year to show. ``reference_interval`` limits the data used, for
example to leave out the year being plotted. The footer states the data actually used: the period of record, or the
reference interval.

Here, the 2024-25 water year of the River Pang at Pangbourne is compared with its daily flow since 1968.
Flow spans a wide range, from low summer flows to winter floods, so ``log_y=True`` draws it on a logarithmic axis:

.. literalinclude:: ../examples/plotting.py
    :language: python
    :start-after: [start:envelope_plot]
    :end-before: [end:envelope_plot]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import plotting

   plotting.envelope_plot()

Envelope colours
~~~~~~~~~~~~~~~~

The default envelope colours of blue above the maximum and red below the minimum suits river flow: wetter or drier than
ever seen. For other data the colours may read the wrong way round. ``envelope_colors`` sets the colours above the
maximum and below the minimum, as any matplotlib colours. For temperature, red above (hotter than ever seen) and blue
below (colder):

.. literalinclude:: ../examples/plotting.py
    :language: python
    :start-after: [start:envelope_colors_plot]
    :end-before: [end:envelope_colors_plot]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import plotting

   plotting.envelope_colors_plot()

Several columns
===============

A list of columns shares one panel. A list of lists gives one panel per inner list, all sharing the time axis.
``subplots=True`` gives every column its own panel. ``show_missing`` shades the background wherever a value is
missing.

.. literalinclude:: ../examples/plotting.py
    :language: python
    :start-after: [start:panels_plot]
    :end-before: [end:panels_plot]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import plotting

   plotting.panels_plot()

Flags
=====

``flags`` adds a strip below the plot with a row for each :doc:`flag column <flagging>`. Each value is coloured by
the most severe flag set on it, on a grey scale that darkens with severity. Severity follows the order of the flag
system. A mapping of flag column to flags, from the least to the most severe, shows only those flags:

.. code-block:: python

   tf.plot("temperature", flags={"temperature_flags": ["SUSPECT"]})

Here, a :doc:`spike check <quality_control>` flags single-day spikes in the temperature as suspect, and missing values
are flagged as missing:

.. literalinclude:: ../examples/plotting.py
    :language: python
    :start-after: [start:flags_plot]
    :end-before: [end:flags_plot]
    :dedent:

.. jupyter-execute::
   :hide-code:

   from examples import plotting

   plotting.flags_plot()

Other options
=============

- ``log_y=True`` uses a logarithmic y axis, as in the river flow example above.
- ``ylim=(bottom, top)`` sets the y axis limits of every panel. Use ``None`` to leave either end automatic.
- ``figsize=(width, height)`` sets the figure size in inches.
