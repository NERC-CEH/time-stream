.. _concepts:

========
Concepts
========

.. rst-class:: lead

    Understand the essential concepts within **Time-Stream**.

Time Properties
===============

Time properties define how your timeseries data is structured and sampled along the **timeline**. These properties
establish the "time grid" that governs when measurements are expected, how they align with each other, and
help establish the temporal patterns that exist in your data.

The core time properties work together:

- **Resolution** sets the sampling interval (e.g., every 15 minutes, daily, yearly)
- **Offset** shifts the sampling points from their natural boundaries (e.g., 9am instead of midnight for daily data)
- **Alignment** is the resulting time grid where all timestamps must fall (derived from resolution + offset)
- **Periodicity** describes the repeating patterns in your data

Getting these properties right ensures your timeseries data is valid, comparable with other datasets,
and ready for analysis.

.. note::

   ``resolution`` and ``periodicity`` are provided to a :class:`~time_stream.TimeFrame` as
   `ISO-8601 duration strings <https://en.wikipedia.org/wiki/ISO_8601#Durations>`_.

   ``offset`` is provided to a :class:`~time_stream.TimeFrame` as a "modified" ISO-8601 string, replacing the "P"
   with a "+".

   ``alignment`` is calculated internally as ``resolution`` + ``offset``

Resolution
----------

**What it is:** defines the sampling interval for your timeseries; the unit of time step allowable between consecutive
data points.

**Examples:**

- ``PT1S`` - 1 second step
- ``PT15M`` - 15 minute step
- ``P1D`` - daily step
- ``P3M`` - quarterly step
- ``P1Y`` - yearly step

**Why it matters:**

- Makes **comparisons deterministic** - data from different sources line up only if they share a resolution.
- Defines the **granularity of analysis** - a 15-minute resolution series has more detail than a daily one,
  and operations like padding or infilling often depend on knowing the intended resolution.
- Combined with ``offset`` it defines the **alignment grid** used to validate timestamps.

Offset
------

**What it is:** the shift from the **natural boundary** of ``resolution`` to position the time steps in your timeseries.

**More detail:**  The **natural boundary** can be thought of as the "start line" from which the ``resolution`` step
repeats. For **calendar step resolutions** (days, months, years), the natural boundary is the start of that calendar
period e.g.:

    - midnight for a day
    - 1st of the month 00:00 for a month
    - 1st Jan 00:00 for a year

For **clock step resolutions** (seconds, minutes, hours) the natural boundary is the start of that *unit*, e.g.:

    - minute = 0 seconds of each minute
    - 15 minutes = 0 seconds at each 15 minute (0, 15, 30, 45)
    - hour = 0 minutes, 0 seconds of the hour

The ``offset`` allows you to modify this natural boundary to reflect the actual time steps when your data was
measured. For example, you may measure daily data (``resolution="P1D"``), but the values are measured
at 9:00am each day. This would be an offset of 9 hours (``"+T9H"``) from the natural boundary of midnight 00:00.

**Examples:**

- ``+T9H`` - offset by 9 hours (e.g. for UK water-day)
- ``+9MT9H`` - offset by 9 months and 9 hours (e.g. for UK water-year)
- ``+T5M`` - offset by 5 minutes (e.g. used with ``PT15M`` resolution, expect values  at **:05, :20, :35, :50**)

**Why it matters:**

- Encodes hydrology conventions (water day/year) cleanly.
- Helps catch accidental drift (e.g. "00:14" instead of "00:15").

Alignment
---------

**What it is:** the **derived time grid** defined by ``resolution + offset`` - i.e. the set of all allowed
timestamp positions.

**Examples:**

- ``resolution="PT15M"`` + ``offset=None`` = ``alignment="PT15M"``. Values expected at **:00, :15, :30, :45**
- ``resolution="PT15M"`` + ``offset=+T5M`` = ``alignment="PT15M+T5M"``. Values expected at **:05, :20, :35, :50**
- ``resolution="P1D"`` + ``offset="+T9H"`` = ``alignment="P1D+T9H"``. Values expected at **09:00 every day**
- ``resolution="P1Y"`` + ``offset="+9MT9H"`` = ``alignment="P1D+9MT9H"``. Values expected on **1 Oct 09:00 each year**

.. mermaid::

    graph LR
        A[Alignment<br/>:05, :20, :35, :50]
        R[Resolution<br/>PT15M] --> Natural
        Natural["Natural Boundary<br/>:00, :15, :30, :45"] --> Shift
        O[Offset<br/>+T5M] --> Shift
        Shift["shift by offset"] --> A

        style R fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
        style O fill:#fff4e1,stroke:#cc6600,stroke-width:2px
        style A fill:#e1ffe1,stroke:#00cc00,stroke-width:3px
        style Natural fill:#f5f5f5,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5

**Why it matters:**

- Ensures valid, aligned timestamps; all timestamps must **snap to this grid**.
- Defines the default ``periodicity``. 1 value expected for each "tick".

Periodicity
-----------

**What it is:** defines how frequently data points appear, or the *spacing* between points,
i.e. how many data points are allowed within a given period of time.

**Examples:**

- ``PT15M`` - At most 1 datetime can occur within any 15-minute duration. Each 15-minute durations starts at
  ("00", "15", "30", "45") minutes past the hour.
- ``P1D`` - At most 1 datetime can occur within any given calendar day (from midnight of first day up to, but
  not including, midnight of the next day).
- ``P1M`` - At most 1 datetime can occur within any given calendar month (from midnight on the 1st of the month
  up to, but not including, midnight on the 1st of the following month).
- ``P3M`` - At most 1 datetime can occur within any given quarterly period.
- ``P1Y+9M9H`` - at most one entry per **UK water year** (09:00 on 1st October each year).

**Why it matters:**

- Enforces **no duplicates in a time window** - e.g. no two daily values for the same day.
- Provides the **expected sampling frequency** - ensures a 15-minute series really has maximum of 96 points/day.
- Allows **missing value detection**
- Allows **robust aggregation** - defines the windows into which multiple values are grouped.

Time Anchor
-----------

**What it is:** defines *where in the period a timestamp is positioned* and therefore
*over which span of time a value is valid*.  Think of it as: *does the timestamp represent the start of the measurement,
the end of the measurement, or a single instant?*.

**Examples:**

- ``point``: A value at a timestamp is considered valid only for the **instant** of that timestamp.
- ``start``: A value at a timestamp is considered valid **starting** at that timestamp (inclusive) and ending
  at the timestamp + dataset resolution (exclusive).
- ``end``: A value at a timestamp is considered valid starting at timestamp - dataset resolution (exclusive)
  and **ending** at the timestamp (inclusive).

.. mermaid::

   gantt
        title Time Anchor
        dateFormat  HH:mm
        axisFormat  %H:%M
        todayMarker off

        POINT  :milestone, point, 12:00, 0m
        ⟵ START  :active, start, 12:00, 59m
        END ⟶   :active, end, 11:01, 59m

**Why it matters:**

- Defines the **semantics of your values**: is it an instantaneous measurement, or does it cover a span of time?
- Makes **aggregation and resampling unambiguous**: without a clear anchor, summing or averaging over periods may
  double-count or misalign.
- Ensures **alignment between datasets**: two series with the same resolution but different anchors
  (e.g. START vs END) are considered to be different.

Flagging system
===============

**What it is:** attaches status codes to data values without expanding the DataFrame schema with many new columns.
Each flag is a named bit in an integer mask, meaning that multiple flags can be combined in one column.

- A ``flag system`` defines the available flags.
- A ``flag column`` accompanies a data column, is governed by a *flag system*, and stores the combined bitmask.
- Flags propagate alongside the data so downstream processes can test them quickly.

.. mermaid::

   flowchart LR
     %% Inputs
     subgraph Data["Data column"]
       D["flow"]
     end

     subgraph Checks["QC checks"]
       C1["Range check<br/>"]
       C2["Spike check<br/>"]
       C3["Error code check<br/>"]
     end

     subgraph MapBits["Map to flags (enum)"]
       M1["RANGE = 1"]
       M2["SPIKE = 2"]
       M3["ERROR_CODE = 4"]
     end

     subgraph Combine["Bitwise combine"]
       OR["bitwise OR ( | )"]
     end

     subgraph Flags["Flag column"]
       F["flow_flag = 3"]
     end

     %% Edges
     D --> C1 --> M1 --> OR
     D --> C2 --> M2 --> OR
     D --> C3

     OR --> F

     %% Decode path (optional)
     F -- decode --> MapBits

     %% Style definitions
     classDef pass fill:#d1fad1,stroke:#237804,color:#000,stroke-width:1px;
     classDef fail fill:#ffe2e2,stroke:#a8071a,color:#000,stroke-width:1px;

     %% Assign colors
     class C1 fail
     class C2 fail
     class C3 pass

**Why it matters:**

- Many flags can be packed into one integer column allowing for **compact storage**.
- Enables **traceability** in your data - you can see which values were infilled, estimated, or failed QC.
- Provides **consistency**, as the same flag system can be reused across datasets and projects.

Aggregation
===========

**What it is:** combine data in a time series from a finer resolution to a coarser one
by summarising values within defined periods (daily, monthly, yearly, etc.).

- Define your aggregation periods using a new **periodicity**.
- Each aggregated value is placed using a **time anchor**.
- Common functions include **sum**, **mean**, **min**, **max**.

**Why it matters:**

- Works with time properties to ensure valid and **consistent** aggregation.
- Supports **domain relevant** aggregation, e.g. hydrological "water day" or "water year" conventions.
- Do aggregation in **one-line**, rather than rolling your own solution.

See more details and examples here: :doc:`aggregation user guide </user_guide/aggregation>`

Infilling
=========

**What it is:** the process of filling missing values in a :class:`~time_stream.TimeFrame` using a defined method.
It ensures continuity of the time axis and can be combined with flagging to make it clear which values are original
and which are estimated.

**Why it matters:**

- Works with time properties of your data to ensure **continuity** in your time series.
- Combining with a flagging system provides **transparency** over which points are original vs infilled.
- Fullly **configurable** to your dataset - set specific time periods to infill between, or minimum number
  of points required for a valid infill.

See more details and examples here: :doc:`infilling user guide </user_guide/infilling>`

Quality control (QC)
====================

**What it is:** a flexible module for defining automated QC checks that assess whether values
in a :class:`~time_stream.TimeFrame` look reasonable. Each check produces a boolean mask (pass/fail) that can be
converted into bitwise flags for permanent storage.

**Why it matters**

- **Configurable validation** allowing you to check your values with data-specific rules.
- The results of QC checks feed into **flagging system** and be stored compactly in bitwise flags.
- Enables you to trace which values failed which checks, **supporting provenance**.

See more details and examples here: :doc:`QC user guide </user_guide/quality_control>`