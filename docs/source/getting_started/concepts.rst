.. _concepts:

========
Concepts
========

.. rst-class:: lead

    Understand the essentials of **Time-Stream**.

This page explains some of the essentials you need before diving deeper.  It covers
how time is modelled (resolution, periodicity, alignment, anchors), how metadata and data fit together,
and what operations can be carried out on your :class:`~time_stream.TimeFrame`.

Time Properties
===============

Resolution
----------

**What it is:** defines the precision of the time values in your time series.

**Examples:**

- ``P1S`` - Resolution of 1 second (no sub-second precision).
- ``PT1M`` - Resolution of 1 minute (seconds must be 0).
- ``P1D`` - Resolution of 1 day (time must be 00:00:00).
- Complex offsets are allowed, e.g. ``P1Y+9M9H`` means only **09:00 on 1st October** each year.

**Why it matters:**

- Ensures **valid, aligned timestamps** - catches accidental drift (e.g. "00:14" instead of "00:15").
- Makes **comparisons deterministic** - data from different sources line up only if they share a resolution.
- Defines the **granularity of analysis** - a 15-minute resolution series has more detail than a daily one,
  and operations like padding or infilling often depend on knowing the intended resolution.

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
- Provides the **expected sampling frequency** - ensures a 15-minute series really has 96 points/day.
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

Quality control (QC)
====================

**What it is:** a flexible module for defining automated QC checks that assess whether values
in a :class:`~time_stream.TimeFrame` look reasonable. Each check produces a boolean mask (pass/fail) that can be
converted into bitwise flags for permanent storage.

**Why it matters**

- **Configurable validation** allowing you to check your values with data-specific rules.
- The results of QC checks feed into **flagging system** and be stored compactly in bitwise flags.
- Enables you to trace which values failed which checks, **supporting provenance**.
