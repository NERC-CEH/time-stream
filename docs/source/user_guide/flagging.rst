.. _flagging:

========
Flagging
========

.. rst-class:: lead

    Define and apply flags to your data

Why use Time-Stream?
====================

Flagging is a common task in environmental data processing, used to mark data that meet certain conditions
(e.g. sensor errors, or extreme values). **Time-Stream** provides a flexible and consistent
framework for defining and applying flags to your data.

Simple example
--------------

Define your flags, link them to a column, and apply with a single call:

.. code-block:: python

    core_flags = {
        "UNCHECKED": 1,
        "MISSING": 2,
        "SUSPECT": 4,
        "CORRECTED": 8,
    }

    # Register flag system into TimeFrame object
    tf.register_flag_system(
        name="CORE_FLAGS",
        flag_system=core_flags
    )
    # Initialise a new flag column, linking it to the 'temperature' column
    tf.init_flag_column(
        base="temperature",
        flag_system="CORE_FLAGS",
        column_name="temperature_flags"
    )
    # Add some flags
    tf.add_flag(
        flag_column_name="temperature_flags",
        flag_value="SUSPECT",
        expr=pl.col("temperature") > 25
    )

A few lines to enrich the data: "I want to *flag* my *temperature* data as *SUSPECT* when values are *greater than 25*"

Complex example
---------------

In practice, you may want to apply several flags to the same column, covering different conditions. For example,
flagging missing values, suspect readings, and corrected data all in one workflow:

.. code-block:: python

    # Show the original data
    print(tf.df)
    """"
    ┌─────────────────────┬─────────────┐
    │ time                ┆ temperature │
    │ ---                 ┆ ---         │
    │ datetime[μs]        ┆ f64         │
    ╞═════════════════════╪═════════════╡
    │ 2023-01-01 00:00:00 ┆ 20.5        │
    │ 2023-01-02 00:00:00 ┆ 21.0        │
    │ 2023-01-03 00:00:00 ┆ null        │
    │ 2023-01-04 00:00:00 ┆ 26.0        │
    │ 2023-01-05 00:00:00 ┆ 24.2        │
    │ 2023-01-06 00:00:00 ┆ 26.6        │
    │ 2023-01-07 00:00:00 ┆ 28.4        │
    │ 2023-01-08 00:00:00 ┆ 30.9        │
    │ 2023-01-09 00:00:00 ┆ 31.0        │
    │ 2023-01-10 00:00:00 ┆ 29.1        │
    └─────────────────────┴─────────────┘
    """

    # Define flag system
    core_flags = {
        "UNCHECKED": 1,
        "MISSING": 2,
        "SUSPECT": 4,
        "CORRECTED": 8,
    }

    # Register flag system and initialize flag column
    tf.register_flag_system("CORE_FLAGS", core_flags)
    tf.init_flag_column("temperature", "CORE_FLAGS", "temperature_flags")
    print(tf.df)
    """
    ┌─────────────────────┬─────────────┬───────────────────┐
    │ time                ┆ temperature ┆ temperature_flags │
    │ ---                 ┆ ---         ┆ ---               │
    │ datetime[μs]        ┆ f64         ┆ i64               │
    ╞═════════════════════╪═════════════╪═══════════════════╡
    │ 2023-01-01 00:00:00 ┆ 20.5        ┆ 0                 │
    │ 2023-01-02 00:00:00 ┆ 21.0        ┆ 0                 │
    │ 2023-01-03 00:00:00 ┆ null        ┆ 0                 │
    │ 2023-01-04 00:00:00 ┆ 26.0        ┆ 0                 │
    │ 2023-01-05 00:00:00 ┆ 24.2        ┆ 0                 │
    │ 2023-01-06 00:00:00 ┆ 26.6        ┆ 0                 │
    │ 2023-01-07 00:00:00 ┆ 28.4        ┆ 0                 │
    │ 2023-01-08 00:00:00 ┆ 30.9        ┆ 0                 │
    │ 2023-01-09 00:00:00 ┆ 31.0        ┆ 0                 │
    │ 2023-01-10 00:00:00 ┆ 29.1        ┆ 0                 │
    └─────────────────────┴─────────────┴───────────────────┘
    """

    # Add flags based on conditions
    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    print(tf.df)
    """
    ┌─────────────────────┬─────────────┬───────────────────┐
    │ time                ┆ temperature ┆ temperature_flags │
    │ ---                 ┆ ---         ┆ ---               │
    │ datetime[μs]        ┆ f64         ┆ i64               │
    ╞═════════════════════╪═════════════╪═══════════════════╡
    │ 2023-01-01 00:00:00 ┆ 20.5        ┆ 0                 │
    │ 2023-01-02 00:00:00 ┆ 21.0        ┆ 0                 │
    │ 2023-01-03 00:00:00 ┆ null        ┆ 2                 │
    │ 2023-01-04 00:00:00 ┆ 26.0        ┆ 4                 │
    │ 2023-01-05 00:00:00 ┆ 24.2        ┆ 0                 │
    │ 2023-01-06 00:00:00 ┆ 26.6        ┆ 4                 │
    │ 2023-01-07 00:00:00 ┆ 28.4        ┆ 4                 │
    │ 2023-01-08 00:00:00 ┆ 30.9        ┆ 4                 │
    │ 2023-01-09 00:00:00 ┆ 31.0        ┆ 4                 │
    │ 2023-01-10 00:00:00 ┆ 29.1        ┆ 4                 │
    └─────────────────────┴─────────────┴───────────────────┘
    """

    # Let's say we correct data points before a particular date.
    correction_date = pl.datetime(2023, 1, 5)
    tf.add_flag(
        "temperature_flags",
        "CORRECTED",
        (pl.col(tf.time_name) < correction_date) & (pl.col("temperature").is_not_null())
    )
    print(tf.df)
    """
    ┌─────────────────────┬─────────────┬───────────────────┐
    │ time                ┆ temperature ┆ temperature_flags │
    │ ---                 ┆ ---         ┆ ---               │
    │ datetime[μs]        ┆ f64         ┆ i64               │
    ╞═════════════════════╪═════════════╪═══════════════════╡
    │ 2023-01-01 00:00:00 ┆ 19.5        ┆ 8                 │
    │ 2023-01-02 00:00:00 ┆ 20.5        ┆ 8                 │
    │ 2023-01-03 00:00:00 ┆ null        ┆ 2                 │ 
    │ 2023-01-04 00:00:00 ┆ 25.5        ┆ 12                │ <-- 4 (SUSPECT) + 8 (CORRECTED) = 12
    │ 2023-01-05 00:00:00 ┆ 24.2        ┆ 0                 │
    │ 2023-01-06 00:00:00 ┆ 26.6        ┆ 4                 │
    │ 2023-01-07 00:00:00 ┆ 28.4        ┆ 4                 │
    │ 2023-01-08 00:00:00 ┆ 30.9        ┆ 4                 │
    │ 2023-01-09 00:00:00 ┆ 31.0        ┆ 4                 │
    │ 2023-01-10 00:00:00 ┆ 29.1        ┆ 4                 │
    └─────────────────────┴─────────────┴───────────────────┘
    """

    # Now let's remove the SUSPECT flag where data has been corrected
    tf.remove_flag("temperature_flags", "SUSPECT", pl.col(tf.time_name) < correction_date)
    print(tf.df)
    """
    ┌─────────────────────┬─────────────┬───────────────────┐
    │ time                ┆ temperature ┆ temperature_flags │
    │ ---                 ┆ ---         ┆ ---               │
    │ datetime[μs]        ┆ f64         ┆ i64               │
    ╞═════════════════════╪═════════════╪═══════════════════╡
    │ 2023-01-01 00:00:00 ┆ 19.5        ┆ 8                 │
    │ 2023-01-02 00:00:00 ┆ 20.5        ┆ 8                 │
    │ 2023-01-03 00:00:00 ┆ null        ┆ 2                 │
    │ 2023-01-04 00:00:00 ┆ 25.5        ┆ 8                 │
    │ 2023-01-05 00:00:00 ┆ 24.2        ┆ 0                 │
    │ 2023-01-06 00:00:00 ┆ 26.6        ┆ 4                 │
    │ 2023-01-07 00:00:00 ┆ 28.4        ┆ 4                 │
    │ 2023-01-08 00:00:00 ┆ 30.9        ┆ 4                 │
    │ 2023-01-09 00:00:00 ┆ 31.0        ┆ 4                 │
    │ 2023-01-10 00:00:00 ┆ 29.1        ┆ 4                 │
    └─────────────────────┴─────────────┴───────────────────┘
    """"


Because flags use `Bitwise values`_, a single data point can carry multiple flags simultaneously — for example,
we saw the value that had both ``SUSPECT`` and ``CORRECTED`` — without any information being lost.

Key benefits
------------

- **You stay in control**
  Flag systems are defined by you.

- **Scalable**
  Multiple flag systems can exist in the same TimeFrame.

- **Safe**
  All flags use *bitwise values*, meaning multiple flags can be combined without losing information.

- **Reversible**
  Flags can be removed as easily as they are added, using the same expression-based interface.


In more detail
==============

Flagging in **Time-Stream** is built around the concept of `Flag systems`_ and `Flag columns`_.


Flag systems
------------

A flag system is a user-defined set of named flags, each associated with unique `Bitwise values`_.
Typically the flag system is defined as a dictionary mapping flag names to their bitwise values.
To initialise a flag system in a **Time-Stream** object, use :meth:`~time_stream.TimeFrame.register_flag_system`:

.. code-block:: python

    # Flags for general status of data
    core_flags = {
        "UNCHECKED": 1,
        "MISSING": 2,
        "SUSPECT": 4,
        "CORRECTED": 8,
        "REMOVED": 16,
        "INFILLED": 32,
    }

    tf.register_flag_system(name="CORE_FLAGS", flag_system=core_flags)

As flags can take many forms, **Time-Stream** allows you to register multiple flag systems.

.. code-block:: python

    # Flags for QC results
    QC_flags = {
        "OUT_OF_RANGE": 1,
        "SPIKE": 2,
        "FLATLINE": 4,
        "ERROR_CODE": 8,
    }

    # Flags describing origin of data
    provenance_flags = {
        "API": 1,
        "USER_INPUT": 2,
        "MODEL_OUTPUT": 4,
        "DERIVED": 8,
    }

    tf.register_flag_system(name="QC_FLAGS", flag_system=QC_flags)
    tf.register_flag_system(name="PROVENANCE_FLAGS", flag_system=provenance_flags)

.. note::

    Each flag value within a flag system must be a power of two (1, 2, 4, 8, 16, …) and must be unique within
    that system. See `Bitwise values`_ for more information. **Time-Stream** will raise an error if you attempt
    to register a flag system with invalid values.


Flag columns
------------

A flag column is a column in the TimeFrame that is linked to a specific flag system, and tied to a specific data
column (the "base" column). The flag column stores the flag values for each data point, which can be a combination
of multiple flags from the linked flag system.

There are two ways to create a flag column:

**1. Initialise a new flag column** using :meth:`~time_stream.TimeFrame.init_flag_column`:

This creates a new integer column in the DataFrame, populated with zeros by default, and registers it as a flag
column linked to the given base column and flag system.

.. code-block:: python

    tf.init_flag_column(
        base="temperature",
        flag_system="CORE_FLAGS",
        column_name="temperature_flags"
    )
.. note::
    The ``column_name`` argument is optional. If omitted, the column will be named automatically as
    ``{base}__flag__{flag_system}`` (e.g. ``temperature__flag__CORE_FLAGS``).

You can also pre-populate the flag column with a default value:

.. code-block:: python

    # Initialise all rows with the UNCHECKED flag (value = 1)
    tf.init_flag_column(
        base="temperature",
        flag_system="CORE_FLAGS",
        column_name="temperature_flags",
        data=1
    )

**2. Register an existing column** using :meth:`~time_stream.TimeFrame.register_flag_column`:

If you already have an integer column in your DataFrame that contains flag values, you can register it as a flag
column without modifying the data:

.. code-block:: python

    tf.register_flag_column(column_name="temperature_flags", base="temperature", flag_system="CORE_FLAGS")


Bitwise values
--------------

Bitwise values are a set of integers where each value is a power of 2 (1, 2, 4, 8, 16, etc.).
The key feature of bitwise values is that any combination of them results in a unique value.
For example, values 1 + 4 = 5. The value 5 cannot be produced by any other combination of bitwise numbers.
To give another example, the value 13 can only be produced by bitwise values 1 + 4 + 8.

.. note::
    Bitwise values come from the binary number system, where each bit represents a power of 2.
    ============  ============
    Decimal       Binary
    ============  ============
    1             1
    2             10
    4             100
    8             1000
    16            10000
    ============  ============

    With this we can see why combinations of bitwise values are unique. For example, 5 in binary is 101, which
    corresponds to the combination of 1 (001) and 4 (100).

For **Time-Stream**, this feature allows any combination of flags from the flag system to be stored as a single
integer value, without losing information about which flags are present. For example, if a data point is flagged
as 48 in the ``CORE_FLAGS`` flag system, we can deduce it has been both ``REMOVED`` (16) and ``INFILLED`` (32).


Adding flags
------------

Use :meth:`~time_stream.TimeFrame.add_flag` to set a flag on rows that match a given condition. The flag is applied
using a bitwise OR operation, so existing flags on a row are preserved.

.. code-block:: python

    # Flag all rows as SUSPECT (4) where temperature exceeds 25
    tf.add_flag(
        flag_column_name="temperature_flags",
        flag_value="SUSPECT",
        expr=pl.col("temperature") > 25)

    # Flag all rows as MISSING (1) where temperature is null
    tf.add_flag(
        flag_column_name="temperature_flags",
        flag_value=1,  # Using integer value instead of name
        expr=pl.col("temperature").is_null()
    )

    # Flag all rows as CORRECTED (8) according to the given mask
    tf.add_flag(
        flag_column_name="temperature_flags",
        flag_value="CORRECTED",
        expr=pl.Series([True, False, True, False, True, False, True, False, True, False])
    )

    # Flag all rows (no condition) as UNCHECKED
    tf.add_flag(
        flag_column_name="temperature_flags",
        flag_value="UNCHECKED"
    )

The ``flag_value`` argument accepts either the flag name as a string (e.g. ``"SUSPECT"``) or its integer value
(e.g. ``4``). The ``expr`` argument is any valid Polars expression that returns a boolean Series, or a boolean
Series directly (which for example could be useful in conjunction with the boolean Series output from :doc:`quality_control`).
If omitted, the flag is applied to all rows.

.. note::

    Adding a flag that is already set on a row has no effect — the bitwise OR operation is idempotent.
    A row flagged as ``SUSPECT`` (4) that receives another ``SUSPECT`` flag remains at 4, not 8.


Removing flags
--------------

Use :meth:`~time_stream.TimeFrame.remove_flag` to clear a flag from rows that match a given condition. The flag is
removed using a bitwise AND NOT operation, so other flags on the same row are unaffected.

.. code-block:: python

    # Remove the SUSPECT flag from all rows where temperature has been corrected
    tf.remove_flag(
        column_name="temperature_flags",
        flag_value="SUSPECT",
        expr=pl.col("temperature") == 26.1
    )

    # Remove the UNCHECKED flag from all rows
    tf.remove_flag(
        column_name="temperature_flags",
        flag_value="UNCHECKED"
    )

As with :meth:`~time_stream.TimeFrame.add_flag`, the ``flag_value`` can be a string name or integer, and ``expr``
is optional (defaults to all rows).

.. note::

    Removing a flag that is not set on a row has no effect — the operation is safe to call even if the flag is
    absent.


Combining multiple flags
------------------------

Because flags are stored as bitwise integers, a single data point can carry multiple flags at once. For example,
a row with a flag value of ``5`` in a system where ``UNCHECKED = 1`` and ``SUSPECT = 4`` is flagged as both
``UNCHECKED`` and ``SUSPECT``.

.. code-block:: python

    # Apply multiple flags to the same column
    tf.add_flag("temperature_flags", "UNCHECKED")
    tf.add_flag("temperature_flags", "SUSPECT", expr=pl.col("temperature") > 25)

A row where temperature is unchecked AND > 25 would have flag value 1 + 4 = 5


Inspecting flag columns
-----------------------

You can retrieve the list of registered flag column names from the TimeFrame:

.. code-block:: python

    print(tf.flag_columns)
    # ['temperature_flags']

To retrieve the full :class:`~time_stream.flag_manager.FlagColumn` object (including its base column and flag
system), use :meth:`~time_stream.TimeFrame.get_flag_column`:

.. code-block:: python

    flag_col = tf.get_flag_column("temperature_flags")
    print(flag_col.name)         # 'temperature_flags'
    print(flag_col.base)         # 'temperature'
    print(flag_col.flag_system)  # <CORE_FLAGS (UNCHECKED=1, MISSING=2, SUSPECT=4, CORRECTED=8, ...)>

To retrieve a registered flag system by name, use :meth:`~time_stream.TimeFrame.get_flag_system`:

.. code-block:: python

    flag_system = tf.get_flag_system("CORE_FLAGS")
    print(flag_system)
    # <CORE_FLAGS (UNCHECKED=1, MISSING=2, SUSPECT=4, CORRECTED=8, REMOVED=16, INFILLED=32)>


Immutable variant
-----------------

If you prefer a functional, immutable style, use :meth:`~time_stream.TimeFrame.with_flag_system` to register a
flag system and return a new TimeFrame rather than modifying in place:

.. code-block:: python

    tf_with_flags = tf.with_flag_system(name="CORE_FLAGS", flag_system=core_flags)

.. note::

    :meth:`~time_stream.TimeFrame.with_flag_system` returns a **new** TimeFrame with the flag system registered.
    The original ``tf`` is unchanged. In contrast, :meth:`~time_stream.TimeFrame.register_flag_system`,
    :meth:`~time_stream.TimeFrame.init_flag_column`, :meth:`~time_stream.TimeFrame.add_flag`, and
    :meth:`~time_stream.TimeFrame.remove_flag` all mutate the TimeFrame in place.


Selecting columns with flags
----------------------------

When using :meth:`~time_stream.TimeFrame.select` to reduce a TimeFrame to a subset of columns, any flag columns
whose base column is included will be carried over automatically:

.. code-block:: python

    # Select only temperature — its flag column is included automatically
    tf_temp = tf.select("temperature")
    print(tf_temp.columns)
    # ['temperature', 'temperature_flags']

To exclude flag columns from the selection, pass ``include_flag_columns=False``:

.. code-block:: python

    tf_temp = tf.select("temperature", include_flag_columns=False)
    print(tf_temp.columns)
    # ['temperature']


API reference
=============

.. autosummary::

   ~time_stream.TimeFrame.register_flag_system
   ~time_stream.TimeFrame.with_flag_system
   ~time_stream.TimeFrame.get_flag_system
   ~time_stream.TimeFrame.init_flag_column
   ~time_stream.TimeFrame.register_flag_column
   ~time_stream.TimeFrame.get_flag_column
   ~time_stream.TimeFrame.add_flag
   ~time_stream.TimeFrame.remove_flag
   ~time_stream.TimeFrame.flag_columns
