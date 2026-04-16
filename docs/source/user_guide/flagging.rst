.. _flagging:

========
Flagging
========

.. rst-class:: lead

    The good, the bad, and the ugly - flexible, reversible flag annotations for every data point.

Why use Time-Stream?
====================

Data values on their own rarely tells the whole story - readings go missing, sensors drift, values get corrected.
Data analysts often need to know those details. **Time-Stream** lets you record every one of those caveats inline with
the data itself, using flag systems **you define**, in a form compact enough to carry multiple annotations per row
without losing detail.

Simple example
--------------

Define your flags, link them to a column, and apply with a single call:

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_1]
   :end-before: [end_block_1]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.simple_example()

A few lines to enrich the data: "I want to *flag* my *temperature* data as *SUSPECT* when values are *greater than 25*."

Key benefits
------------

- **You define the vocabulary.**
  Flag names, values, and meanings are yours to choose.

- **Scalable.**
  Multiple flag systems can coexist on the same TimeFrame.

- **Flexible.**
  Choose *bitwise*, *categorical single*, or *categorical list* flag systems to match your data.

- **Reversible.**
  Flags can be removed as easily as they are added, using the same expression-based interface.

In more detail
==============

Flagging in **Time-Stream** is built around two concepts: `Flag systems`_ and `Flag columns`_.

Flag systems
------------

A flag system is a user-defined set of named flags. **Time-Stream** supports three flag system types, each suited to
a different annotation pattern. You pick the type via the ``flag_type`` argument of
:meth:`~time_stream.TimeFrame.register_flag_system`, with ``"bitwise"`` as the default. Multiple flag systems can
coexist on the same TimeFrame - for example a ``QC_FLAGS`` bitwise system alongside a ``PROVENANCE`` categorical list
system.

``bitwise``
^^^^^^^^^^^

    **What it does:** Stores flags as powers-of-two integers. Multiple flags combine on a single integer per row via
    bitwise OR, so any combination of flags can be recorded without losing detail. See `Bitwise values`_ for the maths.

    **When to use:** The default choice. Reach for this when a row may legitimately carry several compatible flags at
    once (e.g. ``MISSING`` + ``INFILLED``) and you want compact storage.

    **Accepted inputs to** ``register_flag_system``:

    - ``None`` - produces a default system with a single ``FLAGGED`` flag at value ``1``:

      .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
         :language: python
         :start-after: [start_block_3]
         :end-before: [end_block_3]
         :dedent:

      .. jupyter-execute::
         :hide-code:

         import examples_flagging
         examples_flagging.register_default()

    - ``list[str]`` - names are sorted and assigned powers of two automatically:

      .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
         :language: python
         :start-after: [start_block_4]
         :end-before: [end_block_4]
         :dedent:

      .. jupyter-execute::
         :hide-code:

         import examples_flagging
         examples_flagging.register_list()

    - ``dict[str, int]`` - explicit mapping. Values must be powers of two and unique:

      .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
         :language: python
         :start-after: [start_block_5]
         :end-before: [end_block_5]
         :dedent:

      .. jupyter-execute::
         :hide-code:

         import examples_flagging
         examples_flagging.register_bitwise_dict()

    .. note::

        Each value in a bitwise system must be a power of two (1, 2, 4, 8, 16, …) and unique within that system.
        **Time-Stream** raises an error if you try to register an invalid bitwise system.


``categorical``
^^^^^^^^^^^^^^^

    **What it does:** Each row holds exactly one value, or null. Values are arbitrary ``int`` or ``str`` - they do not
    need to be powers of two. Setting a new flag replaces the previous value on each matching row, so flags are
    mutually exclusive.

    **When to use:** For "one verdict per row" annotations - e.g. an overall QC rating of ``good``, ``questionable``
    or ``bad``, or a sensor status code column.

    **Accepted inputs to** ``register_flag_system``:

    - ``dict[str, int]`` with ``flag_type="categorical"`` - arbitrary integer values:

      .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
         :language: python
         :start-after: [start_block_6]
         :end-before: [end_block_6]
         :dedent:

      .. jupyter-execute::
         :hide-code:

         import examples_flagging
         examples_flagging.register_categorical_single()

    - ``dict[str, str]`` - string-valued dicts are inferred as categorical automatically, so ``flag_type`` can be
      omitted:

      .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
         :language: python
         :start-after: [start_block_7]
         :end-before: [end_block_7]
         :dedent:

      .. jupyter-execute::
         :hide-code:

         import examples_flagging
         examples_flagging.register_categorical_string()

    - ``list[str]`` with ``flag_type="categorical"`` - each name is used as both the key and the value:

      .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
         :language: python
         :start-after: [start_block_21]
         :end-before: [end_block_21]
         :dedent:

      .. jupyter-execute::
         :hide-code:

         import examples_flagging
         examples_flagging.register_categorical_name_list()


``categorical_list``
^^^^^^^^^^^^^^^^^^^^

    **What it does:** Defines the same vocabulary as ``categorical``, but applied to a flag column each row holds a
    *list* of values rather than a single one. Adding a flag appends to the list, so multiple flags can coexist on a
    row - stored as distinct entries rather than combined bits.

    **When to use:** When flags need to accumulate on a row but a bitwise encoding doesn't fit - for example a list of
    provenance tags or free-form error codes where values are not powers of two.

    **Accepted inputs to** ``register_flag_system``: The same forms as ``categorical`` above
    (``dict[str, int]``, ``dict[str, str]``, or ``list[str]``), but with ``flag_type="categorical_list"``. The only
    difference is when this flag system is applied to a flag column, a ``categorical_list`` system will allow
    multiple flags per row.

Flag columns
------------

A flag column is a column in the TimeFrame that has been registered against a specific flag system. They are where
flags are stored for a given row in your data. Flag columns are what :meth:`~time_stream.TimeFrame.add_flag`,
:meth:`~time_stream.TimeFrame.remove_flag`, :meth:`~time_stream.TimeFrame.filter_by_flag` methods operate on.

There are two ways to create a flag column:

**1. Initialise a new flag column** using :meth:`~time_stream.TimeFrame.init_flag_column`.

This creates a new column in the DataFrame, populated with a sensible default, and registers it against the given
flag system. The data type of the resulting column depends on the flag system type:

- Bitwise: ``Int64``, initialised to ``0``.
- Categorical single (int values): ``Int32``, initialised to ``null``.
- Categorical single (string values): ``Utf8``, initialised to ``null``.
- Categorical list: ``List`` of the underlying value type, initialised to an empty list.

You can also pre-populate the column with a default flag value:

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_10]
   :end-before: [end_block_10]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.init_flag_column_prepopulated()

.. note::
    Normally, you would supply a sensible ``column_name`` so that you can keep track of your flags. However, if you
    do omit it, the column is given a default name of ``__flag__{flag_system_name}``, with an integer suffix appended
    if a name collides.

    .. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
       :language: python
       :start-after: [start_block_9]
       :end-before: [end_block_9]
       :dedent:

    .. jupyter-execute::
       :hide-code:

       import examples_flagging
       examples_flagging.init_flag_column_default_name()

**2. Register an existing column** using :meth:`~time_stream.TimeFrame.register_flag_column`.

If you already have a column of the right data type containing valid flag values, you can register it without
modifying the data:

.. code-block:: python

    tf.register_flag_column("temperature_flags", "CORE_FLAGS")

**Time-Stream** validates that every non-null value in the column is a known flag value for the given flag system,
and raises an error if any are not recognised.

Adding flags
------------

Use :meth:`~time_stream.TimeFrame.add_flag` to set a flag on rows that match a given condition. The operation that
this method does depends on the flag system type:

- **Bitwise** - the flag is applied with a bitwise OR, so any existing flags on the row are preserved.
- **Categorical single** - the flag replaces the current value on each matching row. Pass ``overwrite=False`` to
  update only rows whose current value is null.
- **Categorical list** - the flag is appended to the list on each matching row. Duplicate appends are a no-op.

**Bitwise workflow:**

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_11]
   :end-before: [end_block_11]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.bitwise_flag_workflow()

**Categorical single workflow:**

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_12]
   :end-before: [end_block_12]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.categorical_single_workflow()

Each ``add_flag`` call overwrites the previous verdict on matching rows, so the order of calls matters. Use
``overwrite=False`` to fill in only the rows that have no verdict yet:

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_13]
   :end-before: [end_block_13]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.categorical_single_overwrite()

**Categorical list workflow:**

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_14]
   :end-before: [end_block_14]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.categorical_list_workflow()

The ``flag_value`` argument accepts either the flag name as a string or its underlying value (integer or string,
depending on the system). The ``expr`` argument is any valid Polars expression that returns a boolean Series, or a
boolean Series directly (which is useful in conjunction with the boolean Series output from :doc:`quality_control`).
If omitted, the flag is applied to all rows.


Removing flags
--------------

Use :meth:`~time_stream.TimeFrame.remove_flag` to clear a flag from rows that match a given condition.

- **Bitwise** - clears the bit with a bitwise AND NOT; other flags on the row are untouched.
- **Categorical single** - sets the column value to null on matching rows.
- **Categorical list** - removes all occurrences of the given value from the list on matching rows.

.. code-block:: python

    # Bitwise: clear SUSPECT from corrected rows
    tf.remove_flag("temperature_flags", "SUSPECT", pl.col(tf.time_name) < correction_date)

    # Categorical single: null out the QC verdict on a subset of rows
    tf.remove_flag("temperature_qc", "bad", pl.col("temperature") < 20)

    # Categorical list: drop a value from each matching row's list
    tf.remove_flag("temperature_origin", "USER_INPUT")

As with :meth:`~time_stream.TimeFrame.add_flag`, ``flag_value`` can be a name or underlying value, and ``expr`` is
optional (defaults to all rows).

.. note::

    Removing a flag that is not set on a row has no effect - the operation is safe to call even when the flag is
    absent.


Filtering by flag
-----------------

Use :meth:`~time_stream.TimeFrame.filter_by_flag` to return a new TimeFrame containing only the rows that carry
(or don't carry) given flags.

**Keep only matching rows:**

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_17]
   :end-before: [end_block_17]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.filter_by_flag_include()

**Drop matching rows** by passing ``include=False`` and a list of flags:

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_18]
   :end-before: [end_block_18]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.filter_by_flag_exclude()

For bitwise columns, "matching" means any of the requested flag bits are set. For categorical columns, it means the
row's value (or any element of the list, in list mode) is any of the requested flag values.


Decoding and encoding flag columns
----------------------------------

Raw flag columns are compact but not particularly human-readable. Use :meth:`~time_stream.TimeFrame.decode_flag_column`
to replace the raw values with their flag names:

- **Bitwise** - the integer column becomes a ``List(String)`` column of active flag names (sorted by ascending bit
  value). A value of ``0`` becomes an empty list.
- **Categorical single** - each raw value becomes its flag name.
- **Categorical list** - each value in each list becomes its flag name.

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_15]
   :end-before: [end_block_15]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.decode_bitwise()

Decoded columns remain registered as flag columns: :meth:`~time_stream.TimeFrame.add_flag`,
:meth:`~time_stream.TimeFrame.remove_flag`, and :meth:`~time_stream.TimeFrame.filter_by_flag` all continue to work
transparently on the decoded form.

Use :meth:`~time_stream.TimeFrame.encode_flag_column` to round-trip the column back to raw values:

.. literalinclude:: ../../../src/time_stream/examples/examples_flagging.py
   :language: python
   :start-after: [start_block_16]
   :end-before: [end_block_16]
   :dedent:

.. jupyter-execute::
   :hide-code:

   import examples_flagging
   examples_flagging.encode_bitwise()

Integration with ``TimeStream`` operations
==========================================

The real power of the flagging systems comes into play when integrated with **Time-Stream**'s operations, such as
:doc:`quality_control` and :doc:`infilling`. Both accept a ``flag_params`` tuple of ``(flag_column_name, flag_value)``
that tells the operation to record its outcome directly into an existing flag column, rather than returning a standalone
boolean series or leaving the infilled rows unmarked.

This lets you build a single, self-describing :class:`~time_stream.TimeFrame` where every QC verdict or infilled
value is traceable back to the check or method that produced it - no separate bookkeeping required.

**Quality control checks** write the given flag value onto every row that fails the check:

.. code-block:: python

    tf.register_flag_system("QC_FLAGS", ["OUT_OF_RANGE", "SPIKE"])
    tf.init_flag_column("QC_FLAGS", "temperature_qc")

    # Flag values outside a plausible range
    tf.qc_check(
        "range", "temperature", min_value=-30, max_value=50,
        flag_params=("temperature_qc", "OUT_OF_RANGE"),
    )

    # Flag spikes in the same column
    tf.qc_check(
        "spike", "temperature", threshold=10,
        flag_params=("temperature_qc", "SPIKE"),
    )

See :doc:`quality_control` for the full set of available checks and the boolean-series alternative.

**Infilling** marks rows that were filled in by the chosen method:

.. code-block:: python

    tf.register_flag_system("INFILL_FLAGS", ["INFILLED"])
    tf.init_flag_column("INFILL_FLAGS", "flow_flags")

    tf_infill = tf.infill(
        "linear", "flow", max_gap_size=3,
        flag_params=("flow_flags", "INFILLED"),
    )

See :doc:`infilling` for the full set of infill methods and options.

Because the flag column lives alongside the data, you can then use :meth:`~time_stream.TimeFrame.filter_by_flag` to
inspect or exclude affected rows, or decode the column for a human-readable audit trail.


Additional information
======================

Bitwise values
--------------

Bitwise values are a set of integers where each value is a power of 2 (1, 2, 4, 8, 16, etc.). The key feature of
bitwise values is that any combination of them results in a unique value. For example, ``1 + 4 = 5``. The value ``5``
cannot be produced by any other combination of bitwise numbers. Similarly, the value ``13`` can only be produced by
``1 + 4 + 8``.

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

With this we can see why combinations of bitwise values are unique. For example, ``5`` in binary is ``101``, which
corresponds to the combination of ``1`` (``001``) and ``4`` (``100``).

For **Time-Stream**, this feature allows any combination of flags from a bitwise flag system to be stored as a single
integer value, without losing information about which flags are present. For example, if a data point is flagged as
``48`` in the ``CORE_FLAGS`` flag system, we can deduce it has been both ``REMOVED`` (16) and ``INFILLED`` (32).

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
    ~time_stream.TimeFrame.filter_by_flag
    ~time_stream.TimeFrame.decode_flag_column
    ~time_stream.TimeFrame.encode_flag_column

