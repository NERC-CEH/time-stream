from datetime import datetime, timedelta

import polars as pl

import time_stream as ts


def _temperature_tf() -> ts.TimeFrame:
    """Ten days of daily temperature readings (one null) - the fixture every flagging example builds on."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperature = [20.5, 21.0, None, 26.0, 24.2, 26.6, 28.4, 30.9, 31.0, 29.1]
    df = pl.DataFrame({"time": dates, "temperature": temperature})
    return ts.TimeFrame(df=df, time_name="time")


def simple_example() -> None:
    """End-to-end: register a categorical flag system, make a flag column, flag the warm rows."""
    tf = _temperature_tf()
    # [start:simple_example]
    core_flags = ["UNCHECKED", "MISSING", "SUSPECT", "CORRECTED"]

    # Register the flag system into the TimeFrame
    tf.register_flag_system("CORE_FLAGS", core_flags, flag_type="categorical")

    # Initialise a new flag column tied to the CORE_FLAGS system
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")

    # Flag rows where the temperature exceeds 25 as SUSPECT
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    print(tf.df)
    # [end:simple_example]


def register_default() -> None:
    """Register a flag system with no definition - you get a single bitwise ``FLAGGED`` = 1."""
    tf = _temperature_tf()
    # [start:register_default]
    # Default bitwise system - a single FLAGGED flag at value 1
    tf.register_flag_system("DEFAULT")
    print(tf.get_flag_system("DEFAULT"))
    # [end:register_default]


def register_list() -> None:
    """Register from a list of names; Time-Stream assigns the powers of two."""
    tf = _temperature_tf()
    # fmt: off
    # [start:register_list]
    # Pass a list of names and let Time-Stream assign powers of two
    tf.register_flag_system(
        "QC_FLAGS", ["OUT_OF_RANGE", "SPIKE", "FLATLINE", "ERROR_CODE"]
    )
    print(tf.get_flag_system("QC_FLAGS"))
    # [end:register_list]
    # fmt: on


def register_bitwise_dict() -> None:
    """Register a bitwise system from an explicit ``name -> value`` mapping."""
    tf = _temperature_tf()
    # [start:register_bitwise_dict]
    core_flags = {
        "UNCHECKED": 1,
        "MISSING": 2,
        "SUSPECT": 4,
        "CORRECTED": 8,
        "REMOVED": 16,
        "INFILLED": 32,
    }
    tf.register_flag_system("CORE_FLAGS", core_flags)
    print(tf.get_flag_system("CORE_FLAGS"))
    # [end:register_bitwise_dict]


def register_categorical_single() -> None:
    """Register a categorical system - each row carries exactly one of the values."""
    tf = _temperature_tf()
    # [start:register_categorical_single]
    # A categorical single system - each row holds exactly one value
    qc = {"good": 0, "questionable": 1, "bad": 2}
    tf.register_flag_system("QC", qc, flag_type="categorical")
    print(tf.get_flag_system("QC"))
    # [end:register_categorical_single]


def register_categorical_string() -> None:
    """String flag values imply a categorical system automatically."""
    tf = _temperature_tf()
    # [start:register_categorical_string]
    # String values imply categorical automatically
    codes = {"good": "G", "questionable": "Q", "bad": "B"}
    tf.register_flag_system("CODES", codes)
    print(tf.get_flag_system("CODES"))
    # [end:register_categorical_string]


def register_categorical_name_list() -> None:
    """Register a categorical system from a bare list of names (name is also the value)."""
    tf = _temperature_tf()
    # fmt: off
    # [start:register_categorical_name_list]
    # A list of names, each used as both the key and the value
    tf.register_flag_system(
        "QC", ["good", "questionable", "bad"], flag_type="categorical"
    )
    print(tf.get_flag_system("QC"))
    # [end:register_categorical_name_list]
    # fmt: on


def register_categorical_list() -> None:
    """Register a ``categorical_list`` system - each row holds a list of values."""
    tf = _temperature_tf()

    # A categorical list system - each row holds a list of values
    tf.register_flag_system(
        "ORIGIN",
        ["API", "USER_INPUT", "MODEL_OUTPUT", "DERIVED"],
        flag_type="categorical_list",
    )
    print(tf.get_flag_system("ORIGIN"))


def init_flag_column_default_name() -> None:
    """Create a flag column without naming it - the name is derived from the system."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})
    # [start:init_flag_column_default_name]
    tf.init_flag_column("CORE_FLAGS")
    # [end:init_flag_column_default_name]
    print(tf.flag_columns)


def init_flag_column_prepopulated() -> None:
    """Create a flag column with every row pre-set to a starting flag."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})
    # [start:init_flag_column_prepopulated]
    # Pre-populate every row with the UNCHECKED flag (value 1)
    tf.init_flag_column("CORE_FLAGS", column_name="temperature_flags", data=1)
    # [end:init_flag_column_prepopulated]
    print(tf.df)


def bitwise_flag_workflow() -> None:
    """Bitwise workflow - independent flags OR together on each row."""
    tf = _temperature_tf()
    # [start:bitwise_flag_workflow]
    tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "MISSING": 2, "SUSPECT": 4})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")

    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    tf.add_flag("temperature_flags", "UNCHECKED", pl.col("temperature") < 25)
    # [end:bitwise_flag_workflow]
    print(tf.df)


def categorical_single_workflow() -> None:
    """Categorical-single workflow - each ``add_flag`` replaces the row's previous value."""
    tf = _temperature_tf()
    # [start:categorical_single_workflow]
    tf.register_flag_system("QC", {"good": 0, "questionable": 1, "bad": 2}, flag_type="categorical")
    tf.init_flag_column("QC", "temperature_qc")

    # Each row carries exactly one value; later add_flag calls replace the previous value
    tf.add_flag("temperature_qc", "good")
    tf.add_flag("temperature_qc", "questionable", pl.col("temperature") > 25)
    tf.add_flag("temperature_qc", "bad", pl.col("temperature") > 30)
    # [end:categorical_single_workflow]
    print(tf.df)


def categorical_single_overwrite() -> None:
    """``overwrite=False`` leaves rows that already carry a value untouched."""
    tf = _temperature_tf()
    tf.register_flag_system("QC", {"good": 0, "bad": 2}, flag_type="categorical")
    tf.init_flag_column("QC", "temperature_qc")
    tf.add_flag("temperature_qc", "bad", pl.col("temperature") > 25)
    # [start:categorical_single_overwrite]
    # overwrite=False leaves rows that already have a value untouched
    tf.add_flag("temperature_qc", "good", overwrite=False)
    # [end:categorical_single_overwrite]
    print(tf.df)


def categorical_list_workflow() -> None:
    """Categorical-list workflow - ``add_flag`` appends, so flags coexist on a row."""
    tf = _temperature_tf()
    # [start:categorical_list_workflow]
    tf.register_flag_system(
        "ORIGIN",
        ["API", "USER_INPUT", "MODEL_OUTPUT"],
        flag_type="categorical_list",
    )
    tf.init_flag_column("ORIGIN", "temperature_origin")

    # Append values to each row's list - flags coexist
    tf.add_flag("temperature_origin", "API")
    tf.add_flag("temperature_origin", "USER_INPUT", pl.col("temperature") > 25)
    # [end:categorical_list_workflow]
    print(tf.df)


def decode_bitwise() -> None:
    """Turn a raw integer flag column into human-readable flag names."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4, "CORRECTED": 8})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")
    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    tf.add_flag(
        "temperature_flags",
        "CORRECTED",
        (pl.col(tf.time_name) < datetime(2023, 1, 5)) & pl.col("temperature").is_not_null(),
    )
    # [start:decode_bitwise]
    # Replace the raw integer flag column with human-readable flag names
    tf_decoded = tf.decode_flag_column("temperature_flags")
    # [end:decode_bitwise]
    print(tf_decoded.df)


def encode_bitwise() -> None:
    """Round-trip a decoded flag column back to raw integers."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4, "CORRECTED": 8})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")
    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    tf_decoded = tf.decode_flag_column("temperature_flags")
    # [start:encode_bitwise]
    # Round-trip a decoded flag column back to raw integers
    tf_encoded = tf_decoded.encode_flag_column("temperature_flags")
    # [end:encode_bitwise]
    print(tf_encoded.df)


def filter_by_flag_include() -> None:
    """Keep only the rows carrying a given flag."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")
    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    # [start:filter_by_flag_include]
    # Keep only rows flagged as SUSPECT
    tf_suspect = tf.filter_by_flag("temperature_flags", "SUSPECT")
    # [end:filter_by_flag_include]
    print(tf_suspect.df)


def filter_by_flag_exclude() -> None:
    """Drop the rows carrying any of the given flags."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")
    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    # [start:filter_by_flag_exclude]
    # Drop rows flagged as MISSING or SUSPECT
    tf_clean = tf.filter_by_flag("temperature_flags", ["MISSING", "SUSPECT"], include=False)
    # [end:filter_by_flag_exclude]
    print(tf_clean.df)


def inspect_flag_columns() -> None:
    """Read back a TimeFrame's flag columns and each column's name and system."""
    tf = _temperature_tf()
    tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")

    print(tf.flag_columns)
    flag_col = tf.get_flag_column("temperature_flags")
    print(flag_col.name)
    print(flag_col.flag_system)


def with_flag_system_example() -> None:
    """``with_flag_system`` - the immutable variant that returns a new TimeFrame."""
    tf = _temperature_tf()

    # Immutable variant - returns a new TimeFrame
    tf_with_flags = tf.with_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})
    print(tf_with_flags.flag_systems)
