from datetime import datetime, timedelta

import polars as pl

import time_stream as ts
from time_stream.examples.utils import suppress_output


def _temperature_tf() -> ts.TimeFrame:
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperature = [20.5, 21.0, None, 26.0, 24.2, 26.6, 28.4, 30.9, 31.0, 29.1]
    df = pl.DataFrame({"time": dates, "temperature": temperature})
    return ts.TimeFrame(df=df, time_name="time")


def simple_example() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_1]
    core_flags = ["UNCHECKED", "MISSING", "SUSPECT", "CORRECTED"]

    # Register the flag system into the TimeFrame
    tf.register_flag_system("CORE_FLAGS", core_flags, flag_type="categorical")

    # Initialise a new flag column tied to the CORE_FLAGS system
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")

    # Flag rows where the temperature exceeds 25 as SUSPECT
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    # [end_block_1]
    print(tf.df)


def register_default() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_3]
    # Default bitwise system - a single FLAGGED flag at value 1
    tf.register_flag_system("DEFAULT")
    # [end_block_3]
    print(tf.get_flag_system("DEFAULT"))


def register_list() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_4]
    # Pass a list of names and let Time-Stream assign powers of two
    tf.register_flag_system("QC_FLAGS", ["OUT_OF_RANGE", "SPIKE", "FLATLINE", "ERROR_CODE"])
    # [end_block_4]
    print(tf.get_flag_system("QC_FLAGS"))


def register_bitwise_dict() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_5]
    core_flags = {
        "UNCHECKED": 1,
        "MISSING": 2,
        "SUSPECT": 4,
        "CORRECTED": 8,
        "REMOVED": 16,
        "INFILLED": 32,
    }
    tf.register_flag_system("CORE_FLAGS", core_flags)
    # [end_block_5]
    print(tf.get_flag_system("CORE_FLAGS"))


def register_categorical_single() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_6]
    # A categorical single system - each row holds exactly one value
    qc = {"good": 0, "questionable": 1, "bad": 2}
    tf.register_flag_system("QC", qc, flag_type="categorical")
    # [end_block_6]
    print(tf.get_flag_system("QC"))


def register_categorical_string() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_7]
    # String values imply categorical automatically
    codes = {"good": "G", "questionable": "Q", "bad": "B"}
    tf.register_flag_system("CODES", codes)
    # [end_block_7]
    print(tf.get_flag_system("CODES"))


def register_categorical_name_list() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_21]
    # A list of names, each used as both the key and the value
    tf.register_flag_system("QC", ["good", "questionable", "bad"], flag_type="categorical")
    # [end_block_21]
    print(tf.get_flag_system("QC"))


def register_categorical_list() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_8]
    # A categorical list system - each row holds a list of values
    tf.register_flag_system(
        "ORIGIN",
        ["API", "USER_INPUT", "MODEL_OUTPUT", "DERIVED"],
        flag_type="categorical_list",
    )
    # [end_block_8]
    print(tf.get_flag_system("ORIGIN"))


def init_flag_column_default_name() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})

    # [start_block_9]
    tf.init_flag_column("CORE_FLAGS")
    # [end_block_9]
    print(tf.flag_columns)


def init_flag_column_prepopulated() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})

    # [start_block_10]
    # Pre-populate every row with the UNCHECKED flag (value 1)
    tf.init_flag_column("CORE_FLAGS", column_name="temperature_flags", data=1)
    # [end_block_10]
    print(tf.df)


def bitwise_flag_workflow() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_11]
    tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "MISSING": 2, "SUSPECT": 4})
    tf.init_flag_column("CORE_FLAGS", "temperature_flags")

    tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
    tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
    tf.add_flag("temperature_flags", "UNCHECKED", pl.col("temperature") < 25)
    # [end_block_11]
    print(tf.df)


def categorical_single_workflow() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_12]
    tf.register_flag_system("QC", {"good": 0, "questionable": 1, "bad": 2}, flag_type="categorical")
    tf.init_flag_column("QC", "temperature_qc")

    # Each row carries exactly one value; later add_flag calls replace the previous value
    tf.add_flag("temperature_qc", "good")
    tf.add_flag("temperature_qc", "questionable", pl.col("temperature") > 25)
    tf.add_flag("temperature_qc", "bad", pl.col("temperature") > 30)
    # [end_block_12]
    print(tf.df)


def categorical_single_overwrite() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("QC", {"good": 0, "bad": 2}, flag_type="categorical")
        tf.init_flag_column("QC", "temperature_qc")
        tf.add_flag("temperature_qc", "bad", pl.col("temperature") > 25)

    # [start_block_13]
    # overwrite=False leaves rows that already have a value untouched
    tf.add_flag("temperature_qc", "good", overwrite=False)
    # [end_block_13]
    print(tf.df)


def categorical_list_workflow() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_14]
    tf.register_flag_system(
        "ORIGIN",
        ["API", "USER_INPUT", "MODEL_OUTPUT"],
        flag_type="categorical_list",
    )
    tf.init_flag_column("ORIGIN", "temperature_origin")

    # Append values to each row's list - flags coexist
    tf.add_flag("temperature_origin", "API")
    tf.add_flag("temperature_origin", "USER_INPUT", pl.col("temperature") > 25)
    # [end_block_14]
    print(tf.df)


def decode_bitwise() -> None:
    with suppress_output():
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

    # [start_block_15]
    # Replace the raw integer flag column with human-readable flag names
    tf_decoded = tf.decode_flag_column("temperature_flags")
    # [end_block_15]
    print(tf_decoded.df)


def encode_bitwise() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4, "CORRECTED": 8})
        tf.init_flag_column("CORE_FLAGS", "temperature_flags")
        tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
        tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)
        tf_decoded = tf.decode_flag_column("temperature_flags")

    # [start_block_16]
    # Round-trip a decoded flag column back to raw integers
    tf_encoded = tf_decoded.encode_flag_column("temperature_flags")
    # [end_block_16]
    print(tf_encoded.df)


def filter_by_flag_include() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4})
        tf.init_flag_column("CORE_FLAGS", "temperature_flags")
        tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
        tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)

    # [start_block_17]
    # Keep only rows flagged as SUSPECT
    tf_suspect = tf.filter_by_flag("temperature_flags", "SUSPECT")
    # [end_block_17]
    print(tf_suspect.df)


def filter_by_flag_exclude() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("CORE_FLAGS", {"MISSING": 2, "SUSPECT": 4})
        tf.init_flag_column("CORE_FLAGS", "temperature_flags")
        tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())
        tf.add_flag("temperature_flags", "SUSPECT", pl.col("temperature") > 25)

    # [start_block_18]
    # Drop rows flagged as MISSING or SUSPECT
    tf_clean = tf.filter_by_flag("temperature_flags", ["MISSING", "SUSPECT"], include=False)
    # [end_block_18]
    print(tf_clean.df)


def inspect_flag_columns() -> None:
    with suppress_output():
        tf = _temperature_tf()
        tf.register_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})
        tf.init_flag_column("CORE_FLAGS", "temperature_flags")

    # [start_block_19]
    print(tf.flag_columns)
    flag_col = tf.get_flag_column("temperature_flags")
    print(flag_col.name)
    print(flag_col.flag_system)
    # [end_block_19]


def with_flag_system_example() -> None:
    with suppress_output():
        tf = _temperature_tf()

    # [start_block_20]
    # Immutable variant - returns a new TimeFrame
    tf_with_flags = tf.with_flag_system("CORE_FLAGS", {"UNCHECKED": 1, "SUSPECT": 4})
    # [end_block_20]
    print(tf_with_flags.flag_systems)
