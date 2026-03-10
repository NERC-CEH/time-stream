import time_stream as ts
from time_stream.examples.utils import get_example_df, suppress_output


def rolling_mean_example() -> None:
    with suppress_output():
        df = get_example_df(library="polars")

    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

    # [start_block_1]
    tf_rolling = tf.rolling_aggregate("PT3H", "mean", "flow")
    # [end_block_1]

    print(tf_rolling.df)


def rolling_missing_criteria_example() -> None:
    with suppress_output():
        df = get_example_df(library="polars")

    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

    # [start_block_2]
    tf_rolling = tf.rolling_aggregate(
        "PT3H",
        "mean",
        "flow",
        missing_criteria=("available", 3),
    )
    # [end_block_2]

    print(tf_rolling.df)
