# Time Series Processor

## Developer Setup

### Requirements

#### Python 3.12
```commandline
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12
```
or
```commandline
brew install python@3.12
```

### Setting up and activating a virtual environment
```commandline
python -m venv .venv
source .venv/bin/activate
```

### Installing the App

```commandline
pip install -e '.[dev]'
```

## Running the app

Build the docker container to access local data

```commandline
docker compose up -d
```
Then call the app which can take two command-line arguments:

period (required): The period of time you want to extract data for.\
    - Must be a valid ISO8601 duration\
    - Must not have a time component\
    - Can be a combination of days, weeks, months and years

e.g. P1D: previous days data; P1M: previous months data; P1M14D: previous month + 14 days data; PT4: invalid

end_date (optional): The date to start extraction from.\
    - Must be of the format YYYY-MM-DD\
    - If not provided then todays date is used\
    - If running locally, the value is overwritten by 2024-03-10 to ensure data is always extracted.
    - If running in staging, the value is overwritten by a random date between two specified dates.

sites (optional): The sites to extract data from.\
    - If empty then all sites will be extracted\
    - Entered sites are checked against available sites in the metadata store and removed if not found\
    - sites must be seperated by a comma, by 5 characters long and not contain special characters\
    - sites can be lower or upper case\

Until the live sensor data is available, the `end_date` parameter is hardcoded within the `build_date_range` function to ensure the app can process some data. If the dataset to be processed is changed, then its likely you will need to update these. Check the dates available for each dataset in `parquet-data` (running locally) and in the [level-0 bucket](https://eu-west-2.console.aws.amazon.com/s3/buckets/ukceh-fdri-staging-timeseries-level-0?region=eu-west-2&bucketType=general) (running in staging).

Get the last two days data from hardcoded end dates (as above)
```commandline
python -m time-stream P2D
```

Get the last two days data from 2024-03-05
```commandline
python -m time-stream P2D --end_date=2024-03-05
```

Get the last two days data from 2024-03-05 for ALCI and BUNNY sites
```commandline
python -m time-stream P2D --end_date=2024-03-05 --sites=alic1,bunny
```

## Linting
Linting uses ruff using the config in pyproject.toml
```
ruff check --fix
```

## Formating
Formating uses ruff using the config in pyproject.toml which follows the default black settings.
```
ruff format .
```

## Testing
Testing is done using pytest and tests are in the /tests directory.
```
pytest
```

## Pre commit hooks
Run below to setup the pre-commit hooks.
```
git config --local core.hooksPath .githooks/
```

## Metrics
Metrics are collected using prometheus. The app is run as a cron job and the metrics are pushed to the prometheus [pushgateway](https://prometheus.io/docs/practices/pushing/). These can be accessed locally via `localhost:9091`

## localstack
Local stack is used to create local AWS resources for testing the app locally. `localstack-setup.sh` is run when the container is initialised which creates the buckets and loads the sample parquet files.

Run ```docker compose up``` to build.

# How to Develop This App

## Adding Tests

Each test is represented by a "bit" in a binary number, with one bit for each test. If a test fails, it's bit is set to 1.

For 3 tests called "A", "B", "C" the outcomes of the tests are:

|      | A | B | C | Result (decimal) |
| ---- |---|---|----|-|
| All passed | 0 | 0 | 0 | 0 |
| A failed  | 1 | 0  | 0 |  1|
| B failed  | 0 | 1  | 0 |  2|
| C failed  | 0 | 0  | 1 |  4|
| A+B failed  | 1 | 1  | 0 |  3|
| A+C failed  | 1 | 0  | 1 |  5|
| B+C failed  | 0 | 1  | 1 |  6|
| A+B+C failed  | 1 | 1  | 1 |  7|

In this scheme, any combination of failed tests will always have a unique identifier that can be stored in decimal, binary, or string format. We are likely to store the test QC_FLAG as a 64bit integer in the final database, this gives us 64 potential tests before we need to implement QC check versioning.

### Adding a New Test

Test definitions are stored in [\_\_metadata\_\_.config_quality_control.py](./src/time-stream__metadata__.config_quality_control.py) in a variable called `qc_tests`. To add a new test, simply continue the dictionary of tests, copy the previous entry, and increment the ID by 1. If the last `id` is `1 << 5`, the next one should be `1 << 6`.

Make sure to also add the test to the `qc_test_map` in [quality_config.py](./src/time-stream/quality_control.py)

### Deprecating a Test

Do nothing! When a test is deprecated, that ID is "retired" and cannot be reused. We may add a attribute to mark them as deprecated, but for now that has not been implemented.

If you want to create a new test with a clashing name, you can add a "_DEPRECATED" suffix to the deprecated test name.

Make sure to also remove the test from the `qc_test_map` in [quality_config.py](./src/time-stream/quality_control.py)

# Making Gaps in the Parquet Data

To test the QC behaviour when there is missing data, and to test the infilling processes, the test data has been duplicated and had sections of data removed. This located at [./parquet-data/cosmos-with-gaps](./parquet-data/cosmos-with-gaps) and is only done for LIVE_PRECIP_1MIN for now.

## Types of Gap Creation

The gaps were generated using file removal and `duckdb` manipulation and should be relatively repeatable:

* Randomly removing rows
* Randomly setting values to null
* Removing rows before / after a time

## Use the databuilder package for making gaps

There is a package called `databuilder` that comes with some utilities for copying the original data and a `ParquetBuilder` class that handles data manipulation. The `ParquetBuilder` is constructed as a [builder pattern](https://refactoring.guru/design-patterns/builder/python/example) and is designed to be extendible to handle different data sources.

The code works by reading the data into a dataframe and then performing convential dataframe manipulation in `pandas`. This makes it data source agnostic and delegates reading and writing to concrete implementations of different data sources. Currently there is one building function that runs all manipulation methods based on the inputs given.

How to use it:

```python
from databuilder.builders import ParquetBuilder
from datetime import time

# Initialize the builder
builder = ParquetBuilder(Path("mydata.parquet"))

# Run the build_all method to run all operations
builder.build_all(
    row_removal_percent=30,
    cell_removal_percent=5,
    protected_columns=["time", "SITE_ID", "RECORD"],
    clear_before_time=time(hour=10),
    clear_after_time=time(hour=19, minute=23)
    )

# The results of the builder are stored in `builder._dataframe`
# but haven't been written anywhere yet

# Write the output
builder.write_output()
```
This did the following:
* Loaded `"mydata.parquet"` into the parameter `builder._dataframe`
* Removed a random 30% of rows
* Set 5% of of each column to `NULL` EXCEPT for columns "time", "SITE_ID", and "RECORD"
* Removed all rows before `10:00`
* Removed all rows after `19:23`
* Wrote the output back to `"mydata.parquet"`

### Manipulating Multiple Files
Say you want to change 2 files sequentially:

```python
from databuilder.builders import ParquetBuilder
from datetime import time

# Initialize the builder
target = Path("mydata-1.parquet")
builder = ParquetBuilder(target, output=target)

builder.build_all(row_removal_percent=30)
builder.write_output()

# Reset the builder, this clears builder._dataframe
builder.reset()

# Set the new target, by default the output file
# is set to the new target
builder.target = Path("mydata-2.parquet")
builder.build_all(cell_removal_percent=75)
builder.write_output()
```

If you need the output file to be different to the target:

```python
# If you don't want to overwrite the file you can set the
# output to a different file
...

builder.reset()
builder.target = Path("mydata-3.parquet")
builder.output = Path("mydata-3-test.parquet")

# This sets the target to "mydata-3.parquet" and the output
# to "mydata-3-test.parquet
builder.build_all(cell_removal_percent=4)
builder.write_output()


# Alternatively you can set the output file on instantiation

target = Path("target.parquet")
output = Path("output.parquet")

builder = ParquetBuilder(target, output)
```
## Prebuilt Script for Managing our Existing Files

There is a script at [./bin/build-gapped-dataset.py](./bin/build-gapped-dataset.py) that has been used to generate the files at [./parquet-data/cosmos-with-gaps](./parquet-data/cosmos-with-gaps) and the results committed to this repo. Because of the inherent randomness involved, running the script will change the files and show a git diff.
