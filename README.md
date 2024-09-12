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
python -m venv .myvenv
source .myenv/bin/activate
```

### Installing the App

```commandline
pip install -e '.[dev]'
```

## Running the app
```commandline
python -m dritimeseriesprocessor
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

Test definitions are stored in [\_\_metadata\_\_.config_quality_control.py](./src/dritimeseriesprocessor__metadata__.config_quality_control.py) in a variable called `qc_tests`. To add a new test, simply continue the dictionary of tests, copy the previous entry, and increment the ID by 1. If the last `id` is `1 << 5`, the next one should be `1 << 6`.

Make sure to also add the test to the `qc_test_map` in [quality_config.py](./src/dritimeseriesprocessor/quality_control.py)

### Deprecating a Test

Do nothing! When a test is deprecated, that ID is "retired" and cannot be reused. We may add a attribute to mark them as deprecated, but for now that has not been implemented.

If you want to create a new test with a clashing name, you can add a "_DEPRECATED" suffix to the deprecated test name.

Make sure to also remove the test from the `qc_test_map` in [quality_config.py](./src/dritimeseriesprocessor/quality_control.py)