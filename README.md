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
pip install '.[dev]'
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

Run ```docker compose --profile tsp_localstack up``` to build.
