![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Language](https://img.shields.io/github/languages/top/NERC-CEH/time-stream)
[![tests badge](https://github.com/NERC-CEH/time-stream/actions/workflows/pipeline.yml/badge.svg)](https://github.com/NERC-CEH/time-stream/actions)
[![Docs](https://img.shields.io/badge/docs-%F0%9F%93%9A%20online-blue)](https://nerc-ceh.github.io/time-stream)

# Time-Stream
**Time** **S**eries **T**oolkit for **R**apid **E**nvironmental **A**nalysis and **M**onitoring: A Python library 
for handling and analysing timeseries data with a focus on maintaining the integrity of the temporal properties of the 
data. 

## Overview

Time-Stream provides robust tools for working with timeseries data, built on top of [Polars](https://pola.rs/), 
with special attention to:

- Precise temporal handling with Period-based time manipulations
- Smart temporal aggregation
- Provision of a flexible flagging system
- Column relationships for organising complex timeseries datasets

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

1. Clone the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure your code passes all tests and follows the coding style before submitting a PR.  
See **developer setup** below for more information.

## Developer Setup

This is for active development on the time-stream package itself. 

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


### Clone the repository

```bash
git clone https://github.com/NERC-CEH/time-stream.git
cd time-stream
```

### Setting up and activating a virtual environment

```commandline
python -m venv .venv
source .venv/bin/activate
```

### Installing the development Python environment

```commandline
pip install -e '.[dev]'
```

### Linting
Linting uses ruff using the config in pyproject.toml
```
ruff check --fix
```

### Formating
Formating uses ruff using the config in pyproject.toml which follows the default black settings.
```
ruff format .
```

### Testing
Testing is done using pytest and tests are in the /tests directory.
```
pytest
```

### Pre commit hooks
Run below to setup the pre-commit hooks.
```
git config --local core.hooksPath .githooks/
```
This will set this repo up to use the git hooks in the `.githooks/` directory. 
The hook runs `ruff format --check` and `ruff check` to prevent commits that are not formatted correctly or have errors. 
The hook intentionally does not alter the files, but informs the user which command to run.

## Installing time-stream

Whilst time-stream is under active development, to use time-stream within your project you can do one of two things:

1. Clone the time-stream repository to a location next to your project's repository. Then, you can install time-series using pip's
"development mode" using a relative path.
    
    When you install a package in editable mode, pip creates a special link to the package's source directory 
    instead of copying files to your Python's site-packages directory. Any changes to the source code are immediately 
    available to any project using the package.
    
    **Using pip directly**
    ```commandline
    pip install -e /path/to/time-stream
    ```
    
    **In your project's pyproject.toml**
    ```commandline
    [project]
    dependencies = [
        "time-stream @ file:///path/to/time-stream"
    ]
    ```
    
    Now when changes have been made to time-stream, you can just do a `git pull` in your cloned directory to get the 
    changes, and they will be automatically available in your package. 

2. Use the time-stream git url in your pip install

    **Using pip directly**
    ```commandline
    pip install -e "time-stream @ git+https://github.com/NERC-CEH/time-stream.git"
    ```
    
    **In your project's pyproject.toml**
    ```commandline
    [project]
    dependencies = [
        "time-stream @ git+https://github.com/NERC-CEH/time-stream.git"
    ]
    ```

## Documentation

For full documentation, visit https://nerc-ceh.github.io/time-stream/

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build the documentation
cd docs
make html

# View documentation
open _build/html/index.html
```
