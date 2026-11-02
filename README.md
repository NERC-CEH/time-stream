[![PyPI](https://img.shields.io/pypi/v/ukceh-time-stream.svg)](https://pypi.org/project/ukceh-time-stream/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/NERC-CEH/time-stream/blob/main/LICENSE)
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
- Quality control checks and infilling of missing data
- A flexible flagging system

## Installation

```bash
pip install ukceh-time-stream
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add ukceh-time-stream
```

The package is imported as `time_stream`:

```python
import time_stream as ts
```

See the [documentation](https://nerc-ceh.github.io/time-stream/) for a quick start, user guides and the full API
reference.

## License

This project is licensed under the [MIT license](https://github.com/NERC-CEH/time-stream/blob/main/LICENSE).

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

1. Clone the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure your code passes all tests and follows the coding style before submitting a PR.
See **developer setup** below, and [`CONTRIBUTING.md`](https://github.com/NERC-CEH/time-stream/blob/main/CONTRIBUTING.md), for more information.

## Developer Setup

This is for active development on the time-stream package itself.

### Requirements

#### Install uv

[Official instructions](https://docs.astral.sh/uv/getting-started/installation/)

### Clone the repository

```bash
git clone https://github.com/NERC-CEH/time-stream.git
cd time-stream
```

### Setting up and activating a virtual environment

```commandline
uv sync
source .venv/bin/activate
```

### Checking your changes
To format, lint, type check and test in one go (run this before submitting a PR):
```
make qa
```

Run `make help` to list all the available commands. The individual checks are below.

### Linting
Linting uses ruff with the config in pyproject.toml.
```
uv run ruff check --fix
```

### Formatting
Formatting uses ruff with the config in pyproject.toml, which follows the default black settings.
```
uv run ruff format .
```

### Type checking
Type checking uses pyright, which must report no errors.
```
make type-check
```

### Testing
Testing uses pytest. The tests are in the `tests/` directory, and the examples in the docstrings under `src/` are run
as tests too.
```
make test
```
To run the tests on every supported Python version, use `make testall`.

### Pre commit hooks
Run below to set up the pre-commit hooks.
```
make install-hooks
```
This will set this repo up to use the git hooks in the `.githooks/` directory.
The hook runs `ruff format --check` and `ruff check` to prevent commits that are not formatted correctly or have errors.
The hook intentionally does not alter the files, but informs the user which command to run.

### Using an unreleased version in another project

To use time-stream from a branch, or from a local clone you are editing, rather than a release on PyPI:

```commandline
uv add git+https://github.com/NERC-CEH/time-stream.git@develop
uv add --editable /path/to/time-stream
```

With `--editable`, changes to the clone are immediately available to the project using it.

## Releasing

Merging to `main` with a new version in `pyproject.toml` tags the release, creates a GitHub release from
`CHANGELOG/<version>.md`, and publishes the package to [PyPI](https://pypi.org/project/ukceh-time-stream/).

## Documentation

For full documentation, visit https://nerc-ceh.github.io/time-stream/

To build the documentation locally (the documentation dependencies are installed by `uv sync` by default):

```bash
make docs-build
```

The built documentation is in `docs/_build/html/index.html`. Alternatively, `make docs-serve` serves the
documentation at http://localhost:8000 and rebuilds it as you edit.

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](https://github.com/NERC-CEH/time-stream/blob/main/CITATION.cff).

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [NERC-CEH/fdri-cookiecutter-pypackage](https://github.com/NERC-CEH/fdri-cookiecutter-pypackage) template.
