# Python Project Template

[![tests badge](https://github.com/NERC-CEH/python-template/actions/workflows/pipeline.yml/badge.svg)](https://github.com/NERC-CEH/python-template/actions)
[![docs badge](https://github.com/NERC-CEH/python-template/actions/workflows/deploy-docs.yml/badge.svg)](https://nerc-ceh.github.io/python-template/)

[Read the docs!](https://nerc-ceh.github.io/python-template)

This repository is a template for a basic Python project. Included here is:

* Example Python package
* Tests
* Documentation
* Automatic incremental versioning
* CI/CD
    * Installs and tests the package
    * Builds documentation on branches
    * Deploys documentation on main branch
    * Deploys docker image to AWS ECR
* Githook to ensure linting and code checking

## Getting Started

### Using the Githook

From the root directory of the repo, run:

```
git config --local core.hooksPath .githooks/
```

This will set this repo up to use the git hooks in the `.githooks/` directory. The hook runs `ruff format --check` and `ruff check` to prevent commits that are not formatted correctly or have errors. The hook intentionally does not alter the files, but informs the user which command to run.

### Installing the package

This package is configured to use optional dependencies based on what you are doing with the code.

As a user, you would install the code with only the dependencies needed to run it:

```
pip install .
```

To work on the docs:

```
pip install -e .[docs]
```

To work on tests:

```
pip install -e .[tests]
```

To run the linter and githook:

```
pip install -e .[lint]
```

The docs, tests, and linter packages can be installed together with:

```
pip install -e .[dev]
```

### Making it Your Own

This repo has a single package in the `./src/...` path called `mypackage` (creative I know). Change this to the name of your package and update it in:

* `docs/conf.py`
* `src/**/*.py`
* `tests/**/*.py`
* `pyproject.toml`

To make thing move a bit faster, use the script `./rename-package.sh` to rename all references of `mypackage` to whatever you like. For example:

```
./rename-package.sh "acoolnewname"
```

Will rename the package and all references to "acoolnewname"

After doing this it is recommended to also run:

```
cd docs
make apidoc
```

To keep your documentation in sync with the package name. You may need to delete a file called `mypackage.rst` from `./docs/sources/...`

### Deploying Docs to GitHub Pages

If you want docs to be published to github pages automatically, go to your repo settings and enable docs from GitHub Actions and the workflows will do the rest.

### Building Docs Locally

The documentation is driven by [Sphinx](https://www.sphinx-doc.org/) an industry standard for documentation with a healthy userbase and lots of add-ons. It uses `sphinx-apidoc` to generate API documentation for the codebase from Python docstrings.

To run `sphinx-apidoc` run:

```
# Install your package with optional dependencies for docs
pip install -e .[docs]

cd docs
make apidoc
```

This will populate `./docs/sources/...` with `*.rst` files for each Python module, which may be included into the documentation.

Documentation can then be built locally by running `make html`, or found on the [GitHub Deployment](https://nerc-ceh.github.io/python-template).

### Run the Tests

To run the tests run:

```
#Install package with optional dependencies for testing
pip install -e .[test]

pytest
```

### Automatic Versioning

This codebase is set up using [autosemver](https://autosemver.readthedocs.io/en/latest/usage.html#) a tool that uses git commit history to calculate the package version. Each time you make a commit, it increments the patch version by 1. You can increment by:

* Normal commit. Use for bugfixes and small updates
    * Increments patch version: `x.x.5 -> x.x.6`
* Commit starts with `* NEW:`. Use for new features
    * Increments minor version `x.1.x -> x.2.x`
* Commit starts with `* INCOMPATIBLE:`. Use for API breaking changes
    * Increments major version `2.x.x -> 3.x.x`

### Docker and the ECR

The python code is packaged into a docker image and pushed to the AWS ECR. For the deployment to succeed you must:

* Add 2 secrets to the GitHub Actions:
    * AWS_REGION: \<our-region\>
    * AWS_ROLE_ARN: \<the-IAM-role-used-to-deploy\>
* Add a repository to the ECR with the same name as the GitHub repo
 