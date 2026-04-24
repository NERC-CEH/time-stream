# Contributing

Contributions are welcome and greatly appreciated.

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/NERC-CEH/time-stream/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs -
these are open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features -
these are open to whoever wants to implement it.

### Write Documentation

My Package could always use more documentation, whether as part of the official docs, local
README's or in docstrings, and comments.

To preview the official docs locally:

```sh
make docs-serve
```

This starts a local server at http://localhost:8000 with live reload. Edit files in `docs/` or add docstrings
to your code (the API reference page is auto-generated).

### Submit Feedback

The best way to send feedback is to file an issue at
https://github.com/NERC-CEH/time-stream/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.

## Get Started

Ready to contribute? Here's how to set up time-stream for local development.

1. Fork the time-stream repo on GitHub.

1. Clone your fork locally:

   ```sh
   git clone git@github.com:your_name_here/time-stream.git
   ```

1. Install your local copy with uv:

   ```sh
   cd time-stream/
   uv sync
   ```

1. Create a branch for local development off `develop`:

   ```sh
   git checkout develop
   git pull origin develop
   git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

1. When you're done making changes, check that your changes pass linting and the tests:

   ```sh
   make qa
   ```

1. Commit your changes and push your branch to GitHub:

   ```sh
   git add .
   git commit -m "Your detailed description of your changes."
   git push origin name-of-your-bugfix-or-feature
   ```

1. Open a pull request targeting `develop` through the GitHub website.

   **Branch protection on `develop` and `main`:**
   - At least 1 approving review is required before merge.
   - CI checks (`test-python`, `build-docs`) must pass.
   - All PR review conversations must be resolved.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function
with a docstring, and add the feature to the list in README.md.
3. The pull request should pass all quality checks (`make qa`) and GitHub Actions, making sure that the tests pass for all supported Python versions.

## Tips

To run a subset of tests:

```sh
uv run pytest tests/
```

## Releasing a New Version

Releases go through `develop` -> `main` to keep `main` in sync with what is published.

1. **On `develop`**, bump the version and create a CHANGELOG stub:
   ```bash
   git checkout develop
   git pull origin develop
   make bump-patch   # or bump-minor / bump-major
   ```
   This updates `pyproject.toml`, commits the bump, and creates `CHANGELOG/<version>.md`.

2. **Fill in** `CHANGELOG/<version>.md` with the release notes, then commit and push:
   ```bash
   git add CHANGELOG/<version>.md
   git commit -m "Add release notes for <version>"
   git push origin develop
   ```

3. **Open a release PR** from develop to main:
   ```bash
   make release
   ```
   When run from `develop`, this opens a pull request against `main` using the CHANGELOG
   as the PR description.

4. **After the PR is merged**, switch to `main` and tag:
   ```bash
   git checkout main
   git pull origin main
   make release
   ```
   When run from `main`, this creates an annotated `v*` tag, pushes it to GitHub, and
   creates a GitHub Release.
