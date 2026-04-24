"""Tag the current version and create a GitHub release.

On `main`: creates a git tag and GitHub release.
On `develop`: opens a release PR from develop to main instead.
"""

import subprocess
import sys
import tomllib
from pathlib import Path


def _run(*cmd: str) -> None:
    """Print and execute a command, raising on non-zero exit.

    Args:
        *cmd: Command and arguments to run.
    """
    print(f"$ {' '.join(cmd)}")  # noqa: T201
    subprocess.run(cmd, check=True)


def _current_branch() -> str:
    """Return the name of the currently checked-out git branch.

    Returns:
        Branch name string (e.g. ``"main"``).
    """
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    """Tag the current version and publish a release (or open a release PR from develop)."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    version = pyproject["project"]["version"]
    tag = f"v{version}"
    notes_path = Path(f"CHANGELOG/{version}.md")
    branch = _current_branch()
    name = pyproject["project"]["name"]

    if branch == "develop":
        print(f"On develop - opening release PR for {tag} rather than tagging.")  # noqa: T201
        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                "develop",
                "--title",
                f"Release {tag}",
                "--body-file",
                str(notes_path),
            ],
            check=False,
        )
        if result.returncode != 0:
            print(  # noqa: T201
                "\nA release PR may already exist, or gh pr create failed.\n"
                "Check: gh pr list --base main\n"
                "After merging, run `make release` from main to tag and publish."
            )
        else:
            print(
                f"\nRelease PR opened for {tag}.\nAfter it is merged, run `make release` from main to tag and publish."
            )  # noqa: T201
        return

    if branch != "main":
        print(
            f"Error: `make release` must be run from main or develop, not '{branch}'.\n"
            f"Switch to the correct branch and try again."
        )  # noqa: T201
        sys.exit(1)

    lines = notes_path.read_text().splitlines(keepends=True)
    title = f"{name} {version}"
    if lines and lines[0].startswith("# "):
        title = lines[0].lstrip("# ").strip()
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    notes = "".join(lines).rstrip()

    _run("git", "tag", "-a", tag, "-m", f"Release {tag}")
    _run("git", "push", "origin", tag)
    _run("gh", "release", "create", tag, "--verify-tag", "--title", title, "--notes", notes)


if __name__ == "__main__":
    main()
