"""Make the user guide's example package importable from the tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "docs" / "source"))
