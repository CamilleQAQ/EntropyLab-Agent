"""Allow ``python -m entropy_agent.cli`` without installing the console script."""

from __future__ import annotations

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
