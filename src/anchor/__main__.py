"""Run ANCHOR's deployment and evaluation commands."""
import sys

from .cli import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ANCHOR error: {exc}", file=sys.stderr)
        raise SystemExit(1)
