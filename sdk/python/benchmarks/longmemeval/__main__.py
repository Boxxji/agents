"""CLI entrypoint for `python3 -m benchmarks.longmemeval`."""

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
