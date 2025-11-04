"""Command line interface for running the dummy simulation."""

from __future__ import annotations

from rlbot_pro.sim import run_dummy_frame


def main() -> None:
    """Run a single simulation frame and print controls."""
    controls = run_dummy_frame()
    print(controls)


if __name__ == "__main__":
    main()
