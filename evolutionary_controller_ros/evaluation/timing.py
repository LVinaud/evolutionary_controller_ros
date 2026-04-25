"""Timing log for the GA loop — pure Python, no ROS.

`TimingLog` collects (category, duration_s, ...metadata) events from any
caller and dumps them to CSV on demand. Designed for two consumers:

    * `algorithm.run_ga` — coarse phases per generation: population init,
      population evaluation (sum of all individuals), case-matrix
      assembly, on-generation callback, breeding.
    * `evaluation.episode.run_episode` — fine-grained per-episode phases:
      `setup_param_push`, `setup_reset_call`, `setup_teleport_models`,
      `setup_teleport_robot`, `episode_spin` (with `early_stop` and
      `ticks_collected`), plus the wall-clock between scenarios.

Schema is row-per-event, with a flexible `extras` dict merged into the
row at write time. Columns missing in a given row come out empty.

Why not just use `time.monotonic()` ad-hoc and `print` it? Because the
goal is producing a CSV for offline analysis (and the project report) —
plotting, aggregation, and detecting where wall-clock is spent need
structured data, not log scraping.
"""
import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class TimingLog:
    """Append-only event log, flushed to CSV via `write_csv`."""

    def __init__(self):
        self._events: list[dict[str, Any]] = []

    def event(self, category: str, duration_s: float, **extras: Any) -> None:
        """Record one event row.

        `category` is the only required field besides duration. Any extra
        keyword arguments become extra columns in the resulting CSV.
        """
        row = {
            "wall_ts": time.time(),
            "category": category,
            "duration_s": float(duration_s),
        }
        row.update(extras)
        self._events.append(row)

    @contextmanager
    def timed(self, category: str, **extras: Any):
        """Context manager that records an event with the elapsed wall time.

        Example
        -------
        >>> log = TimingLog()
        >>> with log.timed("setup_reset_call", gen=3, individual=12): ...
        """
        t0 = time.monotonic()
        yield
        self.event(category, time.monotonic() - t0, **extras)

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    def write_csv(self, path: str | Path) -> Path:
        """Write all collected events to a CSV file at `path`.

        The CSV header is the union of every row's keys, sorted with
        `category` and `duration_s` first for readability. Missing
        fields per row are emitted as empty cells.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._events:
            path.write_text("")
            return path

        all_keys = set()
        for row in self._events:
            all_keys.update(row.keys())
        priority = ["wall_ts", "category", "duration_s",
                    "gen", "individual", "scenario"]
        head = [k for k in priority if k in all_keys]
        tail = sorted(k for k in all_keys if k not in head)
        fieldnames = head + tail

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in self._events:
                writer.writerow(row)
        return path
