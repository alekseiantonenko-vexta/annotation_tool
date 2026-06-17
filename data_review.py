#!/usr/bin/env python3
"""Summarize event annotation JSON files.

Accepts one or more directories and/or annotation JSON files. Directories are
searched recursively for `*_event-annotation.json` files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

ANNOTATION_PATTERNS = ("*_event-annotation.json", "*_even-annotation.json")


@dataclass
class DurationStats:
    frames: list[int] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    def add(self, frames: int, fps: float | None) -> None:
        self.frames.append(frames)
        if fps is not None and fps > 0:
            self.seconds.append(frames / fps)


@dataclass
class EventClassStats:
    total: int = 0
    files: set[Path] = field(default_factory=set)
    durations: DurationStats = field(default_factory=DurationStats)


@dataclass
class ReviewStats:
    total_files: int = 0
    files_with_events: int = 0
    files_without_events: int = 0
    classes: dict[str, EventClassStats] = field(
        default_factory=lambda: defaultdict(EventClassStats),
    )
    unreadable: list[tuple[Path, str]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print summary statistics for event annotation JSON files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Directories and/or annotation JSON files to analyze.",
    )
    parser.add_argument(
        "-l",
        "--label",
        help="Show files and statistics for one exact event label.",
    )
    return parser.parse_args()


def find_annotation_files(inputs: list[str]) -> list[Path]:
    found: dict[Path, None] = {}
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            for pattern in ANNOTATION_PATTERNS:
                for child in path.rglob(pattern):
                    if child.is_file():
                        found[child.resolve()] = None
        elif path.is_file():
            found[path.resolve()] = None
        else:
            print(f"WARN: path does not exist: {path}")
    return sorted(found)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_events(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    events = payload.get("events", [])
    return events if isinstance(events, list) else []


def parse_fps(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    try:
        fps = float(payload.get("fps", 0.0))
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def event_duration_frames(event: dict[str, Any]) -> int | None:
    try:
        start = int(event["start"])
        end_raw = event["end"]
        if end_raw is None:
            return None
        end = int(end_raw)
    except (KeyError, TypeError, ValueError):
        return None
    if end < start:
        return None
    return end - start + 1


def analyze(paths: list[Path]) -> ReviewStats:
    stats = ReviewStats()
    for path in paths:
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            stats.unreadable.append((path, str(exc)))
            continue

        stats.total_files += 1
        events = parse_events(payload)
        fps = parse_fps(payload)
        if events:
            stats.files_with_events += 1
        else:
            stats.files_without_events += 1

        seen_in_file: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type", "")).strip()
            if not event_type:
                continue
            class_stats = stats.classes[event_type]
            class_stats.total += 1
            seen_in_file.add(event_type)
            duration = event_duration_frames(event)
            if duration is not None:
                class_stats.durations.add(duration, fps)

        for event_type in seen_in_file:
            stats.classes[event_type].files.add(path)

    return stats


def format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.3f}"


def format_duration(values: list[int] | list[float], unit: str) -> str:
    if not values:
        return f"n/a {unit}"
    return (
        f"min {format_number(min(values))} / "
        f"max {format_number(max(values))} / "
        f"avg {format_number(mean(values))} {unit}"
    )


def print_report(paths: list[Path], stats: ReviewStats, label: str | None = None) -> None:
    print("Annotation data review")
    print("======================")
    print(f"Input annotation files: {len(paths)}")
    print(f"Readable annotation files: {stats.total_files}")
    print(f"Files with events: {stats.files_with_events}")
    print(f"Files without events: {stats.files_without_events}")
    if stats.unreadable:
        print(f"Unreadable files: {len(stats.unreadable)}")
    print()

    if label is not None:
        print(f"Label filter: {label}")
        print()
        print_label_report(label, stats)
        return

    if stats.classes:
        print("Event classes:")
        for event_type in sorted(stats.classes):
            print(f"- {event_type}")
    else:
        print("Event classes: none")
    print()

    if not stats.classes:
        return

    print("Per-class statistics:")
    for event_type in sorted(stats.classes):
        class_stats = stats.classes[event_type]
        frame_stats = format_duration(class_stats.durations.frames, "frames")
        second_stats = format_duration(class_stats.durations.seconds, "seconds")
        print(
            f"- {event_type}: total {class_stats.total}, "
            f"files {len(class_stats.files)}, "
            f"duration frames: {frame_stats}, "
            f"duration seconds: {second_stats}"
        )

    if stats.unreadable:
        print()
        print("Unreadable:")
        for path, error in stats.unreadable:
            print(f"- {path}: {error}")


def print_label_report(label: str, stats: ReviewStats) -> None:
    class_stats = stats.classes.get(label)
    if class_stats is None:
        print(f"No events with label: {label}")
        return

    print(f"Files containing '{label}': {len(class_stats.files)}")
    for path in sorted(class_stats.files):
        print(f"- {path}")
    print()

    frame_stats = format_duration(class_stats.durations.frames, "frames")
    second_stats = format_duration(class_stats.durations.seconds, "seconds")
    print("Label statistics:")
    print(
        f"- {label}: total {class_stats.total}, "
        f"files {len(class_stats.files)}, "
        f"duration frames: {frame_stats}, "
        f"duration seconds: {second_stats}"
    )


def main() -> int:
    args = parse_args()
    paths = find_annotation_files(args.paths)
    stats = analyze(paths)
    label = args.label.strip() if args.label else None
    print_report(paths, stats, label)
    return 1 if stats.unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
