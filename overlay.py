#!/usr/bin/env python3
"""Parsing and querying of `*.mp4.overlay.json` detector-overlay files.

An overlay file holds per-timestamp detection boxes (absolute pixels) for a
clip. The overlay frame list is time-indexed and is NOT aligned 1:1 with the
decoded video frames, so boxes are looked up by timestamp.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from event_annotator import EventAnnotation, clamp_frame, normalize_box

OVERLAY_SUFFIX = ".overlay.json"
ZONE_LABEL = "restricted zone"
CROSSING_LABEL = "restricted zone crossing"


@dataclass(frozen=True)
class OverlayBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    tone: str


@dataclass(frozen=True)
class Overlay:
    width: int
    height: int
    fps: float
    times: tuple[float, ...]  # sorted frame timestamps in seconds
    boxes_by_frame: tuple[tuple[OverlayBox, ...], ...]  # parallel to `times`

    def boxes_at(self, t: float) -> tuple[OverlayBox, ...]:
        """Boxes from the latest overlay frame at or before time `t`."""
        if not self.times:
            return ()
        index = bisect_right(self.times, t) - 1
        if index < 0:
            index = 0
        return self.boxes_by_frame[index]


def overlay_path_for(clip_path: Path) -> Path:
    return clip_path.with_name(clip_path.name + OVERLAY_SUFFIX)


def _parse_box(raw: Any) -> OverlayBox | None:
    if not isinstance(raw, dict):
        return None
    try:
        x1 = float(raw["x1"])
        y1 = float(raw["y1"])
        x2 = float(raw["x2"])
        y2 = float(raw["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    return OverlayBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        label=str(raw.get("label", "")).strip(),
        tone=str(raw.get("tone", "")).strip(),
    )


def load_overlay(path: Path) -> Overlay:
    """Read and validate an overlay file. Raises ValueError on bad structure."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("overlay root is not an object")
    video = payload.get("video")
    if not isinstance(video, dict):
        raise ValueError("overlay is missing a 'video' object")
    try:
        width = int(video["width"])
        height = int(video["height"])
        fps = float(video.get("fps") or 0.0)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"overlay 'video' block is invalid: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("overlay video dimensions must be positive")

    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list):
        raise ValueError("overlay is missing a 'frames' list")

    parsed: list[tuple[float, tuple[OverlayBox, ...]]] = []
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, dict):
            continue
        try:
            t = float(raw_frame.get("t", 0.0))
        except (TypeError, ValueError):
            continue
        raw_boxes = raw_frame.get("boxes", [])
        if not isinstance(raw_boxes, list):
            raw_boxes = []
        boxes = tuple(
            box for box in (_parse_box(rb) for rb in raw_boxes) if box is not None
        )
        parsed.append((t, boxes))

    parsed.sort(key=lambda item: item[0])
    return Overlay(
        width=width,
        height=height,
        fps=fps,
        times=tuple(t for t, _ in parsed),
        boxes_by_frame=tuple(boxes for _, boxes in parsed),
    )


def derive_tp_event(
    overlay: Overlay,
    *,
    event_type: str,
    video_fps: float,
    frame_count: int,
) -> EventAnnotation:
    """Build one annotation event from the crossing detection in an overlay.

    Time range = first..last frame containing a crossing box; bbox = the union
    of all crossing boxes, normalized. Falls back to non-zone boxes, then the
    whole clip/frame, so a TP verdict always yields a usable event.
    """
    selected = _boxes_with_label(overlay, CROSSING_LABEL)
    if not selected:
        selected = _boxes_excluding_label(overlay, ZONE_LABEL)
    if not selected:
        return EventAnnotation(
            event_type=event_type,
            start=0,
            end=max(0, frame_count - 1),
            bbox=(0.0, 0.0, 1.0, 1.0),
        )

    times = [t for t, _ in selected]
    start = clamp_frame(round(min(times) * video_fps), frame_count)
    end = max(start, clamp_frame(round(max(times) * video_fps), frame_count))
    union = (
        min(box.x1 for _, box in selected),
        min(box.y1 for _, box in selected),
        max(box.x2 for _, box in selected),
        max(box.y2 for _, box in selected),
    )
    return EventAnnotation(
        event_type=event_type,
        start=start,
        end=end,
        bbox=normalize_box(union, overlay.width, overlay.height),
    )


def _boxes_with_label(overlay: Overlay, label: str) -> list[tuple[float, OverlayBox]]:
    return [
        (t, box)
        for t, boxes in zip(overlay.times, overlay.boxes_by_frame)
        for box in boxes
        if box.label == label
    ]


def _boxes_excluding_label(overlay: Overlay, label: str) -> list[tuple[float, OverlayBox]]:
    return [
        (t, box)
        for t, boxes in zip(overlay.times, overlay.boxes_by_frame)
        for box in boxes
        if box.label != label
    ]
