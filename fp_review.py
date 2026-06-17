#!/usr/bin/env python3
"""FP-review tool: triage detector clips as true/false positives.

For every `<clip>.mp4` paired with a `<clip>.mp4.overlay.json` under a folder,
the clip is played in a loop at high speed with the detection boxes drawn on
top. The reviewer presses 1 (true positive) or Space (false positive); the
tool records the verdict and auto-advances to the next unreviewed clip.

A true positive additionally exports a `<clip>_event-annotation.json`
annotation in the same schema used by event_annotator, so it drops straight
into the existing pipeline. False positives export an empty but valid
annotation file and are logged in fp_review_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from event_annotator import annotation_path_for, build_annotation_payload, import_cv2
from overlay import Overlay, OverlayBox, derive_tp_event, load_overlay, overlay_path_for
from review_results import (
    VERDICT_FP,
    VERDICT_TP,
    load_results,
    record_verdict,
    results_path_for,
    save_results,
)

WINDOW_NAME = "FP Review"
TRACKBAR_NAME = "Frame"
SPEEDS = (8, 16)
DEFAULT_SPEED = 8
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
STATUS_PANEL_HEIGHT = 50
# Crossing detections map onto the annotator's 'zone' event type.
TP_EVENT_TYPE = "zone"

# Colors are BGR (OpenCV order).
TONE_COLORS = {
    "danger": (60, 60, 235),
    "warning": (40, 180, 240),
}
DEFAULT_BOX_COLOR = (200, 200, 200)
HUD_COLOR = (245, 245, 245)
HUD_SHADOW = (20, 20, 20)
STATUS_BG = (30, 30, 30)
STATUS_MUTED = (185, 185, 185)

KEY_TP = ord("1")
KEY_FP = ord(" ")
KEY_REPLAY = ord("r")
KEY_NEXT = ord("n")
KEY_BACK = ord("b")
KEY_SPEED = ord("t")
KEY_QUIT = {ord("q"), 27}
KEY_NONE = 255


@dataclass(frozen=True)
class ClipContext:
    overlay: Overlay
    video_fps: float
    frame_count: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review detector clips as TP/FP.")
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder with <clip>.mp4 + <clip>.mp4.overlay.json pairs. "
        "If omitted, a folder picker is shown.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        choices=SPEEDS,
        default=DEFAULT_SPEED,
        help=f"Initial playback speed multiplier. Default: {DEFAULT_SPEED}.",
    )
    return parser.parse_args()


def resolve_folder(arg: str | None) -> Path | None:
    if arg:
        return Path(arg).expanduser().resolve()
    return _pick_folder()


def _pick_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askdirectory(title="Select folder with clips + overlays")
    root.destroy()
    return Path(chosen).expanduser().resolve() if chosen else None


def configure_qt_font_dir() -> None:
    """Point OpenCV's Qt backend at system fonts when the wheel has none."""
    if os.environ.get("QT_QPA_FONTDIR"):
        return
    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts"),
    ):
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def find_clips(folder: Path) -> list[Path]:
    return [
        mp4
        for mp4 in sorted(folder.glob("*.mp4"))
        if overlay_path_for(mp4).exists() and not annotation_path_for(mp4).exists()
    ]


def print_controls() -> None:
    print(
        "\nControls:\n"
        "  1        Mark TRUE positive (saves <clip>_event-annotation.json) and advance\n"
        "  Space    Mark FALSE positive and advance\n"
        "  Frame    Drag the OpenCV slider to seek within the current clip\n"
        "  n / b    Next / back without a verdict\n"
        "  r        Replay current clip from start\n"
        "  t        Toggle playback speed (x8 / x16)\n"
        "  q / Esc  Quit\n"
    )


class FPReviewer:
    def __init__(self, cv2: Any, folder: Path, clips: list[Path], speed: int) -> None:
        self.cv2 = cv2
        self.folder = folder
        self.clips = clips
        self.speed = speed
        self.results_path = results_path_for(folder)
        self.results = load_results(self.results_path)
        self.index = 0
        self.should_quit = False
        self.pending_seek: int | None = None
        self.suppress_trackbar = False

    def run(self) -> None:
        cv2 = self.cv2
        cv2.namedWindow(WINDOW_NAME)
        cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, 0, 1, self._on_trackbar_changed)
        try:
            while not self.should_quit and self.clips:
                self.index = max(0, min(self.index, len(self.clips) - 1))
                clip = self.clips[self.index]
                action, ctx = self._review_clip(clip)
                self._apply(action, clip, ctx)
        finally:
            cv2.destroyAllWindows()

    # --- per-clip playback ------------------------------------------------

    def _review_clip(self, clip: Path) -> tuple[str, ClipContext | None]:
        cv2 = self.cv2
        try:
            overlay = load_overlay(overlay_path_for(clip))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"SKIP {clip.name}: bad overlay ({exc})")
            return "next", None

        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"SKIP {clip.name}: cannot open video")
            return "next", None
        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            print(f"SKIP {clip.name}: cannot read frames")
            return "next", None

        height, width = first.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = overlay.fps if overlay.fps > 0 else 15.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            frame_count = len(overlay.times) or 1
        self._configure_trackbar(frame_count)
        scale = min(1.0, MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
        ctx = ClipContext(overlay, fps, frame_count, width, height)
        interval = max(1, round(1000.0 / fps))

        pos = 0
        self._set_trackbar_pos(pos)
        try:
            while True:
                pending_seek = self._consume_pending_seek(frame_count)
                if pending_seek is not None:
                    pos = pending_seek
                frame = self._read_at(cap, pos)
                if frame is None:
                    frame = self._read_at(cap, 0)
                    pos = 0
                    if frame is None:
                        print(f"SKIP {clip.name}: playback read error")
                        return "next", ctx
                self._update_window_title(clip, pos, frame_count)
                display = self._render(
                    frame,
                    scale,
                    overlay.boxes_at(pos / fps),
                    clip,
                    pos,
                    frame_count,
                )
                cv2.imshow(WINDOW_NAME, display)
                self._set_trackbar_pos(pos)
                key = cv2.waitKey(interval) & 0xFF

                if key in KEY_QUIT:
                    return "quit", ctx
                if key == KEY_TP:
                    return "tp", ctx
                if key == KEY_FP:
                    return "fp", ctx
                if key == KEY_NEXT:
                    return "next", ctx
                if key == KEY_BACK:
                    return "back", ctx
                if key == KEY_REPLAY:
                    pos = 0
                    continue
                if key == KEY_SPEED:
                    self.speed = SPEEDS[1] if self.speed == SPEEDS[0] else SPEEDS[0]
                    self._update_window_title(clip, pos, frame_count)
                    continue

                pending_seek = self._consume_pending_seek(frame_count)
                if pending_seek is not None:
                    pos = pending_seek
                    continue
                pos += self.speed
                if pos >= frame_count:
                    pos = 0
        finally:
            cap.release()

    def _status_text(self, clip: Path, pos: int, frame_count: int) -> str:
        total = len(self.clips)
        last_frame = max(0, frame_count - 1)
        return (
            f"clip {self.index + 1}/{total} | frame {pos}/{last_frame} | "
            f"{clip.name} | speed x{self.speed}"
        )

    def _update_window_title(self, clip: Path, pos: int, frame_count: int) -> None:
        text = self._status_text(clip, pos, frame_count)
        if hasattr(self.cv2, "setWindowTitle"):
            try:
                self.cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} - {text}")
            except self.cv2.error:
                pass

    def _configure_trackbar(self, frame_count: int) -> None:
        max_pos = max(1, frame_count - 1)
        if hasattr(self.cv2, "setTrackbarMin"):
            self.cv2.setTrackbarMin(TRACKBAR_NAME, WINDOW_NAME, 0)
        self.cv2.setTrackbarMax(TRACKBAR_NAME, WINDOW_NAME, max_pos)
        self.pending_seek = None

    def _on_trackbar_changed(self, pos: int) -> None:
        if self.suppress_trackbar:
            return
        self.pending_seek = pos

    def _set_trackbar_pos(self, pos: int) -> None:
        self.suppress_trackbar = True
        try:
            self.cv2.setTrackbarPos(TRACKBAR_NAME, WINDOW_NAME, int(pos))
        finally:
            self.suppress_trackbar = False

    def _consume_pending_seek(self, frame_count: int) -> int | None:
        if self.pending_seek is None:
            return None
        pos = max(0, min(int(self.pending_seek), max(0, frame_count - 1)))
        self.pending_seek = None
        return pos

    def _read_at(self, cap: Any, pos: int) -> Any | None:
        cap.set(self.cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        return frame if ok and frame is not None else None

    # --- rendering --------------------------------------------------------

    def _render(
        self,
        frame: Any,
        scale: float,
        boxes: tuple[OverlayBox, ...],
        clip: Path,
        pos: int,
        frame_count: int,
    ) -> Any:
        cv2 = self.cv2
        if scale != 1.0:
            height, width = frame.shape[:2]
            display = cv2.resize(
                frame, (max(1, round(width * scale)), max(1, round(height * scale)))
            )
        else:
            display = frame.copy()
        self._draw_boxes(display, boxes, scale)
        return self._append_status_panel(display, clip, pos, frame_count)

    def _append_status_panel(
        self,
        image: Any,
        clip: Path,
        pos: int,
        frame_count: int,
    ) -> Any:
        cv2 = self.cv2
        display = cv2.copyMakeBorder(
            image,
            0,
            STATUS_PANEL_HEIGHT,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=STATUS_BG,
        )
        top = image.shape[0]
        self._text(
            display,
            self._status_text(clip, pos, frame_count),
            10,
            top + 20,
            HUD_COLOR,
        )
        self._text(
            display,
            "Frame slider=seek  1=TP  SPACE=FP  n/b=clip  r=replay  t=speed  q=quit",
            10,
            top + 42,
            STATUS_MUTED,
        )
        return display

    def _draw_boxes(
        self, image: Any, boxes: tuple[OverlayBox, ...], scale: float
    ) -> None:
        cv2 = self.cv2
        for box in boxes:
            x1 = int(round(box.x1 * scale))
            y1 = int(round(box.y1 * scale))
            x2 = int(round(box.x2 * scale))
            y2 = int(round(box.y2 * scale))
            color = TONE_COLORS.get(box.tone, DEFAULT_BOX_COLOR)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if box.label:
                self._text(image, box.label, x1 + 3, max(14, y1 - 6), color)

    def _text(self, image: Any, text: str, x: int, y: int, color: Any) -> None:
        cv2 = self.cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (x + 1, y + 1), font, 0.5, HUD_SHADOW, 2, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), font, 0.5, color, 1, cv2.LINE_AA)

    # --- verdict handling -------------------------------------------------

    def _apply(self, action: str, clip: Path, ctx: ClipContext | None) -> None:
        if action == "quit":
            self.should_quit = True
        elif action == "tp" and ctx is not None:
            self._record(clip, ctx, VERDICT_TP)
            self._advance()
        elif action == "fp" and ctx is not None:
            self._record(clip, ctx, VERDICT_FP)
            self._advance()
        elif action == "back":
            self.index -= 1
        else:  # "next", or a verdict on an unreadable clip
            self._advance()

    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.clips):
            self.index = len(self.clips) - 1
            if all(clip.name in self.results for clip in self.clips):
                print("All clips reviewed. Press q to quit (or b to revisit).")

    def _record(self, clip: Path, ctx: ClipContext, verdict: str) -> None:
        annotation = None
        if verdict == VERDICT_TP:
            annotation = self._save_tp_annotation(clip, ctx)
        else:
            annotation = self._save_empty_annotation(clip, ctx)
        self.results = record_verdict(self.results, clip.name, verdict, annotation)
        self._persist()
        suffix = f" -> {annotation}" if annotation else ""
        print(f"{verdict}: {clip.name}{suffix}")

    def _save_tp_annotation(self, clip: Path, ctx: ClipContext) -> str | None:
        event = derive_tp_event(
            ctx.overlay,
            event_type=TP_EVENT_TYPE,
            video_fps=ctx.video_fps,
            frame_count=ctx.frame_count,
        )
        payload = build_annotation_payload(
            videoname=clip.name,
            frame_width=ctx.width,
            frame_height=ctx.height,
            fps=ctx.video_fps,
            events=[event],
        )
        out = annotation_path_for(clip)
        try:
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"WARN: could not write {out.name}: {exc}")
            return None
        return out.name

    def _save_empty_annotation(self, clip: Path, ctx: ClipContext) -> str | None:
        payload = build_annotation_payload(
            videoname=clip.name,
            frame_width=ctx.width,
            frame_height=ctx.height,
            fps=ctx.video_fps,
            events=[],
        )
        out = annotation_path_for(clip)
        try:
            out.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"WARN: could not write {out.name}: {exc}")
            return None
        return out.name

    def _persist(self) -> None:
        try:
            save_results(self.results_path, self.folder, self.results)
        except OSError as exc:
            print(f"WARN: could not save results: {exc}")


def main() -> None:
    args = parse_args()
    folder = resolve_folder(args.folder)
    if folder is None:
        raise SystemExit("No folder selected.")
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    clips = find_clips(folder)
    if not clips:
        raise SystemExit(
            f"No unannotated <clip>.mp4 + overlay pairs found in {folder}"
        )

    configure_qt_font_dir()
    cv2 = import_cv2()
    reviewer = FPReviewer(cv2, folder, clips, args.speed)
    print(
        f"Loaded {len(clips)} unannotated clip(s). "
        f"Starting at #{reviewer.index + 1}."
    )
    print_controls()
    reviewer.run()
    print("Done.")


if __name__ == "__main__":
    main()
