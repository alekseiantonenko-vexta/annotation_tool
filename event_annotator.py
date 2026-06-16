#!/usr/bin/env python3
"""Interactive CCTV event annotation tool.

The GUI uses tkinter for controls and OpenCV for video decoding/drawing.
Videos are opened from the GUI, and annotations are written next to each video.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APP_TITLE = "Vexta Event Annotator"
DEFAULT_MAX_DISPLAY_WIDTH = 1280
DEFAULT_MAX_DISPLAY_HEIGHT = 720
MIN_BOX_SIZE = 3
PLAYBACK_SPEEDS = (1, 2, 4, 8)
EVENT_TYPES = ("fighting", "fire", "smoke", "zone", "throwing")
AUTOSAVE_DELAY_MS = 1000

Color = tuple[int, int, int]
PixelBox = tuple[float, float, float, float]
RelBox = tuple[float, float, float, float]

EVENT_COLORS: tuple[Color, ...] = (
    (30, 180, 255),
    (80, 220, 80),
    (255, 160, 70),
    (220, 110, 255),
    (70, 220, 220),
    (255, 120, 120),
)
SELECTED_COLOR: Color = (0, 255, 255)
DRAFT_COLOR: Color = (255, 255, 255)
ACTIVE_COLOR: Color = (80, 255, 80)
TEXT_COLOR: Color = (245, 245, 245)
TEXT_SHADOW: Color = (20, 20, 20)

tk: Any = None
ttk: Any = None
messagebox: Any = None
filedialog: Any = None


@dataclass
class EventAnnotation:
    event_type: str
    start: int
    end: int | None
    bbox: RelBox

    @property
    def active(self) -> bool:
        return self.end is None


def build_annotation_payload(
    *,
    videoname: str,
    frame_width: int,
    frame_height: int,
    fps: float,
    events: list[EventAnnotation],
) -> dict[str, Any]:
    """Serialize events into the canonical annotation JSON schema.

    Shared by the annotator's Save and the FP-review TP export so both
    write the identical on-disk format.
    """
    return {
        "videoname": videoname,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "events": [
            {
                "type": event.event_type,
                "start": event.start,
                "end": event.end,
                "bbox": [round(value, 6) for value in event.bbox],
            }
            for event in events
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create frame-range event annotations for a video.",
    )
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=DEFAULT_MAX_DISPLAY_WIDTH,
        help=f"Maximum displayed video width. Default: {DEFAULT_MAX_DISPLAY_WIDTH}.",
    )
    parser.add_argument(
        "--max-display-height",
        type=int,
        default=DEFAULT_MAX_DISPLAY_HEIGHT,
        help=f"Maximum displayed video height. Default: {DEFAULT_MAX_DISPLAY_HEIGHT}.",
    )
    return parser.parse_args()


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is not installed. Install dependencies with:\n"
            "  pip install -r requirements.txt"
        ) from exc
    return cv2


def import_tkinter() -> None:
    global filedialog, messagebox, tk, ttk
    try:
        import tkinter as tk_module
        from tkinter import filedialog as filedialog_module
        from tkinter import messagebox as messagebox_module
        from tkinter import ttk as ttk_module
    except ImportError as exc:
        raise SystemExit(
            "tkinter is not installed for this Python. On Ubuntu install it with:\n"
            "  sudo apt-get install -y python3-tk"
        ) from exc
    tk = tk_module
    ttk = ttk_module
    messagebox = messagebox_module
    filedialog = filedialog_module


def annotation_path_for(video_path: Path) -> Path:
    return video_path.with_name(video_path.name + ".json")


def recovery_path_for(video_path: Path) -> Path:
    """Hidden sidecar holding unsaved in-progress work for crash/quit recovery."""
    return video_path.with_name(f".{video_path.name}.recovery.json")


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def clamp_frame(index: int, frame_count: int) -> int:
    if frame_count <= 0:
        return max(0, index)
    return max(0, min(index, frame_count - 1))


def normalize_box(box: PixelBox, width: int, height: int) -> RelBox:
    left, top, right, bottom = ordered_box(box)
    return (
        round(clamp(left / width, 0.0, 1.0), 6),
        round(clamp(top / height, 0.0, 1.0), 6),
        round(clamp(right / width, 0.0, 1.0), 6),
        round(clamp(bottom / height, 0.0, 1.0), 6),
    )


def denormalize_box(box: RelBox, width: int, height: int) -> PixelBox:
    left, top, right, bottom = box
    return (
        clamp(left, 0.0, 1.0) * width,
        clamp(top, 0.0, 1.0) * height,
        clamp(right, 0.0, 1.0) * width,
        clamp(bottom, 0.0, 1.0) * height,
    )


def ordered_box(box: PixelBox) -> PixelBox:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def box_is_big_enough(box: PixelBox) -> bool:
    left, top, right, bottom = ordered_box(box)
    return right - left >= MIN_BOX_SIZE and bottom - top >= MIN_BOX_SIZE


def parse_bbox(value: Any) -> RelBox | None:
    if isinstance(value, list) and len(value) == 4:
        try:
            return tuple(float(item) for item in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        keys = ("left", "top", "right", "bottom")
        try:
            return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def print_controls() -> None:
    print(
        "\nControls:\n"
        "  Space              Play / pause\n"
        "  1 / 2 / 4 / 8      Set playback speed\n"
        "  Left / Right       Move one frame backward / forward\n"
        "  Mouse drag video   Draw a new event bounding box\n"
        "  Enter              Start event from drawn box and type field\n"
        "  End Event button   Finish selected active event at current frame\n"
        "  Delete             Delete selected event\n"
        "  Ctrl+S             Save JSON\n"
        "  Ctrl+O             Open video\n"
        "  q or Esc           Quit\n",
        flush=True,
    )


class EventAnnotator:
    def __init__(
        self,
        *,
        cv2: Any,
        max_display_width: int,
        max_display_height: int,
    ) -> None:
        self.cv2 = cv2
        self.video_path: Path | None = None
        self.output_path: Path | None = None
        self.recovery_path: Path | None = None
        self.dirty = False
        self.autosave_after_id: str | None = None
        self.capture: Any | None = None
        self.max_display_width = max_display_width
        self.max_display_height = max_display_height
        self.last_video_dir = Path.cwd()

        self.fps = 25.0
        self.frame_count = 1
        self.frame_width = max_display_width
        self.frame_height = max_display_height
        self.current_frame: Any | None = None
        self.current_index = 0
        self.scale = 1.0
        self.display_width = max_display_width
        self.display_height = max_display_height

        self.events: list[EventAnnotation] = []
        self.selected_event: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.draft_box: PixelBox | None = None
        self.playing = False
        self.playback_speed = 1
        self.slider_is_updating = False
        self.photo: tk.PhotoImage | None = None
        self.last_status = "Open a video to start annotation."

        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()
        self._bind_keys()
        self._draw_empty_canvas()
        self._set_status(self.last_status)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            left,
            width=self.display_width,
            height=self.display_height,
            background="#111111",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        timeline = ttk.Frame(left)
        timeline.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        timeline.columnconfigure(4, weight=1)

        self.open_button = ttk.Button(timeline, text="Open Video", command=self.open_video_dialog)
        self.open_button.grid(row=0, column=0, padx=(0, 8))
        self.play_button = ttk.Button(timeline, text="Play", command=self.toggle_play)
        self.play_button.grid(row=0, column=1, padx=(0, 8))
        ttk.Label(timeline, text="Speed").grid(row=0, column=2, padx=(0, 4))
        self.speed_var = tk.StringVar(value=self._speed_label(self.playback_speed))
        self.speed_select = ttk.Combobox(
            timeline,
            textvariable=self.speed_var,
            values=[self._speed_label(speed) for speed in PLAYBACK_SPEEDS],
            state="readonly",
            width=4,
        )
        self.speed_select.grid(row=0, column=3, padx=(0, 8))
        self.speed_select.bind("<<ComboboxSelected>>", self._on_speed_selected)

        self.timeline_var = tk.IntVar(value=0)
        self.timeline = ttk.Scale(
            timeline,
            from_=0,
            to=max(0, self.frame_count - 1),
            orient="horizontal",
            command=self._on_timeline_changed,
        )
        self.timeline.grid(row=0, column=4, sticky="ew")

        self.frame_label = ttk.Label(timeline, width=26, anchor="e")
        self.frame_label.grid(row=0, column=5, padx=(8, 0))

        side = ttk.Frame(self.root, padding=(0, 8, 8, 8))
        side.grid(row=0, column=1, sticky="ns")
        side.columnconfigure(0, weight=1)

        type_row = ttk.Frame(side)
        type_row.grid(row=0, column=0, sticky="ew")
        type_row.columnconfigure(1, weight=1)
        ttk.Label(type_row, text="Event type").grid(row=0, column=0, sticky="w")
        self.type_var = tk.StringVar(value=EVENT_TYPES[0])
        self.type_entry = ttk.Combobox(
            type_row,
            textvariable=self.type_var,
            values=list(EVENT_TYPES),
            state="readonly",
            width=24,
        )
        self.type_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        buttons = ttk.Frame(side)
        buttons.grid(row=1, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        ttk.Button(buttons, text="Start Event", command=self.start_event).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(buttons, text="End Event", command=self.end_event).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )

        edit = ttk.LabelFrame(side, text="Selected event", padding=8)
        edit.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        for column in range(2):
            edit.columnconfigure(column, weight=1)

        ttk.Label(edit, text="Start frame").grid(row=0, column=0, sticky="w")
        ttk.Label(edit, text="End frame").grid(row=0, column=1, sticky="w")
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        max_frame = max(0, self.frame_count - 1)
        self.start_spin = ttk.Spinbox(
            edit,
            from_=0,
            to=max_frame,
            textvariable=self.start_var,
            width=10,
        )
        self.start_spin.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(2, 8))
        self.end_spin = ttk.Spinbox(
            edit,
            from_=0,
            to=max_frame,
            textvariable=self.end_var,
            width=10,
        )
        self.end_spin.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(2, 8))

        ttk.Button(edit, text="Update Selected", command=self.update_selected_event).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(edit, text="Delete Selected", command=self.delete_selected_event).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(4, 0),
        )

        list_frame = ttk.LabelFrame(side, text="Events", padding=8)
        list_frame.grid(row=3, column=0, sticky="nsew")
        side.rowconfigure(3, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.event_list = tk.Listbox(list_frame, width=52, height=24, exportselection=False)
        self.event_list.grid(row=0, column=0, sticky="nsew")
        self.event_list.bind("<<ListboxSelect>>", self._on_event_selected)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.event_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.event_list.configure(yscrollcommand=scrollbar.set)

        footer = ttk.Frame(side)
        footer.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="Open Video", command=self.open_video_dialog).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 4),
        )
        ttk.Button(footer, text="Save", command=self.save_annotations).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=4,
        )
        ttk.Button(footer, text="Quit", command=self.close).grid(row=0, column=2, padx=(4, 0))

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.root.bind_all("<space>", self._handle_space)
        self.root.bind_all("<Left>", lambda _event: self.step_frame(-1))
        self.root.bind_all("<Right>", lambda _event: self.step_frame(1))
        self.root.bind_all("<Return>", lambda _event: self.start_event())
        self.root.bind_all("<Delete>", self._handle_delete)
        self.root.bind_all("<Control-s>", lambda _event: self.save_annotations())
        self.root.bind_all("<Control-o>", lambda _event: self.open_video_dialog())
        self.root.bind_all("<Escape>", lambda _event: self.close())
        for speed in PLAYBACK_SPEEDS:
            self.root.bind_all(str(speed), lambda _event, value=speed: self._handle_speed_key(value))
        self.root.bind_all("q", self._handle_q)

    def _focused_text_widget(self) -> bool:
        focused = self.root.focus_get()
        if focused is None:
            return False
        return focused.winfo_class() in {"Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox"}

    def _handle_space(self, _event: tk.Event) -> str | None:
        if self._focused_text_widget():
            return None
        self.toggle_play()
        return "break"

    def _handle_delete(self, _event: tk.Event) -> str | None:
        if self._focused_text_widget():
            return None
        self.delete_selected_event()
        return "break"

    def _handle_q(self, _event: tk.Event) -> str | None:
        if self._focused_text_widget():
            return None
        self.close()
        return "break"

    def _handle_speed_key(self, speed: int) -> str | None:
        if self._focused_text_widget():
            return None
        self.set_playback_speed(speed)
        return "break"

    def run(self) -> None:
        print_controls()
        self.root.mainloop()

    def close(self) -> None:
        if self.dirty and self.video_path is not None:
            answer = messagebox.askyesnocancel(
                APP_TITLE,
                f"You have unsaved changes for {self.video_path.name}.\n"
                "Save before quitting?",
            )
            if answer is None:
                return  # Cancel: abort the quit.
            if answer:
                self.save_annotations(show_status=False)
            else:
                # Discard from the proper file, but keep the recovery cache
                # so the work can still be recovered next time.
                self._flush_autosave()
        else:
            self._cancel_autosave()
            self._delete_recovery_cache()
        if self.capture is not None:
            self.capture.release()
        self.root.destroy()

    def open_video_dialog(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open video",
            initialdir=str(self.last_video_dir),
            filetypes=(
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.mpeg *.mpg"),
                ("All files", "*"),
            ),
        )
        if not path:
            return
        self.load_video(Path(path).expanduser().resolve())

    def load_video(self, video_path: Path) -> None:
        if not video_path.exists():
            messagebox.showerror(APP_TITLE, f"Video file does not exist:\n{video_path}")
            return
        # Persist the previous video's in-progress work to its recovery cache
        # (the proper file is only written on explicit Save).
        self._flush_autosave()
        if self.playing:
            self.playing = False
            self.play_button.configure(text="Play")

        capture = self.cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            messagebox.showerror(APP_TITLE, f"Could not open video file:\n{video_path}")
            return
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            capture.release()
            messagebox.showerror(APP_TITLE, f"Could not read the first frame from:\n{video_path}")
            return

        if self.capture is not None:
            self.capture.release()
        self.video_path = video_path
        self.output_path = annotation_path_for(video_path)
        self.recovery_path = recovery_path_for(video_path)
        self.dirty = False
        self.capture = capture
        self.last_video_dir = video_path.parent
        self.fps = float(capture.get(self.cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            self.fps = 25.0
        self.frame_count = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.frame_count <= 0:
            self.frame_count = 1
        self.current_frame = first_frame
        self.current_index = 0
        self.frame_height, self.frame_width = first_frame.shape[:2]
        self.scale = min(
            1.0,
            self.max_display_width / float(self.frame_width),
            self.max_display_height / float(self.frame_height),
        )
        self.display_width = max(1, int(round(self.frame_width * self.scale)))
        self.display_height = max(1, int(round(self.frame_height * self.scale)))

        self.events = []
        self.selected_event = None
        self.drag_start = None
        self.draft_box = None
        self.canvas.configure(width=self.display_width, height=self.display_height)
        self.timeline.configure(to=max(0, self.frame_count - 1))
        self.start_spin.configure(to=max(0, self.frame_count - 1))
        self.end_spin.configure(to=max(0, self.frame_count - 1))
        self._update_title()
        self._clear_edit_fields()
        self._load_annotations_with_recovery()
        self._seek_to_frame(0)
        self._refresh_event_list()
        loaded_text = f" Loaded {len(self.events)} event(s)." if self.events else ""
        self._set_status(f"Opened {video_path.name}.{loaded_text} JSON: {self.output_path}")
        self.type_entry.focus_set()

    def toggle_play(self) -> None:
        if self.capture is None:
            self._set_status("Open a video first.")
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_next()

    def _play_next(self) -> None:
        if not self.playing:
            return
        if self.current_index >= self.frame_count - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self._seek_to_frame(self.current_index + self.playback_speed)
        delay_ms = max(1, int(round(1000.0 / self.fps)))
        self.root.after(delay_ms, self._play_next)

    def set_playback_speed(self, speed: int) -> None:
        if speed not in PLAYBACK_SPEEDS:
            return
        self.playback_speed = speed
        self.speed_var.set(self._speed_label(speed))
        self._set_status(f"Playback speed: {self._speed_label(speed)}.")

    def _on_speed_selected(self, _event: tk.Event) -> None:
        label = self.speed_var.get()
        try:
            speed = int(label.rstrip("x"))
        except ValueError:
            return
        self.set_playback_speed(speed)

    @staticmethod
    def _speed_label(speed: int) -> str:
        return f"{speed}x"

    def step_frame(self, delta: int) -> None:
        if self.capture is None:
            self._set_status("Open a video first.")
            return
        if self.playing:
            self.toggle_play()
        self._seek_to_frame(self.current_index + delta)

    def _on_timeline_changed(self, value: str) -> None:
        if self.slider_is_updating:
            return
        if self.playing:
            self.toggle_play()
        try:
            frame_index = int(round(float(value)))
        except ValueError:
            return
        self._seek_to_frame(frame_index)

    def _seek_to_frame(self, frame_index: int) -> None:
        if self.capture is None:
            return
        frame_index = clamp_frame(frame_index, self.frame_count)
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._set_status(f"Could not read frame {frame_index}.")
            return
        self.current_index = frame_index
        self.current_frame = frame
        self._render_frame()
        self._update_frame_label()

    def _update_frame_label(self) -> None:
        seconds = self.current_index / self.fps if self.fps > 0 else 0.0
        total = max(0, self.frame_count - 1)
        self.frame_label.configure(
            text=f"frame {self.current_index}/{total}  {seconds:0.2f}s  {self.fps:0.2f} fps",
        )
        self.slider_is_updating = True
        self.timeline.set(self.current_index)
        self.slider_is_updating = False

    def _render_frame(self) -> None:
        if self.current_frame is None:
            self._draw_empty_canvas()
            return
        frame = self.current_frame.copy()
        overlay = frame.copy()
        for index, event in enumerate(self.events):
            if not self._event_visible_at_current_frame(event):
                continue
            color = self._event_color(index, event)
            box = denormalize_box(event.bbox, self.frame_width, self.frame_height)
            left, top, right, bottom = self._int_box(box)
            self.cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
            thickness = 3 if index == self.selected_event else 2
            self.cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            label = f"{index + 1}: {event.event_type}"
            if event.active:
                label += " ACTIVE"
            self._draw_label(frame, label, left, max(18, top - 6), color)

        if self.events:
            frame = self.cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

        if self.draft_box is not None and box_is_big_enough(self.draft_box):
            left, top, right, bottom = self._int_box(self.draft_box)
            self.cv2.rectangle(frame, (left, top), (right, bottom), DRAFT_COLOR, 2)
            self._draw_label(frame, "new bbox", left, max(18, top - 6), DRAFT_COLOR)

        if self.scale != 1.0:
            frame = self.cv2.resize(
                frame,
                (self.display_width, self.display_height),
                interpolation=self.cv2.INTER_AREA,
            )
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        header = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255\n".encode("ascii")
        self.photo = tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def _draw_empty_canvas(self) -> None:
        self.canvas.delete("all")
        self.canvas.configure(width=self.display_width, height=self.display_height)
        self.canvas.create_text(
            self.display_width // 2,
            self.display_height // 2,
            text="Open Video",
            fill="#dddddd",
            font=("TkDefaultFont", 24, "bold"),
        )
        self.frame_label.configure(text="no video")

    def _event_visible_at_current_frame(self, event: EventAnnotation) -> bool:
        if event.end is None:
            return self.current_index >= event.start
        return event.start <= self.current_index <= event.end

    def _event_color(self, index: int, event: EventAnnotation) -> Color:
        if index == self.selected_event:
            return SELECTED_COLOR
        if event.active:
            return ACTIVE_COLOR
        return EVENT_COLORS[index % len(EVENT_COLORS)]

    def _draw_label(self, frame: Any, text: str, x: int, y: int, color: Color) -> None:
        self.cv2.putText(
            frame,
            text,
            (x + 1, y + 1),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_SHADOW,
            3,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            text,
            (x, y),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            self.cv2.LINE_AA,
        )

    def _on_mouse_down(self, event: tk.Event) -> None:
        if self.current_frame is None:
            self._set_status("Open a video first.")
            return
        self.drag_start = self._canvas_to_frame(event.x, event.y)
        self.draft_box = None

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        current = self._canvas_to_frame(event.x, event.y)
        self.draft_box = (*self.drag_start, *current)
        self._render_frame()

    def _on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        current = self._canvas_to_frame(event.x, event.y)
        self.draft_box = (*self.drag_start, *current)
        self.drag_start = None
        if not box_is_big_enough(self.draft_box):
            self.draft_box = None
            self._set_status("Bounding box is too small. Drag a larger rectangle.")
        else:
            self._set_status("Box ready. Enter event type and press Enter or Start Event.")
        self._render_frame()

    def _canvas_to_frame(self, x: int, y: int) -> tuple[float, float]:
        return (
            clamp(x / self.scale, 0.0, float(self.frame_width - 1)),
            clamp(y / self.scale, 0.0, float(self.frame_height - 1)),
        )

    def start_event(self) -> None:
        if self.current_frame is None:
            self._set_status("Open a video first.")
            return
        event_type = self.type_var.get().strip()
        if not event_type:
            self._set_status("Enter event type first, for example: fighting.")
            self.type_entry.focus_set()
            return
        if self.draft_box is None or not box_is_big_enough(self.draft_box):
            self._set_status("Draw a bounding box on the video first.")
            return
        event = EventAnnotation(
            event_type=event_type,
            start=self.current_index,
            end=None,
            bbox=normalize_box(self.draft_box, self.frame_width, self.frame_height),
        )
        self.events.append(event)
        self.selected_event = len(self.events) - 1
        self._populate_edit_fields(event)
        self.draft_box = None
        self._refresh_event_list()
        self._mark_dirty()
        self._set_status(f"Started event {self.selected_event + 1}: {event_type}.")
        self._render_frame()

    def end_event(self) -> None:
        if self.current_frame is None:
            self._set_status("Open a video first.")
            return
        index = self._selected_or_last_active_event()
        if index is None:
            self._set_status("No active event selected or available to end.")
            return
        event = self.events[index]
        event.end = max(event.start, self.current_index)
        self.selected_event = index
        self._populate_edit_fields(event)
        self._refresh_event_list()
        self._mark_dirty()
        self._set_status(f"Ended event {index + 1} at frame {event.end}.")
        self._render_frame()

    def update_selected_event(self) -> None:
        if self.current_frame is None:
            self._set_status("Open a video first.")
            return
        if self.selected_event is None:
            self._set_status("Select an event to update.")
            return
        event = self.events[self.selected_event]
        event_type = self.type_var.get().strip()
        if not event_type:
            self._set_status("Event type cannot be empty.")
            return
        try:
            start = int(self.start_var.get())
            end_text = self.end_var.get().strip()
            end = None if end_text == "" else int(end_text)
        except ValueError:
            self._set_status("Start/end must be frame numbers. Leave end empty for active event.")
            return
        start = clamp_frame(start, self.frame_count)
        if end is not None:
            end = max(start, clamp_frame(end, self.frame_count))
        event.event_type = event_type
        event.start = start
        event.end = end
        if self.draft_box is not None and box_is_big_enough(self.draft_box):
            event.bbox = normalize_box(self.draft_box, self.frame_width, self.frame_height)
            self.draft_box = None
        self._populate_edit_fields(event)
        self._refresh_event_list()
        self._mark_dirty()
        self._set_status(f"Updated event {self.selected_event + 1}.")
        self._render_frame()

    def delete_selected_event(self) -> None:
        if self.current_frame is None:
            self._set_status("Open a video first.")
            return
        if self.selected_event is None:
            self._set_status("Select an event to delete.")
            return
        deleted = self.selected_event
        del self.events[deleted]
        if not self.events:
            self.selected_event = None
            self._clear_edit_fields()
        else:
            self.selected_event = min(deleted, len(self.events) - 1)
            self._populate_edit_fields(self.events[self.selected_event])
        self._refresh_event_list()
        self._mark_dirty()
        self._set_status(f"Deleted event {deleted + 1}.")
        self._render_frame()

    def _selected_or_last_active_event(self) -> int | None:
        if self.selected_event is not None and self.events[self.selected_event].active:
            return self.selected_event
        for index in range(len(self.events) - 1, -1, -1):
            if self.events[index].active:
                return index
        return None

    def _on_event_selected(self, _event: tk.Event) -> None:
        selection = self.event_list.curselection()
        if not selection:
            return
        self.selected_event = int(selection[0])
        event = self.events[self.selected_event]
        self._populate_edit_fields(event)
        self._seek_to_frame(event.start)
        self._set_status(f"Selected event {self.selected_event + 1}.")

    def _populate_edit_fields(self, event: EventAnnotation) -> None:
        self.type_var.set(event.event_type)
        self.start_var.set(str(event.start))
        self.end_var.set("" if event.end is None else str(event.end))

    def _clear_edit_fields(self) -> None:
        self.type_var.set(EVENT_TYPES[0])
        self.start_var.set("")
        self.end_var.set("")

    def _refresh_event_list(self) -> None:
        current_selection = self.selected_event
        self.event_list.delete(0, tk.END)
        for index, event in enumerate(self.events):
            state = "ACTIVE" if event.active else f"{event.start}-{event.end}"
            left, top, right, bottom = event.bbox
            self.event_list.insert(
                tk.END,
                (
                    f"{index + 1:03d}  {state:<14}  {event.event_type:<18}  "
                    f"[{left:.3f}, {top:.3f}, {right:.3f}, {bottom:.3f}]"
                ),
            )
        if current_selection is not None and current_selection < len(self.events):
            self.event_list.selection_set(current_selection)
            self.event_list.see(current_selection)

    def _events_from_payload(self, payload: Any) -> list[EventAnnotation]:
        """Parse and validate the events list from a saved/recovery payload."""
        loaded: list[EventAnnotation] = []
        if not isinstance(payload, dict):
            return loaded
        for raw_event in payload.get("events", []):
            if not isinstance(raw_event, dict):
                continue
            event_type = str(raw_event.get("type", "")).strip()
            bbox = parse_bbox(raw_event.get("bbox"))
            if not event_type or bbox is None:
                continue
            try:
                start = int(raw_event.get("start", 0))
                raw_end = raw_event.get("end")
                end = None if raw_end is None else int(raw_end)
            except (TypeError, ValueError):
                continue
            start = clamp_frame(start, self.frame_count)
            if end is not None:
                end = max(start, clamp_frame(end, self.frame_count))
            loaded.append(EventAnnotation(event_type=event_type, start=start, end=end, bbox=bbox))
        return loaded

    def _load_annotations_with_recovery(self) -> None:
        """Offer unsaved work from a prior session, else load the saved file."""
        if self._maybe_recover_from_cache():
            return
        self._load_existing_annotations()
        self.dirty = False
        self._update_title()

    def _maybe_recover_from_cache(self) -> bool:
        if self.recovery_path is None or not self.recovery_path.exists():
            return False
        try:
            payload = json.loads(self.recovery_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showwarning(APP_TITLE, f"Could not read recovery cache:\n{exc}")
            self._delete_recovery_cache()
            return False
        recovered = self._events_from_payload(payload)
        if not recovered:
            self._delete_recovery_cache()
            return False
        should_recover = messagebox.askyesno(
            APP_TITLE,
            "Unsaved work from a previous session was found for this video.\n"
            "Recover it?\n\n"
            "Choosing No discards the cached work and loads the saved file.",
        )
        if not should_recover:
            self._delete_recovery_cache()
            return False
        self.events = recovered
        self.selected_event = 0
        self._populate_edit_fields(self.events[0])
        self.dirty = True
        self._update_title()
        self._set_status(f"Recovered {len(self.events)} unsaved event(s).")
        return True

    def _load_existing_annotations(self) -> None:
        if self.output_path is None or not self.output_path.exists():
            return
        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showwarning(APP_TITLE, f"Could not load existing JSON:\n{exc}")
            return
        self.events = self._events_from_payload(payload)
        if self.events:
            self.selected_event = 0
            self._populate_edit_fields(self.events[0])
            self._set_status(f"Loaded {len(self.events)} event(s) from {self.output_path}.")

    def _build_payload(self) -> dict[str, Any]:
        return build_annotation_payload(
            videoname=self.video_path.name if self.video_path else "",
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            fps=self.fps,
            events=self.events,
        )

    def save_annotations(self, *, show_status: bool = True) -> None:
        if self.video_path is None or self.output_path is None:
            return
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(
                json.dumps(self._build_payload(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self._set_status(f"Could not save JSON: {exc}")
            return
        self.dirty = False
        self._cancel_autosave()
        self._delete_recovery_cache()
        self._update_title()
        if show_status:
            self._set_status(f"Saved {len(self.events)} event(s) to {self.output_path}.")

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()
        self._schedule_autosave()

    def _schedule_autosave(self) -> None:
        if self.autosave_after_id is not None:
            self.root.after_cancel(self.autosave_after_id)
        self.autosave_after_id = self.root.after(AUTOSAVE_DELAY_MS, self._write_recovery_cache)

    def _cancel_autosave(self) -> None:
        if self.autosave_after_id is not None:
            self.root.after_cancel(self.autosave_after_id)
            self.autosave_after_id = None

    def _flush_autosave(self) -> None:
        """Cancel any pending debounce and write the cache now if dirty."""
        self._cancel_autosave()
        if self.dirty:
            self._write_recovery_cache()

    def _write_recovery_cache(self) -> None:
        self.autosave_after_id = None
        if self.recovery_path is None or self.video_path is None:
            return
        try:
            self.recovery_path.write_text(
                json.dumps(self._build_payload(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            self._set_status(f"Could not write recovery cache: {exc}")

    def _delete_recovery_cache(self) -> None:
        if self.recovery_path is None:
            return
        try:
            self.recovery_path.unlink(missing_ok=True)
        except OSError as exc:
            self._set_status(f"Could not remove recovery cache: {exc}")

    def _update_title(self) -> None:
        if self.video_path is None:
            self.root.title(APP_TITLE)
            return
        prefix = "*" if self.dirty else ""
        self.root.title(f"{prefix}{APP_TITLE} - {self.video_path.name}")

    def _set_status(self, message: str) -> None:
        self.last_status = message
        self.status_var.set(message)
        print(message, flush=True)

    @staticmethod
    def _int_box(box: PixelBox) -> tuple[int, int, int, int]:
        left, top, right, bottom = ordered_box(box)
        return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))


def main() -> int:
    args = parse_args()
    import_tkinter()
    cv2 = import_cv2()
    app = EventAnnotator(
        cv2=cv2,
        max_display_width=args.max_display_width,
        max_display_height=args.max_display_height,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
