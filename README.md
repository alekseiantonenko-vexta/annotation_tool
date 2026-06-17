# CCTV Event Annotator

Small standalone GUI for marking event intervals and bounding boxes on CCTV
videos. It uses OpenCV for decoding frames and tkinter for buttons, text fields,
the timeline, and the event list.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Ubuntu you may also need tkinter and OpenCV runtime packages:

```bash
sudo apt-get install -y python3-tk ffmpeg libgl1 libglib2.0-0
```

## Run

```bash
python event_annotator.py
```

Then press `Open Video` and choose a file in the dialog. After you finish a
video, press `Open Video` again to choose the next one. The dialog opens in the
directory of the last selected video, so repeated annotation of one folder is
fast.

By default the tool reads/writes annotation JSON next to each selected video as:

```text
<video file name>_event-annotation.json
```

For example:

```text
camera_video.mp4_event-annotation.json
```

## Workflow

1. Drag a rectangle on the video.
2. Type an event name, for example `fighting`, `smoking`, `fire`, or `phone_call`.
3. Press Enter or `Start Event`; the event starts at the current frame.
4. Move/play the video until the event ends.
5. Select the active event and press `End Event`.
6. Press `Open Video` to continue with the next file, or `Quit` to exit.

The proper `<video>_event-annotation.json` is written only when you **Save**
(`Ctrl+S` or the `Save` button), or when you choose **Save** on the prompt shown
if you quit with unsaved changes.

While you edit, in-progress work is continuously auto-saved to a hidden recovery
cache next to the video (`.<video>.recovery.json`), so an accidental quit or
crash won't lose it. Reopening a video that has cached unsaved work prompts you
to recover it; saving (or quitting cleanly without unsaved changes) clears the
cache.

## Controls

```text
Space              Play / pause
1 / 2 / 4 / 8      Set playback speed
Left / Right       Move one frame backward / forward
Mouse drag video   Draw a new event bounding box
Enter              Start event from drawn box and type field
End Event button   Finish selected active event at current frame
Delete             Delete selected event
Ctrl+S             Save JSON
Ctrl+O             Open video
q or Esc           Quit
```

The JSON format is:

```json
{
  "videoname": "video.mp4",
  "frame_width": 1920,
  "frame_height": 1080,
  "fps": 25.0,
  "events": [
    {
      "type": "fighting",
      "start": 123,
      "end": 456,
      "bbox": [0.25, 0.1, 0.55, 0.75]
    }
  ]
}
```

## FP Review tool (`fp_review.py`)

A separate triage tool for reviewing detector output as true/false positives.
It pairs each `<clip>.mp4` with its sibling `<clip>.mp4.overlay.json` (detector
boxes), skips clips that already have `<clip>_event-annotation.json`, plays each
remaining clip in a loop at high speed with the boxes drawn on top, and captures
a verdict per clip.

```bash
python3 fp_review.py /path/to/clips_folder      # or omit the path for a folder picker
python3 fp_review.py /path/to/clips_folder --speed 16
```

Keys:

```text
1        Mark TRUE positive (writes <clip>_event-annotation.json) and advance
Space    Mark FALSE positive and advance
Frame bar Drag to seek within the current clip
n / b    Next / back without a verdict
r        Replay current clip from start
t        Toggle playback speed (x8 / x16)
q / Esc  Quit
```

Behavior:

- **Overlay sync** — overlay frames are time-indexed, so boxes are matched to
  each video frame by timestamp (not by frame index). Box color follows the
  `tone` field (`danger` = red, `warning` = amber).
- **Seeking** — the OpenCV window has a `Frame` slider for clip boundaries and
  manual rewinding/fast-forwarding. The current clip/frame counter is shown in
  a status strip below the video frame, not drawn over the frame itself.
- **TP** — a `<clip>_event-annotation.json` annotation is written in the same
  schema as above. A single `zone` event is derived from the
  `restricted zone crossing` detection: its active frame range and the union of
  its boxes (normalized). These files open directly in the annotator.
- **FP** — a `<clip>_event-annotation.json` annotation is written with an empty
  `events` list.
- **Resume / audit** — every verdict is recorded in `fp_review_results.json` in
  the folder. On launch, clips with existing annotation files are skipped.
