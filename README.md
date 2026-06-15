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
<video file name>.json
```

For example:

```text
camera_video.mp4.json
```

## Workflow

1. Drag a rectangle on the video.
2. Type an event name, for example `fighting`, `smoking`, `fire`, or `phone_call`.
3. Press Enter or `Start Event`; the event starts at the current frame.
4. Move/play the video until the event ends.
5. Select the active event and press `End Event`.
6. Press `Open Video` to continue with the next file, or `Quit` to exit.

The JSON is saved immediately when an event is ended, and also after edits or
deletes of existing events.

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
