# image-messrs

A local web app for image and video glitch art: content-aware seam carving,
still-image glitch effects (pixel sorting, channel shifting, block
displacement, byte corruption), cross-image blending, and video datamoshing.

## Setup

Requires [Homebrew](https://brew.sh) and `ffmpeg` (already on your PATH if
you're reading this after running the project setup). If you're starting
from scratch:

```bash
brew install python@3.12 ffmpeg
/usr/local/bin/python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Run

```bash
./venv/bin/python run.py
```

Opens on `http://localhost:5050` (override with the `PORT` env var).

## Test

```bash
./venv/bin/pytest
```

Tests marked `slow` shell out to `ffmpeg`/`ffprobe` and need them on PATH.

## Project layout

- `imagemessrs/` - the effects engine (pure Python/numpy/OpenCV, no Flask
  dependency, unit-testable on its own).
    - `core/` - shared types, the effect registry, image I/O helpers.
    - `effects/` - one module per still-image effect (`seam_carve`,
      `pixel_sort`, `channel_shift`, `block_displace`, `byte_corrupt`) plus
      `effects/blend/` for two-image effects (`optical_flow_blend`,
      `poisson_blend`, `seam_merge`, `energy_warp`). Each effect registers
      itself via `@register_effect(...)`; the web UI's controls are generated
      entirely from that registration, so adding a new effect module and
      importing it in `effects/__init__.py` is enough to make it show up.
    - `video/` - the ffmpeg wrapper, frame extraction/writing, and the three
      datamosh techniques (`frame_blend`, `feedback_loop`, `iframe_smear`).
- `webapp/` - the Flask app: routes, in-memory session stores, templates,
  and static JS/CSS. Thin adapter layer over `imagemessrs/`.
- `tests/` - one file per effect/module, plus fixtures in `conftest.py`.

## Notes on the video effects

`frame_blend` and `feedback_loop` are ordinary ffmpeg operations (frame
averaging, repeated re-encodes) and are reliable.

`iframe_smear` is a genuine bitstream-level datamosh: it transcodes both
clips to MPEG-4 Part 2 elementary streams, strips the second clip's I-frame,
and splices its P-frames directly after the first clip's stream - so the
decoder applies the second clip's motion vectors against the first clip's
last frame. It's marked experimental in the UI because results depend on
clip content and aren't guaranteed to decode identically in every player.
