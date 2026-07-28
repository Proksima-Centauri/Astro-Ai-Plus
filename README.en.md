# Astro Ai Plus

Desktop astrophotography processing application built with Python (PyQt5 + OpenCV + NumPy).

The project combines classic editing tools (Levels, Curves, Histogram, Blur, Crop, Rotate) with astronomy-specific features: star analysis (FWHM/SNR), plate solving, star removal (StarNet++), denoise (deepSNR), frame mosaics, and the 3D FLY animation filter.

## Key features

- Image I/O: PNG, JPG, TIFF, and FITS (`.fits`, `.fit`, `.fts`).
- Layers and history: undo/redo, thumbnails, basic layer operations.
- Tone and color tools: Levels, Curves (LUT), Histogram, GHS, RGB/HSL correction.
- Astro tools:
  - image analysis (FWHM, SNR, background noise),
  - plate solving (`solve-field` locally or Astrometry.net fallback),
  - StarNet++ star removal,
  - deepSNR external denoise,
  - frame mosaic stitching,
  - 3D FLY clip rendering (optional audio).
- Altair AI assistant (Gemini) and optional speech input.
- PL/EN UI with configurable preferences and workspaces.

## Requirements

- Python 3.10+ (recommended 3.11-3.13).
- Linux or Windows (on Linux the app enables software OpenGL).
- Packages from `requirements.txt`: `numpy`, `opencv-python`, `sep`, `astropy`, `photutils`, `SpeechRecognition`, `onnxruntime`, `PyQt5`, `pyserial`.

Optional external tools:

- `solve-field` (Astrometry.net) for local plate solving.
- StarNet++ CLI.
- deepSNR CLI.
- `ffmpeg` for attaching audio in 3D FLY output.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 "Astro Ai Plus.py"
```

## Configuration

Settings are stored in `config` (JSON without extension) in the project root.

Example fields:

- tool paths (`starnet_path`, `deepsnr_path`),
- deepSNR args (`deepsnr_args`),
- plate solving (`api_key`, `pixel_size_um`, `focal_length_mm`),
- AI settings (`gemini_api_key`, `gemini_model`),
- language, theme, core count, workspace.

Security note: config may contain API keys. Do not publish it in public repositories.

## Quick workflow

1. Open an image (`Open`) or drag-and-drop a file.
2. Apply base corrections (Levels/Curves/Histogram/Correction).
3. Run `Analyze` to compute metrics (FWHM, SNR).
4. Optional: `StarNet++`, `deepSNR`, `Mosaic`, `Plate Solve`.
5. For animation, run `3D FLY`.
6. Save output (`Save` / `Save As`).

## 3D FLY quick guide

1. Stage 1 `remove stars` (optional): run StarNet++.
2. Stage 2 `mark sections`: cut layer zones (`Cut`) or use `Load layers`.
3. Stage 3 `edge smoothing`: select a layer, set blur, click `Apply`.
4. Stage 4 `clip setup`: set duration and FPS.
5. Stage 5/6 `motion and position`: set direction, speed, and zoom per layer.
6. Final stage `add music`: choose audio and click `Render 3D FLY`.

Common issues:

- `No layers for motion` -> add layers in stage 2.
- Blur not visible -> in stage 3 select layer and click `Apply`.
- No render output -> open the final tab and set a valid output path.

## Useful in-app console commands

- `help`
- `open [path]`
- `save` / `save as [path]`
- `magic`
- `starnet++`
- `deepsnr`
- `3d fly`
- `analyze`
- `mosaic`
- `levels`, `curves`, `histogram`, `ghs`

## Project structure

- `Astro Ai Plus.py` - main app file.
- `deep_sky_catalog.py` - offline deep-sky object catalog.
- `3d_fly_help.md` - detailed 3D FLY manual.
- `requirements.txt` - Python dependencies.c
- `assets/` - UI icons and resources.
