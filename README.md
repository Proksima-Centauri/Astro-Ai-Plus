# Astro Ai Plus

Desktop astrophotography processing application built with Python (PyQt5 + OpenCV + NumPy).

## Multilingual README

- Polski: `README.pl.md`
- English: `README.en.md`
- Espanol: `README.es.md`
- Deutsch: `README.de.md`
- Francais: `README.fr.md`
- Italiano: `README.it.md`
- Portugues: `README.pt.md`
- Nederlands: `README.nl.md`
- Cesky: `README.cs.md`
- Turkce: `README.tr.md`
- Ukrainska: `README.uk.md`
- Russkiy: `README.ru.md`
- Nihongo: `README.ja.md`

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
  - frame stacking with file list, Bayer pattern selection, and CFA-safe debayer,
  - animated stacking progress dialog with per-stage process visualization,
  - 3D FLY clip rendering (optional audio).
- AutoStretch tuned to a softer, PixInsight AutoSTF-like response.
- Altair AI assistant (Gemini) and optional speech input.

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
