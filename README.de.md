# Astro Ai Plus

Desktop-Anwendung fuer Astrofotografie-Bearbeitung mit Python (PyQt5 + OpenCV + NumPy).

## Hauptfunktionen

- Bildformate: PNG, JPG, TIFF, FITS.
- Werkzeuge: Levels, Curves, Histogram, GHS, Blur, Crop, Rotate.
- Astro-Module: Analyse (FWHM/SNR), Plate Solving, StarNet++, deepSNR, Mosaic, 3D FLY.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start

```bash
python3 "Astro Ai Plus.py"
```

## Wichtige Dateien

- `Astro Ai Plus.py` - Hauptanwendung.
- `deep_sky_catalog.py` - lokaler DSO-Katalog.
- `3d_fly_help.md` - 3D FLY Anleitung.
