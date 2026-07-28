# Astro Ai Plus

Application desktop de traitement d'astrophotographie en Python (PyQt5 + OpenCV + NumPy).

## Fonctions principales

- Formats image: PNG, JPG, TIFF, FITS.
- Outils: Levels, Curves, Histogram, GHS, Blur, Crop, Rotate.
- Fonctions astro: analyse (FWHM/SNR), plate solving, StarNet++, deepSNR, mosaic, 3D FLY.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement

```bash
python3 "Astro Ai Plus.py"
```

## Fichiers importants

- `Astro Ai Plus.py` - application principale.
- `deep_sky_catalog.py` - catalogue DSO local.
- `3d_fly_help.md` - guide 3D FLY.
