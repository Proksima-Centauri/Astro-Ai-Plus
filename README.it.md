# Astro Ai Plus

Applicazione desktop per l'astrofotografia sviluppata in Python (PyQt5 + OpenCV + NumPy).

## Funzionalita principali

- Formati immagine: PNG, JPG, TIFF, FITS.
- Strumenti: Levels, Curves, Histogram, GHS, Blur, Crop, Rotate.
- Moduli astro: analisi (FWHM/SNR), plate solving, StarNet++, deepSNR, mosaic, 3D FLY.

## Installazione

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Avvio

```bash
python3 "Astro Ai Plus.py"
```

## File importanti

- `Astro Ai Plus.py` - applicazione principale.
- `deep_sky_catalog.py` - catalogo DSO locale.
- `3d_fly_help.md` - guida 3D FLY.
