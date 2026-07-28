# Astro Ai Plus

Desktopova aplikace pro zpracovani astrofotografie v Pythonu (PyQt5 + OpenCV + NumPy).

## Hlavni funkce

- Obrazove formaty: PNG, JPG, TIFF, FITS.
- Nastroje: Levels, Curves, Histogram, GHS, Blur, Crop, Rotate.
- Astro funkce: analyza (FWHM/SNR), plate solving, StarNet++, deepSNR, mosaic, 3D FLY.

## Instalace

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Spusteni

```bash
python3 "Astro Ai Plus.py"
```

## Dulezite soubory

- `Astro Ai Plus.py` - hlavni aplikace.
- `deep_sky_catalog.py` - lokalni DSO katalog.
- `3d_fly_help.md` - navod pro 3D FLY.
