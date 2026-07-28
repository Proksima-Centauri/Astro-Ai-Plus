# Astro Ai Plus

Desktop aplikatsiia dlia obrobky astrofotohrafii na Python (PyQt5 + OpenCV + NumPy).

## Osnovni mozhlyvosti

- Formaty zobrazhen: PNG, JPG, TIFF, FITS.
- Instrumenty: Levels, Curves, Histogram, GHS, Blur, Crop, Rotate.
- Astro-funktsii: analiz (FWHM/SNR), plate solving, StarNet++, deepSNR, mosaic, 3D FLY.

## Vstanovlennia

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Zapusk

```bash
python3 "Astro Ai Plus.py"
```

## Vazhlyvi faily

- `Astro Ai Plus.py` - holovnyi zastosunok.
- `deep_sky_catalog.py` - lokalnyi kataloh DSO.
- `3d_fly_help.md` - instruktsiia 3D FLY.
