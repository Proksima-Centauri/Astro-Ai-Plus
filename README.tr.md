# Astro Ai Plus

Python ile yazilmis masaustu astrofotografi duzenleme uygulamasi (PyQt5 + OpenCV + NumPy).

## Temel ozellikler

- Goruntu formatlari: PNG, JPG, TIFF, FITS.
- Araclar: Levels, Curves, Histogram, GHS, Blur, Crop, Rotate.
- Astro modulleri: analiz (FWHM/SNR), plate solving, StarNet++, deepSNR, mosaic, 3D FLY.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Calistirma

```bash
python3 "Astro Ai Plus.py"
```

## Onemli dosyalar

- `Astro Ai Plus.py` - ana uygulama.
- `deep_sky_catalog.py` - yerel DSO katalogu.
- `3d_fly_help.md` - 3D FLY kilavuzu.
