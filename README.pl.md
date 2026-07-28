# Astro Ai Plus

Desktopowa aplikacja do obrobki astrofotografii napisana w Pythonie (PyQt5 + OpenCV + NumPy).

Projekt laczy klasyczne narzedzia (Levels, Curves, histogram, blur, crop, rotate) z funkcjami astro: analiza gwiazd (FWHM/SNR), plate solving, usuwanie gwiazd (StarNet++), denoise (deepSNR), mozaika klatek i filtr animacyjny 3D FLY.

## Najwazniejsze funkcje

- Otwieranie i zapis obrazow: PNG, JPG, TIFF oraz FITS (`.fits`, `.fit`, `.fts`).
- Warstwy i historia zmian: undo/redo, miniatury krokow, podstawowe operacje warstw.
- Narzedzia tonalne i kolorystyczne: Levels, Curves (LUT), Histogram, GHS, korekcja RGB/HSL.
- Narzedzia astro:
  - analiza obrazu (m.in. FWHM, SNR, szum tla),
  - plate solving (lokalnie `solve-field` lub fallback do Astrometry.net),
  - StarNet++ (usuwanie gwiazd),
  - deepSNR (zewnetrzny denoise),
  - mozaika kadrow,
  - 3D FLY (render klipu z warstw, opcjonalnie z audio).
- Asystent AI Altair (Gemini) oraz sterowanie glosowe (opcjonalnie).
- Interfejs PL/EN, konfigurowalne preferencje i workspace.

## Wymagania

- Python 3.10+ (najlepiej 3.11-3.13).
- System Linux/Windows (na Linuxie aplikacja uruchamia software OpenGL).
- Pakiety z `requirements.txt`: `numpy`, `opencv-python`, `sep`, `astropy`, `photutils`, `SpeechRecognition`, `onnxruntime`, `PyQt5`, `pyserial`.

Opcjonalne narzedzia zewnetrzne:

- `solve-field` (Astrometry.net) do lokalnego plate solvingu.
- StarNet++ (CLI) do usuwania gwiazd.
- deepSNR (CLI).
- `ffmpeg` do dolaczania audio do klipu 3D FLY.

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uruchomienie

```bash
python3 "Astro Ai Plus.py"
```

## Konfiguracja

Aplikacja zapisuje ustawienia w pliku `config` (JSON, bez rozszerzenia) w katalogu projektu.

Przykladowe pola:

- sciezki do narzedzi (`starnet_path`, `deepsnr_path`),
- argumenty deepSNR (`deepsnr_args`),
- plate solving (`api_key`, `pixel_size_um`, `focal_length_mm`),
- ustawienia AI (`gemini_api_key`, `gemini_model`),
- jezyk, motyw, liczba rdzeni, workspace.

Uwaga bezpieczenstwa: plik konfiguracyjny moze zawierac klucze API. Nie publikuj go publicznie.

## Szybki start workflow

1. Otworz obraz (`Open`) lub przeciagnij plik do okna.
2. Wykonaj podstawowa korekcje (Levels/Curves/Histogram/Correction).
3. Uruchom `Analyze`, aby policzyc metryki (FWHM, SNR).
4. Opcjonalnie: `StarNet++`, `deepSNR`, `Mosaic`, `Plate Solve`.
5. Dla animacji uruchom `3D FLY`.
6. Zapisz wynik (`Save` / `Save As`).

## 3D FLY - skrocona instrukcja

1. Etap 1 `usun gwiazdy` (opcjonalnie): uruchom StarNet++.
2. Etap 2 `zaznacz sekcje`: potnij obraz na warstwy (`Wytnij`) albo uzyj `Wczytaj warstwy`.
3. Etap 3 `wygladzanie krawedzi`: wybierz warstwe, ustaw blur i kliknij `Zastosuj`.
4. Etap 4 `laczenie w klip`: ustaw czas klipu i FPS.
5. Etap 5/6 `definiowanie ruchu i pozycji`: ustaw kierunek, predkosc i zoom warstw.
6. Ostatni etap `dodaj muzyke`: wybierz audio i kliknij `Renderuj 3D FLY`.

Najczestsze problemy:

- `Brak warstw do ruchu` -> dodaj warstwy w etapie 2.
- Blur nie jest widoczny -> w etapie 3 wybierz warstwe i kliknij `Zastosuj`.
- Brak renderu -> przejdz do ostatniej zakladki i ustaw sciezke wyjsciowa.

## Przydatne komendy konsoli

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

## Struktura projektu

- `Astro Ai Plus.py` - glowny plik aplikacji.
- `deep_sky_catalog.py` - lokalny katalog obiektow DSO.
- `3d_fly_help.md` - instrukcja filtra 3D FLY.
- `requirements.txt` - zaleznosci Python.
- `assets/` - ikony i zasoby UI.
