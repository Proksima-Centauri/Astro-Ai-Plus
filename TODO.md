# TODO - plan uporzadkowania repozytorium

Lista zadan od najwazniejszego do najmniej pilnego. Kazde zadanie to dobra
okazja na jeden osobny commit - dzieki temu historia w gicie bedzie czytelna.

Zaznaczaj zrobione zadania, zmieniajac `[ ]` na `[x]`.

---

## [ ] 1. Usun klucze API z repozytorium (NAJPILNIEJSZE!)

**Dlaczego:** w plikach `config` i `astro_magic_config.json` sa zapisane Twoje
klucze API (`api_key` i `gemini_api_key`). Kazdy, kto wejdzie na Twojego
GitHuba, moze je skopiowac i uzywac ich NA TWOJ KOSZT. Klucz raz wrzucony do
gita zostaje w historii na zawsze, nawet jesli go potem skasujesz z pliku -
dlatego trzeba go uniewaznic u zrodla.

**Co zrobic:**

1. Wygeneruj NOWE klucze (stare uniewaznij):
   - Gemini: https://aistudio.google.com/apikey - skasuj stary klucz, zrob nowy
   - astrometry.net: w ustawieniach konta na https://nova.astrometry.net
2. Powiedz gitowi, zeby przestal sledzic pliki z konfiguracja
   (pliki zostana na dysku, znikna tylko z repo):

   ```
   git rm --cached config astro_magic_config.json
   git commit -m "Usun pliki konfiguracyjne z kluczami API z repo"
   git push
   ```

3. Wpisz nowe klucze do swojego lokalnego pliku `config` - od teraz git go
   ignoruje (jest wpisany w `.gitignore`), wiec klucze zostana tylko u Ciebie.

**Jak sprawdzic, ze dziala:** `git status` nie pokazuje `config`, nawet gdy
cos w nim zmienisz.

---

## [ ] 2. Wyrzuc katalog `__pycache__` z gita

**Dlaczego:** pliki `.pyc` to skompilowane smieci, ktore Python tworzy sam
przy kazdym uruchomieniu. Nie wrzuca sie ich do repo - kazdy komputer
generuje wlasne. W repo lezy nawet `.pyc` po pliku, ktorego juz nie ma
(`background_neutralization`).

**Co zrobic:**

```
git rm -r --cached __pycache__
git commit -m "Usun __pycache__ z repo"
git push
```

Plik `.gitignore` juz zawiera wpis `__pycache__/`, wiec te pliki nie wroca.

---

## [ ] 3. Napisz README.md

**Dlaczego:** README to pierwsza rzecz, jaka widzi kazdy na GitHubie. Bez
niego nikt (nawet Ty za pol roku) nie wie, co to za program i jak go odpalic.

**Co zrobic:** stworz plik `README.md` z sekcjami:

- Co to jest Astro Ai Plus (2-3 zdania + moze zrzut ekranu?)
- Jak zainstalowac:

  ```
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Jak uruchomic: `python "Astro Ai Plus.py"`
- Wymagany Python: napisz, ze potrzebny jest Python 3.14. Na starszych
  (np. 3.12) program sie nie uruchomi, bo adnotacja typu `ImageLayer`
  (okolice linii 4022) uzywa klasy zdefiniowanej dopiero nizej w pliku.
  Ciekawostka: dopisanie `from __future__ import annotations` na samej gorze
  pliku naprawia to takze dla starszych Pythonow.
- Co jest potrzebne opcjonalnie (StarNet++, klucz Gemini, klucz astrometry.net)

---

## [ ] 4. Dopisz wersje bibliotek w requirements.txt

**Dlaczego:** samo `numpy` znaczy "jakikolwiek numpy". Za rok nowa wersja
moze zepsuc program i nie bedzie wiadomo dlaczego. Zapis `numpy==2.3.1`
mowi dokladnie, ktora wersja na pewno dziala.

**Co zrobic:** przy aktywnym venv, w ktorym program dziala, sprawdz wersje
komenda `pip show numpy opencv-python` (i tak dalej dla kazdej biblioteki)
i dopisz je w `requirements.txt` w formacie `nazwa==wersja`.
Dopisz tez `pytest` - jest potrzebny do testow (zadanie 6).

---

## [ ] 5. Refactor: podziel wielki plik na mniejsze moduly

**Dlaczego:** "Astro Ai Plus.py" ma prawie 18 000 linii. W takim pliku
ciezko cokolwiek znalezc, a kazda zmiana grozi zepsuciem czegos obok.
Lepiej: jeden plik = jedna rzecz.

**Przyklad juz zrobiony - obejrzyj go:** klasa `ColorProcessor` zostala
przeniesiona do `processing/color_processor.py`. Zobacz commit z ta zmiana.
Schemat byl taki:

1. Znajdz klase, ktora NIE korzysta z Qt ani z reszty aplikacji
   (`ColorProcessor` uzywala tylko numpy i cv2 - idealna na poczatek).
2. Skopiuj ja do nowego pliku w katalogu `processing/` razem z jej importami.
3. W wielkim pliku usun klase i dodaj na gorze:
   `from processing.color_processor import ColorProcessor`
4. Uruchom program i sprawdz, ze wszystko dziala tak samo.
5. Napisz proste testy (patrz `tests/test_color_processor.py`).
6. Osobny commit: "Wyciagnij ColorProcessor do processing/".

**Dobre nastepne kandydatki (od najlatwiejszej):**

- [ ] funkcje `apply_levels`, `build_curve_lut`, `natural_cubic_spline`,
      `apply_curves_lut` -> `processing/levels_curves.py` (czysta matematyka)
- [ ] funkcje `star_shrink_pixinsight`, `neutralize_background`
      -> `processing/stars.py`
- [ ] funkcje `_compute_star_metrics_*` i `compute_image_analysis_metrics`
      -> `processing/star_metrics.py`
- [ ] funkcje `load_config`, `save_config` -> `app_config.py`

Rob JEDNA rzecz na commit. Po kazdej zmianie odpal program i testy.

---

## [ ] 6. Uruchamiaj testy i dopisuj nowe

**Dlaczego:** testy to automatyczny sprawdzacz, czy refactor niczego nie
zepsul. Zamiast klikac po calym programie, wpisujesz jedna komende.

**Jak uruchomic:**

```
source .venv/bin/activate
pytest
```

Powinno byc: `6 passed`. Pierwsze testy sa w `tests/test_color_processor.py` -
kazdy ma komentarz, co sprawdza. Gdy wyciagniesz kolejny modul (zadanie 5),
dopisz do niego chocby 2-3 testy na tej samej zasadzie.

---

## [ ] 7. Zmien nazwe glownego pliku (na koniec)

**Dlaczego:** spacje w nazwie pliku Pythona utrudniaja zycie - trzeba wszedzie
pisac cudzyslowy i nie da sie takiego pliku zaimportowac jako modul.

**Co zrobic:** uzyj `git mv`, zeby git wiedzial, ze to ten sam plik:

```
git mv "Astro Ai Plus.py" astro_ai_plus.py
git commit -m "Zmien nazwe glownego pliku na astro_ai_plus.py"
```

Pamietaj o poprawieniu nazwy w README i w konfiguracji VS Code
(`.vscode/launch.json`), jesli jest tam wpisana.

---

## Sciagawka gitowa na dzis

```
git status                  # co sie zmienilo?
git add <plik>              # dodaj plik do "paczki" na commit
git commit -m "Opis zmiany" # zapisz paczke w historii
git push                    # wyslij na GitHuba
git log --oneline           # pokaz historie commitow
git diff                    # pokaz dokladnie, co zmieniles
```

Zlota zasada: male commity, kazdy robi jedna rzecz, opis mowi CO i PO CO.
"Poprawki" to zly opis. "Usun __pycache__ z repo" to dobry opis.
