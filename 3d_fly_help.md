# 3D FLY - instrukcja obslugi

Ten dokument jest baza wiedzy dla uzytkownika i dla asystenta Altair.
Opisuje, co robi kazdy etap filtra 3D FLY i jak uzyskac poprawny render.

## Szybki workflow

1. Etap 1 `usun gwiazdy` (opcjonalnie): uruchom StarNet++, jesli chcesz czyste tlo bez gwiazd.
2. Etap 2 `zaznacz sekcje`: potnij obraz na warstwy (`Wytnij`) albo `Wczytaj warstwy` z dysku.
3. Etap 3 `wygladzanie krawedzi`: wybierz warstwe, ustaw blur, kliknij `Zastosuj`.
4. Etap 4 `laczenie w klip`: ustaw czas i FPS.
5. Etap 5/6 `definiowanie ruchu i pozycji`: ustaw ruch warstw i zoom.
6. Ostatni etap `dodaj muzyke`: ustaw audio i plik wyjsciowy, potem `Renderuj 3D FLY`.

## Etap 1 - usun gwiazdy

- `Usun gwiazdy`: uruchamia StarNet++ dla obrazu zrodlowego.
- `Zaawansowane ustawienia` -> `Stride`:
  - tylko wartosci parzyste,
  - zakres 2-512,
  - wyzsze wartosci sa szybsze, ale moga dac gorsza jakosc.

## Etap 2 - zaznacz sekcje

- Narzedzia malowania stref:
  - LPM: malowanie,
  - PPM: gumka,
  - `Wytnij (Ctrl+X)`: zapisuje aktualna strefe jako PNG z alfa.
- `Wczytaj warstwy`: import gotowych pociętych warstw (PNG/TIF/itd).
- `Akceptowalny rozmiar dziury (px)`:
  - to zakres pola dziur, ktore maja byc automatycznie domykane.

Uwaga: panel `Wyciete sekcje (PNG)` jest widoczny tylko w etapie 2.

## Etap 3 - wygladzanie krawedzi

- Podglad sekcji jest per warstwa (nie caly obraz).
- `Blur strefy` jest ustawiany indywidualnie dla wybranej warstwy.
- `Zastosuj`: zapisuje blur tylko dla aktualnej warstwy.
- `Zastosuj dla wszystkich`: kopiuje aktualna wartosc blur na wszystkie warstwy.

## Etap 4 - laczenie w klip

- `Czas (s)`: dlugosc finalnego klipu.
- `FPS`: liczba klatek na sekunde.

## Etap 5/6 - definiowanie ruchu i pozycji

- `Main zoom speed`: globalny zoom calej sceny.
- Pod podgladem sa parametry per warstwa:
  - `Warstwa`: wybor warstwy,
  - `Kierunek`:
    - `Strzalka 2D`: warstwa porusza sie w XY,
    - `Do widza (zoom)`: brak ruchu XY, tylko zoom warstwy,
  - `Predkosc`: szybkosc ruchu XY,
  - `Predkosc zoomu`: zoom konkretnej warstwy.

Sterowanie podgladem:
- klik + przeciaganie: zmiana kierunku,
- rolka myszy nad warstwa: zmiana predkosci.

Wazne:
- gdy kierunek = `Do widza (zoom)`, strzalka nie jest rysowana,
- warstwa wtedy nie idzie w lewo/prawo/gora/dol, tylko zoomuje.

## Ostatni etap - dodaj muzyke i render

- Dodaj muzyke z bazy lub importuj z dysku.
- Ustaw start i glosnosc klipu audio.
- Wybierz plik wyjsciowy (`Plik...`).
- `Renderuj 3D FLY` dziala tylko w ostatniej zakladce.

## Najczestsze problemy

- Komunikat `Brak warstw do ruchu`:
  - dodaj warstwy w etapie 2 (`Wytnij` lub `Wczytaj warstwy`).
- Blur nie widoczny:
  - wybierz warstwe w etapie 3 i kliknij `Zastosuj`.
- Brak renderu po kliknieciu:
  - przejdz do ostatniej zakladki i sprawdz sciezke pliku wyjsciowego.
