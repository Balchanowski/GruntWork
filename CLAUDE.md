# Grunt Work

Webowa platforma do analizy i statystyk meczów z **Warcraft II Remastered** (online). Hasło: *Track, review and improve your Warcraft II games: stats, timestamps and match breakdowns in one place.* AI to jeden z wielu feature'ów, nie cały projekt.

## O mnie i jak ze mną pracować

- Studiuję na AGH, na co dzień piszę w **TypeScripcie**, front to moja mocna strona.
- Pythona znam podstawowo. Backendu i serwerów chcę się nauczyć na tym projekcie.
- **Nie chcę vibe codingu.** Konfigi i boilerplate mogą być gotowe, ale z wyjaśnieniem, co robi każda linijka. **Logikę piszę sam**: Ty tłumaczysz, podpowiadasz i robisz review, ale nie dajesz gotowca.
- **Jeden krok naraz.** Nie zasypuj mnie planami na pięć etapów ani pojęciami, których nie znam. Krok działa → następny.
- **Solidnie, nie szybko.** Wolę przewidywalne rozwiązanie niż szybki efekt.

## Plan projektu (warstwy, każda działa sama)

1. **MVP – ekonomia z nagrania.** Co sekundę odczytać z górnego paska zasoby (złoto, drewno, olej, jedzenie np. `12/16`), zapisać do CSV/JSON, zrobić wykres. Pokazuje np. „od 4:30 do 6:00 trzymałeś 1500 złota”.
2. **Wykrywanie stanu gry (ML).** Model (np. YOLO) rozpoznaje jednostki i budynki → oś czasu armii z timestampami. Później.
3. **Warstwa „coacha”.** Porównanie z lepszymi grami, flagowanie momentów, ewentualnie podsumowanie przez LLM. Później.

Architektura docelowa: rdzeń w Pythonie (wideo → JSON) jako osobny moduł → backend FastAPI (upload, zadania w tle, status) → baza (SQLite, potem Postgres) → deploy (Docker). Front w TS: wykres + odtwarzacz wideo, klik w punkt przeskakuje do momentu w nagraniu. Front można robić równolegle na zmockowanym JSON-ie.

Znajomy streamer, który gra w WC2 na wysokim poziomie, może kiedyś pomóc (VOD-y, wiedza, testy). Najpierw działające MVP, dopiero potem do niego piszę.

## Decyzje techniczne

- **Windows**, bez dual boota. WSL2 dopiero przy backendzie i Dockerze.
- **uv** jako menedżer paczek. Projekt w src layout. Komendy odpalam z folderu `gruntWorkPython`:
  `uv run python -m gruntworkpython.<moduł>`
- **Odczyt liczb: template matching w OpenCV**, a nie OCR. Czcionka cyfr w grze jest stała, więc to najpewniejsze. EasyOCR i torch na razie nie są instalowane.
- W `pyproject.toml` jest już config indeksu PyTorcha z CUDA (`cu128`, nazwa indeksu `pytorch-cuda`) na później, pod model jednostek. GPU: RTX 4070.
- **Współrzędne jako ułamki szerokości i wysokości**, nie piksele. Rozdzielczość zależy od trybu (okno ~2424×1367, full screen ~2553×1435), ale układ UI jest ten sam, tylko przeskalowany. Złoto zaczyna się w ok. 44% szerokości, na samej górze ekranu.
- Gram i nagrywam na **full screenie**.
- Gdy otwarte jest menu, cała gra robi się szara (ważne później przy kolorach).
- OpenCV trzyma kolory w kolejności **BGR**.
- Nie używać słowa „Warcraft” w nazwie ani logo (znak towarowy Blizzarda).

## Struktura

```
GruntWork/
├── .gitignore
├── CLAUDE.md
├── readme.md
└── gruntWorkPython/
    ├── .venv/
    ├── data/                 # screeny i nagrania (screen1.png, screen2.png)
    ├── src/gruntworkpython/
    │   ├── __init__.py
    │   └── resources.py
    ├── pyproject.toml
    └── uv.lock
```

## Stan obecny

- `resources.py` wczytuje screen przez `cv2.imread` i działa. Ścieżka jest liczona względem pliku:
  `Path(__file__).resolve().parents[2] / "data"`. Po `imread` jest sprawdzenie `None`, bo przy złej ścieżce OpenCV po cichu zwraca `None`.
- Uwagi z review do wprowadzenia: klasa `Resources` nie jest potrzebna (wystarczy funkcja w module), literówka `Resourse` → `Resource`, przejście na snake_case (`get_resources_from_image`, `image_path`).

## Następne kroki

1. Skrypt `pick_coords.py`: kliknięciem wyznaczyć prostokąt z cyframi złota i przeliczyć go na ułamki rozmiaru obrazka.
2. Wyciąć region złota i podejrzeć go.
3. Wyciąć szablony cyfr 0–9 z gry i zrobić template matching.
4. Rozszerzyć na drewno, olej i jedzenie.
5. Przepuścić nagranie klatka co sekundę → CSV → wykres.
