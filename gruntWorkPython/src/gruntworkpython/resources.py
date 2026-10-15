from pathlib import Path
import os
import cv2
import easyocr
import numpy as np
from .config import DATA_DIR, REF_H, BAR_HEIGHT, MIN_SCORE, RESOURCE_NAMES, RESOURCES_DIR, DEBUG, OCR_HEIGHT


# ---------- wzorce ----------

def load_templates(version: str) -> dict[str, np.ndarray]:
    """Wczytuje wzorce raz i od razu zamienia je na szare."""
    templates = {}
    for name in RESOURCE_NAMES:
        path = RESOURCES_DIR / version / f"{name}.png"
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        templates[name] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return templates


# ---------- szukanie ikonek ----------

def find_icon(gray: np.ndarray, template: np.ndarray, scales):
    """Szuka wzorca w różnych skalach. Zwraca (score, x, y, w, h, scale) albo None."""
    best = None
    h_img, w_img = gray.shape[:2]

    for scale in scales:
        scaled = cv2.resize(template, None, fx=scale, fy=scale)
        h, w = scaled.shape[:2]
        if h > h_img or w > w_img:
            continue

        result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)

        if best is None or score > best[0]:
            best = (score, x, y, w, h, scale)
    return best


def locate(region: np.ndarray, template: np.ndarray, scales, offset_x: int = 0):
    """find_icon + zamiana wyniku na prostokąt (x1, y1, x2, y2) w układzie pełnego screena."""
    found = find_icon(region, template, scales)
    if found is None:
        return None, 0.0, None
    score, x, y, w, h, scale = found
    x += offset_x
    return (x, y, x + w, y + h), score, scale


def calibration_failed(name: str, score: float):
    if DEBUG: 
        print(f"Kalibracja nieudana: {name} (score {score:.2f})")
        return None

    return None


def find_resource_bar(img: np.ndarray, templates: dict) -> dict | None:
    """Zwraca {nazwa: (x1, y1, x2, y2)} dla czterech ikonek albo None, jeśli się nie udało."""
    img_h = img.shape[0]
    bar = cv2.cvtColor(img[: int(BAR_HEIGHT * img_h), :], cv2.COLOR_BGR2GRAY)
    icons = {}

    # 1. złoto: skala wyliczona z wysokości ekranu, ±15%
    expected = img_h / REF_H
    gold_scales = np.arange(expected * 0.85, expected * 1.15, 0.01)
    icons["gold"], score, gold_scale = locate(bar, templates["gold"], gold_scales)
    gold_score = score

    if score < MIN_SCORE:
        return calibration_failed("gold", score)

    # 2. ropa i jedzenie: tylko skale blisko skali złota
    narrow = np.arange(gold_scale - 0.05, gold_scale + 0.051, 0.01)
    for name in ["oil", "hunger"]:
        icons[name], score, _ = locate(bar, templates[name], narrow)
        if icons[name] is None:
                return calibration_failed(name, 0)
        
        if score < MIN_SCORE:
            print(f"UWAGA: {name} słabo dopasowany ({score:.2f})")

    # 3. drewno: tylko między złotem a ropą
    gold_right = icons["gold"][2]
    oil_left = icons["oil"][0]
    
    if oil_left <= gold_right:
        return calibration_failed("ropa przed złotem", 0)
    
    between = bar[:, gold_right:oil_left]
    icons["lumber"], score, _ = locate(between, templates["lumber"], narrow, offset_x=gold_right)

    if icons["lumber"] is None:
        return calibration_failed("lumber", 0)

    if score < MIN_SCORE:
        print(calibration_failed("lumber", score))

    order = [icons[n][0] for n in ["gold", "lumber", "oil", "hunger"]]
    if order != sorted(order):
        return calibration_failed("kolejność ikonek", 0)

    gold_y = icons["gold"][1]
    if any(abs(icons[n][1] - gold_y) > 5 for n in RESOURCE_NAMES):
        return calibration_failed("ikonki nie w jednym rzędzie", 0)
    return icons, gold_score


def quantity_boxes(icons: dict, img_w: int) -> dict:
    """Obszar liczby: od prawej krawędzi ikonki do lewej krawędzi następnej."""
    ordered = sorted(icons.items(), key=lambda item: item[1][0])   # od lewej do prawej

    gaps = []
    for (_, cur), (_, nxt) in zip(ordered, ordered[1:]):
        gaps.append(nxt[0] - cur[2])
    mean_gap = sum(gaps) // len(gaps)

    boxes = {}
    for i, (name, (_, y1, x2, y2)) in enumerate(ordered):
        gap = gaps[i] if i < len(gaps) else mean_gap
        boxes[name] = (x2, y1, min(x2 + gap, img_w), y2)
    return boxes


# ---------- OCR ----------

def read_values(img: np.ndarray, boxes: dict, ocr: easyocr.Reader) -> dict:
    values = {}
    for name, (x1, y1, x2, y2) in boxes.items():
        crop = img[y1:y2, x1:x2]

        scale = OCR_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        pad = OCR_HEIGHT // 4
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        texts = ocr.readtext(crop, detail=0, allowlist="0123456789/")
        values[name] = "".join(texts)
    return values


# ---------- podgląd ----------

def draw_debug(img: np.ndarray, icons: dict, boxes: dict, path: str):
    podglad = img.copy()
    for x1, y1, x2, y2 in icons.values():
        cv2.rectangle(podglad, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x1, y1, x2, y2 in boxes.values():
        cv2.rectangle(podglad, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(path, podglad)


# ---------- całość: kalibracja raz, potem tylko odczyt ----------

def debug_find(path, out):
    img = cv2.imread(str(path))
    print(img.shape)
    templates = load_templates("classic")
    bar = cv2.cvtColor(img[: int(BAR_HEIGHT * img.shape[0]), :], cv2.COLOR_BGR2GRAY)
    podglad = img.copy()

    for name, tpl in templates.items():
        found = find_icon(bar, tpl, np.arange(0.3, 1.6, 0.05))
        if found is None:
            print(name, "nie znaleziono")
            continue
        score, x, y, w, h, scale = found
        print(f"{name}: score {score:.2f}, skala {scale:.2f}, x={x}, y={y}")
        cv2.rectangle(podglad, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(podglad, f"{name} {score:.2f}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(out, podglad)


class ResourceReader:
    def __init__(self):
        self.templates = {"remastered": load_templates("remastered"), "classic": load_templates("classic")}
        self.ocr = easyocr.Reader(["en"], gpu=True)
        self.icons = None
        self.boxes = None
        self.frame_shape = None
        self.version = None

    def _is_version_choose(self, frame) -> bool:
        if self.version is not None:
            result = find_resource_bar(frame, self.templates[self.version])
            if result is not None:
                icons, _ = result
                self._save(icons, frame)
                return True
        return False

    def _save(self, best_icons, frame):
        self.icons = best_icons
        self.boxes = quantity_boxes(best_icons, frame.shape[1])
        self.frame_shape = frame.shape

    def calibrate(self, frame: np.ndarray) -> bool:
        if self._is_version_choose(frame):
            return True

        best_score = -1
        best_icons = None
        best_version = None

        for version, template in self.templates.items():
            result = find_resource_bar(frame, template)
            if result is None: continue
            icons, gold_score = result

            if gold_score > best_score:
                best_icons = icons
                best_version = version
                best_score = gold_score

        if best_icons is None:
            self.icons = self.boxes = self.frame_shape = self.version = None
            return False
        
        self._save(best_icons, frame)
        self.version = best_version
        return True

    def read(self, frame: np.ndarray) -> dict | None:
        # kalibrujemy, jeśli jeszcze nie było albo zmienił się rozmiar klatki
        if self.boxes is None or frame.shape != self.frame_shape:
            if not self.calibrate(frame):
                return None
        return read_values(frame, self.boxes, self.ocr)


if __name__ == "__main__":
    # debug_find(DATA_DIR / "screen6.png", "podglad6.png")
    reader = ResourceReader()

    files = os.listdir(str(DATA_DIR / "screens"))
    print(files)
    
    for i in range(1, len(files)+1):
        frame = cv2.imread(str(DATA_DIR / "screens" / f"screen{i}.png"))
        if frame is None:
            raise FileNotFoundError(f"screen{i}.png")

        values = reader.read(frame)
        print(f"screen{i}: {values}")

        if reader.boxes is not None:
            draw_debug(frame, reader.icons, reader.boxes, str(DATA_DIR / "output" / f"podglad{i}.png"))