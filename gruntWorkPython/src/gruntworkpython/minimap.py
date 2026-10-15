import os

import cv2
from .config import DATA_DIR, REF_H, MINIMAP_REF, MINIMAP_MARGIN
import numpy as np

def find_minimap(frame: np.ndarray, filename: str) -> tuple[int, int, int, int] | None:
    """Znajduje minimapę w podanej klatce. Zwraca współrzędne prostokąta (x1, y1, x2, y2) lub None jeśli nie znaleziono."""
    h_frame, w_frame, _ = frame.shape
    rect = cv2.rectangle(frame, (0, 0), (int(0.31 * h_frame), int(0.33 * h_frame)), (0, 255, 0), 2)
    cv2.imwrite(str(DATA_DIR / "minimapoutput" / f"podglad_minimap_{filename}"), rect)  # Zapisz wynik do debugowania



def sizes_of_minimap(imagePath: str) -> tuple[int, int, int, int] | None:
    """Zwraca rozmiary minimapy w podanej klatce. Zwraca (x1, y1, x2, y2) lub None jeśli nie znaleziono."""
    frame = cv2.imread(imagePath)
    minimap = cv2.imread(str(DATA_DIR / "resources" / "minimap" / "minimap.png"))
    if minimap is None:
        print(f"Nie można wczytać wzorca minimapy: {DATA_DIR / 'resources' / 'minimap' / 'minimap.png'}")
        return None
    if frame is None:
        print(f"Nie można wczytać obrazu: {imagePath}")
        return None

    result = cv2.matchTemplate(frame, minimap, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    h_minimap, w_minimap, _ = minimap.shape
    print(x, y, h_minimap, w_minimap)

    h_minimap = 367
    w_minimap = 367

    cv2.rectangle(frame, (x, y), (x + w_minimap, y + h_minimap), (0, 255, 0), 2)
    cv2.imwrite(str(DATA_DIR / "minimapoutput" / f"sizes_of_minimap_debug.png"), frame)  # Zapisz wynik do debugowania
    
if __name__ == "__main__":

    files = os.listdir(str(DATA_DIR / "screens"))
    for i in files:
        frame = cv2.imread(str(DATA_DIR / "screens" / i))
        find_minimap(frame, i)
    