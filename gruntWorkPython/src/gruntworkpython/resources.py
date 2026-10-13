import cv2
from pathlib import Path
from .pick_coords import pick_coords


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def getResourseFromImage(imagePath: str):
    print(imagePath)
    
    img = cv2.imread(imagePath)
    # print(img.shape)

    if img is None:
        raise FileNotFoundError(imagePath)
    
    # h, w, _ = img.shape
    # from_x = int(0.40 * w)
    # to_x = int(0.75 * w)
    # from_h = 0
    # to_h = int(0.04 * h)
    # roi = img[from_h: to_h, from_x:to_x]
    # print(from_x, to_x, from_h, to_h)

    
    # resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    # pick_coords(roi)
    
    # gold
    gold = cv2.imread(str(DATA_DIR / "resources" / "gold.png"))
    # print(gold)

    resource_array = cv2.matchTemplate(img, gold, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(resource_array)
    print(max_val, max_loc)

    w_gold, h_gold, _ = gold.shape
    print(w_gold, h_gold)
    podglad = img.copy()
    x, y = max_loc

    left_down_corner_loc = (x + w_gold, y + h_gold)

    rect = cv2.rectangle(podglad, max_loc, left_down_corner_loc, (0, 255, 0))
    cv2.imshow("rect", rect)
    cv2.waitKey(5000)


getResourseFromImage(str(DATA_DIR / "screen1.png"))
