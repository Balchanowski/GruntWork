import cv2
from pathlib import Path
from .pick_coords import pick_coords
import os
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def getResourseFromImage(imagePath: str):
    print(imagePath)
    
    img = cv2.imread(imagePath)
    # print(img.shape)

    if img is None:
        raise FileNotFoundError(imagePath)
    
    h, w, _ = img.shape
    from_x = int(0.40 * w)
    to_x = int(0.75 * w)
    from_h = 0
    to_h = int(0.04 * h)
    roi = img[from_h: to_h, from_x:to_x]
    print(from_x, to_x, from_h, to_h)

    
    # resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    # pick_coords(roi)
    
    resources = os.listdir(str(DATA_DIR / "resources"))
    print(resources)

    rectangles = []
    podglad = img.copy()

    for resource in resources:
        resource_path = str(DATA_DIR / "resources" / resource)
        print(resource_path)
        resource_im = cv2.imread(resource_path)

        if resource_im is None:
            raise FileNotFoundError(resource_path)
        
        

        resource_array = cv2.matchTemplate(img, resource_im, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(resource_array)
        print(max_val, max_loc)

        h_gold, w_gold, _ = resource_im.shape
        print(w_gold, h_gold)
        x, y = max_loc

        rectangles.append(((x,y),(x + w_gold, y + h_gold)))

        #laptop podgląd
        # podglad = cv2.resize(podglad, (1400, 800))
            
    sorted_rectangles = sorted(rectangles, key=lambda r: r[0][0])
    print(sorted_rectangles)

    selectRectanglesForImageOfResources(sorted_rectangles, podglad)
    value_rectangles = setRectanglesForQuantityOfResources(sorted_rectangles, podglad)

    podglad = cv2.resize(podglad, (1400, 800))
    cv2.imshow('Wiele prostokatow', podglad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def setRectanglesForQuantityOfResources(sorted_rectangles, podglad):
    gaps = []
    value_rectangles = []
    for cur, nxt in zip(sorted_rectangles, sorted_rectangles[1:]):
        cur_right = cur[1][0]
        next_left = nxt[0][0]
        gaps.append(next_left - cur_right)

    mean_gap = sum(gaps) // len(gaps)

    for i, ((_, y1), (x2, y2)) in enumerate(sorted_rectangles):
        gap = gaps[i] if i < len(gaps) else mean_gap
        value_rectangles.append((x2, y1, x2 + gap, y2))
        cv2.rectangle(podglad, (x2, y1), (x2 + gap, y2), (0, 0, 255), 3)

    return value_rectangles

def selectRectanglesForImageOfResources(sorted_rectangles, podglad):
    for pt1, pt2 in sorted_rectangles:
        cv2.rectangle(podglad, pt1, pt2, (0, 255, 0), 3)


getResourseFromImage(str(DATA_DIR / "screen1.png"))
