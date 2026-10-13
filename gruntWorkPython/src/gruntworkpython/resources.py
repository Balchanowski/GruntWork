import cv2
from pathlib import Path
from .pick_coords import pick_coords
import os

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

        w_gold, h_gold, _ = resource_im.shape
        print(w_gold, h_gold)
        x, y = max_loc

        rectangles.append(((x,y),(x + w_gold, y + h_gold)))

        #laptop podgląd
        # podglad = cv2.resize(podglad, (1400, 800))
            
    print(rectangles)
    for pt1, pt2 in rectangles:
        cv2.rectangle(podglad, pt1, pt2, (0, 255, 0), 3)
    cv2.imshow('Wiele prostokatow', podglad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



getResourseFromImage(str(DATA_DIR / "screen1.png"))
