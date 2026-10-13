import cv2
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def getResourseFromImage(imagePath: str):
    
    img = cv2.imread(imagePath)
    

    if img is None:
        raise FileNotFoundError(imagePath)
    
    h, w, _ = img.shape
    from_x = int(0.40 * w)
    to_x = int(0.75 * w)

    from_h = 0
    to_h = int(0.04 * h)

    roi = img[from_h: to_h, from_x:to_x]

    resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("ok", resized)
    cv2.waitKey(5000);
    cv2.destroyAllWindows()






getResourseFromImage(str(DATA_DIR / "screen2.png"))