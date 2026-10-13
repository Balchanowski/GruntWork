import cv2

def pick_coords(str):
    img = cv2.imread(str)

    if img is None:
        raise "IMG is NONE"

    x, y, w, h = cv2.selectROI("Zaznacz obszar", img, showCrosshair=True)

    print(f"x={x}, y={y}, szerokosc={w}, wysokosc={h}")
    print(f"lewy gorny: ({x}, {y}), prawy dolny: ({x + w}, {y + h})")

    wycinek = img[y:y+h, x:x+w]
    return wycinek