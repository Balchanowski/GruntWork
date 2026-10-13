import cv2

def pick_coords(str):

    # rois = cv2.selectROIs("Zaznacz obszar", str, showCrosshair=True)
    x, y, w, h = cv2.selectROI("Zaznacz obszar", str, showCrosshair=True)
    cv2.destroyAllWindows()
    print(f"x: {x}, y: {y}, w: {w}, h: {h}")

    roi = str[y:y+h, x:x+w]
    cv2.imwrite("hunger.png", roi)

    # for i, (x, y, w, h) in enumerate(rois, start=1):
    #     print(f"Obszar {i}: x={x}, y={y}, w={w}, h={h}")
        
    