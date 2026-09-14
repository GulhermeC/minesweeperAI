# screencapture
import mss

# for raw pixel data
import numpy as np

# image processing
import cv2
from random import randrange

# Settings for Easy difficulty, Zoom = 200%
# Might need to calibrate for other resolutions
cols = 9
rows = 9
size = 32
start_x = 736
start_y = 221

def capture_region(left, top, width, height):

    # opens screencapture session
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        # sct.grab captures the defined rectangle of the screen
        # np.array turns that into a numpy array of shape (height, width, channels)
        img = np.array(sct.grab(region))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    width = size * cols
    height = size * rows
    img = capture_region(start_x, start_y, width, height)
    # saves the img to disk
    cv2.imwrite("calibration_raw.png", img)

    # copy so we dont draw on top of the original
    grid_img = img.copy()

    # draw the grid (vertical lines first, then horizontal lines)
    for c in range(cols + 1):
        x = c * size
        cv2.line(grid_img, (x, 0), (x, height), (0, 0, 255), 1)
    for r in range(rows + 1):
        y = r * size
        cv2.line(grid_img, (0, y), (width, y), (0, 0, 255), 1)

    cv2.imwrite("calibration_grid.png", grid_img)
    print(f"Saved calibration_raw.png ({width}x{height})")
    print("Saved calibration_grid.png -- open it and check the red lines")
    print("land exactly on the cell borders. Adjust start_x/start_y/size if not.")


if __name__ == "__main__":
    main()