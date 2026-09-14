import mss
import numpy as np
import cv2
import os

# Settings for Easy difficulty, Zoom = 200%
# Might need to calibrate for other resolutions
cols = 9
rows = 9
size = 32
start_x = 737
start_y = 223

def capture_region(left, top, width, height):
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        img = np.array(sct.grab(region))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def main():
    # this just creates a folder
    os.makedirs("cells", exist_ok=True)
    width = size * cols
    height = size * rows
    img = capture_region(start_x, start_y, width, height)

    # slicing the board into individual cells and saving them individualy as pngs
    for r in range(rows):
        for c in range(cols):
            cell = img[r*size:(r+1)*size, c*size:(c+1)*size]
            cv2.imwrite(f"cells/{r}_{c}.png", cell)

    print(f"Saved {rows*cols} cell images to ./cells/")
    print("Look through them, then copy one clean example of each state")
    print("(unopened, flag, 0,1,2,3,4,5,6,7,8) into a templates/ folder,")
    print("named like: unopened.png, flag.png, 0.png, 1.png, ... 8.png")

if __name__ == "__main__":
    main()