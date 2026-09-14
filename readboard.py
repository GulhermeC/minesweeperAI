import mss
import numpy as np
import cv2
import os

cols = 9
rows = 9
size = 32
start_x = 737
start_y = 223

MATCH_THRESHOLD = 0.40
DEBUG = False

def capture_region(left, top, width, height):
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        img = np.array(sct.grab(region))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_templates(folder="templates"):
    templates = {}
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"'{folder}/' not found. Run 2_extract_cells.py first and "
            f"build your template images."
        )
    # loops over files in templates/ and keeps only pngs, adding to a dict
    for fname in os.listdir(folder):
        if fname.lower().endswith(".png"):
            label = os.path.splitext(fname)[0]
            templates[label] = cv2.imread(os.path.join(folder, fname))
    if not templates:
        raise ValueError(f"No .png templates found in '{folder}/'")
    return templates

# compares images agaisnt every template one at a time, keeping track of the highest score
def classify_cell(cell_img, templates):
    best_label, best_score = None, -1.0
    for label, template in templates.items():
        if template.shape != cell_img.shape:
            template = cv2.resize(template, (cell_img.shape[1], cell_img.shape[0]))
        res = cv2.matchTemplate(cell_img, template, cv2.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_score:
            best_score, best_label = score, label
    return best_label, best_score

# same process as extractcells but instead of saving each cell it feeds into the classify_cell
def read_board(templates):
    width = size * cols
    height = size * rows
    img = capture_region(start_x, start_y, width, height)

    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell_img = img[r*size:(r+1)*size, c*size:(c+1)*size]
            label, score = classify_cell(cell_img, templates)
            if DEBUG:
                print(f"({r},{c}) -> {label} ({score:.3f})")
            if score < MATCH_THRESHOLD:
                row.append("?")
            else:
                row.append(label)
        grid.append(row)
    return grid


# translates internal labels into single-character symbols
def print_grid(grid):
    symbol_map = {"unopened": ".", "flag": "F", "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "?": "?"}
    for row in grid:
        line = []
        for cell in row:
            line.append(symbol_map.get(cell, cell))
        print("".join(line))

def main():
    templates = load_templates()
    grid = read_board(templates)
    print_grid(grid)

if __name__ == "__main__":
    main()