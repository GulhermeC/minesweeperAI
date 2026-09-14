import mss
import numpy as np
import cv2
import os
import time
import pyautogui

cols = 9
rows = 9
size = 32
start_x = 737
start_y = 223

MATCH_THRESHOLD = 0.40
DEBUG = False
MAX_ITERATIONS = 100
ACTION_DELAY = 0.15
TOTAL_MINES = 10

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

    for fname in os.listdir(folder):
        if fname.lower().endswith(".png"):
            label = os.path.splitext(fname)[0]
            templates[label] = cv2.imread(os.path.join(folder, fname))
    if not templates:
        raise ValueError(f"No .png templates found in '{folder}/'")
    return templates

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
            elif label in ("unopened", "flag"):
                row.append(label)
            else:
                row.append(int(label))
        grid.append(row)
    return grid


def print_grid(grid):
    symbol_map = {"unopened": ".", "flag": "F"}
    for row in grid:
        line = [str(symbol_map.get(cell, cell)) for cell in row]
        print("".join(line))

# gets all the neighbours of a cell also skipping 0,0 which is its own cell
def get_neighbors(r, c, n_rows, n_cols):
    neighbors = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                neighbors.append((nr, nc))
    return neighbors

def solve_step(grid):
    n_rows = len(grid)
    n_cols = len(grid[0])

    # these are sets not lists so duplicates dont matter
    safe = set()
    mines = set()

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            # checks if a cell has a number aka skips everything else because only numbers carry info
            if not isinstance(cell, int):
                continue

            # get neighbours and splits them into two lists, unopened for hidden and flagged for mines
            neighbors = get_neighbors(r, c, n_rows, n_cols)
            unopened = [n for n in neighbors if grid[n[0]][n[1]] == "unopened"]
            flagged = [n for n in neighbors if grid[n[0]][n[1]] == "flag"]

            # if a cell has no unopened neighbours then it doesnt matter we skip
            if not unopened:
                continue

            # check if the number is the same as the flagged mines around it, if its not then there are more mines unflagged
            remaining_mines = cell - len(flagged)

            # if the number of remaining mines is the same as the unopened neighbours then all of them are mines
            if remaining_mines == len(unopened) and remaining_mines > 0:
                mines.update(unopened)
            # if the number of remaining mines is 0 then all neighbours are safe
            if remaining_mines == 0:
                safe.update(unopened)

    return safe, mines

# just so we can click the center of the cell
# start_x + c * size goes to the left edge then size // 2 moves to the horizontal center
# then same with y
def cell_center(r, c):
    return (start_x + c * size + size // 2, start_y + r * size + size // 2)

# left click
def click_safe(r, c):
    x, y = cell_center(r, c)
    pyautogui.click(x, y)

# rightclick
def click_mine(r, c):
    x, y = cell_center(r, c)
    pyautogui.rightClick(x, y)

def board_has_unopened(grid):
    return any(cell == "unopened" for row in grid for cell in row)
 
 
def board_has_unknown(grid):
    return any(cell == "?" for row in grid for cell in row)

def estimate_probabilities(grid, total_mines):
    n_rows = len(grid)
    n_cols = len(grid[0])

    unopened_cells = [
        (r, c) for r in range(n_rows) for c in range(n_cols)
        if grid[r][c] == "unopened"
    ]
    flagged_count = sum(row.count("flag") for row in grid)

    cell_probs = {}

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            if not isinstance(cell, int):
                continue

            neighbors = get_neighbors(r, c, n_rows, n_cols)
            unopened_neighbors = [n for n in neighbors if grid[n[0]][n[1]] == "unopened"]
            flagged_neighbors = [n for n in neighbors if grid[n[0]][n[1]] == "flag"]

            if not unopened_neighbors:
                continue

            # same as solve_step but we turn it into a ratio
            remaining = cell - len(flagged_neighbors)
            local_prob = remaining / len(unopened_neighbors)
            local_prob = max(0.0, min(1.0, local_prob))

            # merging estimates when a cell touches multiple numbers
            for n in unopened_neighbors:
                if n not in cell_probs or local_prob > cell_probs[n]:
                    cell_probs[n] = local_prob

    # frontier cells are ones at got touched by at least one number
    frontier_cells = set(cell_probs.keys())
    non_frontier_cells = [c for c in unopened_cells if c not in frontier_cells]

    # add every frontier cells probability gives an expected number of mines sitting on the frontier
    # total_mines - flagged_count - expected_frontier_mines gives total mines in the whole game minus the one hiding on the frontier
    if non_frontier_cells:
        expected_frontier_mines = sum(cell_probs.values())
        remaining_mines_est = total_mines - flagged_count - expected_frontier_mines
        remaining_mines_est = max(0, remaining_mines_est)
        global_density = remaining_mines_est / len(non_frontier_cells)
        global_density = max(0.0, min(1.0, global_density))
        for cell in non_frontier_cells:
            cell_probs[cell] = global_density

    return cell_probs

# chooses the least risky cell
def choose_guess(grid, total_mines):
    probs = estimate_probabilities(grid, total_mines)
    if not probs:
        return None
    return min(probs, key=probs.get)


def main():
    templates = load_templates()

    for iteration in range(MAX_ITERATIONS):
        grid = read_board(templates)
        print(f"--- iteration {iteration} ---")
        print_grid(grid)

        if board_has_unknown(grid):
            print("Found unclassified ('?') cells -- check templates/threshold.")
 
        if not board_has_unopened(grid):
            print("No unopened cells left -- looks solved!")
            break

        # first move
        all_unopened = all(
            cell == "unopened" for row in grid for cell in row
        )
        if all_unopened:
            print("Empty board detected -- clicking center to start.")
            click_safe(rows // 2, cols // 2)
            time.sleep(ACTION_DELAY)
            continue

        # runs the solver, if both come back empty then its a stall
        safe, mines = solve_step(grid)

        if not safe and not mines:
            guess = choose_guess(grid, TOTAL_MINES)
            if guess is None:
                print("No unopened cells left to guess -- stopping.")
                break
            r, c = guess
            print(f"No certain moves -- guessing lowest-risk cell ({r}, {c}).")
            click_safe(r, c)
            time.sleep(ACTION_DELAY)
            continue

        for (r, c) in safe:
            click_safe(r, c)
        for (r, c) in mines:
            click_mine(r, c)

        time.sleep(ACTION_DELAY)

    else:
        print("Hit MAX_ITERATIONS -- stopping as a safety net.")

if __name__ == "__main__":
    print("Starting in 3 seconds -- switch to the game window now...")
    time.sleep(3)
    main()