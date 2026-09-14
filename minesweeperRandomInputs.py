import pyautogui
import time
from random import randrange

cols = 9
rows = 9
size = 32
start_x = 736
start_y = 221

# Random clicks

for i in range(10):
    random_x = randrange(cols)
    random_y = randrange(rows)
    pyautogui.click(start_x+size*random_x, start_y+size*random_y)
    time.sleep(0.5)