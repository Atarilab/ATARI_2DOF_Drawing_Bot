from drawing_bot_api import Drawing_Bot
from drawing_bot_api import shapes
from math import cos, sin, pi
import time

ROTATION_SPEED = 2000 # defines how many milliseconds one rotation takes
X0 = 0
Y0 = 0.11
R = 0.03

drawing_bot = Drawing_Bot()

def main():
    while not drawing_bot.is_ready():
        time.sleep(0.5)

    while(1):
        drawing_bot.draw(shapes.Circle([X0, Y0], R))
    
#main()
