from drawing_bot_api import Drawing_Bot
from drawing_bot_api import shapes
from math import cos, sin, pi
import time

ROTATION_SPEED = 2000 # defines how many milliseconds one rotation takes
X0 = 0
Y0 = 110
R = 30

drawing_bot = Drawing_Bot()

def main():
    #while not drawing_bot.is_ready():
        #time.sleep(0.5)
    #while(1):
    drawing_bot.add_shape(shapes.Line([-50, 80], [50, 80]))
    drawing_bot.add_shape(shapes.Partial_circle([50, 80], [50, 50], 15, -1))
    drawing_bot.add_shape(shapes.Line([50, 50], [-50, 50]))
    drawing_bot.add_shape(shapes.Partial_circle([-50, 50], [-50, 20], 15, 1))
    drawing_bot.add_shape(shapes.Line([-50, 20], [0, 20]))
    drawing_bot.plot()
    #drawing_bot.execute()
    
main()
