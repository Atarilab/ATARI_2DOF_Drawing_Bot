from drawing_bot_api import Drawing_Bot
from drawing_bot_api import shapes
from math import cos, sin, pi
import time
from drawing_bot_api import config

drawing_bot = Drawing_Bot(unit='mm')

from drawing_bot_api.delta_utils import ik_delta
from drawing_bot_api.delta_utils import plt
ik_delta([0.03, 0.07])

def main2():
    #drawing_bot.restart()

    while(1):
        if drawing_bot.is_ready():
            print('yay')
        time.sleep(0.5)

def main():
    while not drawing_bot.is_ready():
        time.sleep(0.5)

    drawing_bot.add_shape(shapes.Line([-40, 120], [40, 120]))
    drawing_bot.add_shape(shapes.Partial_circle([40, 120], [40, 100], 10, -1))
    drawing_bot.add_shape(shapes.Line([40, 100], [-40, 100]))
    drawing_bot.add_shape(shapes.Partial_circle([-40, 100], [-40, 80], 10, 1))
    drawing_bot.add_shape(shapes.Line([-40, 80], [0, 80]))
    #drawing_bot.plot(blocking=True)
    drawing_bot.execute(promting=True)
    
main()
