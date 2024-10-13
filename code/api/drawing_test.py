from drawing_bot_api import Drawing_Bot
from drawing_bot_api import shapes
from math import cos, sin, pi
import time
from drawing_bot_api import config

drawing_bot = Drawing_Bot(unit='mm')

def main():
    #while not drawing_bot.is_ready():
        #time.sleep(0.5)

    drawing_bot.add_shape(shapes.Line([-40, 120], [40, 120]))
    drawing_bot.add_shape(shapes.Partial_circle([40, 120], [40, 100], 10, -1))
    drawing_bot.add_shape(shapes.Line([40, 100], [-40, 100]))
    drawing_bot.add_shape(shapes.Partial_circle([-40, 100], [-40, 80], 10, 1))
    drawing_bot.add_shape(shapes.Line([-40, 80], [0, 80]))
    drawing_bot.add_shape(shapes.Line([0, 80], [0, 150]))
    drawing_bot.plot(blocking=False)
    drawing_bot.execute(promting=False)
    
main()
