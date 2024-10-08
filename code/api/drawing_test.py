from drawing_bot_api import Drawing_Bot
from drawing_bot_api import shapes
from math import cos, sin, pi
import time

drawing_bot = Drawing_Bot(unit='mm')

from drawing_bot_api.delta_utils import ik_delta
from drawing_bot_api.delta_utils import plt
ik_delta([0.03, 0.07])

def main_2():
    fig, ax = plt.subplots()
    ax.set_xlim(-120, 120)
    ax.set_ylim(0, 200)
    x_val = -100
    y_val = 50
    for y in range(100):
        for x in range(100):
            try:
                ik_delta([x_val/1000, y_val/1000])
                plt.plot(x_val, y_val, marker="o", markersize=5, markeredgecolor='black', markerfacecolor='black')
            except:
                print('fail')
            x_val+=2
        y_val += 1.5
        x_val = -100

    drawing_bot.plot()
    
    plt.show()

def main():
    #while not drawing_bot.is_ready():
        #time.sleep(0.5)

    drawing_bot.add_shape(shapes.Line([-40, 120], [40, 120]))
    drawing_bot.add_shape(shapes.Partial_circle([40, 120], [40, 100], 10, -1))
    drawing_bot.add_shape(shapes.Line([40, 100], [-40, 100]))
    drawing_bot.add_shape(shapes.Partial_circle([-40, 100], [-40, 80], 10, 1))
    drawing_bot.add_shape(shapes.Line([-40, 80], [0, 80]))
    drawing_bot.plot(blocking=True)
    drawing_bot.execute(promting=True)
    
main()
