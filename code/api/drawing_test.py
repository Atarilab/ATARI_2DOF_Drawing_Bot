#!/usr/bin/env python3
from drawing_bot_api import Drawing_Bot
from drawing_bot_api import shapes

drawing_bot = Drawing_Bot(unit='mm', speed=200)

#################################
#             DOMAIN            #
#                               #
#  +160    ............         #
#        ..            ..       #
#  +120 .                .      #
#       .                .      #
#  +70  ..................      #
#      -70             +70      #
#################################

def limits_test():
    drawing_bot.add_shape(shapes.Line([-60, 120], [60, 120]))
    drawing_bot.add_shape(shapes.Partial_circle([60, 120], [60, 100], 10, -1))
    drawing_bot.add_shape(shapes.Line([60, 100], [-60, 100]))
    drawing_bot.add_shape(shapes.Partial_circle([-60, 100], [-60, 72], 20, 1))
    drawing_bot.add_shape(shapes.Line([-60, 72], [0, 72]))
    drawing_bot.add_shape(shapes.Line([0, 72], [0, 160]))

def heart():
    drawing_bot.add_shape(shapes.Partial_circle([0, 135], [-40, 110], 25, 1, big_angle=True))
    drawing_bot.add_shape(shapes.Line([-40, 110], [0, 75]))
    drawing_bot.add_shape(shapes.Line([0, 75], [40, 110]))
    drawing_bot.add_shape(shapes.Partial_circle([40, 110], [0, 135], 25, 1, big_angle=True))

def partial_circle():
    drawing_bot.add_shape(shapes.Partial_circle([0, 120], [0, 80], 30, -1, big_angle=True))

def main():

    #limits_test()
    heart()
    #partial_circle()
    drawing_bot.plot(blocking=False)
    drawing_bot.execute(promting=False)

main()
