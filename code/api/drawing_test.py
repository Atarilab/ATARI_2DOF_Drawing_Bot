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

def square(width, center):
    side = width/2
    drawing_bot.add_shape(shapes.Line([center[0]-side, center[1]+side], [center[0]+side, center[1]+side]))
    drawing_bot.add_shape(shapes.Line([center[0]+side, center[1]+side], [center[0]+side, center[1]-side]))
    drawing_bot.add_shape(shapes.Line([center[0]+side, center[1]-side], [center[0]-side, center[1]-side]))
    drawing_bot.add_shape(shapes.Line([center[0]-side, center[1]-side], [center[0]-side, center[1]+side]))

def main():
    #drawing_bot.hard_reset()
    for _ in range(5):
        heart()

    #square(30, [0, 110])
    #square(40, [0, 110])
    #square(50, [0, 110])
    #square(60, [0, 110])

    drawing_bot.plot(blocking=True)
    drawing_bot.execute(promting=True, clear_buffer=False)

main()
