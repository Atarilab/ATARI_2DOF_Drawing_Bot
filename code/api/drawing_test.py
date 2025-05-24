#!/usr/bin/env python3
from drawing_bot_api import DrawingBot
from drawing_bot_api.shapes import *

# Here you can adjust the speed of the robot and change the unit if you prefer another: meters (m), centimeters (cm), millimeters (mm)
drawing_bot = DrawingBot(speed=200)

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

# These are the predefined shapes you can draw and add together:

# Line(starting_point, end_point)
# e.g. Line([-30, 80], [40, 100])

# Circle(center_point, radius)
# e.g. Circle([-10, 110], 20)

# Partial_circle(start_point, end_point, radius, direction) optional: big_angle
# e.g. Partial_circle([-20, 120], [10, 100], 25, -1, big_angle=True)


# EXAMPLE FUNCTIONS:
def heart():
    drawing_bot.add_shape(PartialCircle([0, 135], [-40, 110], 25, 1, big_angle=True))
    drawing_bot.add_shape(Line([-40, 110], [0, 75]))
    drawing_bot.add_shape(Line([0, 75], [40, 110]))
    drawing_bot.add_shape(PartialCircle([40, 110], [0, 135], 25, 1, big_angle=True))

def square(width, center):
    side = width/2
    drawing_bot.add_shape(Line([center[0]-side, center[1]+side], [center[0]+side, center[1]+side]))
    drawing_bot.add_shape(Line([center[0]+side, center[1]+side], [center[0]+side, center[1]-side]))
    drawing_bot.add_shape(Line([center[0]+side, center[1]-side], [center[0]-side, center[1]-side]))
    drawing_bot.add_shape(Line([center[0]-side, center[1]-side], [center[0]-side, center[1]+side]))

########################################################
# THE FOLLOWING CODE IS THE CODE THAT WILL BE EXECUTED #
########################################################

def main():
    # Use the function below (drawing_bot.hard_reset()) to hard reset the whole system if the connection to the drawing bot fails
    # This will restart the drawing bot which means it will do it's start up routine again
    # Using this every time you run a programm is possible but annoying because of the start up routine
    
    #drawing_bot.hard_reset()
    

    # ENTER YOUR PROGRAM HERE:
    ##################################################
    
    #heart()
    #square(50, [0, 110])

    # T
    drawing_bot.add_shape(Line([-60, 100], [-60, 120]))
    drawing_bot.add_shape(Line([-60, 120], [-65, 120]))
    drawing_bot.add_shape(Line([-65, 120], [-55, 120]))
    # h
    drawing_bot.add_shape(Line([-55, 120], [-55, 100]))
    drawing_bot.add_shape(Line([-55, 100], [-55, 107]))
    drawing_bot.add_shape(PartialCircle([-55, 107], [-45, 107], 5.5, -1, big_angle=False))
    drawing_bot.add_shape(Line([-45, 107], [-45, 100]))
    # a
    drawing_bot.add_shape(Line([-45, 100], [-30, 100]))
    drawing_bot.add_shape(PartialCircle([-30, 103], [-30, 107], 5, -1, big_angle=True))
    drawing_bot.add_shape(Line([-30, 107], [-30, 100]))
    drawing_bot.add_shape(Line([-30, 100], [-25, 100]))
    # n
    drawing_bot.add_shape(Line([-25, 100], [-25, 110]))
    drawing_bot.add_shape(Line([-25, 110], [-25, 107]))
    drawing_bot.add_shape(PartialCircle([-25, 107], [-15, 104], 6, -1, big_angle=False))
    drawing_bot.add_shape(Line([-15, 104], [-15, 100]))
    drawing_bot.add_shape(Line([-15, 100], [-10, 100]))
    # k
    drawing_bot.add_shape(Line([-10, 100], [-10, 120]))
    drawing_bot.add_shape(Line([-10, 120], [-10, 105]))
    drawing_bot.add_shape(Line([-10, 105], [-3, 110]))
    drawing_bot.add_shape(PartialCircle([-3, 110], [-3, 105], 3, -1, big_angle=True))
    drawing_bot.add_shape(Line([-3, 105], [-8, 106]))
    drawing_bot.add_shape(Line([-8, 106], [0, 100]))
    # space
    drawing_bot.add_shape(Line([0, 100], [25, 100]))
    # y
    drawing_bot.add_shape(PartialCircle([25, 100], [20, 110], 7, -1, big_angle=False))
    drawing_bot.add_shape(PartialCircle([20, 110], [30, 110], 6, 1, big_angle=True))
    drawing_bot.add_shape(Line([30, 110], [25, 90]))
    drawing_bot.add_shape(PartialCircle([25, 90], [22, 92], 2, -1, big_angle=True))
    drawing_bot.add_shape(Line([22, 92], [43, 110]))
    # o
    drawing_bot.add_shape(PartialCircle([43, 110], [44, 109], 5, 1, big_angle=True))
    # u
    drawing_bot.add_shape(Line([44, 109], [50, 110]))
    drawing_bot.add_shape(Line([50, 110], [50, 105]))
    drawing_bot.add_shape(PartialCircle([50, 105], [60, 105], 5, 1, big_angle=False))
    drawing_bot.add_shape(Line([60, 105], [60, 110]))
    drawing_bot.add_shape(Line([60, 110], [60, 100]))
    drawing_bot.add_shape(Line([60, 100], [0, 40]))


    
    ###################################################
    
    # This function creates a graph showing the path you want to draw
    drawing_bot.plot()
    
    # This function sends the commands to the robot
    # Set promting=True if you want to be asked for confirmation before the commands are sent to the robot; promting=False otherwise
    # clear_buffer=True means that all commands previously handed to the robot will be erased from it's memory after the execution
    # comment this out if, you only want to look at the plots
    drawing_bot.execute(promting=True)

main()
