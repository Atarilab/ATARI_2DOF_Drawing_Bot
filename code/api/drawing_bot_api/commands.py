from drawing_bot_api.delta_utils import plt
import matplotlib.patches as mpatches
from drawing_bot_api.delta_utils import plot_delta
from drawing_bot_api.delta_utils import ik_delta
from drawing_bot_api.delta_utils import plot_box
from math import cos, sin, pi
import time
import serial
from drawing_bot_api.logger import Log, Error_handler, ErrorCode
from drawing_bot_api import shapes
from drawing_bot_api.config import *
from drawing_bot_api.serial_handler import Serial_handler

class Drawing_Bot:
    def __init__(self, baud=115200, verbose=2, unit='mm', speed=200):
        # unit: Define which unit the user is using
        # speed is measured in unit/s

        self.log = Log((verbose-1)>0)
        self.error_handler = Error_handler(verbose)

        self.current_position = [0, 0]
        self.busy = 0
        self.speed = speed
        self.unit = 1000
        self.shapes = []
        
        if unit == 'm' or unit == 'meter':
            self.unit = 1
            self.log('Unit set to "m".')
        elif unit == 'cm' or unit == 'centimeter':
            self.unit = 100
            self.log('Unit set to "cm".')
        elif unit == 'mm' or unit == 'millimeter':
            self.unit = 1000
            self.log('Unit set to "mm".')
        else:
            self.error_handler(f'Invalid unit ("{unit}"). Reverting to default ("mm").', warning=True)

    def get_angles(self, position):
        x=position[0]/self.unit
        y=position[1]/self.unit

        try:
            angles = ik_delta([x, y])
            return angles
        except:
            self.error_handler("Targeted position is outside of robots domain.", ErrorCode.DOMAIN_ERROR)
            exit()

    def send_angle(self, angle, side, serial_handler):
        message = f'{side}{3*float(angle)}\n'
        serial_handler(message)

    # OLD AND UNUSED #########
    def update_position(self, position, serial_handler):
        angles = self.get_angles(position)
        self.log(f'Position: {position}, Angles: {angles}', clear=False)
        self.send_angle(angles[0], 'W', serial_handler)
        self.send_angle(angles[1], 'E', serial_handler)
        time.sleep(SERIAL_DELAY)
    ##########################

    def add_position(self, position, serial_handler):
        angles = self.get_angles(position)
        self.log(f'Position: {position}, Angles: {angles}', clear=False)
        self.send_angle(angles[0], 'W', serial_handler)
        self.send_angle(angles[1], 'E', serial_handler)
        #time.sleep(SERIAL_DELAY)


    def add_shape(self, shape):
        self.shapes.append(shape)

    def __plot_domain(self, resolution=PLOTTING_RESOLUTION):
        shapes.Line(DOMAIN_BOX[0], DOMAIN_BOX[1]).plot(color=DOMAIN_COLOR, resolution=resolution)
        shapes.Line(DOMAIN_BOX[2], DOMAIN_BOX[3]).plot(color=DOMAIN_COLOR, resolution=resolution)
        shapes.Line(DOMAIN_BOX[0], DOMAIN_BOX[3]).plot(color=DOMAIN_COLOR, resolution=resolution)
        shapes.Partial_circle(DOMAIN_DOME[0], DOMAIN_DOME[1], DOMAIN_DOME[2], DOMAIN_DOME[3]).plot(color=DOMAIN_COLOR, resolution=resolution)

    def plot(self, blocking=True, resolution=PLOTTING_RESOLUTION):
        _, ax = plt.subplots()
        ax.set_xlim((PLOT_XLIM[0]*(self.unit/1000), PLOT_XLIM[1]*(self.unit/1000)))
        ax.set_ylim((PLOT_YLIM[0]*(self.unit/1000), PLOT_YLIM[1]*(self.unit/1000)))

        self.__plot_domain(resolution=resolution)

        if not self.shapes:
            plt.show(block=blocking)
            return 0

        plt.plot(self.shapes[0].start_point[0], self.shapes[0].start_point[1], marker="o", markersize=START_END_DOT_SIZE, markeredgecolor=START_DOT_COLOR, markerfacecolor=START_DOT_COLOR, label='Start point')

        previous_shape = shapes.Line([0, 0], self.shapes[0].start_point)

        for shape in self.shapes:
            if shape.start_point != previous_shape.end_point:
                __bridge_line = shapes.Line(previous_shape.end_point, shape.start_point)
                __bridge_line.plot(color=BRIDGE_COLOR, resolution=resolution)

            shape.plot(resolution=resolution)
            previous_shape = shape

        plt.plot(previous_shape.end_point[0], previous_shape.end_point[1], marker="o", markersize=START_END_DOT_SIZE, markeredgecolor=END_DOT_COLOR, markerfacecolor=END_DOT_COLOR, label='End point')
        plt.plot(0, 0, color=BRIDGE_COLOR, label='Bridging lines')
        plt.plot(0, 0, color=SHAPE_COLOR, label='User defined drawings')
        plt.plot(0, 0, color=DOMAIN_COLOR, label='Domain')

        #plt.legend(['Start point', 'Bridge lines', 'User defined drawings', 'End point'], loc="lower right")
        plt.legend(bbox_to_anchor=(1, 1.15), ncol=3)

        plt.show(block=blocking)

    def plot_sampled_domain(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-120, 120)
        ax.set_ylim(0, 200)
        x_val = -125
        y_val = 20
        for y in range(50):
            for x in range(50):
                try:
                    ik_delta([x_val/1000, y_val/1000])
                    plt.plot(x_val, y_val, marker="o", markersize=3, markeredgecolor='black', markerfacecolor='black')
                except:
                    pass
                x_val+=5
            y_val += 4
            x_val = -125
        
        plt.show()

    def execute(self, promting=True, clear_buffer=False): # time defines how long the drawing process should take
        serial_handler = Serial_handler()

        for shape in self.shapes:
            __duration = (shape.circumference / self.speed) # in seconds
            __number_of_points = int(__duration / SERIAL_DELAY)

            for i in range(__number_of_points):
                point = shape.get_point(i * (1/__number_of_points))
                self.add_position(point, serial_handler=serial_handler)

            '''
            while(self.busy):
                __t = (self.millis() - __time) / __duration
                if __t > 1:
                    __t = 1

                __target_position = shape.get_point(__t)
                self.update_position(__target_position, serial_handler=serial_handler)

                if self.millis() - __time >= __duration:
                    self.busy = False
            '''

        self.add_position(self.shapes[-1].get_point(1), serial_handler=serial_handler) # Add last point 

        if clear_buffer:
            self.shapes.clear()
        
        serial_handler.send_buffer(promting)
        plt.show()

    def hard_reset(self):
        serial_handler = Serial_handler()
        serial_handler.kill_serial_script()
        serial_handler.start_serial_script()

    def millis(self):
        return time.time()*1000
    
if __name__ == '__main__':
    drawing_bot = Drawing_Bot()
    drawing_bot.add_shape(shapes.Line([0, 0], [1, 5]))
    drawing_bot.plot()