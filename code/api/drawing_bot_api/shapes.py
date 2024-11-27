from math import sin, cos, pi, sqrt, pow, asin, atan2
import matplotlib.pyplot as plt
from drawing_bot_api.config import *

class Line:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.circumference = sqrt(pow(self.end_point[0] - self.start_point[0], 2) + pow(self.end_point[1] - self.start_point[1], 2))

    def get_point(self, t): # t determines which point on the curve defined by the shape is selected, t=0 is start point, t=1 is end point
        x = self.start_point[0] + ( (self.end_point[0] - self.start_point[0]) * t)
        y = self.start_point[1] + ( (self.end_point[1] - self.start_point[1]) * t)
        return [x, y]
    
    def plot(self, color=SHAPE_COLOR, label=None, resolution=PLOTTING_RESOLUTION):
        sample_number = int(resolution * self.circumference)
        for t in range(sample_number):
            point = self.get_point(t/sample_number)
            plt.plot(point[0], point[1], marker="o", markersize=PLOT_THICKNESS, markeredgecolor=color, markerfacecolor=color, label=label)

class Circle:
    def __init__(self, center_point, radius):
        self.center_point = center_point
        self.radius = radius
        self.circumference = 2 * pi * self.radius
        self.start_point = [self.center_point[0]+self.radius, self.center_point[1]]
        self.end_point = self.start_point

    def get_point(self, t):
        x = cos(2 * pi * t) * self.radius + self.center_point[0]
        y = sin(2 * pi * t) * self.radius + self.center_point[1]
        return [x, y]
    
    def plot(self, color=SHAPE_COLOR, label=None, resolution=PLOTTING_RESOLUTION):
        sample_number = int(resolution * self.circumference)
        for t in range(sample_number):
            point = self.get_point(t/sample_number)
            plt.plot(point[0], point[1], marker="o", markersize=PLOT_THICKNESS, markeredgecolor=color, markerfacecolor=color)

class PartialCircle:
    def __init__(self, start_point, end_point, radius, direction, big_angle=False):
        # direction: clockwise or anti-clockwise
        self.start_point = start_point
        self.end_point = end_point
        self.radius = radius
        self.direction = direction

        _xy_distance = self.__calc_xy_distance(start_point, end_point)
        _abs_distance = self.__abs_distance(_xy_distance)

        self.section_angle = 2*asin(_abs_distance/(2*self.radius))
        if big_angle:
            self.section_angle = 2 * pi - self.section_angle
            
        self.circumference = self.section_angle * self.radius

        self.center_point = self.__calc_center_point(self.start_point, self.end_point, self.direction, self.radius, _xy_distance, _abs_distance)

        _center_to_start_vector = self.__calc_xy_distance(self.center_point, self.start_point)
        self.offset = atan2(_center_to_start_vector[1], _center_to_start_vector[0])

    def __calc_xy_distance(self, point_1, point_2):
        return [point_2[0] - point_1[0], point_2[1] - point_1[1]]
    
    def __abs_distance(self, xy_distance):
        return sqrt(pow(xy_distance[0], 2) + pow(xy_distance[1], 2))
    
    def __calc_center_point(self, start_point, end_point, direction, radius, xy_distance, abs_distance):
        _normal_point = [self.start_point[0]+xy_distance[0]/2, self.start_point[1]+xy_distance[1]/2]
        _normal_vector = [(-direction)*(xy_distance[1]/abs_distance), direction*(xy_distance[0]/abs_distance)]
        _normal_distance = self.radius * cos(self.section_angle/2)
        return [_normal_point[0] + (_normal_vector[0] * _normal_distance), _normal_point[1] + (_normal_vector[1] * _normal_distance)]

    def get_point(self, t):
        x = self.radius * cos(self.offset + (t * self.direction * self.section_angle)) + self.center_point[0]
        y = self.radius * sin(self.offset + (t * self.direction * self.section_angle)) + self.center_point[1]
        return [x, y]
    
    def plot(self, color=SHAPE_COLOR, label=None, resolution=PLOTTING_RESOLUTION):
        sample_number = int(resolution * self.circumference)
        for t in range(sample_number):
            point = self.get_point(t/sample_number)
            plt.plot(point[0], point[1], marker="o", markersize=PLOT_THICKNESS, markeredgecolor=color, markerfacecolor=color)

if __name__ == '__main__':
    circ = PartialCircle([1, 0], [3, 8], 5, 1)
    line = Line([0, 1], [3,6])
    circ2 = Circle([4, 4], 3)

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    ax.set_xlim((-10, 10))
    ax.set_ylim((0, 16))

    circ.plot()
    line.plot()
    circ2.plot()

    plt.show()