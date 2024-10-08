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
    
    def plot(self):
        sample_number = int(PLOTTING_RESOLUTION * self.circumference)
        for t in range(sample_number):
            point = self.get_point(t/sample_number)
            plt.plot(point[0], point[1], marker="o", markersize=PLOT_THICKNESS, markeredgecolor=SHAPE_COLOR, markerfacecolor=SHAPE_COLOR)

class Circle:
    def __init__(self, center_point, radius):
        self.center_point = center_point
        self.radius = radius
        self.circumference = 2 * pi * self.radius

    def get_point(self, t):
        x = cos(2 * pi * t) * self.radius + self.center_point[0]
        y = sin(2 * pi * t) * self.radius + self.center_point[1]
        return [x, y]
    
    def plot(self):
        sample_number = int(PLOTTING_RESOLUTION * self.circumference)
        for t in range(sample_number):
            point = self.get_point(t/sample_number)
            plt.plot(point[0], point[1], marker="o", markersize=PLOT_THICKNESS, markeredgecolor=SHAPE_COLOR, markerfacecolor=SHAPE_COLOR)

class Partial_circle:
    def __init__(self, start_point, end_point, radius, direction):
        # direction: clockwise or anti-clockwise
        self.start_point = start_point
        self.end_point = end_point
        self.radius = radius
        self.direction = direction

        x_distance = self.end_point[0] - self.start_point[0]
        y_distance = self.end_point[1] - self.start_point[1]
        abs_distance = sqrt(pow(x_distance, 2) + pow(y_distance, 2))

        self.section_angle = 2*asin(abs_distance/(2*self.radius))
        self.circumference = self.section_angle * self.radius

        normal_point = [self.start_point[0]+x_distance/2, self.start_point[1]+y_distance/2]
        normal_vector = [(-direction)*(y_distance/abs_distance), direction*(x_distance/abs_distance)]
        normal_distance = self.radius * cos(self.section_angle/2)

        self.center_point = [normal_point[0] + (normal_vector[0] * normal_distance), normal_point[1] + (normal_vector[1] * normal_distance)]

        center_to_start_vector = [self.start_point[0]-self.center_point[0], self.start_point[1]-self.center_point[1]]
        self.offset = atan2(center_to_start_vector[1], center_to_start_vector[0])

    def get_point(self, t):
        x = self.radius * cos(self.offset + t * self.section_angle) + self.center_point[0]
        y = self.radius * sin(self.offset + t * self.section_angle) + self.center_point[1]
        return [x, y]
    
    def plot(self):
        sample_number = int(PLOTTING_RESOLUTION * self.circumference)
        for t in range(sample_number):
            point = self.get_point(t/sample_number)
            plt.plot(point[0], point[1], marker="o", markersize=PLOT_THICKNESS, markeredgecolor=SHAPE_COLOR, markerfacecolor=SHAPE_COLOR)

if __name__ == '__main__':
    circ = Partial_circle([1, 0], [3, 8], 5, 1)
    line = Line([0, 1], [3,6])
    circ2 = Circle([4, 4], 3)

    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    ax.set_xlim((-10, 10))
    ax.set_ylim((0, 16))

    circ.plot()
    line.plot()
    circ2.plot()

    plt.show()