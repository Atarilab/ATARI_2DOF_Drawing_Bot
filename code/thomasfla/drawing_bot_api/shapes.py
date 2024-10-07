from math import sin, cos, pi, abs
class Line:
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.circumference = abs(self.end_point - self.start_point)

    def get_point(self, t): # t determines which point on the curve defined by the shape is selected, t=0 is start point, t=1 is end point
        x = self.start_point[0] + ( (self.end_point[0] - self.start_point[0]) * t)
        y = self.start_point[1] + ( (self.end_point[1] - self.start_point[1]) * t)
        return [x, y]

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.circumference = 2 * pi * self.radius

    def get_point(self, t):
        x = cos(2 * pi * t) * self.radius + self.center[0]
        y = sin(2 * pi * t) * self.radius + self.center[1]
        return [[x, y]]

class Partial_circle:
    def __init__(self, start_point, end_point, radius):
        self.start_point = start_point
        self.end_point = end_point
        self.radius = radius
        self.circumferences = 0

    def get_point(self, t):
        pass