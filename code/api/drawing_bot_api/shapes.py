from math import sin, cos, pi, sqrt, pow, asin, acos

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
    def __init__(self, start_point, end_point, radius, direction):
        # direction: clockwise or anti-clockwise
        self.start_point = start_point
        self.end_point = end_point
        self.radius = radius
        self.direction = direction
        self.circumferences = 0

        x_distance = self.end_point[0] - self.start_point[0]
        y_distance = self.end_point[1] - self.start_point[1]
        abs_distance = sqrt(pow(x_distance, 2) + pow(y_distance, 2))

        self.section_angle = 2*asin(abs_distance/(2*self.radius))

        normal_point = [self.start_point[0]+x_distance/2, self.start_point[1]+y_distance/2]
        normal_vector = [(-direction)*(y_distance/abs_distance), direction*(x_distance/abs_distance)]
        normal_distance = self.radius * cos(self.section_angle/2)

        self.center_point = [normal_point[0] + (normal_vector[0] * normal_distance), normal_point[1] + (normal_vector[1] * normal_distance)]

        # offset calculation
        distance_center_to_start = [(self.center_point[0] + self.radius)-self.start_point[0], self.center_point[1]-self.start_point[1]]
        abs_distance_center_to_start = sqrt(pow(distance_center_to_start[0], 2) + pow(distance_center_to_start[1], 2))
        self.offset = 2*asin(abs_distance_center_to_start/(2*self.radius)) # NOT WORKING YET !!!!!!!!!!


    def get_point(self, t):
        print(self.section_angle)
        print(self.offset)
        return self.center_point

circ = Partial_circle([1, 4], [3, 8], 5, 1)
print(circ.get_point(0))
