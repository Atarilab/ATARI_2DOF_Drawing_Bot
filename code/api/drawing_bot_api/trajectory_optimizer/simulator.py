import numpy as np
from scipy.ndimage import gaussian_filter1d

class PatternErrorSim:
    def __init__(self, strength=100, pattern_length=20):
        self.pattern_length = pattern_length

        _random_offsets_x = np.random.normal(0, 0.1, pattern_length)
        _random_offsets_y = np.random.normal(0, 0.1, pattern_length)
        # Smooth the offsets
        self.pattern_x = gaussian_filter1d(_random_offsets_x, sigma=1.5) * strength
        self.pattern_y = gaussian_filter1d(_random_offsets_y, sigma=1.5) * strength

    def __call__(self, points):
        # Here a pattern of offsets is added to the point sequence in a repeating order
        # e.g. offset1 = 1; offset2 = 5; offset3 = 8; And then start over with offset1, etc...
        _points = np.array(points).T
        _extended_offset_x = np.resize(self.pattern_x, len(_points[0]))
        _extended_offset_y = np.resize(self.pattern_x, len(_points[1]))
        _points[0] = _points[0] + _extended_offset_x
        _points[1] = _points[1] + _extended_offset_y

        return _points.T

if __name__ == '__main__':
    error_simulator = PatternErrorSim(strength=100, pattern_length=10)
    #print(error_simulator.pattern_x)
    points = [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1]]
    points = [[0.0,0.0], [1.0, 1.0], [2.0,2.0], [3.0,3.0], [4.0, 4.0], [5.0,5.0], [6.0, 6.0], [7.0,7.0], [8.0,8.0], [9.0, 9.0]]
    print(error_simulator(points))