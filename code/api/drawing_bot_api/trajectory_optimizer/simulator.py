import numpy as np
from scipy.ndimage import gaussian_filter1d
from math import sqrt, pow, atan2, sin, cos, pi, exp

DECAY_FACTOR = 1.8
DAMPING_FACTOR = 0.95

class PatternErrorSim:
    def __init__(self, strength=100, pattern_length=20, seed=None):
        self.pattern_length = pattern_length

        rng = np.random.default_rng(seed)
        _random_offsets_x = rng.normal(0, 0.1, pattern_length)
        _random_offsets_y = rng.normal(0, 0.1, pattern_length)
        # Smooth the offsets
        self.pattern_x = gaussian_filter1d(_random_offsets_x, sigma=1.5) * strength
        self.pattern_y = gaussian_filter1d(_random_offsets_y, sigma=1.5) * strength

    def __call__(self, points):
        # Here a pattern of offsets is added to the point sequence in a repeating order
        # e.g. offset1 = 1; offset2 = 5; offset3 = 8; And then start over with offset1, etc...
        _points = self._apply_error_rule(points)

        return _points
    
    def _old_method(self, points):
        _points = np.array(points).T
        _extended_offset_x = np.resize(self.pattern_x, len(_points[0]))
        _extended_offset_y = np.resize(self.pattern_x, len(_points[1]))
        _points[0] = _points[0] + _extended_offset_x
        _points[1] = _points[1] + _extended_offset_y

        return _points.T
    
    def _get_phase(self, point, prev_point):
        _pointing_vector = [point[0]-prev_point[0], point[1]-prev_point[1]]
        _phase = atan2(_pointing_vector[1], _pointing_vector[0])
        return _phase
    
    def _get_point_from_phase(self, phase, radius):
        _scaler = radius
        _x_offset = _scaler * cos(phase)
        _y_offset = _scaler * sin(phase)
        return [_x_offset, _y_offset]
    
    def _apply_error_rule(self, points):
        _new_points = [[0, 0], points[0]]
        _prev_phase_offset = 0

        for _index in range(1, len(points)):
            # add offset to current point
            _point = points[_index]
            _radius = sqrt(pow(_point[0]-_new_points[-1][0], 2) + pow(_point[1] - _new_points[-1][1], 2))
            print(f'Current point: {_point};    Previous point: {_new_points[-1]}')
            print(f'Radius: {_radius}')

            # Calculate phase of vector between current point and last point; Same for previous point and the one before
            _phase = self._get_phase(_point, _new_points[-1])
            _prev_phase = self._get_phase(_new_points[-1], _new_points[-2])
            print(f'current phase: {_phase};    Prev phase: {_prev_phase}')

            # Calculate difference between the last two phases
            _phase_difference = _phase-_prev_phase
            print(f'Phase difference before: {_phase_difference}')
            if abs(_phase_difference) > pi:
                _phase_difference = -np.sign(_phase_difference) * ((2 * pi) - abs(_phase_difference))
            print(f'Phase difference after: {_phase_difference}')

            damping = -(1/(1 + exp(DAMPING_FACTOR * abs(_prev_phase_offset))))
            _phase_offset = damping * (_phase_difference) * DECAY_FACTOR
            #_phase_offset = -0.65 * (_phase_difference)
            print(f'Phase offset: {_phase_offset}')

            # Calculate offset for next point
            _new_vector = self._get_point_from_phase(_phase+_phase_offset, _radius)
            print(f'Offset vector: {_new_vector}')

            _new_point = [_new_points[-1][0]+_new_vector[0], _new_points[-1][1]+_new_vector[1]]
            print(f'New point: {_new_point}')

            # overwritting all 'prev'-parameters with current parameters
            _new_points.append(_new_point)
            _prev_phase_offset = _phase_offset
            print(f'----------------------------------')
        
        _new_points.pop(0)
        return _new_points


if __name__ == '__main__':
    error_simulator = PatternErrorSim(strength=100, pattern_length=10)
    #print(error_simulator.pattern_x)
    #points = [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [1,1]]
    points = [[0.0,0.0], [1.0, 2.0], [2.0,3.0], [3.0,4.0], [4.0, 3.0], [5.0,6.0], [6.0, 9.0], [7.0,8.0], [8.0,9.0], [9.0, 0]]
    print(error_simulator(points))