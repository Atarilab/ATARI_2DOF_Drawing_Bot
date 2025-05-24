import numpy as np
from math import sqrt, pow, atan2, sin, cos, pi, exp
from drawing_bot_api.trajectory_optimizer.config_fourier import *

class FourierCompensator:
    def __call__(self, points, type, **kwargs):
        _points = None
        if type == 'fourier':
            _points, _data1, _data2 = self.apply_compensation_fourier(points, **kwargs)
        else:
            _points, _data1, _data2 = self.apply_compensation(points, **kwargs)
        return _points, _data1, _data2
        
    def _get_phase(self, point, prev_point):
        _pointing_vector = [point[0]-prev_point[0], point[1]-prev_point[1]]
        _phase = atan2(_pointing_vector[1], _pointing_vector[0])
        return _phase
    
    def _get_point_from_phase(self, phase, radius):
        _scaler = radius
        _x_offset = _scaler * cos(phase)
        _y_offset = _scaler * sin(phase)
        return [_x_offset, _y_offset]
    
    def _phase_offsets_to_points(self, phase_offsets, normals, points):
        _points_x = points[:, :1] + normals[:, :1] * phase_offsets
        _points_y = points[:, 1:] + normals[:, 1:] * phase_offsets
        return np.hstack([_points_x, _points_y])
    
    def _get_phase_difference(self, phases):
        _phase_differences = []

        for _index in range(1, len(phases)):
            
            _phase = phases[_index]
            _prev_phase = phases[_index - 1]
            _phase_difference = _phase-_prev_phase

            if abs(_phase_difference) > pi:
                _phase_difference = -np.sign(_phase_difference) * ((2 * pi) - abs(_phase_difference))
            
            _phase_differences.append(_phase_difference)
        
        return np.array(_phase_differences)
    
    def _points_to_phases(self, trajectory):
        _points = np.array(trajectory)[1:]
        _prev_points = np.array(trajectory)[:-1]
        _direction_vectors = _points - _prev_points
        _direction_vectors = _direction_vectors.T
        _phases = np.arctan2(_direction_vectors[1], _direction_vectors[0])
        return _phases
    
    def calc_fourier_element(self, a, b, n, sample_points):
        return a * np.cos(n * sample_points) + b * np.sin(n * sample_points)
    
    def get_fourier_series(self, coefficients, length):
        _sample_points = np.arange(length, dtype='float64')

        _fourier_series = np.zeros(length)

        if ADD_CONSTANT_COEFFICIENT:
            _fourier_series += coefficients[0]
            coefficients = coefficients[1:]
        
        if FOURIER_PRE_SCALING:
            _pre_scaler = coefficients[-1]
        else:
            _pre_scaler = 1
        
        _sample_points *= _pre_scaler

        for _index in range(0, len(coefficients), 2):
            _a = coefficients[_index]
            _b = coefficients[_index + 1]
            _n = (_index + 2) // 2
            _fourier_series += self.calc_fourier_element(_a, _b, _n, _sample_points)

        return _fourier_series
    
    def _reverse_sigmoid(self, scaler, value):
        return 1 / (1 + np.exp(scaler * value))
    
    def _soft_rectangular_function(self, strength, values):
        return ( 1 / ( 1 + np.exp( 10 * strength * (np.abs(values)-1) ) ) )
    
    def apply_compensation_fourier(self, points, parameters, length=FOURIER_LENGTH):
        # format all parameters
        _points = np.array(points)

        _fourier_coefficients_indices = 2*NUM_OF_FOURIER_HARMONICS+1
        _fourier_coefficients = parameters[:_fourier_coefficients_indices]
        _exponential_decay = parameters[_fourier_coefficients_indices]

        if PHASE_DIFFERENCE_POWER is None:
            _power = parameters[_fourier_coefficients_indices+1]
        else:
            _power = PHASE_DIFFERENCE_POWER

        if ADD_EDGE_FALL_OFF:
            _edge_fall_off_strength = parameters[_fourier_coefficients_indices+2]

        # calculate and process phases
        _phases = self._points_to_phases(points)
        _normals_of_phases = np.array([np.cos(_phases + np.pi/2), np.sin(_phases + np.pi/2)]).T
        _normals_of_phases = np.append(_normals_of_phases, np.zeros((1, 2)), axis=0)

        # calculate and process phase differences
        _phase_differences = self._get_phase_difference(_phases)
        _normalized_phase_differences = _phase_differences / np.pi
        _weighted_normalized_phase_differences = np.power(np.abs(_normalized_phase_differences),np.abs(_power))
        _weighted_normalized_phase_differences = np.where(_normalized_phase_differences < 0, -_weighted_normalized_phase_differences, _weighted_normalized_phase_differences)
        if ADD_EDGE_FALL_OFF:
            _weighted_normalized_phase_differences = self._soft_rectangular_function(_edge_fall_off_strength, _weighted_normalized_phase_differences)
        # calculate fourier series
        _fourier_series = self.get_fourier_series(_fourier_coefficients, length)

        # apply exponential decay to fourier series
        _sample_points = np.arange(length)
        _exponential_fourier_series = (1 / np.exp((_exponential_decay) * _sample_points)) * _fourier_series

        # calculate super position of fourier series functions weighted by phase differences
        _point_offsets = np.zeros(np.shape(_phases))
        for _index in range(len(_point_offsets)-1):
            _points_left = len(_point_offsets) - _index
            _range = length if _points_left >= length else _points_left
            _point_offsets[_index:_index+length] = _point_offsets[_index:_index+length] + (_exponential_fourier_series[:_range] * _weighted_normalized_phase_differences[_index])

        # format offsets
        _point_offsets = np.append(0, _point_offsets)
        _point_offsets = _point_offsets.reshape(-1, 1)

        # apply offsets to points
        _new_points = self._phase_offsets_to_points(_point_offsets, _normals_of_phases, _points)
        
        return _new_points, _exponential_fourier_series, _point_offsets
    
    def apply_compensation(self, points, gain=5, damping=0.25, freq=1, fade_in=0, tanh_scaling=1, length=60):
        _points = np.array(points)

        _phases = self._points_to_phases(points)

        _normals_to_phases = np.array([np.cos(_phases + np.pi/2), np.sin(_phases + np.pi/2)]).T
        _normals_to_phases = np.append(_normals_to_phases, np.zeros((1, 2)), axis=0)

        _phase_differences = self._get_phase_difference(_phases)
        _normalized_phase_differences = _phase_differences / np.pi
        #_weighted_normalized_phase_differences = 0.5 + np.tanh(_normalized_phase_differences * tanh_scaling) * 0.55

        _sample_points = np.arange(length)
        _oscillation = (1 / np.exp((damping) * _sample_points)) * np.sin(freq * _sample_points) * np.tanh((1/fade_in) * _sample_points)
        _oscillation = (_oscillation / np.max(_oscillation)) * -gain

        _point_offsets = np.zeros(np.shape(_phases))
        for _index in range(len(_point_offsets)-length):
            _point_offsets[_index:_index+length] = _point_offsets[_index:_index+length] + (_oscillation * _normalized_phase_differences[_index])

        _point_offsets = np.append(0, _point_offsets)
        _point_offsets = _point_offsets.reshape(-1, 1)

        _new_points = self._phase_offsets_to_points(_point_offsets, _normals_to_phases, _points)
        
        return _new_points, _oscillation