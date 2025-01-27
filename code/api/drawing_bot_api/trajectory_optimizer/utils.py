import numpy as np
import keras

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.buffer = []
        self.call_counter = 0
        self.type = type

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

class Scheduler:
    def __init__(self, base_value, gamma):
        self.base_value = base_value
        self.gamma = gamma
        self.call_counter = 0

    def __call__(self, count_up=True):
        _value = self.base_value * pow(self.gamma, self.call_counter)
        if count_up:
            self.call_counter += 1
        return _value
    
def normalize_to_normal_distribution_subtract_mean(data):
    (data - np.mean(data)) / (np.std(data) + 1e-8)

def normalize_to_normal_distribution_keep_mean(data):
    data / (np.std(data) + 1e-8)

def normalize_linear_incl_neg(self, data):
    """(-inf, inf) -> [-1, 1]"""
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

def normalize_linear_only_pos(self, data):
    """(-inf, inf) -> [0, 1]"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def shift_to_range(self, data, inv_factor):
    """Input data must be in range [0, 1]"""
    return (data * (1 - inv_factor)) + inv_factor