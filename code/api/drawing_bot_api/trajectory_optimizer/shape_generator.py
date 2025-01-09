import random
from drawing_bot_api.shapes import Line, PartialCircle
from math import sqrt
from drawing_bot_api.delta_utils import ik_delta
from drawing_bot_api.logger import Log
import time

MIN_NUM_OF_SHAPES = 2
MAX_NUM_OF_SHAPES = 6
RESTING_POINT = [0, 40]
START_POINT = [0, 100]

MIN_CIRCLE_RADIUS = 10

DOMAIN = [-70, 70, 90, 160] # left, right, bottom, top

class ShapeGenerator:
    shapes = []
    num_of_shapes = 0
    logger = Log(0)

    def __call__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        #self.logger('Generating shapes...')
        self._init()

        #self.logger(f'Number of Shapes: {self.num_of_shapes}')

        for _index in range(self.num_of_shapes):
            while True:
                _toss = random.randint(0, 10)
                _shape = None

                if not _toss:
                    #self.logger(f'Generating line...')
                    _shape = self._get_line()
                    #self.logger(f'Generated line: {self.shapes[-1]}')
                else:
                    #self.logger(f'Generating partial circle...')
                    _shape = self._get_partial_circle()
                    #self.logger(f'Generated partial circle: {self.shapes[-1]}')
                
                self.logger(f'Testing_shape {_index}')
                if _shape and not self._test_shape(_shape):
                    self.shapes.append(_shape)
                    break

        self.shapes.append(Line(self.shapes[-1].end_point, START_POINT))
        self.shapes.append(Line(START_POINT, RESTING_POINT))
        
        return self.shapes

    def _init(self):
        self.num_of_shapes = random.randint(MIN_NUM_OF_SHAPES, MAX_NUM_OF_SHAPES)
        self.shapes = [Line(RESTING_POINT, START_POINT)]

    def _get_partial_circle(self):
        _shape = None
        _start_point = self.shapes[-1].end_point
        _end_point = [random.randint(DOMAIN[0], DOMAIN[1]), random.randint(DOMAIN[2], DOMAIN[3])]
        _delta = [_end_point[0] - _start_point[0], _end_point[1] - _start_point[1]]
        _length = sqrt(pow(_delta[0], 2) + pow(_delta[1], 2))

        if _length > 2*MIN_CIRCLE_RADIUS:
            _radius = random.randint(int(_length/1.5), int(1.5*_length))
            _direction = (random.randint(0, 1) - 0.5) * 2
            _big_angle = (random.randint(0, 1) - 0.5) * 2
            _shape = PartialCircle(_start_point, _end_point, _radius, _direction, big_angle=_big_angle)

        return _shape

    def _get_line(self):
        _shape = None
        _start_point = self.shapes[-1].end_point
        _end_point = [random.randint(DOMAIN[0], DOMAIN[1]), random.randint(DOMAIN[2], DOMAIN[3])]
        _shape = Line(_start_point, _end_point)
        return _shape
    
    def _test_shape(self, shape):
        for i in range(100):
            _point = shape.get_point(i/100)

            if _point[1] < DOMAIN[2]: # check lower bound
                time.sleep(0.001)
                return 1
            
            _point = [_point[0]/1000, _point[1]/1000]

            try:
                ik_delta(_point)
            except:
                time.sleep(0.001)
                return 1
        time.sleep(0.001)
        return 0