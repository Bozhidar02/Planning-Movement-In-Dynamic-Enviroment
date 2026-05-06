import numpy as np


class Robot:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.speed = 1
        self.radius = 10
