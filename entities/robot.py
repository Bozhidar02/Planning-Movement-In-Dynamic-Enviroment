import numpy as np


class Robot:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.speed = 0.5
        self.radius = 10
