import numpy as np
from config import WIDTH


class StaticObstacle:
    def __init__(self, rect):
        self.rect = rect
        self.radius = 10


class DynamicObstacle:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([2.0, 0])  # horizontal movement

    def update(self):
        self.pos += self.vel

        if self.pos[0] > WIDTH:
            self.pos[0] = 0
        if self.pos[0] < 0:
            self.pos[0] = WIDTH
