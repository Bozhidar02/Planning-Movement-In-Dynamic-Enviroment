import numpy as np
from config import WIDTH


class StaticObstacle:
    def __init__(self, rect):
        self.rect = rect
        self.radius = 10


class DynamicObstacle:
    def __init__(self, pos, radius):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius
        self.vel = np.array([2.0, 1.5])
        self.goal = None
        self.path = []
        self.path_index = 0


class ObstacleGoal:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
