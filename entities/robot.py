import numpy as np


class Robot:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.speed = 2

    def update(self, goal):
        if goal is None:
            return

        direction = goal - self.pos
        dist = np.linalg.norm(direction)

        if dist > 1:
            direction /= dist
            self.pos += direction * self.speed
