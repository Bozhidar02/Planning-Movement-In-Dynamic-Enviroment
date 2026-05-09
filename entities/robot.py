import numpy as np


class Robot:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.speed = 1
        self.radius = 10
        self.heading = 0.0
        self.lidar_range = 150
        self.lidar_rays = 36
        self.lidar_data = []
