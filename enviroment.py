import numpy as np


class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.static_obstacles = []
        self.dynamic_obstacles = []

    def add_static_obstacle(self, obstacle):
        self.static_obstacles.append(obstacle)

    def add_dynamic_obstacle(self, obstacle):
        self.dynamic_obstacles.append(obstacle)

    def update(self):
        for obs in self.dynamic_obstacles:
            obs.update(self)

    def is_collision(self, point):
        for obs in self.static_obstacles:
            if obs.contains(point):
                return True

        for obs in self.dynamic_obstacles:
            if obs.contains(point):
                return True

        return False
