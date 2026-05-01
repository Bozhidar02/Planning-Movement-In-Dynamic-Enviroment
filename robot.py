import numpy as np


class Robot:
    def __init__(self, start, goal, speed=1.0, radius=1.0):
        self.position = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.speed = speed
        self.radius = radius
        self.path = [self.position.copy()]

    def move_towards_goal(self, environment):
        direction = self.goal - self.position
        dist = np.linalg.norm(direction)

        if dist < 1e-3:
            return  # reached goal

        direction = direction / dist
        new_pos = self.position + direction * self.speed

        if not environment.is_collision(new_pos):
            self.position = new_pos
            self.path.append(self.position.copy())
