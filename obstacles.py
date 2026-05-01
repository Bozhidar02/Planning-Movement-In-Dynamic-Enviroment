import numpy as np

class StaticRectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, point):
        px, py = point
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


class DynamicObstacle:
    def __init__(self, position, velocity, radius=2.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius

    def update(self, environment):
        next_pos = self.position + self.velocity

        # Check wall boundaries
        if not (0 <= next_pos[0] <= environment.width):
            self.velocity[0] *= -1
            next_pos = self.position + self.velocity

        if not (0 <= next_pos[1] <= environment.height):
            self.velocity[1] *= -1
            next_pos = self.position + self.velocity

        # Check collision with static obstacles
        collision = False
        for obs in environment.static_obstacles:
            if obs.contains(next_pos):
                collision = True
                break

        if collision:
            # Reverse direction if hitting obstacle
            self.velocity *= -1
            next_pos = self.position + self.velocity

        self.position = next_pos

    def contains(self, point):
        return np.linalg.norm(point - self.position) <= self.radius
