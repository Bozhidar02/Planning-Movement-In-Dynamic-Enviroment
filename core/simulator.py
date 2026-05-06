import numpy as np

from utils.collision import circle_circle_collision, circle_rect_collision


class Simulator:
    def __init__(self, environment):
        self.env = environment
        self.robot = None
        self.goal = None
        self.mode = "static"
        self.running = False

    def _collides_with_static(self, pos, radius):
        for static in self.env.static_obstacles:
            rect = (static.rect.x, static.rect.y,
                    static.rect.width, static.rect.height)

            if circle_rect_collision(pos, radius, rect):
                return True
        return False

    def _collides_with_dynamic(self, pos, moving_obj):
        for obs in self.env.dynamic_obstacles:
            if obs is moving_obj:
                continue

            if circle_circle_collision(pos, moving_obj.radius, obs.pos, obs.radius):
                return True
        return False

    def _update_dynamic_obstacles(self):
        for obs in self.env.dynamic_obstacles:
            # Sub-stepping for dynamic obstacles to prevent tunneling
            steps = 4
            step_vel = obs.vel / steps

            for _ in range(steps):
                proposed = obs.pos + step_vel

                # Check collision at the proposed step
                if self._collides_with_static(proposed, obs.radius) or \
                        self._collides_with_dynamic(proposed, obs):

                    # COLLISION DETECTED:
                    # 1. Reverse velocity
                    obs.vel *= -1
                    # 2. Stop moving for this frame to avoid getting stuck
                    # inside the collision zone
                    break
                else:
                    # Path is clear, update position
                    obs.pos = proposed

    def _update_robot(self):
        print("Robo update: ")
        if self.robot is None or self.goal is None:
            return

        direction = self.goal - self.robot.pos
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return

        direction = direction / dist
        remaining = self.robot.speed
        step_size = 0.1  # Smaller steps increase precision

        while remaining > 0:
            step = min(step_size, remaining)
            proposed = self.robot.pos + direction * step
            print("Proposed next position: ", proposed)
            # If the next tiny step hits something, we stop moving this frame
            if self._collides_with_static(proposed, self.robot.radius) or \
                    self._collides_with_dynamic(proposed, self.robot):
                # Optimization: You could implement "sliding" here
                # by projecting the direction onto the obstacle tangent.
                break

            self.robot.pos = proposed
            remaining -= step

    def update(self):
        print("Running simulator...")
        if not self.running:
            print("Simulator not running yet")
            return
        print("Updating everything")
        self._update_dynamic_obstacles()
        self._update_robot()

    def reset(self):
        self.env.reset()
        self.robot = None
        self.goal = None
        self.running = False
