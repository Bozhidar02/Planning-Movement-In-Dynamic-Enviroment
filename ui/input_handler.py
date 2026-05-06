import pygame
import numpy as np
from utils.helpers import snap_to_grid
from entities.robot import Robot
from entities.obstacles import StaticObstacle, DynamicObstacle


class InputHandler:
    def __init__(self, simulator):
        self.sim = simulator

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_click(pygame.mouse.get_pos())

    def _handle_key(self, key):
        if key == pygame.K_1:
            self.sim.mode = "static"
        elif key == pygame.K_2:
            self.sim.mode = "dynamic"
        elif key == pygame.K_3:
            self.sim.mode = "robot"
        elif key == pygame.K_4:
            self.sim.mode = "goal"
        elif key == pygame.K_s:
            self.sim.running = True
        elif key == pygame.K_r:
            self.sim.reset()

    def _handle_click(self, pos):
        x, y = snap_to_grid(pos)

        if self.sim.mode == "static":
            rect = pygame.Rect(x - 10, y - 10, 20, 20)
            self.sim.env.static_obstacles.append(StaticObstacle(rect))

        elif self.sim.mode == "dynamic":
            self.sim.env.dynamic_obstacles.append(DynamicObstacle((x, y)))

        elif self.sim.mode == "robot":
            self.sim.robot = Robot((x, y))

        elif self.sim.mode == "goal":
            self.sim.goal = np.array([x, y], dtype=float)
