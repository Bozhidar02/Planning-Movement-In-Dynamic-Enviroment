import pygame
import numpy as np
from utils.helpers import snap_to_grid
from entities.robot import Robot
from entities.obstacles import StaticObstacle, DynamicObstacle


class InputHandler:
    def __init__(self, simulator):
        self.sim = simulator
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.dragging = True
            self._handle_click(pygame.mouse.get_pos())
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

    def _handle_key(self, key):
        if key == pygame.K_1:
            self.sim.mode = "static"
        elif key == pygame.K_2:
            self.sim.mode = "dynamic"
        elif key == pygame.K_3:
            self.sim.mode = "robot"
        elif key == pygame.K_4:
            self.sim.mode = "goal"
        elif key == pygame.K_5:
            self.sim.mode = "obstacle_goal"
        elif key == pygame.K_s:
            self.sim.running = True
        elif key == pygame.K_r:
            self.sim.reset()

    def _handle_click(self, pos):
        x, y = snap_to_grid(pos)
        m_pos = np.array(pos, dtype=float)

        if self.sim.mode == "static":
            new_rect = pygame.Rect(x - 10, y - 10, 20, 20)

            # Prevent overlapping duplicates while dragging
            for obs in self.sim.env.static_obstacles:
                if obs.rect.colliderect(new_rect):
                    return

            self.sim.env.static_obstacles.append(StaticObstacle(new_rect))

        elif self.sim.mode == "dynamic":
            self.sim.env.dynamic_obstacles.append(DynamicObstacle((x, y), 15))

        elif self.sim.mode == "robot":
            self.sim.robot = Robot((x, y))

        elif self.sim.mode == "goal":
            self.sim.goal = np.array([x, y], dtype=float)

        elif self.sim.mode == "obstacle_goal":
            print("Click with new mode")
            # STEP 1: Try selecting obstacle
            for obs in self.sim.env.dynamic_obstacles:

                if np.linalg.norm(obs.pos - m_pos) <= obs.radius:
                    self.sim.selected_obstacle = obs
                    print("Selected obstacle")
                    return

            # STEP 2: Place goal for selected obstacle
            if self.sim.selected_obstacle is not None:
                from entities.obstacles import ObstacleGoal

                self.sim.selected_obstacle.goal = ObstacleGoal(m_pos)

                # Force A* recompute
                self.sim.selected_obstacle.path = []
                self.sim.selected_obstacle.path_index = 0

                print("Placed obstacle goal")

                # Optional: deselect after placing
                self.sim.selected_obstacle = None
