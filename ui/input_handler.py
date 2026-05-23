import pygame
import numpy as np
from utils.helpers import snap_to_grid
from entities.robot import Robot
from entities.obstacles import StaticObstacle, DynamicObstacle, ObstacleGoal


class InputHandler:
    def __init__(self, simulator):
        self.sim = simulator
        self.dragging = False
        self.drag_button = None

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Left click only for placement; right click erases static obstacles
            if event.button == 1:
                self.dragging = True
                self.drag_button = 1
                self._handle_click(pygame.mouse.get_pos())
            elif event.button == 3:
                self.dragging = True
                self.drag_button = 3
                self._erase_static(pygame.mouse.get_pos())

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            self.drag_button = None

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging and self.drag_button == 1 and self.sim.mode == "static":
                self._handle_click(pygame.mouse.get_pos())
            elif self.dragging and self.drag_button == 3:
                self._erase_static(pygame.mouse.get_pos())

    def _handle_key(self, key):
        key_map = {
            pygame.K_1: "static",
            pygame.K_2: "dynamic",
            pygame.K_3: "robot",
            pygame.K_4: "goal",
            pygame.K_5: "obstacle_goal",
        }
        if key in key_map:
            self.sim.mode = key_map[key]
        elif key == pygame.K_s:
            self.sim.running = True
        elif key == pygame.K_r:
            self.sim.reset()
        elif key == pygame.K_c:
            # Clear all static obstacles
            self.sim.env.static_obstacles.clear()
            self.sim._cached_grid = None
            self.sim._cached_dynamic_grid = None

    def _handle_click(self, pos):
        # Don't place on toolbar area
        if pos[1] < 48:
            return

        x, y = snap_to_grid(pos)
        m_pos = np.array(pos, dtype=float)

        if self.sim.mode == "static":
            new_rect = pygame.Rect(x - 10, y - 10, 20, 20)
            for obs in self.sim.env.static_obstacles:
                if obs.rect.colliderect(new_rect):
                    return
            self.sim.env.static_obstacles.append(StaticObstacle(new_rect))
            # Invalidate grid cache when map changes
            self.sim._cached_grid = None
            self.sim._cached_dynamic_grid = None

        elif self.sim.mode == "dynamic":
            self.sim.env.dynamic_obstacles.append(DynamicObstacle((x, y), 15))

        elif self.sim.mode == "robot":
            self.sim.robot = Robot((x, y))

        elif self.sim.mode == "goal":
            self.sim.goal = np.array([x, y], dtype=float)

        elif self.sim.mode == "obstacle_goal":
            for obs in self.sim.env.dynamic_obstacles:
                if np.linalg.norm(obs.pos - m_pos) <= obs.radius:
                    self.sim.selected_obstacle = obs
                    return
            if self.sim.selected_obstacle is not None:
                self.sim.selected_obstacle.goal = ObstacleGoal(m_pos)
                self.sim.selected_obstacle.path = []
                self.sim.selected_obstacle.path_index = 0
                self.sim.selected_obstacle = None

    def _erase_static(self, pos):
        if pos[1] < 48:
            return
        m_pos = np.array(pos, dtype=float)
        before = len(self.sim.env.static_obstacles)
        self.sim.env.static_obstacles = [
            obs for obs in self.sim.env.static_obstacles
            if not obs.rect.collidepoint(pos)
        ]
        if len(self.sim.env.static_obstacles) != before:
            self.sim._cached_grid = None
            self.sim._cached_dynamic_grid = None