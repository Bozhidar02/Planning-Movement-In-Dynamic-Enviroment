import numpy as np
import pygame
from config import *


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, sim):
        self.screen.fill(WHITE)

        # Static obstacles
        for obs in sim.env.static_obstacles:
            pygame.draw.rect(self.screen, GRAY, obs.rect)

        # Dynamic obstacles
        for obs in sim.env.dynamic_obstacles:
            pygame.draw.circle(
                self.screen,
                RED,
                obs.pos.astype(int),
                obs.radius
            )
            if obs is sim.selected_obstacle:
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 0),
                    obs.pos.astype(int),
                    obs.radius + 4,
                    2
                )
            if obs.goal is not None:
                pygame.draw.circle(
                    self.screen,
                    (0, 255, 0),
                    obs.goal.pos.astype(int),
                    6
                )
            if hasattr(obs, "path") and obs.path:
                for cell in obs.path:
                    pos = (np.array(cell) * 10).astype(int)  # 10 = grid resolution
                    pygame.draw.circle(self.screen, (0, 0, 255), pos, 3)

        # Robot
        if sim.robot:
            for _, _, hit in sim.robot.lidar_data:
                pygame.draw.line(
                    self.screen,
                    (0, 255, 255),
                    sim.robot.pos.astype(int),
                    hit.astype(int),
                    1
                )

                pygame.draw.circle(
                    self.screen,
                    (255, 255, 0),
                    hit.astype(int),
                    2
                )
            pygame.draw.circle(
                self.screen,
                BLUE,
                sim.robot.pos.astype(int),
                sim.robot.radius
            )
            heading_end = (
                sim.robot.pos
                + np.array([
                    np.cos(sim.robot.heading),
                    np.sin(sim.robot.heading)
                ]) * 20
            )

            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                sim.robot.pos.astype(int),
                heading_end.astype(int),
                2
            )

        # Goal
        if sim.goal is not None:
            pygame.draw.circle(self.screen, GREEN, sim.goal.astype(int), 10)

        # UI text
        text = self.font.render(
            f"Mode: {sim.mode} | 1:Static 2:Dynamic 3:Robot 4:Goal 5:Obstacle Goal | S:Start R:Reset",
            True,
            BLACK,
        )
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
