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
            pygame.draw.circle(self.screen, RED, obs.pos.astype(int), 8)

        # Robot
        if sim.robot:
            pygame.draw.circle(self.screen, BLUE, sim.robot.pos.astype(int), 10)

        # Goal
        if sim.goal is not None:
            pygame.draw.circle(self.screen, GREEN, sim.goal.astype(int), 10)

        # UI text
        text = self.font.render(
            f"Mode: {sim.mode} | 1:Static 2:Dynamic 3:Robot 4:Goal | S:Start R:Reset",
            True,
            BLACK,
        )
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
