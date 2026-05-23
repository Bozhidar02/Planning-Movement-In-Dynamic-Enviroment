import numpy as np
import pygame
from config import *

TOOLBAR_H = 48
TOOLBAR_BG = (30, 30, 35)
TOOLBAR_ACTIVE = (80, 120, 220)
TOOLBAR_HOVER = (55, 55, 65)
TEXT_DIM = (160, 160, 175)
TEXT_BRIGHT = (230, 230, 240)

MODES = [
    ("1", "static",       "Static wall"),
    ("2", "dynamic",      "Dynamic obs"),
    ("3", "robot",        "Robot"),
    ("4", "goal",         "Goal"),
    ("5", "obstacle goal","Obs goal"),
]

STATUS_COLORS = {
    "navigate":   (80,  180, 120),
    "wall_align": (220, 160,  50),
    "wall_follow":(220, 110,  50),
}


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font_sm = pygame.font.SysFont("segoeui", 13)
        self.font_md = pygame.font.SysFont("segoeui", 15, bold=True)
        self.font_lg = pygame.font.SysFont("segoeui", 18, bold=True)
        self._btn_rects = {}  # mode → Rect, for hover detection

    def draw(self, sim):
        self.screen.fill((20, 20, 25))
        self._draw_grid()
        self._draw_static(sim)
        self._draw_dynamic(sim)
        self._draw_robot(sim)
        self._draw_goal(sim)
        self._draw_toolbar(sim)
        self._draw_status(sim)
        pygame.display.flip()

    # ------------------------------------------------------------------ #
    #  Scene elements
    # ------------------------------------------------------------------ #
    def _draw_grid(self):
        w, h = self.screen.get_size()
        grid_color = (35, 35, 42)
        for x in range(0, w, 20):
            pygame.draw.line(self.screen, grid_color, (x, TOOLBAR_H), (x, h), 1)
        for y in range(TOOLBAR_H, h, 20):
            pygame.draw.line(self.screen, grid_color, (0, y), (w, y), 1)

    def _draw_static(self, sim):
        for obs in sim.env.static_obstacles:
            pygame.draw.rect(self.screen, (90, 95, 110), obs.rect)
            pygame.draw.rect(self.screen, (120, 125, 145), obs.rect, 1)

    def _draw_dynamic(self, sim):
        for obs in sim.env.dynamic_obstacles:
            # Body
            pygame.draw.circle(self.screen, (200, 70, 70), obs.pos.astype(int), obs.radius)
            pygame.draw.circle(self.screen, (230, 100, 100), obs.pos.astype(int), obs.radius, 2)

            # Selection ring
            if obs is sim.selected_obstacle:
                pygame.draw.circle(self.screen, (255, 220, 50),
                                   obs.pos.astype(int), obs.radius + 5, 2)

            # Goal marker
            if obs.goal is not None:
                gp = obs.goal.pos.astype(int)
                pygame.draw.circle(self.screen, (80, 200, 120), gp, 6)
                pygame.draw.circle(self.screen, (120, 230, 160), gp, 6, 1)
                # Line from obs to goal
                pygame.draw.line(self.screen, (80, 200, 120, 80),
                                 obs.pos.astype(int), gp, 1)

            # Path dots
            if hasattr(obs, "path") and obs.path:
                for cell in obs.path:
                    pos = (np.array(cell) * 10).astype(int)
                    pygame.draw.circle(self.screen, (60, 100, 200), pos, 2)

    def _draw_robot(self, sim):
        if sim.robot is None:
            return

        robot = sim.robot

        # LiDAR rays
        for _, dist, hit in robot.lidar_data:
            col = (0, 80, 80) if dist >= robot.lidar_range else (0, 160, 160)
            pygame.draw.line(self.screen, col,
                             robot.pos.astype(int), hit.astype(int), 1)
            if dist < robot.lidar_range:
                pygame.draw.circle(self.screen, (0, 220, 200),
                                   hit.astype(int), 2)

        # Robot body
        mode_col = STATUS_COLORS.get(robot.mode, (100, 100, 220))
        pygame.draw.circle(self.screen, (50, 100, 200),
                           robot.pos.astype(int), robot.radius)
        pygame.draw.circle(self.screen, mode_col,
                           robot.pos.astype(int), robot.radius, 3)

        # Heading arrow
        heading_end = robot.pos + np.array([
            np.cos(robot.heading), np.sin(robot.heading)
        ]) * (robot.radius + 10)
        pygame.draw.line(self.screen, (255, 255, 255),
                         robot.pos.astype(int), heading_end.astype(int), 2)

        # Mode label above robot
        label = self.font_sm.render(robot.mode, True, mode_col)
        self.screen.blit(label, (robot.pos[0] - label.get_width() // 2,
                                 robot.pos[1] - robot.radius - 18))

    def _draw_goal(self, sim):
        if sim.goal is None:
            return
        gp = sim.goal.astype(int)
        pygame.draw.circle(self.screen, (60, 200, 100), gp, 10)
        pygame.draw.circle(self.screen, (120, 240, 160), gp, 10, 2)
        # Crosshair
        pygame.draw.line(self.screen, (120, 240, 160), (gp[0] - 14, gp[1]), (gp[0] + 14, gp[1]), 1)
        pygame.draw.line(self.screen, (120, 240, 160), (gp[0], gp[1] - 14), (gp[0], gp[1] + 14), 1)

    # ------------------------------------------------------------------ #
    #  Toolbar
    # ------------------------------------------------------------------ #
    def _draw_toolbar(self, sim):
        w = self.screen.get_width()
        pygame.draw.rect(self.screen, TOOLBAR_BG, (0, 0, w, TOOLBAR_H))
        pygame.draw.line(self.screen, (60, 60, 75), (0, TOOLBAR_H), (w, TOOLBAR_H), 1)

        mouse_pos = pygame.mouse.get_pos()
        btn_w = 110
        x = 8

        self._btn_rects.clear()

        for key, mode, label in MODES:
            rect = pygame.Rect(x, 6, btn_w, 36)
            self._btn_rects[mode] = rect
            active = sim.mode == mode
            hovered = rect.collidepoint(mouse_pos)

            bg = TOOLBAR_ACTIVE if active else (TOOLBAR_HOVER if hovered else TOOLBAR_BG)
            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            if active:
                pygame.draw.rect(self.screen, (120, 160, 255), rect, 1, border_radius=6)

            key_surf = self.font_sm.render(key, True, (180, 180, 200) if not active else (200, 220, 255))
            lbl_surf = self.font_sm.render(label, True, TEXT_BRIGHT if active else TEXT_DIM)
            self.screen.blit(key_surf, (x + 6, 12))
            self.screen.blit(lbl_surf, (x + 18, 12))

            x += btn_w + 4

        # Right side: controls hint + start/reset
        hint_parts = [
            ("S", "start"),
            ("R", "reset"),
            ("C", "clear walls"),
            ("RMB", "erase"),
        ]
        rx = w - 8
        for key, action in reversed(hint_parts):
            a_surf = self.font_sm.render(action, True, TEXT_DIM)
            k_surf = self.font_sm.render(key, True, (200, 200, 220))
            rx -= a_surf.get_width() + 4
            self.screen.blit(a_surf, (rx, 16))
            rx -= k_surf.get_width() + 6
            self.screen.blit(k_surf, (rx, 16))
            rx -= 14

        # Running indicator
        if sim.running:
            ind_col = (80, 220, 120)
            pygame.draw.circle(self.screen, ind_col, (x + 8, TOOLBAR_H // 2), 5)
            run_surf = self.font_sm.render("running", True, ind_col)
            self.screen.blit(run_surf, (x + 18, 16))

    def _draw_status(self, sim):
        if sim.robot is None:
            return
        lines = [
            f"mode: {sim.robot.mode}",
            f"stuck: {sim.robot.stuck}",
        ]
        y = TOOLBAR_H + 8
        for line in lines:
            surf = self.font_sm.render(line, True, (120, 120, 140))
            self.screen.blit(surf, (8, y))
            y += 16