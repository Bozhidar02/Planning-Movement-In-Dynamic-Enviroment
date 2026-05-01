import pygame
import sys
import numpy as np

# --- SETTINGS ---
WIDTH, HEIGHT = 800, 600
FPS = 60
GRID = 20

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Motion Planning Simulator")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 24)

# --- COLORS ---
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE = (50, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 100)

# --- DATA ---
static_obstacles = []
dynamic_obstacles = []
robot = None
goal = None

mode = "static"
running_sim = False


# --- HELPERS ---
def snap(pos):
    x, y = pos
    return (round(x / GRID) * GRID, round(y / GRID) * GRID)


# --- CLASSES ---
class Robot:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.speed = 2

    def update(self):
        if goal is None:
            return

        direction = goal - self.pos
        dist = np.linalg.norm(direction)

        if dist > 1:
            direction /= dist
            self.pos += direction * self.speed


class DynamicObstacle:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([2.0, 0])  # horizontal "car"

    def update(self):
        self.pos += self.vel

        if self.pos[0] > WIDTH:
            self.pos[0] = 0
        if self.pos[0] < 0:
            self.pos[0] = WIDTH


# --- MAIN LOOP ---
while True:
    screen.fill(WHITE)

    # --- EVENTS ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                mode = "static"
            elif event.key == pygame.K_2:
                mode = "dynamic"
            elif event.key == pygame.K_3:
                mode = "robot"
            elif event.key == pygame.K_4:
                mode = "goal"
            elif event.key == pygame.K_s:
                running_sim = True
            elif event.key == pygame.K_r:
                static_obstacles.clear()
                dynamic_obstacles.clear()
                robot = None
                goal = None
                running_sim = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = snap(pygame.mouse.get_pos())

            if mode == "static":
                rect = pygame.Rect(x - 10, y - 10, 20, 20)
                static_obstacles.append(rect)

            elif mode == "dynamic":
                dynamic_obstacles.append(DynamicObstacle((x, y)))

            elif mode == "robot":
                robot = Robot((x, y))

            elif mode == "goal":
                goal = np.array([x, y], dtype=float)

    # --- UPDATE ---
    if running_sim:
        if robot:
            robot.update()

        for obs in dynamic_obstacles:
            obs.update()

    # --- DRAW ---
    # Static obstacles
    for rect in static_obstacles:
        pygame.draw.rect(screen, GRAY, rect)

    # Dynamic obstacles
    for obs in dynamic_obstacles:
        pygame.draw.circle(screen, RED, obs.pos.astype(int), 8)

    # Robot
    if robot:
        pygame.draw.circle(screen, BLUE, robot.pos.astype(int), 10)

    # Goal
    if goal is not None:
        pygame.draw.circle(screen, GREEN, goal.astype(int), 10)

    # UI text
    text = font.render(
        f"Mode: {mode} | 1:Static 2:Dynamic 3:Robot 4:Goal | S:Start R:Reset",
        True,
        (0, 0, 0),
    )
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)