import pygame
import numpy as np
import core.enviroment
import core.simulator
import entities.robot
import ui.renderer

# --- Configuration (or import from config) ---
WIDTH, HEIGHT = 800, 600
FPS = 60


class StaticObstacle:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)


class DynamicObstacle:
    def __init__(self, pos, radius):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius
        self.vel = np.array([2.0, 1.5])  # Initial velocity


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # 1. Initialize our components
    env = core.enviroment.Environment()
    sim = core.simulator.Simulator(env)
    renderer = ui.renderer.Renderer(screen)

    running_app = True
    while running_app:
        # 2. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_app = False

            # Switch Modes
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: sim.mode = "static"
                if event.key == pygame.K_2: sim.mode = "dynamic"
                if event.key == pygame.K_3: sim.mode = "robot"
                if event.key == pygame.K_4: sim.mode = "goal"
                if event.key == pygame.K_s: sim.running = not sim.running  # Toggle Start/Stop
                if event.key == pygame.K_r: sim.reset()

            # Add objects with Mouse
            if event.type == pygame.MOUSEBUTTONDOWN:
                m_pos = pygame.mouse.get_pos()

                if sim.mode == "static":
                    # Create a 50x50 block at mouse click
                    new_rect = pygame.Rect(m_pos[0] - 25, m_pos[1] - 25, 50, 50)
                    env.static_obstacles.append(StaticObstacle(new_rect))

                elif sim.mode == "dynamic":
                    env.dynamic_obstacles.append(DynamicObstacle(m_pos, 15))

                elif sim.mode == "robot":
                    sim.robot = entities.robot.Robot(m_pos)

                elif sim.mode == "goal":
                    sim.goal = np.array(m_pos, dtype=float)

        # 3. CRITICAL: The Simulator Update
        # This calls the collision-aware logic we wrote.
        # DO NOT update robot.pos or obstacle.pos outside of this call.
        sim.update()

        # 4. Rendering
        renderer.draw(sim)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()