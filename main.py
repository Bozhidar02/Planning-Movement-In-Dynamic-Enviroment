import pygame
import numpy as np
import core.enviroment
import core.simulator
import entities.robot
import ui.renderer
import entities.obstacles
import ui.input_handler

WIDTH, HEIGHT = 800, 600
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Initialize components
    env = core.enviroment.Environment(WIDTH, HEIGHT)
    sim = core.simulator.Simulator(env)
    renderer = ui.renderer.Renderer(screen)
    input_handler = ui.input_handler.InputHandler(sim)

    running_app = True
    while running_app:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_app = False

            # Switch Modes
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_1: sim.mode = "static"
            #     if event.key == pygame.K_2: sim.mode = "dynamic"
            #     if event.key == pygame.K_3: sim.mode = "robot"
            #     if event.key == pygame.K_4: sim.mode = "goal"
            #     if event.key == pygame.K_5: sim.mode = "obstacle_goal"
            #     if event.key == pygame.K_s: sim.running = not sim.running  # Toggle Start/Stop
            #     if event.key == pygame.K_r: sim.reset()
            input_handler.handle_event(event)
            # Add objects with Mouse
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     m_pos = pygame.mouse.get_pos()
            #
            #     if sim.mode == "static":
            #
            #         new_rect = pygame.Rect(m_pos[0] - 25, m_pos[1] - 25, 50, 50)
            #         env.static_obstacles.append(entities.obstacles.StaticObstacle(new_rect))
            #
            #     elif sim.mode == "dynamic":
            #         env.dynamic_obstacles.append(entities.obstacles.DynamicObstacle(m_pos, 15))
            #
            #     elif sim.mode == "robot":
            #         sim.robot = entities.robot.Robot(m_pos)
            #
            #     elif sim.mode == "goal":
            #         sim.goal = np.array(m_pos, dtype=float)

        sim.update()

        # 4. Rendering
        renderer.draw(sim)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
