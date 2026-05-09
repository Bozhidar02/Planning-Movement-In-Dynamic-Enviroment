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

            input_handler.handle_event(event)

        sim.update()

        # 4. Rendering
        renderer.draw(sim)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
