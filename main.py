import pygame
import numpy as np
import core.enviroment
import core.simulator
import entities.robot
import ui.renderer
import entities.obstacles
import ui.input_handler

WIDTH, HEIGHT = 900, 650
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Robot Simulator")
    clock = pygame.time.Clock()

    env = core.enviroment.Environment(WIDTH, HEIGHT)
    sim = core.simulator.Simulator(env)
    renderer = ui.renderer.Renderer(screen)
    input_handler = ui.input_handler.InputHandler(sim)

    running_app = True
    while running_app:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_app = False

            # Toolbar mode switching via mouse click
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for mode, rect in renderer._btn_rects.items():
                    if rect.collidepoint(event.pos):
                        sim.mode = mode
                        break

            input_handler.handle_event(event)

        sim.update()
        renderer.draw(sim)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()