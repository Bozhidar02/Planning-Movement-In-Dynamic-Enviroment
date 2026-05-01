from enviroment import Environment
from robot import Robot
from obstacles import StaticRectangle, DynamicObstacle
from visualizer import Visualizer
import matplotlib

matplotlib.use('QtAgg')


def main():
    env = Environment(100, 100)

    # Static obstacles
    env.add_static_obstacle(StaticRectangle(30, 30, 10, 40))
    env.add_static_obstacle(StaticRectangle(60, 20, 20, 10))

    # Dynamic obstacles
    env.add_dynamic_obstacle(DynamicObstacle([20, 80], [0.5, -0.3]))
    env.add_dynamic_obstacle(DynamicObstacle([70, 70], [-0.4, 0.2]))

    # Robot
    robot = Robot(start=[10, 10], goal=[90, 90], speed=0.8)

    # Visualizer
    vis = Visualizer(env, robot)
    vis.animate()


if __name__ == "__main__":
    main()
