import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Visualizer:
    def __init__(self, environment, robot):
        self.env = environment
        self.robot = robot

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, environment.width)
        self.ax.set_ylim(0, environment.height)

        self.robot_plot, = self.ax.plot([], [], 'bo', label="Robot")
        self.goal_plot, = self.ax.plot(robot.goal[0], robot.goal[1], 'gx', markersize=10)

        self.dynamic_plots = []

        # Draw static obstacles
        for obs in self.env.static_obstacles:
            rect = plt.Rectangle((obs.x, obs.y), obs.w, obs.h, color='gray')
            self.ax.add_patch(rect)

        # Prepare dynamic obstacle plots
        for _ in self.env.dynamic_obstacles:
            plot, = self.ax.plot([], [], 'ro')
            self.dynamic_plots.append(plot)

    def update(self, frame):
        self.env.update()
        self.robot.move_towards_goal(self.env)

        # Update robot
        self.robot_plot.set_data([self.robot.position[0]], [self.robot.position[1]])

        # Update dynamic obstacles
        for i, obs in enumerate(self.env.dynamic_obstacles):
            self.dynamic_plots[i].set_data([obs.position[0]], [obs.position[1]])

        return [self.robot_plot] + self.dynamic_plots

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=300, interval=50)
        plt.legend()
        plt.show()
