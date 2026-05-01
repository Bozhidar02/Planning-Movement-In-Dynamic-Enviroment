import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

from obstacles import StaticRectangle, DynamicObstacle
from robot import Robot


class Visualizer:
    def __init__(self, environment):
        self.env = environment
        self.robot = None

        self.mode = "static"  # modes: static, dynamic, robot, goal
        self.running = False

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        self.ax.set_xlim(0, environment.width)
        self.ax.set_ylim(0, environment.height)
        self.ax.set_aspect('equal')

        # Plots
        self.robot_plot, = self.ax.plot([], [], 'bo')
        self.goal_plot, = self.ax.plot([], [], 'gx', markersize=10)

        self.dynamic_plots = []

        # Buttons
        self._create_buttons()

        # Mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.title = self.ax.set_title(f"Mode: {self.mode}")

    # ---------------- BUTTONS ---------------- #
    def _create_buttons(self):
        ax_static = plt.axes([0.1, 0.1, 0.1, 0.075])
        ax_dynamic = plt.axes([0.21, 0.1, 0.1, 0.075])
        ax_robot = plt.axes([0.32, 0.1, 0.1, 0.075])
        ax_goal = plt.axes([0.43, 0.1, 0.1, 0.075])
        ax_start = plt.axes([0.6, 0.1, 0.1, 0.075])
        ax_reset = plt.axes([0.71, 0.1, 0.1, 0.075])

        self.btn_static = Button(ax_static, 'Static')
        self.btn_dynamic = Button(ax_dynamic, 'Dynamic')
        self.btn_robot = Button(ax_robot, 'Robot')
        self.btn_goal = Button(ax_goal, 'Goal')
        self.btn_start = Button(ax_start, 'Start')
        self.btn_reset = Button(ax_reset, 'Reset')

        self.btn_static.on_clicked(lambda e: self.set_mode("static"))
        self.btn_dynamic.on_clicked(lambda e: self.set_mode("dynamic"))
        self.btn_robot.on_clicked(lambda e: self.set_mode("robot"))
        self.btn_goal.on_clicked(lambda e: self.set_mode("goal"))
        self.btn_start.on_clicked(self.start)
        self.btn_reset.on_clicked(self.reset)

    def set_mode(self, mode):
        self.mode = mode
        self.title.set_text(f"Mode: {mode}")
        self.fig.canvas.draw_idle()

    # ---------------- CLICK HANDLER ---------------- #
    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if self.mode == "static":
            size = 6
            x0 = x - size / 2
            y0 = y - size / 2
            rect = StaticRectangle(x0, y0, size, size)
            self.env.add_static_obstacle(rect)
            self.ax.add_patch(plt.Rectangle((x0, y0), size, size, color='gray'))

        elif self.mode == "dynamic":
            obs = DynamicObstacle([x, y], [0.5, 0.3])
            self.env.add_dynamic_obstacle(obs)

            plot, = self.ax.plot([x], [y], 'ro')
            self.dynamic_plots.append(plot)

        elif self.mode == "robot":
            if self.robot is None:
                self.robot = Robot(start=[x, y], goal=[x, y])
            else:
                self.robot.position = np.array([x, y])

        elif self.mode == "goal":
            if self.robot is None:
                print("Place robot first!")
                return

            self.robot.goal = np.array([x, y])

            self.goal_plot.set_data([x], [y])

        self.fig.canvas.draw()

    # ---------------- SIMULATION ---------------- #
    def start(self, event):
        if not self.robot:
            print("Place robot first!")
            return

        self.running = True
        self.ani = FuncAnimation(self.fig, self.update, interval=50)

    def reset(self, event):
        self.running = False

        self.env.static_obstacles.clear()
        self.env.dynamic_obstacles.clear()
        self.dynamic_plots.clear()
        self.robot = None

        self.ax.cla()
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')

        # redraw title
        self.title = self.ax.set_title(f"Mode: {self.mode}")

        self.fig.canvas.draw_idle()

    def update(self, frame):
        if not self.running or not self.robot:
            return

        self.env.update()
        self.robot.move_towards_goal(self.env)

        # Robot
        self.robot_plot.set_data([self.robot.position[0]], [self.robot.position[1]])

        # Dynamic obstacles
        for i, obs in enumerate(self.env.dynamic_obstacles):
            if i >= len(self.dynamic_plots):
                plot, = self.ax.plot([], [], 'ro')
                self.dynamic_plots.append(plot)

            self.dynamic_plots[i].set_data(
                [obs.position[0]],
                [obs.position[1]]
            )

        return [self.robot_plot] + self.dynamic_plots

    def show(self):
        plt.show()
