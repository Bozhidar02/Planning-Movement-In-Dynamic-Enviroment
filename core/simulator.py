class Simulator:
    def __init__(self, environment):
        self.env = environment
        self.robot = None
        self.goal = None
        self.mode = "static"
        self.running = False

    def update(self):
        if not self.running:
            return

        if self.robot:
            self.robot.update(self.goal)

        for obs in self.env.dynamic_obstacles:
            obs.update()

    def reset(self):
        self.env.reset()
        self.robot = None
        self.goal = None
        self.running = False
