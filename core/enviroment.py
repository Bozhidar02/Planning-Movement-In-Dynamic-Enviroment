class Environment:
    def __init__(self):
        self.static_obstacles = []
        self.dynamic_obstacles = []

    def reset(self):
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()