import numpy as np


class Robot:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.speed = 2
        self.radius = 15
        self.heading = 0.0

        self.lidar_range = 150
        self.lidar_rays = 36
        self.lidar_data = []

        self.max_turn_rate = 0.15
        self.prediction_time = 10
        self.candidate_angles = 15

        self.stuck = False
        self._pos_history = []

        self.mode = "navigate"  # "navigate" | "wall_align" | "wall_follow"
        self.wall_follow_dir = 1  # 1 = left-hand rule, -1 = right-hand rule
        self.wall_follow_steps = 0
        self.max_wall_follow_steps = 300  # safety exit
        self.dynamic_wait_steps = 10
