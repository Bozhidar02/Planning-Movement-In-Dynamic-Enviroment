import numpy as np

from utils.collision import circle_circle_collision, circle_rect_collision


class Simulator:
    def __init__(self, environment):
        self.env = environment
        self.robot = None
        self.goal = None
        self.mode = "static"
        self.running = False
        self.selected_obstacle = None

    def _collides_with_static(self, pos, radius):
        for static in self.env.static_obstacles:
            rect = (static.rect.x, static.rect.y,
                    static.rect.width, static.rect.height)

            if circle_rect_collision(pos, radius, rect):
                return True
        return False

    def _collides_with_dynamic(self, pos, moving_obj):
        for obs in self.env.dynamic_obstacles:
            if obs is moving_obj:
                continue

            if circle_circle_collision(pos, moving_obj.radius, obs.pos, obs.radius):
                return True
        return False

    def _update_dynamic_obstacles(self):
        grid, res = self._build_grid()

        for obs in self.env.dynamic_obstacles:

            # Recompute path if needed
            if not hasattr(obs, "path") or not obs.path:
                start = (int(obs.pos[0] // res), int(obs.pos[1] // res))
                if obs.goal is None:
                    continue

                goal = (
                    int(obs.goal.pos[0] // res),
                    int(obs.goal.pos[1] // res)
                )

                obs.path = self._astar_path(start, goal, grid, res)
                obs.path_index = 0

            # Follow path
            if obs.path and obs.path_index < len(obs.path):
                target_cell = obs.path[obs.path_index]
                target = np.array(target_cell) * res

                direction = target - obs.pos
                dist = np.linalg.norm(direction)

                if dist < 5:  # reached node
                    obs.path_index += 1
                else:
                    step = (direction / max(dist, 1e-6)) * np.linalg.norm(obs.vel)

                    proposed = obs.pos + step

                    if not self._collides_with_static(proposed, obs.radius):
                        obs.pos = proposed
                    else:
                        obs.path = []

    def _update_robot(self):
        if self.robot is None or self.goal is None:
            return

        direction = self.goal - self.robot.pos
        dist = np.linalg.norm(direction)

        if dist < 1e-6:
            return

        direction = direction / dist
        remaining = self.robot.speed
        step_size = 0.1  # Smaller steps increase precision

        while remaining > 0:
            step = min(step_size, remaining)
            proposed = self.robot.pos + direction * step
            # If the next tiny step hits something, we stop moving this frame
            if self._collides_with_static(proposed, self.robot.radius) or \
                    self._collides_with_dynamic(proposed, self.robot):
                break

            self.robot.pos = proposed
            remaining -= step

    def update(self):
        if not self.running:
            return
        self._update_dynamic_obstacles()
        self._update_robot()

    def reset(self):
        self.env.reset()
        self.robot = None
        self.goal = None
        self.running = False

    def _astar_path(self, start, goal, grid, grid_size):
        import heapq

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        def neighbors(node):
            x, y = node
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    yield (nx, ny)

        start = tuple(start)
        goal = tuple(goal)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for n in neighbors(current):
                if grid[n] == 1:  # blocked
                    continue

                tentative = g_score[current] + heuristic(current, n)

                if n not in g_score or tentative < g_score[n]:
                    came_from[n] = current
                    g_score[n] = tentative
                    f = tentative + heuristic(n, goal)
                    heapq.heappush(open_set, (f, n))

        return []

    def _build_grid(self, resolution=10):
        width, height = self.env.width, self.env.height

        grid_w = int(width // resolution)
        grid_h = int(height // resolution)

        grid = np.zeros((grid_w, grid_h), dtype=int)

        for obs in self.env.static_obstacles:
            x0 = int(obs.rect.x // resolution)
            y0 = int(obs.rect.y // resolution)
            x1 = int((obs.rect.x + obs.rect.width) // resolution)
            y1 = int((obs.rect.y + obs.rect.height) // resolution)

            grid[x0:x1+1, y0:y1+1] = 1

        return grid, resolution
