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

        self._sense_obstacles()
        self._detect_stuck()

        robot = self.robot
        direction = np.zeros(2)

        if robot.mode == "navigate":
            if robot.stuck:
                robot.mode = "wall_align"
                robot.wall_follow_dir = 1
                robot.wall_follow_steps = 0
                robot.stuck = False
                robot._pos_history.clear()

            direction = self._choose_best_direction()
            if np.linalg.norm(direction) < 1e-6:
                return
            proposed = robot.pos + direction * robot.speed
            if not self._trajectory_collision(proposed, robot.radius):
                robot.pos = proposed

        elif robot.mode == "wall_align":
            aligned = self._align_to_wall()
            if aligned:
                robot.mode = "wall_follow"

        elif robot.mode == "wall_follow":
            robot.wall_follow_steps += 1

            # Stuck detection during wall follow
            if robot.stuck:
                robot.wall_follow_dir *= -1  # try the other side
                robot.wall_follow_steps = 0
                robot.stuck = False
                robot._pos_history.clear()
                return

            min_steps = 60
            can_exit = robot.wall_follow_steps > min_steps

            if can_exit and (self._goal_is_reachable() or
                             robot.wall_follow_steps > robot.max_wall_follow_steps):
                robot.mode = "navigate"
                robot._pos_history.clear()
                return

            direction = self._wall_follow_direction()
            if np.linalg.norm(direction) < 1e-6:
                return
            proposed = robot.pos + direction * robot.speed
            if not self._trajectory_collision(proposed, robot.radius):
                robot.pos = proposed

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
            dirs = [
                (1, 0), (-1, 0),
                (0, 1), (0, -1),
                (1, 1), (1, -1),
                (-1, 1), (-1, -1)
            ]
            for dx, dy in dirs:

                nx, ny = x + dx, y + dy

                if not (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]):
                    continue

                if grid[nx, ny] == 1:
                    continue

                # Prevent diagonal corner cutting
                if dx != 0 and dy != 0:
                    if grid[x + dx, y] == 1 or grid[x, y + dy] == 1:
                        continue

                yield nx, ny

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

    def _build_grid(self, resolution=10, inflation_radius=15):

        width, height = self.env.width, self.env.height

        grid_w = int(width // resolution)
        grid_h = int(height // resolution)

        grid = np.zeros((grid_w, grid_h), dtype=int)

        inflation_cells = int(np.ceil(inflation_radius / resolution))

        for obs in self.env.static_obstacles:
            x0 = int(obs.rect.x // resolution) - inflation_cells
            y0 = int(obs.rect.y // resolution) - inflation_cells

            x1 = int((obs.rect.x + obs.rect.width) // resolution) + inflation_cells
            y1 = int((obs.rect.y + obs.rect.height) // resolution) + inflation_cells

            x0 = max(0, x0)
            y0 = max(0, y0)

            x1 = min(grid_w - 1, x1)
            y1 = min(grid_h - 1, y1)

            grid[x0:x1 + 1, y0:y1 + 1] = 1

        return grid, resolution

    def _sense_obstacles(self):

        robot = self.robot

        if robot is None:
            return

        robot.lidar_data = []

        angles = np.linspace(
            0,
            2 * np.pi,
            robot.lidar_rays,
            endpoint=False
        )

        for angle in angles:

            direction = np.array([
                np.cos(angle),
                np.sin(angle)
            ])

            hit_point = robot.pos + direction * robot.lidar_range
            min_dist = robot.lidar_range

            # Ray marching
            for d in np.linspace(0, robot.lidar_range, 100):

                point = robot.pos + direction * d

                collided = False

                # Static obstacles
                for obs in self.env.static_obstacles:

                    rect = (
                        obs.rect.x,
                        obs.rect.y,
                        obs.rect.width,
                        obs.rect.height
                    )

                    if circle_rect_collision(point, 1, rect):
                        min_dist = d
                        hit_point = point
                        collided = True
                        break

                # Dynamic obstacles
                for obs in self.env.dynamic_obstacles:

                    if np.linalg.norm(point - obs.pos) <= obs.radius:
                        min_dist = d
                        hit_point = point
                        collided = True
                        break

                if collided:
                    break

            robot.lidar_data.append(
                (angle, min_dist, hit_point)
            )

    def _compute_navigation_vector(self):

        robot = self.robot

        if robot is None or self.goal is None:
            return np.zeros(2)

        # -----------------------------
        # GOAL ATTRACTION
        # -----------------------------
        goal_vector = self.goal - robot.pos

        goal_dist = np.linalg.norm(goal_vector)

        if goal_dist > 1e-6:
            goal_vector /= goal_dist

        # -----------------------------
        # OBSTACLE REPULSION
        # -----------------------------
        avoidance = np.zeros(2)

        # Temporary variable for testing
        influence = 25
        # -----------------------------
        for angle, dist, _ in robot.lidar_data:

            if dist < influence:
                strength = ((influence - dist) / influence) ** 2

                direction = np.array([
                    np.cos(angle),
                    np.sin(angle)
                ])

                avoidance -= direction * strength

        # -----------------------------
        # STUCK ESCAPE
        # -----------------------------
        if robot.stuck:

            tangent = np.zeros(2)

            for angle, dist, _ in robot.lidar_data:

                influence = 60

                if dist < influence:
                    direction = np.array([
                        np.cos(angle),
                        np.sin(angle)
                    ])

                    # Perpendicular vector
                    perp = np.array([
                        -direction[1],
                        direction[0]
                    ])

                    strength = (
                            (influence - dist)
                            / influence
                    )

                    tangent += perp * strength

            avoidance += tangent * 1.5

        # -----------------------------
        # COMBINE
        # -----------------------------
        final = goal_vector * 1.5 + avoidance

        mag = np.linalg.norm(final)

        if mag > 1e-6:
            final /= mag

        return final

    def _trajectory_collision(self, pos, radius):

        # Static obstacles
        if self._collides_with_static(pos, radius):
            return True

        # Dynamic obstacles
        for obs in self.env.dynamic_obstacles:

            if np.linalg.norm(pos - obs.pos) <= (radius + obs.radius):
                return True

        return False

    def _choose_best_direction(self):

        if self.robot is None or self.goal is None:
            return np.zeros(2)

        best_score = -1e9
        best_direction = None
        best_heading = self.robot.heading

        # Goal direction
        goal_vector = self.goal - self.robot.pos

        goal_dist = np.linalg.norm(goal_vector)

        if goal_dist < 1e-6:
            return np.zeros(2)

        goal_direction = goal_vector / goal_dist

        # Candidate steering angles
        angles = np.linspace(
            -self.robot.max_turn_rate,
            self.robot.max_turn_rate,
            self.robot.candidate_angles
        )

        for delta in angles:

            candidate_heading = (
                    self.robot.heading + delta
            )

            direction = np.array([
                np.cos(candidate_heading),
                np.sin(candidate_heading)
            ])

            simulated_pos = self.robot.pos.copy()

            collision = False

            # Simulate future motion
            for _ in range(self.robot.prediction_time):

                simulated_pos += (
                        direction * self.robot.speed
                )

                if self._trajectory_collision(
                        simulated_pos,
                        self.robot.radius
                ):
                    collision = True
                    break

            # Reject collisions immediately
            if collision:
                continue

            # -------------------------
            # SCORE COMPONENTS
            # -------------------------

            # 1. Goal alignment
            alignment = np.dot(
                direction,
                goal_direction
            )

            # 2. Goal progress
            final_goal_dist = np.linalg.norm(
                self.goal - simulated_pos
            )

            progress = -final_goal_dist

            # 3. Turning penalty
            smoothness = -abs(delta)

            # 4. Obstacle clearance from lidar
            clearance = 0.0
            for angle, dist, _ in self.robot.lidar_data:
                ray_dir = np.array([np.cos(angle), np.sin(angle)])
                alignment_to_ray = np.dot(direction, ray_dir)
                if alignment_to_ray > 0.5:  # ray roughly ahead in this direction
                    clearance -= (1.0 / max(dist, 1.0)) * alignment_to_ray

            # Continuity: penalise deviation from current heading
            continuity = -abs(candidate_heading - self.robot.heading)
            # -------------------------
            # FINAL SCORE
            # -------------------------
            score = (
                    alignment * 200
                    + progress * 1.5
                    + smoothness * 10
                    + clearance * 300  # penalise directions toward lidar hits
                    + continuity * 80
                    + (500 if self.robot.stuck else 0) * (1 - abs(delta) / self.robot.max_turn_rate) # reward turning when stuck
            )

            if score > best_score:
                best_score = score
                best_direction = direction
                best_heading = candidate_heading

        # No safe direction found
        if best_direction is None:
            return np.zeros(2)

        self.robot.heading = best_heading

        return best_direction

    def _detect_stuck(self, window=20, threshold=8.0):
        robot = self.robot
        robot._pos_history.append(robot.pos.copy())

        if len(robot._pos_history) > window:
            robot._pos_history.pop(0)

        if len(robot._pos_history) == window:
            displacement = np.linalg.norm(
                robot._pos_history[-1] - robot._pos_history[0]
            )
            robot.stuck = displacement < threshold
        else:
            robot.stuck = False

    def _goal_is_reachable(self, clearance_threshold=None):
        if clearance_threshold is None:
            clearance_threshold = self.robot.radius * 2

        goal_dir = self.goal - self.robot.pos
        goal_dist = np.linalg.norm(goal_dir)
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0])

        for angle, dist, _ in self.robot.lidar_data:
            angular_diff = abs((angle - goal_angle + np.pi) % (2 * np.pi) - np.pi)
            if angular_diff < 0.5:
                # Only blocked if the obstacle is between us and the goal
                if dist < min(goal_dist, clearance_threshold):
                    return False

        return True

    def _wall_follow_direction(self):
        robot = self.robot

        # Same approach: use closest lidar hit as the wall normal
        closest = min(robot.lidar_data, key=lambda r: r[1])
        closest_angle, closest_dist, _ = closest

        if closest_dist >= robot.lidar_range:
            return np.array([np.cos(robot.heading), np.sin(robot.heading)])

        tangent_angle = closest_angle + (np.pi / 2) * robot.wall_follow_dir
        desired = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

        # Goal pull after min steps (keep your existing logic)
        if robot.wall_follow_steps > 60:
            goal_dir = self.goal - robot.pos
            goal_dist = np.linalg.norm(goal_dir)
            if goal_dist > 1e-6:
                goal_dir /= goal_dist
            extra_steps = robot.wall_follow_steps - 60
            goal_pull = min(extra_steps / 120.0, 0.6)
            desired = desired * (1 - goal_pull) + goal_dir * goal_pull
            desired /= np.linalg.norm(desired)

        desired_heading = np.arctan2(desired[1], desired[0])
        heading_error = (desired_heading - robot.heading + np.pi) % (2 * np.pi) - np.pi
        robot.heading += np.clip(heading_error, -robot.max_turn_rate * 5, robot.max_turn_rate * 5)

        return np.array([np.cos(robot.heading), np.sin(robot.heading)])

    def _align_to_wall(self):
        robot = self.robot

        # Average the 5 closest hits for a stable wall normal
        sorted_hits = sorted(robot.lidar_data, key=lambda r: r[1])
        close_hits = [h for h in sorted_hits[:5] if h[1] < robot.lidar_range]

        if not close_hits:
            return True

        # Average the ray angles weighted by proximity
        tangent_angles = []
        for angle, dist, _ in close_hits:
            tangent_angles.append(angle + (np.pi / 2) * robot.wall_follow_dir)

        # Circular mean to avoid angle wrapping issues
        sin_mean = np.mean([np.sin(a) for a in tangent_angles])
        cos_mean = np.mean([np.cos(a) for a in tangent_angles])
        tangent_angle = np.arctan2(sin_mean, cos_mean)

        heading_error = (tangent_angle - robot.heading + np.pi) % (2 * np.pi) - np.pi
        robot.heading += np.clip(heading_error, -robot.max_turn_rate * 4, robot.max_turn_rate * 4)

        return abs(heading_error) < np.radians(5)
