import numpy as np
import heapq
from utils.collision import circle_circle_collision, circle_rect_collision


class Simulator:
    def __init__(self, environment):
        self.env = environment
        self.robot = None
        self.goal = None
        self.mode = "static"
        self.running = False
        self.selected_obstacle = None
        self._cached_grid = None
        self._cached_grid_res = None
        self._predicted_obs = []
        self._cached_dynamic_grid  = None

    def _collides_with_static(self, pos, radius):
        px, py = pos
        r2 = radius * radius
        for obs in self.env.static_obstacles:
            ox = obs.rect.x;
            oy = obs.rect.y
            cx = ox if px < ox else (ox + obs.rect.width if px > ox + obs.rect.width else px)
            cy = oy if py < oy else (oy + obs.rect.height if py > oy + obs.rect.height else py)
            dx = px - cx;
            dy = py - cy
            if dx * dx + dy * dy < r2:
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
        grid, res = self._build_dynamic_grid()

        for obs in self.env.dynamic_obstacles:
            # Cooldown so failed A* doesn't retry every frame
            if not hasattr(obs, 'path_retry_cooldown'):
                obs.path_retry_cooldown = 0
            if obs.path_retry_cooldown > 0:
                obs.path_retry_cooldown -= 1
                continue

            if not hasattr(obs, "path") or not obs.path:
                if obs.goal is None:
                    continue
                start = (int(obs.pos[0] // res), int(obs.pos[1] // res))
                goal = (int(obs.goal.pos[0] // res), int(obs.goal.pos[1] // res))
                obs.path = self._astar_path(start, goal, grid, res)
                obs.path_index = 0
                if not obs.path:
                    obs.path_retry_cooldown = 30
                    continue

            if obs.path and obs.path_index < len(obs.path):
                target = np.array(obs.path[obs.path_index]) * res
                diff = target - obs.pos
                dist = np.hypot(diff[0], diff[1])  # faster than linalg.norm for 2D

                if dist < 5:
                    obs.path_index += 1
                else:
                    speed = np.hypot(obs.vel[0], obs.vel[1])
                    step = (diff / max(dist, 1e-6)) * speed
                    proposed = obs.pos + step

                    if not self._collides_with_static(proposed, obs.radius):
                        obs.vel = step  # already the delta, no copy needed
                        obs.pos = proposed
                    else:
                        obs.path = []
                        obs.path_retry_cooldown = 10

    def _update_robot(self):
        if self.robot is None or self.goal is None:
            return

        self._sense_obstacles()
        self._detect_stuck()
        self._precompute_predictions()

        robot = self.robot
        direction = np.zeros(2)

        if robot.mode == "navigate":
            if robot.stuck:
                robot.mode = "wall_align"
                robot.wall_follow_dir = 1
                robot.wall_follow_steps = 0
                robot.stuck = False
                robot._pos_history.clear()

            wait, back_up, backup_direction = self._assess_dynamic_threats()

            if back_up or wait:
                # Increment stall counter
                robot.dynamic_wait_steps = getattr(robot, 'dynamic_wait_steps', 0) + 1

                if robot.dynamic_wait_steps > 60:  # ~1 second at 60fps — tune this
                    self._circle_back()
                    return

                if back_up:
                    backup_direction /= np.linalg.norm(backup_direction)
                    proposed = robot.pos + backup_direction * robot.speed
                    if not self._trajectory_collision(proposed, robot.radius):
                        robot.pos = proposed
                return  # wait: stand still

            else:
                robot.dynamic_wait_steps = 0  # clear counter when path is free

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
        self._cached_grid = None
        self._cached_grid_res = None
        self._cached_dynamic_grid = None

    def _astar_path(self, start, goal, grid, grid_size):

        def heuristic(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return (dx * dx + dy * dy) ** 0.5

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

    def _build_grid(self, resolution=10, inflation_radius=10):
        if self._cached_grid is not None:
            return self._cached_grid, self._cached_grid_res

        # Default inflation = just enough to keep dynamic obs centre
        # away from walls — use their radius, not the robot's
        if inflation_radius is None:
            inflation_radius = resolution  # 1 cell minimum clearance

        width, height = self.env.width, self.env.height
        grid_w = int(width // resolution)
        grid_h = int(height // resolution)
        grid = np.zeros((grid_w, grid_h), dtype=int)

        inflation_cells = max(1, int(np.ceil(inflation_radius / resolution)))

        for obs in self.env.static_obstacles:
            x0 = max(0, int(obs.rect.x // resolution) - inflation_cells)
            y0 = max(0, int(obs.rect.y // resolution) - inflation_cells)
            x1 = min(grid_w - 1, int((obs.rect.x + obs.rect.width) // resolution) + inflation_cells)
            y1 = min(grid_h - 1, int((obs.rect.y + obs.rect.height) // resolution) + inflation_cells)
            grid[x0:x1 + 1, y0:y1 + 1] = 1

        self._cached_grid = grid
        self._cached_grid_res = resolution
        return grid, resolution

    def _build_dynamic_grid(self, resolution=10):
        """Separate grid for dynamic obstacle pathfinding with smaller inflation."""
        if hasattr(self, '_cached_dynamic_grid') and self._cached_dynamic_grid is not None:
            return self._cached_dynamic_grid, self._cached_dynamic_grid_res

        # Use the smallest dynamic obstacle radius as inflation
        if self.env.dynamic_obstacles:
            min_radius = min(obs.radius for obs in self.env.dynamic_obstacles)
            inflation_radius = max(resolution, int(min_radius * 0.5))  # half radius clearance
        else:
            inflation_radius = resolution

        width, height = self.env.width, self.env.height
        grid_w = int(width // resolution)
        grid_h = int(height // resolution)
        grid = np.zeros((grid_w, grid_h), dtype=int)
        inflation_cells = max(1, int(np.ceil(inflation_radius / resolution)))

        for obs in self.env.static_obstacles:
            x0 = max(0, int(obs.rect.x // resolution) - inflation_cells)
            y0 = max(0, int(obs.rect.y // resolution) - inflation_cells)
            x1 = min(grid_w - 1, int((obs.rect.x + obs.rect.width) // resolution) + inflation_cells)
            y1 = min(grid_h - 1, int((obs.rect.y + obs.rect.height) // resolution) + inflation_cells)
            grid[x0:x1 + 1, y0:y1 + 1] = 1

        self._cached_dynamic_grid = grid
        self._cached_dynamic_grid_res = resolution
        return grid, resolution

    def _sense_obstacles(self):
        robot = self.robot
        if robot is None:
            return

        angles = np.linspace(0, 2 * np.pi, robot.lidar_rays, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        rx, ry = robot.pos
        lr = robot.lidar_range
        steps = np.linspace(0, lr, 60)  # reduced from 100 — 60 is plenty

        # Pre-filter: only obstacles within lidar range (avoid checking distant ones)
        nearby_static = [
            (obs.rect.x, obs.rect.y, obs.rect.width, obs.rect.height)
            for obs in self.env.static_obstacles
            if abs(obs.rect.x + obs.rect.width / 2 - rx) - obs.rect.width / 2 < lr
               and abs(obs.rect.y + obs.rect.height / 2 - ry) - obs.rect.height / 2 < lr
        ]
        nearby_dynamic = [
            (obs.pos, obs.radius)
            for obs in self.env.dynamic_obstacles
            if (obs.pos[0] - rx) ** 2 + (obs.pos[1] - ry) ** 2 < (lr + obs.radius) ** 2
        ]

        robot.lidar_data = []

        for i in range(len(angles)):
            dx = cos_a[i]
            dy = sin_a[i]
            min_dist = lr
            hit_x = rx + dx * lr
            hit_y = ry + dy * lr

            for d in steps:
                if d >= min_dist:
                    break

                px = rx + dx * d
                py = ry + dy * d

                hit = False

                for rect in nearby_static:
                    ox, oy, ow, oh = rect
                    cx = max(ox, min(px, ox + ow))
                    cy = max(oy, min(py, oy + oh))
                    ex = px - cx
                    ey = py - cy
                    if ex * ex + ey * ey < 1:
                        min_dist = d
                        hit_x, hit_y = px, py
                        hit = True
                        break

                if not hit:
                    for obs_pos, obs_rad in nearby_dynamic:
                        ex = px - obs_pos[0]
                        ey = py - obs_pos[1]
                        if ex * ex + ey * ey <= obs_rad * obs_rad:
                            min_dist = d
                            hit_x, hit_y = px, py
                            hit = True
                            break

                if hit:
                    break

            robot.lidar_data.append((angles[i], min_dist, np.array([hit_x, hit_y])))

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

    def _precompute_predictions(self, lookahead=10):
        # Call once per frame before choose_best_direction
        self._predicted_obs = []
        for obs in self.env.dynamic_obstacles:
            positions = [obs.pos + obs.vel * t for t in range(lookahead)]
            self._predicted_obs.append((positions, obs.radius))

    def _trajectory_collision(self, pos, radius):
        if self._collides_with_static(pos, radius):
            return True

        px, py = pos
        for positions, obs_radius in self._predicted_obs:
            combined = (radius + obs_radius) ** 2
            for opos in positions:
                dx = px - opos[0]
                dy = py - opos[1]
                if dx * dx + dy * dy <= combined:
                    return True
        return False

    def _assess_dynamic_threats(self):
        robot = self.robot
        robot_speed = robot.speed

        wait = False
        back_up = False
        backup_direction = np.zeros(2)
        threat_found = False

        for obs in self.env.dynamic_obstacles:
            rel_pos = obs.pos - robot.pos
            dist = np.linalg.norm(rel_pos)
            obs_speed = np.linalg.norm(obs.vel)

            if dist > robot.lidar_range * 0.6:
                continue

            threat_frames = None
            for t in range(1, 30):
                predicted = obs.pos + obs.vel * t
                if np.linalg.norm(predicted - robot.pos) <= (robot.radius + obs.radius + 5):
                    threat_frames = t
                    break

            if threat_frames is None:
                continue

            threat_found = True

            if obs_speed > robot_speed:
                if dist < (robot.radius + obs.radius) * 3:
                    back_up = True
                    backup_direction -= (rel_pos / max(dist, 1e-6))
                else:
                    wait = True

        # Reset wait counter if no threat
        if not threat_found:
            robot.dynamic_wait_steps = 0

        return wait, back_up, backup_direction

    def _circle_back(self):
        """Find a perpendicular path around the blocking dynamic obstacle."""
        robot = self.robot

        # Find the closest threatening obstacle
        closest_obs = None
        closest_dist = float('inf')
        for obs in self.env.dynamic_obstacles:
            dist = np.linalg.norm(obs.pos - robot.pos)
            if dist < closest_dist and dist < robot.lidar_range * 0.6:
                closest_dist = dist
                closest_obs = obs

        if closest_obs is None:
            robot.dynamic_wait_steps = 0
            robot.mode = "navigate"
            return

        # Vector toward obstacle
        to_obs = closest_obs.pos - robot.pos
        to_obs_norm = to_obs / max(np.linalg.norm(to_obs), 1e-6)

        # Try both perpendicular directions, pick the one closer to goal
        perp_a = np.array([-to_obs_norm[1], to_obs_norm[0]])
        perp_b = np.array([to_obs_norm[1], -to_obs_norm[0]])

        goal_dir = self.goal - robot.pos
        goal_dir /= max(np.linalg.norm(goal_dir), 1e-6)

        # Pick whichever perpendicular aligns better with goal
        direction = perp_a if np.dot(perp_a, goal_dir) >= np.dot(perp_b, goal_dir) else perp_b

        proposed = robot.pos + direction * robot.speed
        if not self._trajectory_collision(proposed, robot.radius):
            robot.pos = proposed
        else:
            # Blocked that way too — try the other perpendicular
            direction = perp_b if direction is perp_a else perp_a
            proposed = robot.pos + direction * robot.speed
            if not self._trajectory_collision(proposed, robot.radius):
                robot.pos = proposed
            # If both blocked, do nothing this frame — next frame may clear

    def _choose_best_direction(self):
        if self.robot is None or self.goal is None:
            return np.zeros(2)

        robot = self.robot
        goal_vector = self.goal - robot.pos
        goal_dist = np.hypot(goal_vector[0], goal_vector[1])
        if goal_dist < 1e-6:
            return np.zeros(2)
        goal_direction = goal_vector / goal_dist

        # Precompute lidar ray directions once for the clearance check
        lidar_dirs = np.array([[np.cos(a), np.sin(a)] for a, _, _ in robot.lidar_data])
        lidar_dists = np.array([d for _, d, _ in robot.lidar_data])

        angles = np.linspace(-robot.max_turn_rate, robot.max_turn_rate, robot.candidate_angles)
        best_score = -1e9
        best_direction = None
        best_heading = robot.heading

        for delta in angles:
            candidate_heading = robot.heading + delta
            direction = np.array([np.cos(candidate_heading), np.sin(candidate_heading)])

            simulated_pos = robot.pos.copy()
            collision = False
            for _ in range(robot.prediction_time):
                simulated_pos += direction * robot.speed
                if self._trajectory_collision(simulated_pos, robot.radius):
                    collision = True
                    break
            if collision:
                continue

            alignment = np.dot(direction, goal_direction)
            progress = -np.hypot(*(self.goal - simulated_pos))
            smoothness = -abs(delta)
            continuity = -abs(candidate_heading - robot.heading)

            # Vectorised clearance — no per-ray loop
            alignments_to_rays = lidar_dirs @ direction
            mask = alignments_to_rays > 0.5
            clearance = -np.sum(alignments_to_rays[mask] / np.maximum(lidar_dists[mask], 1.0)) if mask.any() else 0.0

            score = (
                    alignment * 200
                    + progress * 1.5
                    + smoothness * 10
                    + clearance * 300
                    + continuity * 80
                    + (500 if robot.stuck else 0) * (1 - abs(delta) / robot.max_turn_rate)
            )

            if score > best_score:
                best_score = score
                best_direction = direction
                best_heading = candidate_heading

        if best_direction is None:
            return np.zeros(2)

        robot.heading = best_heading
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

        # Same approach: use the closest lidar hit as the wall normal
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
