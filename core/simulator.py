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
        self._cached_dynamic_grid = None
        # Visited map: coarse grid, cell size 20px
        self._visited_map = {}
        self._visited_cell_size = 20

    def _mark_visited(self):
        cs = self._visited_cell_size
        cell = (int(self.robot.pos[0] // cs), int(self.robot.pos[1] // cs))
        self._visited_map[cell] = self._visited_map.get(cell, 0) + 1

    def _visited_penalty(self, pos):
        cs = self._visited_cell_size
        cell = (int(pos[0] // cs), int(pos[1] // cs))
        count = self._visited_map.get(cell, 0)
        return -min(count * 8.0, 200.0)  # caps at -200 so it doesn't override everything

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

        goal_dist = np.hypot(self.robot.pos[0] - self.goal[0],
                             self.robot.pos[1] - self.goal[1])
        if goal_dist <= self.robot.radius:
            self.robot.stuck = False
            self.robot.mode = "navigate"
            self.running = False
            return

        self._sense_obstacles()
        self._detect_stuck()
        self._precompute_predictions()
        self._mark_visited()

        robot = self.robot

        if robot.mode == "navigate":
            if robot.stuck:
                robot.mode = "wall_align"
                robot.wall_follow_steps = 0
                robot.stuck = False
                robot._pos_history.clear()

                goal_dir = self.goal - robot.pos
                goal_dir /= max(np.linalg.norm(goal_dir), 1e-6)

                wall_normal = np.zeros(2)
                for angle, dist, _ in robot.lidar_data:
                    if dist < 80:
                        ray_dir = np.array([np.cos(angle), np.sin(angle)])
                        wall_normal -= ray_dir * ((80 - dist) / 80)

                if np.linalg.norm(wall_normal) > 1e-6:
                    wall_normal /= np.linalg.norm(wall_normal)
                    tangent_a = np.array([-wall_normal[1], wall_normal[0]])
                    tangent_b = np.array([wall_normal[1], -wall_normal[0]])

                    def openness_in_dir(tangent):
                        total, count = 0.0, 0
                        for a, d, _ in robot.lidar_data:
                            if np.dot(np.array([np.cos(a), np.sin(a)]), tangent) > 0.5:
                                total += d
                                count += 1
                        return total / count if count > 0 else 0.0

                    open_a = openness_in_dir(tangent_a)
                    open_b = openness_in_dir(tangent_b)
                    score_a = np.dot(tangent_a, goal_dir) * 0.5 + (open_a / robot.lidar_range) * 0.5
                    score_b = np.dot(tangent_b, goal_dir) * 0.5 + (open_b / robot.lidar_range) * 0.5
                    robot.wall_follow_dir = 1 if score_a >= score_b else -1
                else:
                    robot.wall_follow_dir = 1

            wait, back_up, backup_direction = self._assess_dynamic_threats()

            if back_up or wait:
                robot.dynamic_wait_steps = getattr(robot, 'dynamic_wait_steps', 0) + 1
                if robot.dynamic_wait_steps > 60:
                    self._circle_back()
                    return
                if back_up:
                    backup_direction /= np.linalg.norm(backup_direction)
                    proposed = robot.pos + backup_direction * robot.speed
                    if not self._trajectory_collision(proposed, robot.radius):
                        robot.pos = proposed
                return
            else:
                robot.dynamic_wait_steps = 0

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
                robot.wall_entry_goal_dist = np.linalg.norm(self.goal - robot.pos)
                robot.wall_follow_entry_pos = robot.pos.copy()

        elif robot.mode == "wall_follow":
            robot.wall_follow_steps += 1

            # Dead end check — blocked on wall side, ahead, and behind
            if self._detect_dead_end():
                robot.wall_follow_dir *= -1
                robot.wall_follow_steps = 0
                robot.stuck = False
                robot._pos_history.clear()
                robot.wall_entry_goal_dist = np.linalg.norm(self.goal - robot.pos)
                return

            if robot.stuck and robot.wall_follow_steps > 15:
                robot.wall_follow_dir *= -1
                robot.wall_follow_steps = 0
                robot.stuck = False
                robot._pos_history.clear()
                robot.wall_entry_goal_dist = np.linalg.norm(self.goal - robot.pos)
                return

            can_exit = robot.wall_follow_steps > 60
            current_goal_dist = np.linalg.norm(self.goal - robot.pos)
            past_entry_point = current_goal_dist < robot.wall_entry_goal_dist

            # Only exit toward a gap if it's actually wide enough to fit through
            wall_ended = self._detect_wall_end()
            gap_passable = wall_ended and self._gap_is_passable()

            if can_exit and (gap_passable or past_entry_point) and (
                    self._goal_is_reachable() or
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
        self._visited_map = {}

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
            visited = self._visited_penalty(simulated_pos)

            alignments_to_rays = lidar_dirs @ direction
            mask = alignments_to_rays > 0.5
            clearance = (
                -np.sum(alignments_to_rays[mask] / np.maximum(lidar_dists[mask], 1.0))
                if mask.any() else 0.0
            )

            score = (
                    alignment * 200
                    + progress * 1.5
                    + smoothness * 10
                    + clearance * 300
                    + continuity * 80
                    + visited  # penalise already-visited cells
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

            # Also catch oscillation: high heading variance + low displacement
            if not hasattr(robot, '_heading_history'):
                robot._heading_history = []
            robot._heading_history.append(robot.heading)
            if len(robot._heading_history) > window:
                robot._heading_history.pop(0)

            if len(robot._heading_history) == window:
                # Circular variance of heading
                sin_mean = np.mean([np.sin(h) for h in robot._heading_history])
                cos_mean = np.mean([np.cos(h) for h in robot._heading_history])
                heading_consistency = np.hypot(sin_mean, cos_mean)  # 1=steady, 0=chaotic
                oscillating = heading_consistency < 0.3 and displacement < threshold * 3

                robot.stuck = displacement < threshold or oscillating
            else:
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

        wall_side_angle = robot.heading + (np.pi / 2) * (-robot.wall_follow_dir)

        wall_hits = []
        for angle, dist, hit in robot.lidar_data:
            angular_diff = abs((angle - wall_side_angle + np.pi) % (2 * np.pi) - np.pi)
            if angular_diff < np.radians(45) and dist < robot.lidar_range * 0.9:
                wall_hits.append(hit)

        if len(wall_hits) >= 2:
            pts = np.array(wall_hits)
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            wall_tangent = Vt[0]

            # Orient by the INTENDED follow tangent, not current heading
            # This is stable even when heading oscillates
            intended_tangent_angle = wall_side_angle + (np.pi / 2) * robot.wall_follow_dir
            intended = np.array([np.cos(intended_tangent_angle), np.sin(intended_tangent_angle)])
            if np.dot(wall_tangent, intended) < 0:
                wall_tangent = -wall_tangent

            desired = wall_tangent.copy()
        else:
            closest = min(robot.lidar_data, key=lambda r: r[1])
            closest_angle, closest_dist, _ = closest
            if closest_dist >= robot.lidar_range:
                return np.array([np.cos(robot.heading), np.sin(robot.heading)])
            tangent_angle = closest_angle + (np.pi / 2) * robot.wall_follow_dir
            desired = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

        if robot.wall_follow_steps > 60:
            goal_dir = self.goal - robot.pos
            goal_dist = np.linalg.norm(goal_dir)
            if goal_dist > 1e-6:
                goal_dir /= goal_dist
            extra_steps = robot.wall_follow_steps - 60
            goal_pull = min(extra_steps / 120.0, 0.4)
            desired = desired * (1 - goal_pull) + goal_dir * goal_pull
            desired /= np.linalg.norm(desired)

        ahead_cone = np.radians(25)
        wall_side_angle = robot.heading + (np.pi / 2) * (-robot.wall_follow_dir)

        fwd_dists, side_dists = [], []
        for angle, dist, _ in robot.lidar_data:
            fwd_diff = abs((angle - robot.heading + np.pi) % (2 * np.pi) - np.pi)
            side_diff = abs((angle - wall_side_angle + np.pi) % (2 * np.pi) - np.pi)
            if fwd_diff < ahead_cone:
                fwd_dists.append(dist)
            if side_diff < np.radians(30):
                side_dists.append(dist)

        if fwd_dists and side_dists:
            fwd_clear = min(fwd_dists)  # worst-case ahead
            side_clear = np.mean(side_dists)  # average wall distance

            # Ridge criterion: something is blocking the path ahead while
            # the wall is still present beside us (not a corner/gap).
            ridge_threshold = robot.radius * 3.5
            wall_still_present = side_clear < robot.lidar_range * 0.7

            if fwd_clear < ridge_threshold and wall_still_present:
                # Step away from wall: blend in an outward component
                outward = np.array([np.cos(wall_side_angle + np.pi),
                                    np.sin(wall_side_angle + np.pi)])
                # Scale step strength by how blocked we are
                step_strength = 1.0 - (fwd_clear / ridge_threshold)  # 0–1
                desired = desired * (1 - step_strength * 0.6) + outward * (step_strength * 0.6)
                mag = np.linalg.norm(desired)
                if mag > 1e-6:
                    desired /= mag
        # ── END RIDGE DETECTION ──────────────────────────────────────────────────

        # (rest of existing heading update is unchanged)
        desired_heading = np.arctan2(desired[1], desired[0])
        heading_error = (desired_heading - robot.heading + np.pi) % (2 * np.pi) - np.pi
        robot.heading += np.clip(heading_error, -robot.max_turn_rate * 5, robot.max_turn_rate * 5)

        return np.array([np.cos(robot.heading), np.sin(robot.heading)])

    def _align_to_wall(self):
        robot = self.robot

        sorted_hits = sorted(robot.lidar_data, key=lambda r: r[1])
        close_hits = [h for h in sorted_hits[:5] if h[1] < robot.lidar_range]

        if not close_hits:
            return True

        tangent_angles = [
            angle + (np.pi / 2) * robot.wall_follow_dir
            for angle, dist, _ in close_hits
        ]

        sin_mean = np.mean([np.sin(a) for a in tangent_angles])
        cos_mean = np.mean([np.cos(a) for a in tangent_angles])
        tangent_angle = np.arctan2(sin_mean, cos_mean)

        heading_error = (tangent_angle - robot.heading + np.pi) % (2 * np.pi) - np.pi
        robot.heading += np.clip(heading_error, -robot.max_turn_rate * 4, robot.max_turn_rate * 4)

        return abs(heading_error) < np.radians(5)

    def _detect_wall_end(self):
        robot = self.robot
        if not robot.lidar_data:
            return False

        wall_side_angle = robot.heading + (np.pi / 2) * (-robot.wall_follow_dir)
        open_rays, checked_rays = 0, 0

        for angle, dist, _ in robot.lidar_data:
            angular_diff = abs((angle - wall_side_angle + np.pi) % (2 * np.pi) - np.pi)
            if angular_diff < np.radians(30):
                checked_rays += 1
                if dist >= robot.lidar_range * 0.85:
                    open_rays += 1

        if checked_rays == 0:
            return False
        return (open_rays / checked_rays) > 0.6

    def _gap_is_passable(self):
        """
        Measures the physical width of the open space on the wall side.
        Returns True only if the gap is wide enough for the robot to pass.
        """
        robot = self.robot
        wall_side_angle = robot.heading + (np.pi / 2) * (-robot.wall_follow_dir)
        min_gap = robot.radius * 2.5

        # Collect open rays in the wall-side cone
        open_ray_angles = []
        gap_dists = []
        for angle, dist, _ in robot.lidar_data:
            angular_diff = abs((angle - wall_side_angle + np.pi) % (2 * np.pi) - np.pi)
            if angular_diff < np.radians(40) and dist >= robot.lidar_range * 0.85:
                open_ray_angles.append(angle)
                gap_dists.append(dist)

        if not open_ray_angles:
            return False

        # Angular span of the opening × distance = approximate physical width
        span = max(open_ray_angles) - min(open_ray_angles)
        avg_dist = np.mean(gap_dists)
        physical_width = avg_dist * np.tan(span / 2) * 2

        return physical_width >= min_gap

    def _detect_dead_end(self):
        """
        Returns True when the robot is boxed in on three sides —
        ahead, wall side, and behind — indicating a dead end pocket.
        """
        robot = self.robot
        if not robot.lidar_data:
            return False

        close_thresh = robot.radius * 4

        def avg_dist_in_cone(center_angle, half_width):
            dists = []
            for angle, dist, _ in robot.lidar_data:
                diff = abs((angle - center_angle + np.pi) % (2 * np.pi) - np.pi)
                if diff < half_width:
                    dists.append(dist)
            return np.mean(dists) if dists else robot.lidar_range

        ahead_dist = avg_dist_in_cone(robot.heading, np.radians(30))
        wall_side = robot.heading + (np.pi / 2) * (-robot.wall_follow_dir)
        side_dist = avg_dist_in_cone(wall_side, np.radians(30))
        behind_dist = avg_dist_in_cone(robot.heading + np.pi, np.radians(30))

        return ahead_dist < close_thresh and side_dist < close_thresh

