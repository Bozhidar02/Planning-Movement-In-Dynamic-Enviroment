import numpy as np


def circle_circle_collision(pos1, rad1, pos2, rad2):
    # Calculate distance between centers
    # Ensure pos1 and pos2 are numpy arrays or use math.dist
    dist = np.linalg.norm(np.array(pos1) - np.array(pos2))

    # Collision occurs if distance is less than the sum of radii
    return dist < (rad1 + rad2)


def circle_rect_collision(circle_pos, radius, rect):
    rx, ry, rw, rh = rect
    cx, cy = circle_pos

    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))

    dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
    hit = dist_sq < (radius ** 2)

    # DEBUG LOG
    #if hit: print(f"HIT! Dist Sq: {dist_sq} | Rad Sq: {radius**2}")

    return hit
