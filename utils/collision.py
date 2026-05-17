def circle_circle_collision(pos1, rad1, pos2, rad2):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    combined = rad1 + rad2
    return (dx*dx + dy*dy) < (combined * combined)


def circle_rect_collision(circle_pos, radius, rect):
    rx, ry, rw, rh = rect
    cx, cy = circle_pos
    closest_x = max(rx, min(cx, rx + rw))
    closest_y = max(ry, min(cy, ry + rh))
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx*dx + dy*dy) < (radius * radius)
