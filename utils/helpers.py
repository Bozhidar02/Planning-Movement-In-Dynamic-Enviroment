from config import GRID_SIZE


def snap_to_grid(pos):
    x, y = pos
    return (
        round(x / GRID_SIZE) * GRID_SIZE,
        round(y / GRID_SIZE) * GRID_SIZE,
    )