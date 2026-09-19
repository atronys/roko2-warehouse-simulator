import random
import math
import time
import matplotlib.pyplot as plt
from heapq import heappush, heappop


def generate_maze(width, height):
    """Generate maze with walls exactly at x=0, x=30, y=0, y=30"""
    # 30x30 grid with wider passages (4 meters between walls)
    maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # y=0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # shelf row
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # shelf row
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # shelf row
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # shelf row
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # wide passage
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # y=29
    ]
    
    return maze


def get_shelves():
    """Returns list of shelves with control points - now 20 shelves (10 per row)"""
    return [
        # Lower row (Y=6.0)
        (1, 5.4, 6.0),   # Shelf 1
        (2, 9.4, 6.0),   # Shelf 2
        (3, 13.4, 6.0),  # Shelf 3
        (4, 17.4, 6.0),  # Shelf 4
        (5, 21.4, 6.0),  # Shelf 5
        (6, 25.4, 6.0),  # Shelf 6
        (7, 5.4, 12.0),  # Shelf 7 - second row
        (8, 9.4, 12.0),  # Shelf 8
        (9, 13.4, 12.0), # Shelf 9
        (10, 17.4, 12.0),# Shelf 10
        (11, 21.4, 12.0),# Shelf 11
        (12, 25.4, 12.0),# Shelf 12
        (13, 5.4, 18.0), # Shelf 13 - third row
        (14, 9.4, 18.0), # Shelf 14
        (15, 13.4, 18.0),# Shelf 15
        (16, 17.4, 18.0),# Shelf 16
        (17, 21.4, 18.0),# Shelf 17
        (18, 25.4, 18.0),# Shelf 18
        (19, 5.4, 24.0), # Shelf 19 - fourth row
        (20, 9.4, 24.0), # Shelf 20
        (21, 13.4, 24.0),# Shelf 21
        (22, 17.4, 24.0),# Shelf 22
        (23, 21.4, 24.0),# Shelf 23
        (24, 25.4, 24.0),# Shelf 24
    ]


def maze_to_walls(maze):
    """Convert maze grid to wall line segments with exact coordinates"""
    walls = []
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell == 0:  # free space
                x, y = j, i
                # Check walls around free cell
                if maze[i-1][j] == 1:  # wall above
                    walls.append(((x, x+1), (y, y)))
                if maze[i+1][j] == 1:  # wall below
                    walls.append(((x, x+1), (y+1, y+1)))
                if maze[i][j-1] == 1:  # wall left
                    walls.append(((x, x), (y, y+1)))
                if maze[i][j+1] == 1:  # wall right
                    walls.append(((x+1, x+1), (y, y+1)))
    return walls


def sense_wall(walls, point, angle_rad):  
    x, y = point
    dx = math.cos(angle_rad) * 0.1
    dy = math.sin(angle_rad) * 0.1

    distance = 0.0
    while True:
        x += dx
        y += dy
        distance += 0.1

        for wall in walls:
            x1, x2 = wall[0]
            y1, y2 = wall[1]

            if x1 == x2:  # vertical wall
                if min(y1, y2) <= y <= max(y1, y2) and abs(x - x1) < 0.2:
                    return distance
            if y1 == y2:  # horizontal wall
                if min(x1, x2) <= x <= max(x1, x2) and abs(y - y1) < 0.2:
                    return distance

        if distance > 1000:
            return None


def segment_intersect_walls(walls, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    segment_length = math.sqrt(dx*dx + dy*dy)
    if segment_length == 0:
        return False
    dx /= segment_length
    dy /= segment_length
    step = 0.15

    distance = 0.0
    while distance <= segment_length:
        x = x1 + dx * distance
        y = y1 + dy * distance
        distance += step

        for wall in walls:
            wx1, wx2 = wall[0]
            wy1, wy2 = wall[1]

            if wx1 == wx2:  # vertical wall
                if min(wy1, wy2) - 0.1 <= y <= max(wy1, wy2) + 0.1 and abs(x - wx1) < 0.15:
                    return True
            if wy1 == wy2:  # horizontal wall
                if min(wx1, wx2) - 0.1 <= x <= max(wx1, wx2) + 0.1 and abs(y - wy1) < 0.15:
                    return True

    return False


def add_custom_walls(walls, custom_walls):
    if custom_walls:
        for w in custom_walls:
            if len(w) == 2:
                walls.append(tuple(w))
    return walls


def plot_maze(walls, points=None, angles_rad=None, distances=None):
    for wall in walls:
        plt.plot(wall[0], wall[1], 'k-', lw=2)

    if points and angles_rad and distances:
        for point, angle_rad, distance in zip(points, angles_rad, distances):
            if point and angle_rad is not None and distance:
                x, y = point
                end_x = x + math.cos(angle_rad) * distance
                end_y = y + math.sin(angle_rad) * distance
                plt.plot([x, end_x], [y, end_y], 'r-', lw=2)

    plt.axis('off')
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    maze = generate_maze(4,4)
    walls = maze_to_walls(maze)
    print(f"Maze generated (30x30). Shelves available: {len(get_shelves())}")