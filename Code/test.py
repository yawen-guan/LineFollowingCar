import numpy as np
import math
from typing import Tuple


def tranCoordinate(pos: Tuple):
    return (pos[1], -pos[0])


def azimuthAngle(pos1: Tuple, pos2: Tuple):
    pos1, pos2 = tranCoordinate(pos1), tranCoordinate(pos2)
    x1, y1 = pos1[0], pos1[1]
    x2, y2 = pos2[0], pos2[1]
    angle = 0.0
    dx = x2 - x1
    dy = y2 - y1
    if x2 == x1:
        angle = math.pi / 2.0
        if y2 == y1:
            angle = 0.0
        elif y2 < y1:
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif x2 > x1 and y2 < y1:
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif x2 < x1 and y2 < y1:
        angle = math.pi + math.atan(dx / dy)
    elif x2 < x1 and y2 > y1:
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)

    # (-pi, pi]
    if angle > math.pi:
        angle -= 2.0 * math.pi
    if angle <= -math.pi:
        angle += 2.0 * math.pi

    return np.rad2deg(angle)


print(azimuthAngle((479, 319), (450, 0)))

# print(azimuthAngle(0, 0, 479, 319))

# print(azimuthAngle(319, -479, 0, 0))

# print(azimuthAngle(0, 0, 1, 0))

# print(azimuthAngle(0, 0, 0, 1))

# print(azimuthAngle(0, 0, 2, 1))

# print(azimuthAngle(0, 0, -2, 1))

# print(azimuthAngle(0, 0, -2, -1))

# print(azimuthAngle(0, 0, -1, 0))
