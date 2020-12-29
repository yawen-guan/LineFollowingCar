from matplotlib import pyplot as plt
import math
from typing import Tuple
import numpy as np


def CalAngle(v1, v2):
    # v1旋转到v2，逆时针为正，顺时针为负
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    # 叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
    if rho < 0:
        return - theta
    else:
        return theta


def vector_delta_theta(p0, p1, p2):
    y0, x0 = p0
    y1, x1 = p1
    y2, x2 = p2
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x1, y2 - y1)
    print('v1 = ', v1, 'v2 = ', v2)
    # v1旋转到v2，逆时针为正，顺时针为负
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    deg_cos = np.dot(v1, v2) / TheNorm
    if deg_cos > 1.0 and deg_cos < 1.0 + 0.000000001:
        deg_cos = 1.0
    if deg_cos < -1.0 and deg_cos > -1.0 - 0.000000001:
        deg_cos = -1.0
    # 叉乘
    rho = np.rad2deg(np.arcsin(deg_cos))
    # 点乘
    theta = np.rad2deg(np.arccos(deg_cos))
    if rho > 0:
        return - theta
    else:
        return theta


print(CalAngle((8, -13), (10, -10)))

print(vector_delta_theta((373, 241), (378, 221), (387, 204)))


# def tranCoordinate(pos: Tuple):
#     return (pos[1], -pos[0])

# def azimuthAngle(pos1: Tuple, pos2: Tuple):
#     pos1, pos2 = tranCoordinate(pos1), tranCoordinate(pos2)
#     y1, x1 = pos1
#     y2, x2 = pos2

#     angle = math.pi / 2 * (-1 if x2 > x1 else 1)
#     if y2 != y1:
#         angle = math.atan((x2-x1)/(y2-y1))

#     # (-pi, pi]
#     if angle > math.pi:
#         angle -= 2.0 * math.pi
#     if angle <= -math.pi:
#         angle += 2.0 * math.pi

#     return np.rad2deg(angle)


# print(azimuthAngle((500, 320), (479, 348)))
rows = 480
cols = 640


def printPoints(points):
    fig, ax = plt.subplots(1, 1)
    x = []
    y = []
    for point in points:
        y.append(point[0])
        x.append(point[1])
    plt.plot(x, y, '.', color='b')
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    ax.invert_yaxis()
    plt.show()


points = [(479, 322), (459, 320), (439, 317), (419, 313), (401, 306),
          (384, 296), (371, 282), (365, 263), (367, 244), (374, 225), (384, 208)]

printPoints(points)
