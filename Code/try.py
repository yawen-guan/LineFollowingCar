from matplotlib import pyplot as plt
import math
from typing import Tuple
import numpy as np


# def vector_delta_theta(p0, p1, p2):
#     x0, y0 = p0
#     x1, y1 = p1
#     x2, y2 = p2
#     v1 = (x1 - x0, y1 - y0)
#     v2 = (x2 - x1, y2 - y1)
#     # v1旋转到v2，逆时针为正，顺时针为负
#     # 2个向量模的乘积
#     TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
#     deg_cos = np.dot(v1, v2) / TheNorm
#     if deg_cos > 1.0 and deg_cos < 1.0 + 0.000000001:
#         deg_cos = 1.0
#     if deg_cos < -1.0 and deg_cos > -1.0 - 0.000000001:
#         deg_cos = -1.0
#     # 叉乘
#     rho = np.rad2deg(np.arcsin(deg_cos))
#     # 点乘
#     theta = np.rad2deg(np.arccos(deg_cos))
#     if rho < 0:
#         # if rho > 0:
#         return - theta
#     else:
#         return theta


def vector_delta_theta(p0, p1, p2):
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x1, y2 - y1)
    # v1旋转到v2，逆时针为正，顺时针为负
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    vcos = np.dot(v1, v2) / TheNorm
    if vcos > 1.0 and vcos < 1.0 + 0.000000001:
        vcos = 1.0
    if vcos < -1.0 and vcos > -1.0 - 0.000000001:
        vcos = -1.0
    flag = np.sign(v1[0] * v2[1] - v1[1] * v2[0])
    # 叉乘
    # rho = np.rad2deg(np.arcsin(vcos))
    # 点乘
    theta = np.rad2deg(np.arccos(vcos))  # [0, 180)
    return theta * flag


print(vector_delta_theta((0, 0), (1, 0), (1, 2)))


# def tranCoordinate(pos: Tuple):
#     return (pos[1], -pos[0])

def correctAngle(angle):  # (-180, 180]
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle


def azimuthAngle(pos1: Tuple, pos2: Tuple):  # stand
    virtual_point = (pos1[0], pos1[1] - 5)
    return vector_delta_theta(virtual_point, pos1, pos2)
    # x1, y1 = pos1
    # x2, y2 = pos2

    # angle = math.pi / 2 * np.sign(x2 - x1)
    # if y2 != y1:
    #     angle = -math.atan((x2 - x1) / (y2 - y1))

    # angle = np.rad2deg(angle)
    # return correctAngle(angle)


print(azimuthAngle((320, -20), (299, 1)))

rows = 480
cols = 640


def printPoints(points):  # stand
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, '.', color='b')
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.show()


points = [(308, 1), (308, 20), (308, 39), (308, 58), (308, 77), (308, 96), (308, 115), (308, 134), (309, 154), (312, 174), (315, 194), (318, 214), (323, 233), (329, 252),
          (338, 270), (347, 288), (360, 303), (377, 315), (394, 325), (413, 328), (433, 328), (452, 328), (472, 322), (490, 315), (508, 308), (526, 300), (544, 290), (561, 278)]

for i in range(2, len(points)):
    print('i = ', i, 'theta = ', vector_delta_theta(
        points[i-2], points[i-1], points[i]))

# printPoints(points)
