import numpy as np
import sim
import cv2
import time
import math
import enum
import threading
import imutils
from matplotlib import pyplot as plt
import numpy as np
from typing import List, Tuple

print('Program started')
# close all opened connections
sim.simxFinish(-1)
# Connect to CoppeliaSim
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID == -1:
    raise Exception("Failed to connect to remote API server")

print('Connected to remote API server')

_, bot = sim.simxGetObjectHandle(
    clientID, "Pioneer_p3dx", sim.simx_opmode_oneshot_wait)
_, left_motor = sim.simxGetObjectHandle(
    clientID, "Pioneer_p3dx_leftMotor", sim.simx_opmode_oneshot_wait)
_, right_motor = sim.simxGetObjectHandle(
    clientID, "Pioneer_p3dx_rightMotor", sim.simx_opmode_oneshot_wait)
_, vision_sensor = sim.simxGetObjectHandle(
    clientID, "Vision_sensor", sim.simx_opmode_oneshot_wait)
wheel_diameter = 0.195
wheel_radius = wheel_diameter / 2
wheel_distance = 0.381  # distance(leftWheel, rightWheel)
center_distance = wheel_distance / 2  # distance(centerLine, wheel)


class wUnit(enum.Enum):
    rad = 1
    deg = 2


cur_v = 0
cur_w = 0
cur_wUnit = wUnit.deg


def init():
    """Initialize the simulation.
    """
    sim.simxGetVisionSensorImage(
        clientID, vision_sensor, 0, sim.simx_opmode_streaming)
    time.sleep(1)


def distance(x1: float, y1: float, x2: float, y2: float):
    """Calculate the distance between (x1, y1) and (x2, y2).

    :param x1: x-coordinate of (x1, y1)
    :param y1: y-coordinate of (x1, y1)
    :param x2: x-coordinate of (x2, y2)
    :param y2: y-coordinate of (x2, y2)
    :return: the distance between (x1, y1) and (x2, y2)
    """
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)


def move(v, w, w_unit: wUnit):
    """
    Differential Drive Movement.

    :param v: desired velocity of car, unit: m
    :param w: desired angular velocity of car (Relative to ICR), unit: rad or deg
    """
    v_l = v - w * center_distance
    v_r = v + w * center_distance
    if w_unit == wUnit.rad:
        w_l = np.rad2deg(v_l / wheel_radius)
        w_r = np.rad2deg(v_r / wheel_radius)
    else:
        w_l = v_l / wheel_radius
        w_r = v_r / wheel_radius
    sim.simxSetJointTargetVelocity(
        clientID, left_motor, w_l, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(
        clientID, right_motor, w_r, sim.simx_opmode_oneshot)


def handleGraph(img):
    img = np.flip(img, 0)
    # cv2.imshow('image', img)

    # Convert image to greyscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    # Process image using Gaussian blur.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('blur', blur)

    # Process image using Color Threshold.
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)

    # Erode and dilate to remove accidental line detections.
    mask = cv2.erode(thresh, None, iterations=1)
    # cv2.imshow('mask_1', mask)
    mask = cv2.dilate(mask, None, iterations=3)
    # cv2.imshow('mask_2', mask)

    contour_img = cv2.Canny(mask, 50, 150)

    cv2.imshow('contour_img', contour_img)

    return contour_img


def isRoad(img, row, col):
    return img[row][col] >= 128


def getMin(a, b):
    if a is None:
        return b
    elif b is None:
        return a
    return min(a, b)


def getRoadMid(road_cols):
    width = None
    road_mid = None
    for i in range(len(road_cols)):
        if i == 0:
            continue
        cur_width = road_cols[i] - road_cols[i - 1]
        if cur_width < 5:
            continue
        if width is None or cur_width < width:
            width = road_cols[i] - road_cols[i - 1]
            road_mid = (road_cols[i] + road_cols[i - 1]) / 2
    return road_mid


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

    return np.rad2deg(angle) * (-1.0)


section_dist_weight = {
    0: 1,  # [0, 60)
    1: 2,  # [60, 120)
    2: 4,  # [120, 180)
    3: 8,  # [180, 240)
    4: 16,  # [240, 300)
    5: 32,  # [300, 360)
    6: 32,  # [360, 420)
    7: 16,  # [420, 480)
}

section_theta_weight = {
    0: 1,  # [0, 60)
    1: 1,  # [60, 120)
    2: 1,  # [120, 180)
    3: 1,  # [180, 240)
    4: 2,  # [240, 300)
    5: 2,  # [300, 360)
    6: 2,  # [360, 420)
    7: 2,  # [420, 480)
}


def calIndicator(img):
    rows, cols = img.shape[0], img.shape[1]
    mid = cols / 2
    car_pos = (rows - 1, mid - 1)

    indicator_dist = 0
    indicator_theta = 0
    dist_weight_sum = 0
    theta_weight_sum = 0

    valid_count = 0
    dist_sum = 0
    road_mid_sum = 0
    for i in range(rows):
        if i != 0 and (i % 60) == 0:
            if valid_count != 0:
                dist = (dist_sum / valid_count)
                dist_weight = section_dist_weight[(i - 1) // 60]
                indicator_dist += dist * dist_weight
                dist_weight_sum += dist_weight

                # car_pos = (i - 1, mid - 1)
                target_pos = (i - 30 - 1, (road_mid_sum / valid_count))
                theta = azimuthAngle(car_pos, target_pos)
                theta_weight = section_theta_weight[(i - 1) // 60]
                indicator_theta += theta * theta_weight
                theta_weight_sum += theta_weight

                # print("section ", (i - 1) // 60,
                #       " dist_sum = ", dist_sum,
                #       " valid_count = ", valid_count,
                #       " dist = ", dist,
                #       " theta = ", theta)

            valid_count = 0
            dist_sum = 0
            road_mid_sum = 0

        road_cols = []
        for j in range(cols):
            if isRoad(img, i, j):
                road_cols.append(j)
        road_mid = getRoadMid(road_cols)
        if road_mid is None:
            continue

        valid_count += 1
        dist_sum += road_mid - mid
        road_mid_sum += road_mid

    if dist_weight_sum == 0 or theta_weight_sum == 0:
        print("indicator_dist=None, indicator_theta=None")
        return None, None
    indicator_dist /= dist_weight_sum
    indicator_theta /= theta_weight_sum
    print("indicator_dist = ", indicator_dist,
          ", indicator_theta = ", indicator_theta)
    return indicator_dist, indicator_theta


def pidController(kp: float, ki: float, kd: float):
    """PID Control.

    :param kp: the proportional factor
    :param ki: the integral factor
    :param kd: the derivative factor
    :return: a function that processes the error
    """
    prev_error = 0
    integral = 0
    derivative = 0

    def pid(error: float):
        nonlocal prev_error
        nonlocal integral
        nonlocal derivative
        integral = integral + error
        derivative = error - prev_error
        prev_error = error
        return kp * error + ki * integral + kd * derivative

    return pid


def main():
    # move(10, 0.1, wUnit.deg)
    # while (True):
    # pass
    # angle = azimuthAngle(0, 0, 2, 1)
    # print("angle = ", angle)
    # move(10, 0, wUnit.deg)
    # return 0
    # getImage(vision_sensor)

    pid_v = pidController(kp=0.1, ki=0, kd=0)
    pid_w = pidController(kp=0.01, ki=0, kd=0.001)

    ############### Get Image ###############
    err, resolution, image = sim.simxGetVisionSensorImage(
        clientID, vision_sensor, 0, sim.simx_opmode_streaming)
    while (sim.simxGetConnectionId(clientID) != -1):
        err, resolution, image = sim.simxGetVisionSensorImage(
            clientID, vision_sensor, 0, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = handleGraph(img)
            indicator_dist, indicator_theta = calIndicator(img)
            if indicator_dist is None or indicator_theta is None:
                continue

            # break

            # v = pid_v(1.0 - indicator_dist / 320)
            v = 0.1
            w = pid_w(indicator_theta)
            print("v = ", v, " w = ", w)
            move(v, w, wUnit.deg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elif err == sim.simx_return_novalue_flag:
            print("Getting image: no image yet")
        else:
            print("Gettimg image: error with code = ", err)


if __name__ == '__main__':
    main()
