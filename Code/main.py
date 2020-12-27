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
    print("v_l = ", v_l, " v_r = ", v_r, " w_l = ", w_l, " w_r = ", w_r)
    sim.simxSetJointTargetVelocity(
        clientID, left_motor, w_l, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(
        clientID, right_motor, w_r, sim.simx_opmode_oneshot)

# def smoothMove(v, w, w_unit: wUnit):
#     """
#     Smooth Differential Drive Movement.

#     :param v: desired velocity of car, unit: m
#     :param w: desired angular velocity of car (Relative to ICR), unit: rad or deg
#     """
#     while True:


def world_coordinate(coord: Tuple[float, float]):
    """Convert Sensor-Coordinate to World-Coordinate.

    :param coord: Coordinate related to Vision Sensor
    :return: World-Coordinate
    """
    k = 64 * math.sqrt(3)
    x, y = (256 - coord[0]) / k, (coord[1] - 256) / k + 0.8
    return x, y


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


def getImage(sensor):
    """Retrieve an image from Vision Sensor.

    :return: an image represented by numpy.ndarray from Vision Sensor
    """
    err, resolution, image = sim.simxGetVisionSensorImage(
        clientID, sensor, 0, sim.simx_opmode_streaming)
    while (sim.simxGetConnectionId(clientID) != -1):
        err, resolution, image = sim.simxGetVisionSensorImage(
            clientID, sensor, 0, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])

            # Convert image to greyscale.
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Process image using Gaussian blur.
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            # Process image using Color Threshold.
            _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('thresh', thresh)
            # Erode and dilate to remove accidental line detections.
            mask = cv2.erode(thresh, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=5)
            # mask = cv2.dilate(thresh, None, iterations=2)
            cv2.imshow('mask', mask)
            # # Find the contours of the image.
            # contours = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
            # # Use imutils to unpack contours.
            # contours = imutils.grab_contours(contours)
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif err == sim.simx_return_novalue_flag:
            print("in getImage: no image yet")
        else:
            print("in getImage: err = ", err)


def isRoad(img, row, col):
    return img[row][col] >= 128


def getNearestCol(cur_col, mid,  nearest_len, nearest_col):
    cur_len = abs(cur_col - mid)
    if cur_len < nearest_len:
        return cur_len, cur_col
    return nearest_len, nearest_col


def calIndicator_0(img):
    # print("img = ", img)
    rows, cols = img.shape[0], img.shape[1]
    mid = cols / 2
    inf = cols * 2

    indicator = 0
    level_diff = 5
    sum_weight = 0
    for i in range(rows):
        nearest_len = inf
        nearest_col = None
        for j in range(cols):
            if isRoad(img, i, j) and (j == 0 or not isRoad(img, i, j - 1)):
                nearest_len, nearest_col = getNearestCol(
                    j, mid, nearest_len, nearest_col)
            elif isRoad(img, i, j) and (j == cols - 1 or not isRoad(img, i, j + 1)):
                nearest_len, nearest_col = getNearestCol(
                    j, mid, nearest_len, nearest_col)
        if nearest_len == inf:
            continue
        weight = (i + 1) * level_diff
        sum_weight += weight
        indicator += (nearest_col - mid) * weight

    if sum_weight == 0:
        print("indicator = None")
        return None

    indicator /= sum_weight

    print("indicator = ", indicator)

    return indicator


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

    return np.rad2deg(angle)


road_section_weight = {
    0: 1,  # [0, 60)
    1: 2,  # [60, 120)
    2: 4,  # [120, 180)
    3: 8,  # [180, 240)
    4: 16,  # [240, 300)
    5: 32,  # [300, 360)
    6: 32,  # [360, 420)
    7: 16,  # [420, 480)
}


def calIndicator_1(img):
    rows, cols = img.shape[0], img.shape[1]
    mid = cols / 2
    car_pos = (rows - 1, mid - 1)

    indicator_dist = 0
    indicator_theta = 0
    weight_sum = 0

    valid_count = 0
    dist_sum = 0
    road_mid_sum = 0
    for i in range(rows):
        if i != 0 and (i % 60) == 0:
            if valid_count != 0:
                weight = road_section_weight[(i - 1) // 60]
                dist = (dist_sum / valid_count)
                indicator_dist += dist * weight

                mid_pos = (i - 30, (road_mid_sum / valid_count))
                theta = azimuthAngle(car_pos, mid_pos)
                indicator_theta += theta * weight
                # print("section ", (i - 1) // 60,
                #       " dist_sum = ", dist_sum,
                #       " valid_count = ", valid_count,
                #       " dist = ", (dist_sum / valid_count))
                weight_sum += weight
            valid_count = 0
            dist_sum = 0

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

    if weight_sum == 0:
        print("indicator_dist = None")
        return None, None
    indicator_dist /= weight_sum
    indicator_theta /= weight_sum
    print("indicator_dist = ", indicator_dist)
    print("indicator_theta = ", indicator_theta)
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
    pid_w = pidController(kp=0.01, ki=0, kd=0.01)

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
            indicator_dist, indicator_theta = calIndicator_1(img)
            if indicator_dist is None or indicator_theta is None:
                continue

            # indicator = indicator / 640

            # v = pid_v(1.0 - indicator_dist / 320)
            v = 0.1
            w = pid_w(-indicator_theta)
            # print("v = ", v, " w = ", w)
            move(v, w, wUnit.deg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # break

        elif err == sim.simx_return_novalue_flag:
            print("Getting image: no image yet")
        else:
            print("Gettimg image: error with code = ", err)


if __name__ == '__main__':
    main()
