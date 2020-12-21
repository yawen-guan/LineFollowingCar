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
    # Process image using Gaussian blur.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Process image using Color Threshold.
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh', thresh)
    # Erode and dilate to remove accidental line detections.
    mask = cv2.erode(thresh, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=8)
    # mask = cv2.dilate(thresh, None, iterations=2)
    # cv2.imshow('mask', mask)
    # # Find the contours of the image.
    # contours = cv2.findContours(mask.copy(), 1, cv2.CHAIN_APPROX_NONE)
    # # Use imutils to unpack contours.
    # contours = imutils.grab_contours(contours)
    return mask


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


def calIndicator(img):
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
    # move(0.1, -0.1, wUnit.deg)
    # return 0
    # getImage(vision_sensor)

    pid_v = pidController(kp=0.1, ki=0.000001, kd=-0.000001)
    pid_w = pidController(kp=-1, ki=0.05, kd=0.1)

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
            indicator = calIndicator(img)
            if indicator is None:
                continue

            # indicator = indicator / 640

            v = pid_v(1 - abs(indicator) / 320)
            # v = 1
            w = pid_w(indicator / 320)
            print("v = ", v, " w = ", w)
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
