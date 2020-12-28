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
from weight import weights
from utility import *

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

start_row = 108


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
    # Convert image to greyscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Process image using Gaussian blur.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Process image using Color Threshold.
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    # Transfer perspective
    pts1 = np.float32([[0, 0], [639, 0], [0, 479], [639, 479]])
    pts2 = np.float32([[0, 110], [639, 110], [250, 479], [389, 479]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    trans = cv2.warpPerspective(thresh, M, (640, 480))
    # Erode and dilate to remove accidental line detections.
    mask = cv2.erode(trans, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)
    # Get contour
    contour_img = cv2.Canny(mask, 50, 150)
    # Dilate contour image
    mask = cv2.dilate(contour_img, None, iterations=1)

    cv2.imwrite("./img.png", mask)
    return mask

############# Handle Road ################


min_road_width = 3
max_road_width = 50
road_parallel_eps = 30


def isRoad(img, row, col):
    return img[row][col] >= 128


def getRoadMids(road_cols):
    row_road_mids = []
    row_road_dists = []
    for i in range(1, len(road_cols)):
        width = road_cols[i] - road_cols[i - 1]
        if width >= min_road_width and width <= max_road_width:
            row_road_mids.append((road_cols[i - 1] + road_cols[i]) / 2)
    for i in range(1, len(row_road_mids)):
        row_road_dists.append(row_road_mids[i] - row_road_mids[i - 1])
    return row_road_mids, row_road_dists


def recognizeRoad(img):
    """
    img: perspective transformed contour image
    """
    rows, cols = img.shape[0], img.shape[1]

    road_mids = []
    road_dists = []
    isMultiroad = False
    isParallel = False
    isTurn = False
    turnPoint = None

    for i in range(rows):
        road_cols = []
        for j in range(cols):
            if isRoad(img, i, j):
                road_cols.append(j)
        # print("row = ", i, " road_cols = ", road_cols)
        row_road_mids, row_road_dists = getRoadMids(road_cols)
        # print("row = ", i, " row_road_mids = ", row_road_mids)
        road_mids.append(row_road_mids)
        road_dists.append(row_road_dists)
        if len(row_road_dists) > 0:
            isMultiroad = True

    if isMultiroad == True:
        ###### check parallel ######
        isParallel = True
        parallelFirst = None
        for i in range(0, rows):
            if len(road_dists[i]) == 0:
                continue
            if parallelFirst is None:
                parallelFirst = i
                continue
            for j in range(min(len(road_dists[parallelFirst]), len(road_dists[i]))):
                if abs(road_dists[i][j] - road_dists[parallelFirst][j]) > road_parallel_eps:
                    isParallel = False
                    break
            if isParallel == False:
                break
        if parallelFirst is None:
            isParallel = False

        ###### check turn ######
        isEmpty = False
        empty_cnt = 0
        empty_idx = -1
        for i in range(start_row, rows):
            if len(road_mids[i]) == 0:
                if i == empty_idx + 1:
                    empty_cnt += 1
                    if empty_cnt == 20:
                        isEmpty = True
                else:
                    empty_cnt = 1
                empty_idx = i

            if isEmpty and i >= 2 and len(road_mids[i-1]) == 1 and len(road_mids[i]) > 1:
                isTurn = True
                turnPoint = (i-1, road_mids[i - 1][0])
                break

    return road_mids, isMultiroad, isParallel, isTurn, turnPoint


############## Calculate indicator #################


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


def getNearestRoadMid(row_road_mids):
    mid = 320
    nearest = None
    for road_mid in row_road_mids:
        if nearest is None or abs(road_mid - mid) < abs(nearest - mid):
            nearest = road_mid
    return nearest


def getSecondNearestRoadMid(row_road_mids):
    mid = 640 / 2
    first = None
    second = None
    for road_mid in row_road_mids:
        if first is None:
            first = road_mid
        elif second is None:
            second = road_mid
            if abs(first - mid) > abs(second - mid):
                first, second = second, first
        elif abs(road_mid - mid) < abs(first - mid):
            first = road_mid
        elif abs(road_mid - mid) < abs(second - mid):
            second = road_mid
    return first, second


section_size = 30


def calIndicator(road_mids, v):
    rows = 480
    cols = 640
    mid = cols / 2
    car_pos = (rows, cols / 2)
    global section_size
    global weights
    v = findNearestV(v, weights)

    valid_count = 0
    road_mid_sum = 0
    weight_sum = 0
    indicator = 0

    turnDirection = None
    dist_valid_count = 0
    dist_sum = 0

    inCurrentRoad = False

    for i, row_road_mids in enumerate(road_mids):
        road_mid, road_mid_2 = getSecondNearestRoadMid(row_road_mids)

        if road_mid is not None:
            valid_count += 1
            road_mid_sum += road_mid

        if road_mid_2 is not None:
            dist_valid_count += 1
            dist_sum += road_mid_2 - mid

        if ((i + 1) % section_size) == 0:
            section_id = i // section_size

            if valid_count != 0:
                target_pos = (i - section_size // 2,
                              (road_mid_sum / valid_count))
                theta = azimuthAngle(car_pos, target_pos)
                weight = weights[v][section_id]
                indicator += theta * weight
                weight_sum += weight
                if section_id >= (rows - 1 - section_size * 3) // section_size:
                    inCurrentRoad = True
                # print("section = ", i//section_size,
                #   " target_pos = ", target_pos, " theta = ", theta)
            valid_count = 0
            road_mid_sum = 0

    if weight_sum == 0:
        return None, None, False
    indicator /= weight_sum
    turnDirection = getLabel(dist_sum) * (-1)  # left: +, right: -
    return indicator, turnDirection, inCurrentRoad

################ PID Control ###################


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

    def clear():
        nonlocal integral
        integral = 0

    return pid, clear


################# main function ###################

def main():

    # pid_v = pidController(kp=0.3, ki=0, kd=0)
    pid_w_5, clear_w_5 = pidController(kp=0.01, ki=0, kd=0.015)
    pid_w_3, clear_w_3 = pidController(kp=0.008, ki=0, kd=0.005)  # 0.006
    pid_w_1, clear_w_1 = pidController(kp=0.005, ki=0.00, kd=0.01)

    ############### Initial ###############
    v = 0
    w = 0
    pre_idx = 0
    pre_size = 5
    pres = {
        "v": [0] * pre_size,
        "w": [0] * pre_size
    }
    pre_labels = {
        "isMultiroad": [False] * pre_size,
        "isParallel": [False] * pre_size,
        "isTurn": [False] * pre_size,
        "turnPoint": [(0, 0)]*pre_size,
        "turnDirection": [None] * pre_size
    }

    ############### Get Image ###############
    # lastCmdTime = sim.simxGetLastCmdTime(clientID)
    err, resolution, image = sim.simxGetVisionSensorImage(
        clientID, vision_sensor, 0, sim.simx_opmode_streaming)
    while (sim.simxGetConnectionId(clientID) != -1):
        # currCmdTime = sim.simxGetLastCmdTime(clientID)
        # dt = currCmdTime - lastCmdTime
        err, resolution, image = sim.simxGetVisionSensorImage(
            clientID, vision_sensor, 0, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = handleGraph(img)
            # cv2.imwrite("./img.png", img)
            road_mids, isMultiroad, isParallel, isTurn, turnPoint = recognizeRoad(
                img)
            indicator, turnDirection, inCurrentRoad = calIndicator(
                road_mids, v)
            print("indicator = ", indicator, " isMultiroad = ",
                  isMultiroad, " isParallel = ", isParallel, " isTurn = ", isTurn, " turnPoint = ", turnPoint, " turnDirection = ", turnDirection, " inCurrentRoad = ", inCurrentRoad)

            break

            """
            并道： isMultiroad and isParallel
            交叉或转弯： isMultiroad and not isParallel, 此时判断未来可能的转弯方向为 turnDirection
            转弯： isTurn, 转弯顶点为 turnPoint
            当前小车是否在路上：inCurrentRoad
            """

            # 急转弯
            if inCurrentRoad == False and getMost(pre_labels["isTurn"]) == True:
                v = 0
                w = getTurnDirection(pre_labels, pre_size) * 0.6
                move(v, w, wUnit.deg)
                print("-- v = ", v, " w = ", w)
                continue

            # 无路
            if indicator is None:
                v = sum(pres["v"]) / pre_size
                w = sum(pres["w"]) / pre_size
                move(v, w, wUnit.deg)
                continue

            # break

            # if abs(indicator) < 20 and ((not isMultiroad) or (isMultiroad and isParallel)):
            #     v = 0.5
            #     w = pid_w_5(indicator)
            # elif

            # if abs(indicator) < 40 and ((not isMultiroad) or (isMultiroad and isParallel)):
            #     v = 0.3
            #     w = pid_w_3(indicator)
            # else:
            #     v = 0.1
            #     w = pid_w_1(indicator)

            # v = 0.3
            # w = pid_w_3(indicator)
            # v = 0.5
            # w = pid_w_5(indicator)
            # if abs(indicator) >= 20 or (isTurn and turnPoint[0] <= 300):
            v = 0.3
            w = pid_w_3(indicator)
            if abs(indicator) >= 50:
                v = 0.1
                w = pid_w_1(indicator)
            print("v = ", v, " w = ", w)

            move(v, w, wUnit.deg)

            pres["v"][pre_idx] = v
            pres["w"][pre_idx] = w
            pre_labels["isMultiroad"][pre_idx] = isMultiroad
            pre_labels["isParallel"][pre_idx] = isParallel
            pre_labels["isTurn"][pre_idx] = isTurn
            pre_labels["turnPoint"][pre_idx] = turnPoint
            pre_labels["turnDirection"][pre_idx] = turnDirection
            pre_idx = (pre_idx + 1) % pre_size

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elif err == sim.simx_return_novalue_flag:
            print("Getting image: no image yet")
        else:
            print("Gettimg image: error with code = ", err)


if __name__ == '__main__':
    main()
