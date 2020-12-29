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
sim.simxFinish(-1)  # close all opened connections
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

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


# 设置仿真步长，为了保持API端与V-rep端相同步长
tstep = 0.01
sim.simxSetFloatingParameter(
    clientID, sim.sim_floatparam_simulation_time_step, tstep, sim.simx_opmode_oneshot)
# 打开同步模式
sim.simxSynchronous(clientID, True)
sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)

wheel_diameter = 0.195
wheel_radius = wheel_diameter / 2
wheel_distance = 0.381  # distance(leftWheel, rightWheel)
center_distance = wheel_distance / 2  # distance(centerLine, wheel)

rows = 480
cols = 640
near_points_to_zero = {}
car_theta_weight = 3
front_weight = 3


def init():
    """Initialize the simulation.
    """
    sim.simxGetVisionSensorImage(
        clientID, vision_sensor, 0, sim.simx_opmode_streaming)
    time.sleep(1)


############ Move ###############

def move(v, w):
    """
    Differential Drive Movement.

    :param v: desired velocity of car, unit: m/s
    :param w: desired angular velocity of car (Relative to ICR), unit: deg/s
    """

    # global car_theta_weight
    # global front_weight
    # if v <= 0.5:
    #     car_theta_weight = 3
    #     front_weight = 3
    # else:
    #     car_theta_weight = 2
    #     front_weight = 2

    # w: deg/tstep
    w *= 1 / tstep

    v_l = v - w * center_distance
    v_r = v + w * center_distance

    w_l = v_l / wheel_radius
    w_r = v_r / wheel_radius

    sim.simxSetJointTargetVelocity(
        clientID, left_motor, w_l, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(
        clientID, right_motor, w_r, sim.simx_opmode_oneshot)

############## Handle Graph #################


def isPointInContours(contours, point):
    for i, contour in enumerate(contours):
        flag = cv2.pointPolygonTest(contour, point, False)
        if flag == 1:
            return True, i
    return False, None


def isCornerInContours(contours, corner, fills):
    check_size = 5
    contour_id = None
    cnt = 0
    for i in range(check_size):
        flag, cur_id = isPointInContours(
            contours, (corner[1] - i, corner[0] - i))
        if flag == 1 and (contour_id is None or cur_id == contour_id):
            cnt += 1
            contour_id = cur_id
    if cnt >= (check_size // 2) + 1 and contours[contour_id] not in fills:
        area = cv2.contourArea(contours[contour_id])
        if area <= rows * cols / 4:
            fills.append(contours[contour_id])


def fillImage(img, fills):
    cv2.drawContours(img, fills, -1, (0, 0, 0), cv2.FILLED)
    return img


def handleGraphEdges(img, contours):
    fills = []
    isCornerInContours(contours, (10, 630), fills)
    isCornerInContours(contours, (10, 10), fills)
    if len(fills) > 0:
        img = fillImage(img, fills)
    return img


def isRoad(img, pos):
    return img[pos[0]][pos[1]] >= 32


def handleGraph(img):
    img = np.flip(img, 0)
    # cv2.imwrite("../Images/origin.png", img)

    # Convert image to greyscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Process image using Gaussian blur.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Process image using Color Threshold.
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_NONE)
    thresh = handleGraphEdges(thresh, contours)
    # cv2.imwrite("../Images/thresh.png", thresh)

    # Transfer perspective
    pts1 = np.float32([[0, 0], [639, 0], [0, 479], [639, 479]])
    pts2 = np.float32([[0, 110], [639, 110], [250, 479], [389, 479]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    trans = cv2.warpPerspective(thresh, M, (640, 480))
    # cv2.imwrite("../Images/trans.png", trans)

    # Erode to get a line to represent road
    mask = cv2.erode(trans, None, iterations=2)

    return mask


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

    return pid

############# Handle Road ################


def initRoad(rs):
    for r in rs:
        near_points_to_zero[r] = getNearPointsToZero(r)


def tranCoordinate(pos: Tuple):
    return (pos[1], -pos[0])


def correctAngle(angle):
    while angle >= 180:
        # print(angle)
        angle -= 360
    while angle < -180:
        # print(angle)
        angle += 360
    return angle


def azimuthAngle(pos1: Tuple, pos2: Tuple):
    # pos1, pos2 = tranCoordinate(pos1), tranCoordinate(pos2)
    y1, x1 = pos1
    y2, x2 = pos2

    angle = math.pi / 2 * (-1 if x2 > x1 else 1)
    if y2 != y1:
        angle = math.atan((x2-x1)/(y2-y1))

    # (-pi, pi]
    if angle > math.pi:
        angle -= 2.0 * math.pi
    if angle <= -math.pi:
        angle += 2.0 * math.pi

    return np.rad2deg(angle)


def distance(p0, p1):
    return ((p0[0] - p1[0]) ** 2.0 + (p0[1] - p1[1]) ** 2.0) ** 0.5


def isValidPoint(point):
    return (point[0] >= 0) and (point[0] < rows) and (point[1] >= 0) and (point[1] < cols)


def getNearPointsToZero(r):
    eps = 1
    near_points = []
    for i in range(int(-r)-2, int(r)+2):
        for j in range(int(-r)-2, int(r)+2):
            if abs(distance((0, 0), (i, j)) - r) <= eps:
                near_points.append((i, j))
    return near_points


def moveCoordinates(start_point, points):
    moved_points = []
    for i in range(len(points)):
        moved_points.append((points[i][0] + start_point[0],
                             points[i][1] + start_point[1]))
    return moved_points


def vector_delta_theta(p0, p1, p2):
    y0, x0 = p0
    y1, x1 = p1
    y2, x2 = p2
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x1, y2 - y1)
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
    # if rho < 0:
    if rho > 0:
        return - theta
    else:
        return theta
    # vb = (x1 - x0, y1 - y0)
    # vf = (x2 - x1, y2 - y1)
    # lb = distance((0, 0), vb)
    # lf = distance((0, 0), vf)
    # deg_cos = (vb[0] * vf[0] + vb[1] * vf[1]) / lb / lf
    # # flag = np.sign((vb[0] * vf[1]) - (vf[0] * vb[1]))
    # if deg_cos > 1.0 and deg_cos < 1.0 + 0.000000001:
    #     deg_cos = 1.0
    # if deg_cos < -1.0 and deg_cos > -1.0 - 0.000000001:
    #     deg_cos = -1.0
    # return np.rad2deg(math.acos(deg_cos)
    # # return abs(np.rad2deg(math.acos(deg_cos))) * flag


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


def moveStep(img, start_point, last_point, r):
    global near_points_to_zero
    near_points = moveCoordinates(start_point, near_points_to_zero[r])
    theta = None
    next_point = None
    # print('near_points', near_points)
    for near_point in near_points:
        if isValidPoint(near_point) and isRoad(img, near_point):
            # print('start', start_point, 'near', near_point)
            near_theta = vector_delta_theta(
                last_point, start_point, near_point)
            if theta is None or abs(near_theta) < abs(theta):
                next_point, theta = near_point, near_theta
    return next_point, theta


def scanStartPoint(img, r):
    start_points = []
    for i in range(rows - 1, 0, -1):
        for j in range(cols):
            if isRoad(img, (i, j)):
                start_points.append((i, j))
        if len(start_points) > 0:
            break

    # print('start_points', start_points)
    if len(start_points) == 0:
        return None

    min_theta = None
    real_point = start_points[0]
    for start_point in start_points:
        _, theta = moveStep(
            img, start_point, (start_point[0]+5, start_point[1]), r)
        if theta is None:
            continue
        if min_theta is None or theta < min_theta:
            min_theta = theta
            real_point = start_point

    return real_point


def getRoad(img, start_point, r):
    virtual_point = (start_point[0]+5, start_point[1])
    points = [virtual_point, start_point]
    thetas = [azimuthAngle(virtual_point, start_point)]

    print('in getRoad: start_point =', start_point)
    while True:
        next_point, theta = moveStep(img, points[-1], points[-2], r)
        if next_point is None or theta is None:
            found = False
            for big_r in range(r+1, r+3):
                next_point, theta = moveStep(
                    img, points[-1], points[-2], big_r)
                if next_point is not None and theta is not None:
                    found = True
                    break
            if found is False:
                break
        if abs(theta) > (180 * 7 / 8):
            break

        points.append(next_point)
        thetas.append(theta)
        # if len(points) > 10:
        # printPoints(points)
        # print(points)
        # print("len(points) = ", len(points))
        # last_theta += theta
    return points[1:], thetas

# def simplify(points, thetas):
#     pre_theta = thetas[0]
#     for i in range(1, len(thetas)):
#         if abs()


def handleRoad(img):
    r0 = 20
    r1 = 20
    start_point = scanStartPoint(img, r0)
    if start_point is None:
        return None, None
    points, thetas = getRoad(img, start_point, r1)

    return points, thetas

############# Calculator Indicator ################


def smoothList(ori):
    lst = []
    lst.append(ori[0])
    for i in range(1, len(ori)):
        lst.append((ori[i - 1] + ori[i]) / 2)
    return lst


def getThetasToCar(points, size):
    if len(points) < size:
        raise Exception("Points not enough.")
    car_thetas = []
    car_pos = (rows + 20, cols / 2)
    for i in range(size):
        # print('point = ', points[i], 'azimuthAngle = ', azimuthAngle(car_pos, points[i]))
        car_thetas.append(azimuthAngle(car_pos, points[i]))
    return smoothList(car_thetas)


def getAverageTheta(car_thetas, thetas, front_size, weights):
    theta_sum = 0
    weight_sum = 0
    global car_theta_weight
    for i in range(len(thetas)):
        if i < front_size:
            theta_sum += car_thetas[i] * car_theta_weight
            weight_sum += car_theta_weight
        theta_sum += thetas[i] * weights[i]
        weight_sum += weights[i]

    if weight_sum == 0:
        return None

    print('weights = ', weights)

    return theta_sum / weight_sum


def generateWeights(size, front_size):
    weights = []
    # global front_weight
    global car_theta_weight
    car_theta_weight = 2
    for i in range(size):
        if (i / size) < 0.3 or i < front_size:
            weight = 3
        elif (i / size) < 0.8:
            weight = 2
        else:
            weight = 1
        # weight = 1
        # if i < front_size:
        #     weight = front_weight
        weights.append(weight)

    return weights


front_rate = 0.3


def calIndicator(points, thetas):
    global front_rate
    front_size = max(int(len(thetas) * front_rate), 2)
    if len(points) < front_size:
        return None
    car_thetas = getThetasToCar(points, front_size)
    weights = generateWeights(len(thetas), front_size)
    print('')
    print('points = ', points)
    print('thetas = ', thetas)
    print('car_thetas = ', car_thetas)
    return getAverageTheta(car_thetas, thetas, front_size, weights)


################# main function ###################


def calV_(theta):
    theta = abs(theta)
    k = 1 / 10
    max_v = 2
    min_v = 0.01

    return max(max_v - (max_v - min_v) * k * theta, 0.01)


def calV(theta):
    theta = abs(theta) * 2

    if theta <= 1:
        return 3
    elif theta <= 3:
        return 2.5
    elif theta <= 5:
        return 2
    elif theta <= 8:
        return 1.5
    elif theta <= 10:
        return 1
    elif theta <= 15:
        return 0.7
    elif theta <= 20:
        return 0.5
    elif theta <= 30:
        return 0.2
    else:
        return 0.1


def simMove():
    sim.simxSynchronousTrigger(clientID)  # 进行下一步
    sim.simxGetPingTime(clientID)    # 使得该仿真步走完


def main():

    initRoad([5, 6, 7, 8, 9, 10, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

    print('initial finished')

    pid_quick = pidController(kp=0.1, ki=0.00001, kd=0)
    pid_slow = pidController(kp=0.1, ki=0, kd=0)
    pid = None

    ############### Initial ###############
    v = 0
    w = 0
    ############### Get Image ###############

    sim.simxSynchronousTrigger(clientID)
    err, resolution, image = sim.simxGetVisionSensorImage(
        clientID, vision_sensor, 0, sim.simx_opmode_streaming)

    count = 0
    while (sim.simxGetConnectionId(clientID) != -1):
        err, resolution, image = sim.simxGetVisionSensorImage(
            clientID, vision_sensor, 0, sim.simx_opmode_buffer)
        if err == sim.simx_return_ok:
            count += 1
            img = np.array(image, dtype=np.uint8)
            img.resize([resolution[1], resolution[0], 3])
            img = handleGraph(img)

            cv2.imwrite("../Images/img.png", img)

            points, thetas = handleRoad(img)

            # printPoints(points)

            if points is None or thetas is None:
                v = 0.01
                w = 0.5
                move(v, w)
                print('v = ', v, 'w = ', w)
                simMove()
                continue

            theta = calIndicator(points, thetas)

            if theta is None:
                v = 0.01
                w = 0.5
                move(v, w)
                print('v = ', v, 'w = ', w)
                simMove()
                continue

            v = 0.5

            if v <= 0.5:
                pid = pid_slow
            else:
                pid = pid_quick

            w = pid(theta)
            # v = calV(theta)
            print('theta = ', theta, 'v = ', v, 'w = ', w)
            move(v, w)

            # break

            simMove()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        elif err == sim.simx_return_novalue_flag:
            pass
            print("Getting image: no image yet")
        else:
            print("Gettimg image: error with code = ", err)


if __name__ == '__main__':
    main()
