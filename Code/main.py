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


def img2stand(pos: Tuple):
    return (pos[1], rows - pos[0])


def stand2img(pos: Tuple):
    return (rows-pos[1], pos[0])


def correctAngle(angle):  # (-180, 180]
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle


def azimuthAngle(pos1: Tuple, pos2: Tuple):  # stand
    virtual_point = (pos1[0], pos1[1] - 5)
    return vector_delta_theta(virtual_point, pos1, pos2)


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
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x1, y2 - y1)
    # 点乘
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    vcos = np.dot(v1, v2) / TheNorm
    if vcos > 1.0 and vcos < 1.0 + 0.000000001:
        vcos = 1.0
    if vcos < -1.0 and vcos > -1.0 - 0.000000001:
        vcos = -1.0
    # 叉乘判断正负
    flag = np.sign(v1[0] * v2[1] - v1[1] * v2[0])
    theta = np.rad2deg(np.arccos(vcos))  # [0, 180)
    if flag < 0:
        theta *= -1
    return theta


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


def moveStep(img, start_point, last_point, r):
    global near_points_to_zero
    near_points = moveCoordinates(start_point, near_points_to_zero[r])
    theta = None
    next_point = None
    for near_point in near_points:
        if isValidPoint(stand2img(near_point)) and isRoad(img, stand2img(near_point)):
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
                start_points.append(img2stand((i, j)))
        if len(start_points) > 0:
            break

    print('start_points = ', start_points)
    if len(start_points) == 0:
        return None

    min_theta = None
    real_point = start_points[0]
    for start_point in start_points:
        virtual_point = (start_point[0], start_point[1] - 5)
        _, theta = moveStep(img, start_point, virtual_point, r)
        if theta is None:
            continue
        if min_theta is None or theta < min_theta:
            min_theta = theta
            real_point = start_point

    return real_point


def getRoad(img, start_point, r):
    virtual_point = (start_point[0], start_point[1] - 5)
    points = [virtual_point, start_point]
    thetas = [azimuthAngle(virtual_point, start_point)]

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
    return points[1:], thetas


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
    car_pos = (cols / 2, -30)

    for i in range(size):
        # print('point = ', points[i], 'azimuthAngle = ', azimuthAngle(car_pos, points[i]))
        car_theta = azimuthAngle(car_pos, points[i])
        car_thetas.append(car_theta)
    return car_thetas


def getAverageTheta(car_thetas, thetas, front_size, weights):
    theta_sum = 0
    weight_sum = 0
    global car_theta_weight
    for i in range(len(thetas)):
        if i < front_size:
            theta_sum += car_thetas[i] * weights[i] * 2
            weight_sum += weights[i] * 2
        theta_sum += thetas[i] * weights[i]
        weight_sum += weights[i]

    if weight_sum == 0:
        return None

    print('weights = ', weights)

    return theta_sum / weight_sum


def generateWeights(size, front_size, v):
    weights = []
    # global front_weight
    # global car_theta_weight
    # car_theta_weight = 2

    if v == 0:
        v = 0.01
    k = - 6 / v
    b = (1 - size) * k + 1

    for i in range(size):
        ########
        # if (i / size) < 0.3 or i < front_size:
        #     weight = 3
        # elif (i / size) < 0.8:
        #     weight = 2
        # else:
        #     weight = 1

        ##########
        # weight = 1

        #######
        weight = k * i + b

        # if i < front_size:
        #     weight = front_weight
        weights.append(weight)

    return weights


front_rate = 0.3


def calIndicator(points, thetas, v):
    # global front_rate
    # front_size = max(int(len(thetas) * front_rate), 2)
    # if len(points) < front_size:
    #     return None
    front_size = len(points)
    car_thetas = getThetasToCar(points, front_size)
    weights = generateWeights(len(thetas), front_size, v)
    print('')
    print('points = ', points)
    print('thetas = ', thetas)
    print('car_thetas = ', car_thetas)
    front_size = max(int(len(thetas) * 0.2), 5)
    return getAverageTheta(car_thetas, thetas, front_size, weights)


def analysis(points, thetas):
    theta_sum = 0
    isTurn = False
    turnPoint = None
    for i in range(len(thetas)):
        theta_sum += thetas[i]
        if abs(theta_sum) >= 85:
            isTurn = True
            turnPoint = i
            break
    theta_sum = 0
    isStra = True
    for i in range(1, len(thetas)):
        theta_sum += thetas[i]
        if abs(theta_sum) >= 30:
            isStra = False
            break
    return isTurn, turnPoint, isStra


################# main function ###################


def calV_(theta):
    theta = abs(theta)
    k = 1 / 10
    max_v = 2
    min_v = 0.01

    return max(max_v - (max_v - min_v) * k * theta, 0.01)


def calV(theta, w):
    theta = abs(theta) * 5

    v = 2

    if theta <= 3:
        v = 2
    elif theta <= 5:
        v = 1.8
    elif theta <= 8:
        v = 1.5
    elif theta <= 10:
        v = 1.0
    elif theta <= 15:
        v = 0.8
    elif theta <= 20:
        v = 0.6
    elif theta <= 30:
        v = 0.4
    elif theta <= 40:
        v = 0.2
    else:
        v = 0.1
    
    # if abs(w) <= 0.00:
    #     v = min(v, 2)
    # elif theta <= 5:
    #     v = min(v, 1.8)
    # elif theta <= 8:
    #     v = min(v, 1.5)
    # elif theta <= 10:
    #     v = min(v, 1.0)
    # elif theta <= 15:
    #     v = min(v, 0.8)
    # elif theta <= 20:
    #     v = min(v, 0.6)
    # elif abs(w) <= 0.2:
    #     v = min(v, 0.4)
    # elif abs(w) <= 0.5:
    #     v = min(v, 0.2)
    # else:
    #     v = 0.1


    return v


def simMove():
    sim.simxSynchronousTrigger(clientID)  # 进行下一步
    sim.simxGetPingTime(clientID)    # 使得该仿真步走完


def main():

    initRoad([5, 6, 7, 8, 9, 10, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])

    print('initial finished')

    pid_quick = pidController(kp=0.1, ki=0, kd=0)
    pid_slow = pidController(kp=0.1, ki=0, kd=0)
    pid = pidController(kp=0.3, ki=0.000, kd=0.20)  # kp = 0.3  kd = 0.2

    ############### Initial ###############
    v = 0
    w = 0
    pre_size = 5
    pre_idx = 0
    pre_controls = {
        "w": [0]*pre_size,
        "isTurn": [False]*pre_size,
        "isStra": [False]*pre_size
    }
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
                """
                No road
                """
                v = 0.001
                w = np.sign(sum(pre_controls["w"]) / pre_size) * 10
                move(v, w)
                print('v = ', v, 'w = ', w)
                simMove()
                continue

            theta = calIndicator(points, thetas, v)
            isTurn, turnPoint, isStra = analysis(points, thetas)

            if theta is None:
                v = 0.01
                w = sum(pre_controls["w"]) / pre_size
                move(v, w)
                print('v = ', v, 'w = ', w)
                simMove()
                continue

            w = pid(theta)
            if abs(w) > 3:
                w = np.sign(w) * 3

            # if not isTurn and True in pre_controls["isTurn"]:
            #     if min(pre_controls["turnPoint"]) <= 3:

            v = calV(theta)
            if isTurn and turnPoint <= 5:
                v = min(v, 0.3)

            print('theta = ', theta, 'v = ', v, 'w = ', w,
                  'isTurn = ', isTurn, 'turnPoint = ', turnPoint)

            # printPoints(points)

            move(v, w)

            pre_controls["w"][pre_idx] = w
            pre_controls["isTurn"][pre_idx] = isTurn
            pre_controls["isStra"][pre_idx] = isStra
            pre_idx = (pre_idx + 1) % pre_size

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
