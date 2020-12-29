import cv2

imgfile = "../Images/thresh.png"
img = cv2.imread(imgfile)
h, w, _ = img.shape

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find Contour
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 需要搞一个list给cv2.drawContours()才行！！！！！
c_max = []
max_area = 0
max_cnt = 0
for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    # find max countour
    if (area > max_area):
        if(max_area != 0):
            c_min = []
            c_min.append(max_cnt)
            print("c_min = ", c_min)
            cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)
        max_area = area
        max_cnt = cnt
    else:
        c_min = []
        c_min.append(cnt)
        cv2.drawContours(img, c_min, -1, (0, 0, 0), cv2.FILLED)

c_max.append(max_cnt)


cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)

cv2.imwrite("../Images/mask.png", img)
cv2.imshow('mask', img)
cv2.waitKey(0)


"""
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


def printRoad(road_mids):
    x = []
    y = []
    for i in range(rows):
        for j in range(len(road_mids[i])):
            x.append(road_mids[i][j])
            y.append(480 - i)
    plt.plot(x, y, 'o', color='b')
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.show()


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


"""

# cv2.imwrite("../Images/img.png", img)

     road_mids, isMultiroad, isParallel, isTurn, turnPoint = recognizeRoad(
          img)
      printRoad(road_mids)
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
                move(v, w)
                print("-- v = ", v, " w = ", w)
                sim.simxSynchronousTrigger(clientID)  # 进行下一步
                sim.simxGetPingTime(clientID)    # 使得该仿真步走完
                continue

            # 无路
            if indicator is None:
                v = sum(pres["v"]) / pre_size
                w = sum(pres["w"]) / pre_size
                move(v, w)
                sim.simxSynchronousTrigger(clientID)  # 进行下一步
                sim.simxGetPingTime(clientID)    # 使得该仿真步走完
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

            move(v, w)

            pres["v"][pre_idx] = v
            pres["w"][pre_idx] = w
            pre_labels["isMultiroad"][pre_idx] = isMultiroad
            pre_labels["isParallel"][pre_idx] = isParallel
            pre_labels["isTurn"][pre_idx] = isTurn
            pre_labels["turnPoint"][pre_idx] = turnPoint
            pre_labels["turnDirection"][pre_idx] = turnDirection
            pre_idx = (pre_idx + 1) % pre_size
