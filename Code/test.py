import cv2
import numpy as np

img = cv2.imread("./img.png")


print(img[479][639])
p0 = [0, 0]
p1 = [639, 0]
p2 = [0, 479]
p3 = [639, 479]
pts1 = np.float32([p0, p1, p2, p3])
pts2 = np.float32([[0, 110], [639, 110], [250, 479], [389, 479]])

# pts1 = np.float32([[15, 180], [15, 475], [80, 120], [80, 540]])
# pts2 = np.float32([[0, 0], [0, 100], [100, 0], [100, 100]])

# 生成变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(img, M, (640, 480))

cv2.imwrite("./output.png", dst)

# while (True):
# pass


"""

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
    4: 4,  # [240, 300)
    5: 4,  # [300, 360)
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

    current_distance = None

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

        if i == rows - 1 and road_mid is not None:
            current_distance = road_mid - mid

        if road_mid is None:
            continue

        valid_count += 1
        dist_sum += road_mid - mid
        road_mid_sum += road_mid

    if dist_weight_sum == 0 or theta_weight_sum == 0:
        print("indicator_dist=None, indicator_theta=None")
        return None, None, current_distance
    indicator_dist /= dist_weight_sum
    indicator_theta /= theta_weight_sum
    print("indicator_dist = ", indicator_dist,
          ", indicator_theta = ", indicator_theta)
    return indicator_dist, indicator_theta, current_distance

"""
