import numpy as np
import cv2
import imutils

rows = 480
cols = 640


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


def handleGraph(img):
    img = np.flip(img, 0)
    # img = cv2.imread("../Images/input.png")

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

    src = trans
    contours, _ = cv2.findContours(src, 1, cv2.CHAIN_APPROX_NONE)

    fills = []

    for contour in contours:
        area = cv2.contourArea(contour)
        # print('area = ', area)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print('cX = ', cX, 'cY = ', cY)
        # cv2.circle(src, (cX, cY), 7, (255, 255, 255), -1)
        if (cY < rows / 3 or cX < cols / 3 or cX > cols - cols / 3) and area <= rows * cols / 5:
            fills.append(contour)

    for contour in fills:
        rect = cv2.minAreaRect(contour)
        box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)
        src = fillImage(src, [box])
        # cv2.drawContours(src, [box], 0, (32, 32, 255), 2)
        # cv2.imwrite("../Images/ellipse.png", src)

    trans = src

    cv2.imwrite("../Images/target.png", src)

    # Erode to get a line to represent road
    # mask = cv2.erode(trans, None, iterations=1)
    # mask = cv2.dilate(mask, None, iterations=3)
    # Get contour
    contour_img = cv2.Canny(trans, 50, 150)
    # Dilate contour image
    mask = cv2.dilate(contour_img, None, iterations=1)
    cv2.imwrite("../Images/mask.png", mask)

    return mask


handleGraph(None)
