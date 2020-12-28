import numpy as np
import cv2

print(cv2.__version__)

rMaskgray = cv2.imread('test.png', 0)
(thresh, binRed) = cv2.threshold(rMaskgray, 50, 255, cv2.THRESH_BINARY)

Rcontours, hier_r = cv2.findContours(
    binRed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
r_areas = [cv2.contourArea(c) for c in Rcontours]
max_rarea = np.max(r_areas)
CntExternalMask = np.ones(binRed.shape[:2], dtype="uint8") * 255

for c in Rcontours:
    if((cv2.contourArea(c) > max_rarea * 0.70) and (cv2.contourArea(c) < max_rarea)):
        cv2.drawContours(CntExternalMask, [c], -1, 0, 1)

cv2.imwrite('contour1.jpg', CntExternalMask)
