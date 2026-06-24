import cv2 as cv
import numpy as np
import resistor_utils as ru

img = cv.imread("images/1.jpg")
img, img_hsv = ru.prepareIMG(img)
main_masked = ru.detect_main_bodies(img_hsv)
rectangles = ru.drw_recs(main_masked, img)

cv.imshow("Original", img)
cv.imshow("HSV", img_hsv)
cv.imshow("masked", main_masked)

cv.namedWindow('HSV')
cv.setMouseCallback('HSV', lambda event, x, y, flags, param: ru.get_hsv(event, x, y, flags, img_hsv))

cv.waitKey(0)