import cv2 as cv
import numpy as np
import resistor_utils as ru
from config import IP_ADDRESS

capture = cv.VideoCapture(IP_ADDRESS)

while True:
    isTrue, frame = capture.read()
    img_O, img_hsv = ru.prepareIMG(frame, 0.5)
    main_masked = ru.detect_main_bodies(img_hsv)
    detected_res = ru.drw_recs(main_masked, img_O)
    
    cv.imshow("masked", main_masked)
    cv.imshow("detected", detected_res)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()