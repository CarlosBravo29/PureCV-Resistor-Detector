import cv2 as cv
import numpy as np
import resistor_utils as mfun

ip_addres = "http://192.168.0.38:8080/video"

capture = cv.VideoCapture(ip_addres)

while True:
    isTrue, frame = capture.read()
    img_O, img_hsv = mfun.prepareIMG(frame, 0.5)
    #cv.imshow("Original", img_O)
    main_masked = mfun.detect_main_bodies(img_hsv)
    detected_res = mfun.drw_recs(main_masked, img_O)
    
    cv.imshow("masked", main_masked)
    cv.imshow("detected", detected_res)


    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows