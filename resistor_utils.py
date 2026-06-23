import cv2 as cv
import numpy as np

# Image resizing function
def rescaleFrame(frame, scale = 0.15):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Resizes frame, applies Gaussian blur, and performs histogram equalization on the V channel 
def prepareIMG(img, scale = 0.15):
    img = rescaleFrame(img, scale)
    blur = cv.GaussianBlur(img, (7, 7), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    equalize_v = cv.equalizeHist(v)
    hsv_equalized = cv.merge((h, s, equalize_v))
    return img, hsv_equalized

# Isolates pixels within a color range and applies morphological closing to remove noise/gaps
def apply_main_mask(img, mask_tuple):
    mask = cv.inRange(img, np.array(mask_tuple[0]), np.array(mask_tuple[1]))
    img_masked = cv.bitwise_and(img, img, mask=mask)
    k_morphClose = cv.getStructuringElement(cv.MORPH_RECT, (11, 11))
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_morphClose)
    img_mClosed_masked = cv.bitwise_and(mask_closed, mask_closed, mask=mask_closed)
    return img_masked, img_mClosed_masked

# Detects and combines masks for different resistor colors into a single binary image
def detect_main_bodies(img):
    resistor_masks = {
        "mask_green_res" : ([40, 23, 0], [100, 145, 15]),
        "mask_blue_res" : ([101, 76, 1], [111, 161, 12]),
        "mask_yellow_res" : ([12, 90, 0], [18, 157, 11])
    }
    main_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask_name, mask_val in resistor_masks.items():
        _, closed = apply_main_mask(img, mask_val)
        main_mask = cv.bitwise_or(main_mask, closed)
    return main_mask

# Validates if a contour matches expected resistor dimensions and shape properties
def is_resistor(countour):
    x, y, w, h = cv.boundingRect(countour)
    aspect_ratio = float(w) / h
    area = cv.contourArea(countour)

    if aspect_ratio < 1.0:
        aspect_ratio = 1.0 / aspect_ratio

    if not (1.5 < aspect_ratio < 10.0):
        return False
    
    if area < 100:
        return False

    hull = cv.convexHull(countour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0: return False
    solidity = float(area) / hull_area
    if solidity < 0.5:
        return False
    return True

# Extracts valid contours and draws rotated bounding boxes around detected resistors
def drw_recs(main_mask, img):
    contours, _ = cv.findContours(main_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print(f"Total contours {len(contours)}")
    for c in contours:
        area = cv.contourArea(c) # DEBUG
        x, y, w, h = cv.boundingRect(c) # DEBUG
        print(f"Contour - Area: {area}, W: {w}, H: {h}") # DEBUG
        if is_resistor(c):
            fit_rec = cv.minAreaRect(c)
            box = cv.boxPoints(fit_rec)
            box = np.int32(box)
            cv.drawContours(img, [box], 0, (255, 0, 0), 2)
        else:
            print("Failed")
            continue
    return img
