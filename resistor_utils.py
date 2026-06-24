import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

# Image resizing function
def rescaleFrame(frame, scale = 0.15):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Get HSV of a point
def get_hsv(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        pixel = param[y, x]
        print(f"HSV at ({x}, {y}): {pixel}")

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
        "mask_blue_res" : ([101, 76, 1], [117, 167, 36]),
        "mask_yellow_res" : ([11, 53, 0], [18, 157, 37]) #[11, 53, 0], [18, 157, 18]
    }
    main_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask_name, mask_val in resistor_masks.items():
        _, closed = apply_main_mask(img, mask_val)
        main_mask = cv.bitwise_or(main_mask, closed)
    return main_mask

def extract_ROI(img, rect, target_size=(200, 60)):
    center, (w, h), angle = rect
    if w < h:
        angle = angle + 90
        w, h = h, w
    m = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, m, (img.shape[1], img.shape[0]))
    roi = cv.getRectSubPix(rotated, (int(w), int(h)), center)
    roi_rz = cv.resize(roi, target_size, interpolation=cv.INTER_AREA)
    return roi_rz

def validate_res_content(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    h, w = gray.shape
    body_gray = gray[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
    body_sat  = hsv[:, :, 1][int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
    mean_brightness = np.mean(body_gray)
    mean_saturation = np.mean(body_sat)
    if mean_brightness < 60 or mean_saturation < 30:
        return False, 0
    strip = gray[int(h*0.3):int(h*0.7), int(w * 0.10):int(w * 0.90)]
    intensity_profile = np.mean(strip, axis=0).astype(float)
    profile_norm = (intensity_profile - intensity_profile.min()) / (intensity_profile.max() - intensity_profile.min() + 1e-6) * 255
    smoothed = np.convolve(profile_norm, np.ones(7) / 7, mode='same')
    inverted_profile = 255 - smoothed
    strip_w = inverted_profile.shape[0]
    peaks, props = find_peaks(inverted_profile, distance=8, prominence=3, width=2)
    widths = props["widths"]
    min_band_width = 5.0
    max_band_width = strip_w * 0.20
    left_margin  = int(strip_w * 0.08)
    right_margin = int(strip_w * 0.04)
    # print(f"  strip_w={strip_w}, left={left_margin}, right={right_margin}, min_w={min_band_width}, max_w={max_band_width:.1f}")
    for i, (pw, pk) in enumerate(zip(widths, peaks)):
        c1 = min_band_width <= pw
        c2 = pw < max_band_width  
        c3 = left_margin < pk
        c4 = pk < strip_w - right_margin
        # print(f"    peak[{i}] pos={pk} w={pw:.1f} | w>={min_band_width}:{c1} w<{max_band_width:.1f}:{c2} pos>{left_margin}:{c3} pos<{strip_w-right_margin}:{c4}")
    # print(f"  peak widths: {[f'{pw:.1f}' for pw in widths]}")
    valid_peak_indices = np.array([
        i for i, pw in enumerate(widths)
        if min_band_width <= pw < max_band_width
        and left_margin < peaks[i] < strip_w - right_margin
    ])
    if len(valid_peak_indices) == 0:
        return False, 0
    valid_peaks = peaks[valid_peak_indices]
    if len(valid_peaks) >= 2:
        spread = valid_peaks[-1] - valid_peaks[0]
        if spread < strip_w * 0.25:
            return False, 0
    if len(valid_peaks) >= 3:
        spacings = np.diff(valid_peaks)
        cv_spacing = np.std(spacings) / (np.mean(spacings) + 1e-6)
        # print(f"  spacing cv={cv_spacing:.3f}")
        if cv_spacing < 0.08:
            # print("  REJECTED: peaks too regular (not a resistor)")
            return False, 0
    bands_num = len(valid_peaks)
    is_valid = 4 <= bands_num <= 6
    # print(f"  valid peaks={bands_num} at {valid_peaks.tolist()}")
    return is_valid, int(bands_num)

# Validates if a contour matches expected resistor dimensions and shape properties
def get_res_geometry(countour):
    #x, y, w, h = cv.boundingRect(countour)
    rect = cv.minAreaRect(countour)
    (center, (w, h), angle) = rect

    if w == 0 or h == 0: return False, None
    aspect_ratio = max(w, h) / min(w, h)
    area = cv.contourArea(countour)
    if not (1.8 < aspect_ratio < 12.0):
        return False, None
    if area < 50:
        return False, None
    
    hull = cv.convexHull(countour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0: return False, None
    solidity = float(area) / hull_area
    if solidity < 0.5:
        return False, None
    return True, rect

# Extracts valid contours and draws rotated bounding boxes around detected resistors
def drw_recs(main_mask, img, active_windows=set()):
    contours, _ = cv.findContours(main_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(f"Total contours {len(contours)}")
    current_windows = set()
    count = 0
    for c in contours:
        is_valid, rect = get_res_geometry(c)
        if is_valid:
            roi = extract_ROI(img, rect)
            count = count + 1
            valid_res, bands_num = validate_res_content(roi)
            # print(f"Object {count} | valid={valid_res} | bands={bands_num}")
            window_name = f"ROI_{count}"
            current_windows.add(window_name)
            cv.imshow(window_name, roi)
            fit_rec = rect
            box = cv.boxPoints(fit_rec)
            box = np.int32(box)
            if valid_res:
                cv.drawContours(img, [box], 0, (255, 0, 0), 2)
        else:
            # print("Failed")
            continue
    for old_window in active_windows - current_windows:
        cv.destroyWindow(old_window)
    active_windows.clear()
    active_windows.update(current_windows)

    return img
