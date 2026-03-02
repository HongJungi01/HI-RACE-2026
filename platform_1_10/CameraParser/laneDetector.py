import numpy as np
import cv2
import os

# -----------------------------------------------------------
# [Description] Line Detection and Post-Processing Functions
# Get binary masks for left and right lanes, perform morphological thinning, and fit polynomial curves to lane lines.
# returns: left_lane_points, right_lane_points
# ------------------------------------------------------------

def get_separated_masks(binary_image, min_area):
    height, width = binary_image.shape
    img_center_x = width // 2 
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    mask_left = np.zeros_like(binary_image)
    mask_right = np.zeros_like(binary_image)
    
    candidates = []
    MIN_AREA = min_area 

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > MIN_AREA:
            ys, xs = np.where(labels == i)
            max_y = np.max(ys)
            min_y = np.min(ys)
            bottom_x = int(np.mean(xs[ys == max_y]))
            top_x = int(np.mean(xs[ys == min_y]))
            tilt = top_x - bottom_x
            candidates.append({'idx': i, 'bottom_x': bottom_x, 'tilt': tilt})

    if not candidates:
        return mask_left, mask_right

    if len(candidates) == 1:
        lane = candidates[0]
        TILT_THRESHOLD = 20
        if lane['tilt'] < -TILT_THRESHOLD: 
            mask_right[labels == lane['idx']] = 255
        elif lane['tilt'] > TILT_THRESHOLD: 
            mask_left[labels == lane['idx']] = 255
        else:
            if lane['bottom_x'] < img_center_x:
                mask_left[labels == lane['idx']] = 255
            else:
                mask_right[labels == lane['idx']] = 255
    else:
        left_group = [c for c in candidates if c['bottom_x'] < img_center_x]
        right_group = [c for c in candidates if c['bottom_x'] >= img_center_x]
        
        if left_group:
            best_left = max(left_group, key=lambda c: c['bottom_x'])
            mask_left[labels == best_left['idx']] = 255
        if right_group:
            best_right = min(right_group, key=lambda c: c['bottom_x'])
            mask_right[labels == best_right['idx']] = 255

    return mask_left, mask_right

def morphology_thinning(binary_img):
    skeleton = np.zeros(binary_img.shape, np.uint8)
    img = binary_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0: break
    return skeleton

def post_process_skeleton(skeleton_img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton_img, connectivity=8)
    final_output = np.zeros_like(skeleton_img)
    MIN_LENGTH = 30 
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_LENGTH:
            final_output[labels == i] = 255
    return final_output

def fit_poly(binary_mask, degree=5):
    y_coords, x_coords = np.where(binary_mask > 0)
    if len(y_coords) == 0:
        return None, None, None

    try:
        poly_coeffs = np.polyfit(y_coords, x_coords, degree)
        poly_func = np.poly1d(poly_coeffs)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        return poly_func, y_min, y_max
    except Exception:
        return None, None, None

def generate_resampled_points(poly_func, y_min, y_max, count):
    if poly_func is None:
        return None, None
    y_points = np.linspace(y_min, y_max, count)
    x_points = poly_func(y_points)
    return x_points, y_points

def lane_extractor(binary_image, min_area=100, resample_count=50):
    mask_left, mask_right = get_separated_masks(binary_image, min_area)
    skeleton_left = morphology_thinning(mask_left)
    skeleton_right = morphology_thinning(mask_right)
    processed_left = post_process_skeleton(skeleton_left)
    processed_right = post_process_skeleton(skeleton_right)
    
    left_poly, left_y_min, left_y_max = fit_poly(processed_left)
    right_poly, right_y_min, right_y_max = fit_poly(processed_right)
    
    left_x, left_y = generate_resampled_points(left_poly, left_y_min, left_y_max, resample_count)
    right_x, right_y = generate_resampled_points(right_poly, right_y_min, right_y_max, resample_count)
    
    # 결과를 np.array([[x...], [y...]]) 형식으로 변환
    left_lane = np.array([left_x, left_y]) if left_x is not None else np.array([[], []])
    right_lane = np.array([right_x, right_y]) if right_x is not None else np.array([[], []])
    
    return left_lane, right_lane