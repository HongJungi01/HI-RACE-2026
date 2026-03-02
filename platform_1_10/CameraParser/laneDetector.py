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

def generate_resampled_points(poly_func, y_min, y_max, count, img_width, img_height, meters_per_pixel):
    if poly_func is None:
        return None, None
    
    # 1. 픽셀 좌표계에서의 Y값 샘플링
    # y_max(차량과 가장 가까운 지점)을 시작점으로 설정하여 인덱스 0번이 되도록 함
    y_pixel_points = np.linspace(y_max, y_min, count)
    x_pixel_points = poly_func(y_pixel_points)

    # 2. 월드 좌표계 변환 (차량 중심을 0,0으로 설정)
    # 이미지 하단 중앙을 (0,0)으로 가정:
    # World Y = (img_height - pixel_y) * meters_per_pixel (차량에서 멀어질수록 커짐)
    # World X = (pixel_x - img_width / 2) * meters_per_pixel (중앙 기준 좌/우)
    
    world_y = (img_height - y_pixel_points) * meters_per_pixel
    world_x = (x_pixel_points - img_width / 2) * meters_per_pixel
    
    return world_x, world_y

def lane_extractor(binary_image, min_area=100, resample_count=50, meters_per_pixel=0.01):
    img_height, img_width = binary_image.shape
    mask_left, mask_right = get_separated_masks(binary_image, min_area)
    skeleton_left = morphology_thinning(mask_left)
    skeleton_right = morphology_thinning(mask_right)
    processed_left = post_process_skeleton(skeleton_left)
    processed_right = post_process_skeleton(skeleton_right)
    
    left_poly, left_y_min, left_y_max = fit_poly(processed_left)
    right_poly, right_y_min, right_y_max = fit_poly(processed_right)
    
    # 수정된 resampling 함수 호출
    left_x, left_y = generate_resampled_points(left_poly, left_y_min, left_y_max, resample_count, img_width, img_height, meters_per_pixel)
    right_x, right_y = generate_resampled_points(right_poly, right_y_min, right_y_max, resample_count, img_width, img_height, meters_per_pixel)
    
    left_lane = np.array([left_x, left_y]) if left_x is not None else np.array([[], []])
    right_lane = np.array([right_x, right_y]) if right_x is not None else np.array([[], []])
    
    return left_lane, right_lane