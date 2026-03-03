import numpy as np
import cv2
import os

# -----------------------------------------------------------
# [설정] 파라미터 수정 (360x404 해상도 및 물리 스케일 반영)
# -----------------------------------------------------------
M_PER_PIXEL = 0.013        # 1 픽셀당 거리 (m)
STANDARD_LANE_WIDTH_M = 1.5 # 일반적인 차선 폭 (m)

# 차선 폭을 픽셀로 변환 (3.0m / 0.013 ≈ 230 px)
LANE_WIDTH_PIXELS = int(STANDARD_LANE_WIDTH_M / M_PER_PIXEL)

RESAMPLE_COUNT = 100 

# 차량 앞축 기준 좌표 (Front Axle Center)
AXLE_X = 180
AXLE_Y = 365

def remove_low_intensity_noise(image_array, threshold):
    image = np.array(image_array, dtype=np.uint8)
    if threshold > 0:
        image[image < threshold] = 0
    return image

def preprocess_to_binary(image):
    # 1. Blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Top-Hat
    top_hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    top_hat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, top_hat_kernel)
    
    # 3. Otsu
    _, binary = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Closing
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    
    return binary_closed

def get_separated_masks(binary_image, max_area):
    height, width = binary_image.shape
    img_center_x = width // 2 
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    mask_left = np.zeros_like(binary_image)
    mask_right = np.zeros_like(binary_image)
    
    candidates = []
    MIN_AREA = max_area 

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > MIN_AREA:
            ys, xs = np.where(labels == i)
            max_y_idx = np.argmax(ys)
            bottom_x = xs[max_y_idx]
            min_y_idx = np.argmin(ys)
            top_x = xs[min_y_idx]
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

def calculate_stanley_error(img, center_xs, center_ys):
    """
    Stanley Method 계산 및 시각화
    Return: cte_meter, heading_rad
    """
    if len(center_xs) == 0:
        return 0.0, 0.0
    
    # 1. 차량 앞축(Target)과 가장 가까운 경로점 찾기
    idx = (np.abs(center_ys - AXLE_Y)).argmin()
    
    target_x = center_xs[idx]
    target_y = center_ys[idx]
    
    # 2. CTE 계산
    cte_pixel = target_x - AXLE_X
    cte_meter = cte_pixel * M_PER_PIXEL
    
    # 3. Heading Error 계산
    look_ahead_step = 5 
    next_idx = max(0, idx - look_ahead_step) 
    
    path_dx = center_xs[next_idx] - center_xs[idx]
    path_dy = center_ys[next_idx] - center_ys[idx] 
    
    # atan2(dx, -dy) -> 수직축 기준 각도 (라디안)
    heading_rad = np.arctan2(path_dx, -path_dy)
    heading_deg = np.degrees(heading_rad)
    
    # 4. 시각화
    cv2.circle(img, (AXLE_X, AXLE_Y), 6, (0, 255, 255), -1)
    cv2.line(img, (AXLE_X, AXLE_Y), (int(target_x), int(target_y)), (0, 255, 255), 2)
    
    info_text_cte = f"CTE: {cte_meter:.3f} m"
    info_text_head = f"Head: {heading_deg:.2f} deg"
    
    cv2.putText(img, info_text_cte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, info_text_head, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return cte_meter, heading_rad

def calculate_and_draw_center(img, left_data, right_data):
    """
    중심선 계산, 그리기, 에러 및 가장 먼 지점(Forward/Lateral) 계산
    Return: (cte_m, heading_rad, farthest_fwd_m, farthest_lat_m)
    """
    poly_l, min_l, max_l = left_data
    poly_r, min_r, max_r = right_data
    
    center_xs = []
    center_ys = []
    
    # 차선 존재 여부에 따른 중심선 좌표 생성
    if poly_l is not None and poly_r is not None:
        lx, ly = generate_resampled_points(poly_l, min_l, max_l, RESAMPLE_COUNT)
        rx, ry = generate_resampled_points(poly_r, min_r, max_r, RESAMPLE_COUNT)
        center_xs = (lx + rx) / 2.0
        center_ys = (ly + ry) / 2.0

    elif poly_l is not None:
        lx, ly = generate_resampled_points(poly_l, min_l, max_l, RESAMPLE_COUNT)
        deriv = poly_l.deriv()(ly)
        magnitude = np.sqrt(1 + deriv**2)
        nx = 1.0 / magnitude
        ny = -deriv / magnitude
        offset = LANE_WIDTH_PIXELS / 2.0
        center_xs = lx + (nx * offset)
        center_ys = ly + (ny * offset)

    elif poly_r is not None:
        rx, ry = generate_resampled_points(poly_r, min_r, max_r, RESAMPLE_COUNT)
        deriv = poly_r.deriv()(ry)
        magnitude = np.sqrt(1 + deriv**2)
        nx = 1.0 / magnitude
        ny = -deriv / magnitude
        offset = LANE_WIDTH_PIXELS / 2.0
        center_xs = rx - (nx * offset)
        center_ys = ry - (ny * offset)
        
    else:
        # 차선 없음
        return 0.0, 0.0, 0.0, 0.0

    if len(center_xs) > 0:
        # 중심선 그리기
        points = np.stack([center_xs, center_ys], axis=1).astype(np.int32)
        cv2.polylines(img, [points], isClosed=False, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        
        # 1. CTE & Heading 계산
        cte, head_rad = calculate_stanley_error(img, center_xs, center_ys)
        
        # 2. Farthest Point (가장 먼 지점) 계산
        # 이미지 상단(Y가 작은 쪽)이 전방이므로 min(center_ys)를 찾습니다.
        min_y_idx = np.argmin(center_ys)
        farthest_x_px = center_xs[min_y_idx]
        farthest_y_px = center_ys[min_y_idx]
        
        # 픽셀 -> 미터 변환 (차량 축 기준)
        # Forward: 차량(AXLE_Y)에서 위쪽(작은Y)으로 얼마나 갔는지
        farthest_fwd_m = (AXLE_Y - farthest_y_px) * M_PER_PIXEL
        
        # Lateral: 차량(AXLE_X)에서 좌우로 얼마나 떨어졌는지 (우측+, 좌측-)
        farthest_lat_m = (farthest_x_px - AXLE_X) * M_PER_PIXEL
        
        return cte, head_rad, farthest_fwd_m, farthest_lat_m
    
    return 0.0, 0.0, 0.0, 0.0

def draw_resampled_poly(img, x_pts, y_pts, color):
    if x_pts is None: return
    points = np.stack([x_pts, y_pts], axis=1).astype(np.int32)
    cv2.polylines(img, [points], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

def main(image_array, noise_threshold=180, max_area=1800):
    # 전처리
    denoised_image = remove_low_intensity_noise(image_array, noise_threshold)
    binary_image = preprocess_to_binary(denoised_image)
    
    # 좌우 분리 및 세선화
    mask_left, mask_right = get_separated_masks(binary_image, max_area)
    clean_left = post_process_skeleton(morphology_thinning(mask_left))
    clean_right = post_process_skeleton(morphology_thinning(mask_right))
    
    # 다항 회귀
    left_data = fit_poly(clean_left, degree=5)  
    right_data = fit_poly(clean_right, degree=5)
    
    # 결과 이미지 생성
    height, width = binary_image.shape
    color_result = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 감지된 차선 그리기
    l_pts_x, l_pts_y = generate_resampled_points(*left_data, RESAMPLE_COUNT)
    r_pts_x, r_pts_y = generate_resampled_points(*right_data, RESAMPLE_COUNT)
    
    draw_resampled_poly(color_result, l_pts_x, l_pts_y, (0, 0, 255))
    draw_resampled_poly(color_result, r_pts_x, r_pts_y, (255, 0, 0))
    
    # 중심선 계산 및 데이터 추출 (핵심 변경 부분)
    cte_m, heading_err_rad, farthest_forward_m, farthest_lateral_m = \
        calculate_and_draw_center(color_result, left_data, right_data)
    
    # 이미지 저장
    output_filename = "processed_lane_stanley_v2.png"
    # 디버그용 바이너리 이미지가 필요하다면 아래 주석 해제
    # cv2.imwrite("debug_binary_pattern.png", binary_image)
    
    try:
        cv2.imwrite(output_filename, color_result)
    except Exception as e:
        return f"Error: {str(e)}"
    
    # 요청하신 포맷으로 문자열 반환
    # Format: "절대경로,CTE(m),Heading(rad),FarthestForward(m),FarthestLateral(m)"
    return f"{os.path.abspath(output_filename)},{cte_m},{heading_err_rad},{farthest_forward_m:.3f},{farthest_lateral_m:.3f}"