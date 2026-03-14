import numpy as np
import cv2
import os

# NumPy 2.0+ 호환성을 위한 패치
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid


# ============================================================
# 기본 유틸리티 함수
# ============================================================

def preprocess_to_binary(image):

    # 4. Closing
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    binary_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)
    
    return binary_closed

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


def fit_poly_with_rmse(binary_mask, degree=2):
    """폴리피팅 + RMSE를 함께 반환. 차선 필터링용."""
    y_coords, x_coords = np.where(binary_mask > 0)
    if len(y_coords) == 0:
        return None, None, None, float('inf')

    try:
        poly_coeffs = np.polyfit(y_coords, x_coords, degree)
        poly_func = np.poly1d(poly_coeffs)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        predicted_x = poly_func(y_coords)
        rmse = np.sqrt(np.mean((predicted_x - x_coords) ** 2))

        return poly_func, y_min, y_max, rmse
    except Exception:
        return None, None, None, float('inf')


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


# ============================================================
# 차선 후보 필터링 — RMSE + 스팬 거리 기반
# ============================================================

def filter_lane_candidates(binary_image, min_area=100, min_span=50, max_rmse=10.0, poly_degree=2):
    """
    바이너리 이미지에서 connected component를 분리하고,
    각 클러스터에 폴리피팅을 수행하여 차선일 가능성이 높은 후보만 필터링한다.

    Parameters:
        binary_image : 흰색 바이너리 마스크 (uint8, 0 or 255)
        min_area     : 최소 면적 (이하 무시)
        min_span     : poly 양 끝점 간 최소 유클리드 거리 (픽셀)
        max_rmse     : 폴리피팅 잔차 최대 허용값 (픽셀)
        poly_degree  : 필터링용 폴리피팅 차수 (기본 2)

    Returns:
        candidates : list of dict
            각 dict는 {'label_idx', 'bottom_x', 'top_x', 'tilt', 'y_min', 'y_max', 'rmse', 'span'} 포함
        labels     : cv2.connectedComponentsWithStats 결과 labels 배열
    """
    height, width = binary_image.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    candidates = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # 해당 클러스터만 추출
        cluster_mask = np.uint8(labels == i) * 255

        # 폴리피팅 + RMSE
        poly_func, y_min, y_max, rmse = fit_poly_with_rmse(cluster_mask, degree=poly_degree)
        if poly_func is None:
            continue

        # 스팬 거리: 클러스터 내 두 끝점 간의 유클리드 거리
        ys, xs = np.where(cluster_mask > 0)
        p1 = (xs[0], ys[0])
        p2 = (xs[-1], ys[-1])
        span = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        if span < min_span or rmse > max_rmse:
            continue

        # 좌/우 분류에 필요한 메타데이터
        ys, xs = np.where(labels == i)
        bottom_x = int(np.mean(xs[ys == np.max(ys)]))
        top_x = int(np.mean(xs[ys == np.min(ys)]))
        tilt = top_x - bottom_x

        candidates.append({
            'label_idx': i,
            'bottom_x': bottom_x,
            'top_x': top_x,
            'tilt': tilt,
            'y_min': y_min,
            'y_max': y_max,
            'rmse': rmse,
            'span': span,
        })

    return candidates, labels


# ============================================================
# 좌/우 차선 분류 — laneExtractor.py get_separated_masks() 참고
# ============================================================

def classify_left_right(candidates, labels, img_shape, tilt_threshold=20):
    """
    필터링된 차선 후보들을 좌/우 차선으로 분류한다.

    Parameters:
        candidates     : filter_lane_candidates() 반환 리스트
        labels         : connected components labels 배열
        img_shape      : (height, width) 튜플
        tilt_threshold : tilt 기반 분류 시 임계값 (픽셀)

    Returns:
        mask_left  : 좌측 차선 바이너리 마스크
        mask_right : 우측 차선 바이너리 마스크
    """
    height, width = img_shape
    img_center_x = width // 2
    mask_left = np.zeros((height, width), dtype=np.uint8)
    mask_right = np.zeros((height, width), dtype=np.uint8)

    if not candidates:
        return mask_left, mask_right

    if len(candidates) == 1:
        lane = candidates[0]
        if lane['tilt'] < -tilt_threshold:
            mask_right[labels == lane['label_idx']] = 255
        elif lane['tilt'] > tilt_threshold:
            mask_left[labels == lane['label_idx']] = 255
        else:
            if lane['bottom_x'] < img_center_x:
                mask_left[labels == lane['label_idx']] = 255
            else:
                mask_right[labels == lane['label_idx']] = 255
    else:
        left_group = [c for c in candidates if c['bottom_x'] < img_center_x]
        right_group = [c for c in candidates if c['bottom_x'] >= img_center_x]

        if left_group:
            best_left = max(left_group, key=lambda c: c['bottom_x'])
            mask_left[labels == best_left['label_idx']] = 255
        if right_group:
            best_right = min(right_group, key=lambda c: c['bottom_x'])
            mask_right[labels == best_right['label_idx']] = 255

    return mask_left, mask_right


# ============================================================
# 통합 메인 함수
# ============================================================

def determine_lanes(binary_image,
                    min_area=800,
                    min_span=250,
                    max_rmse=10.0,
                    filter_poly_degree=2,
                    fit_poly_degree=5,
                    tilt_threshold=20,
                    resample_count=50,
                    meters_per_pixel=0.01):
    """
    흰색 바이너리 마스크에서 차선을 검출하고 월드 좌표계로 변환한다.

    Parameters:
        binary_image      : 흰색 바이너리 마스크
        min_area          : 최소 클러스터 면적 (픽셀)
        min_span          : 차선 최소 스팬 거리 (픽셀, 호 길이)
        max_rmse          : 폴리피팅 잔차 최대 허용값 (픽셀)
        filter_poly_degree: 필터링 단계 폴리피팅 차수 (낮은 차수 권장)
        fit_poly_degree   : 최종 피팅 단계 폴리피팅 차수
        tilt_threshold    : 좌/우 분류 시 tilt 임계값 (픽셀)
        resample_count    : 리샘플링 포인트 수
        meters_per_pixel  : 픽셀→미터 변환 계수

    Returns:
        left_lane  : np.array shape (2, resample_count) — [world_x, world_y], 없으면 (2, 0)
        right_lane : np.array shape (2, resample_count) — [world_x, world_y], 없으면 (2, 0)
    """
    img_height, img_width = binary_image.shape

    # 1. 차선 후보 필터링 (RMSE + 스팬)
    candidates, labels = filter_lane_candidates(
        binary_image, min_area, min_span, max_rmse, filter_poly_degree
    )

    # 2. 좌/우 차선 분류
    mask_left, mask_right = classify_left_right(
        candidates, labels, binary_image.shape, tilt_threshold
    )

    # 3. 최종 폴리피팅 (더 높은 차수로 정밀 피팅)
    left_poly, left_y_min, left_y_max = fit_poly(mask_left, degree=fit_poly_degree)
    right_poly, right_y_min, right_y_max = fit_poly(mask_right, degree=fit_poly_degree)

    # 4. 월드 좌표 리샘플링
    left_x, left_y = generate_resampled_points(
        left_poly, left_y_min, left_y_max, resample_count, img_width, img_height, meters_per_pixel
    )
    right_x, right_y = generate_resampled_points(
        right_poly, right_y_min, right_y_max, resample_count, img_width, img_height, meters_per_pixel
    )

    left_lane = np.array([left_x, left_y]) if left_x is not None else np.array([[], []])
    right_lane = np.array([right_x, right_y]) if right_x is not None else np.array([[], []])

    return left_lane, right_lane