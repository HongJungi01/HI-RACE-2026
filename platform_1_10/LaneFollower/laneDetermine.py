import numpy as np
import cv2

# NumPy 2.0+ 호환성을 위한 패치
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid


# ============================================================
# 캘리브레이션 설정 (640x310 해상도 기준)
# ============================================================
STANDARD_LANE_WIDTH_M = 1.5  # 차선 폭 (m)
RESAMPLE_COUNT = 50          # 샘플링 포인트 수

# 캘리브레이션 포인트: (pixel_x, pixel_y) -> (world_x, world_y) [단위: 미터]
_CALIB_PIXEL_PTS = np.array([
    [448, 130],
    [426, 100],
    [193, 129],
    [320, 101],
    [166, 172],
], dtype=np.float32)

_CALIB_WORLD_PTS = np.array([
    [1.5, 0.5],
    [1.5, 0.0],
    [0.0, 0.5],
    [0.75, 0.0],
    [0.0, 1.0],
], dtype=np.float32)


def build_homography(pixel_pts=None, world_pts=None):
    """캘리브레이션 포인트로부터 호모그래피 행렬(3x3)을 계산한다."""
    if pixel_pts is None:
        pixel_pts = _CALIB_PIXEL_PTS
    if world_pts is None:
        world_pts = _CALIB_WORLD_PTS
    H, status = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC)
    return H


def pixel_to_world(pixel_points, H):
    """
    픽셀 좌표 배열을 호모그래피로 월드 좌표로 변환한다.
    pixel_points: shape (N, 2) — [[px_x, px_y], ...]
    return: (world_x, world_y) 각각 shape (N,)
    """
    pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return world[:, 0], world[:, 1]


def world_to_pixel(world_points, H):
    """
    월드 좌표 배열을 호모그래피의 역행렬로 픽셀 좌표로 변환한다.
    world_points: shape (N, 2) — [[world_x, world_y], ...]
    return: (pixel_x, pixel_y) 각각 shape (N,)
    """
    H_inv = np.linalg.inv(H)
    pts = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)
    pixel = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return pixel[:, 0], pixel[:, 1]


# 모듈 로드 시 호모그래피 행렬을 한 번만 계산
_H_raw = build_homography()

# 차량 앞축중심점 (320, 310) 픽셀을 월드 (0, 0)으로 평행이동
_origin_px = np.array([[320, 310]], dtype=np.float32)
_ox, _oy = pixel_to_world(_origin_px, _H_raw)
_T = np.array([
    [1, 0, -_ox[0]],
    [0, 1, -_oy[0]],
    [0, 0,       1]
], dtype=np.float64)
_H = _T @ _H_raw


# ============================================================
# 기본 유틸리티 함수
# ============================================================

def fit_poly(binary_mask, degree=5):
    """바이너리 마스크에서 y→x 다항식 피팅."""
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


def generate_resampled_points_pixel(poly_func, y_min, y_max, count):
    """다항식에서 count개의 균등 샘플링 포인트를 (픽셀 좌표)로 반환한다.
    y_max(차량에 가까운 쪽)부터 y_min(먼 쪽) 순서로 인덱스 0번이 차량 근처."""
    if poly_func is None:
        return None, None
    y_pixel = np.linspace(y_max, y_min, count)
    x_pixel = poly_func(y_pixel)
    return x_pixel, y_pixel


def generate_resampled_points_world(poly_func, y_min, y_max, count, H=None):
    """다항식에서 count개의 균등 샘플링 후 호모그래피로 월드 좌표 변환."""
    if poly_func is None:
        return None, None
    if H is None:
        H = _H
    x_pixel, y_pixel = generate_resampled_points_pixel(poly_func, y_min, y_max, count)
    pixel_pts = np.stack([x_pixel, y_pixel], axis=1)  # (N, 2)
    world_x, world_y = pixel_to_world(pixel_pts, H)
    return world_x, world_y


# ============================================================
# 차선 월드좌표 추출
# ============================================================

def extract_lane_world_points(binary_mask,
                              min_area=100,
                              min_span=50,
                              max_rmse=100.0,
                              poly_degree=2,
                              fit_degree=5,
                              resample_count=RESAMPLE_COUNT,
                              H=None):
    """
    바이너리 마스크에서 차선 후보를 필터링하고, 각 차선을 다항식 피팅 후
    호모그래피로 월드 좌표 변환하여 반환한다.

    1. connectedComponentsWithStats로 클러스터 분리
    2. 각 클러스터: area → RMSE → span 필터링
    3. centroid x 기준 오름차순 정렬 (왼쪽 → 오른쪽)
    4. 최대 3개까지 선택
    5. 각 차선을 fit_degree 다항식 피팅 → resample_count개 월드 좌표 샘플링

    Args:
        binary_mask:    전처리 완료된 uint8 바이너리 이미지 (0 or 255)
        min_area:       최소 클러스터 면적 (px)
        min_span:       최소 스팬 거리 (px)
        max_rmse:       최대 허용 RMSE (px)
        poly_degree:    필터링용 폴리피팅 차수
        fit_degree:     최종 월드좌표 변환용 폴리피팅 차수
        resample_count: 차선당 샘플링 포인트 수
        H:              호모그래피 행렬 (None이면 모듈 기본값 사용)

    Returns:
        list[np.ndarray]: 각 원소는 shape (2, resample_count)의 np.ndarray
                          [[world_x1, world_x2, ...], [world_y1, world_y2, ...]]
                          centroid x 오름차순 정렬, 최대 3개. 감지 없으면 빈 리스트 [].
    """
    if H is None:
        H = _H

    # Step 1: 클러스터 분리
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels <= 1:
        return []

    # Step 2: 각 클러스터 필터링
    candidates = []
    for i in range(1, num_labels):
        # area 필터
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cluster_mask = np.uint8(labels == i) * 255

        # RMSE 필터
        poly_func, y_min, y_max, rmse = fit_poly_with_rmse(cluster_mask, degree=poly_degree)
        if poly_func is None or rmse > max_rmse:
            continue

        # span 필터
        ys, xs = np.where(cluster_mask > 0)
        p1 = (xs[0], ys[0])
        p2 = (xs[-1], ys[-1])
        span = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if span < min_span:
            continue

        centroid_x = centroids[i][0]
        candidates.append((centroid_x, cluster_mask))

    # Step 3: centroid x 기준 오름차순 정렬
    candidates.sort(key=lambda c: c[0])

    # Step 4: 최대 3개 선택
    candidates = candidates[:3]

    # Step 5: 각 차선을 피팅 → 월드 좌표 변환
    result = []
    for _, mask in candidates:
        poly_func, y_min, y_max = fit_poly(mask, degree=fit_degree)
        if poly_func is None:
            continue

        wx, wy = generate_resampled_points_world(
            poly_func, y_min, y_max, resample_count, H
        )
        if wx is None:
            continue

        result.append(np.array([wx, wy]))  # shape (2, resample_count)

    return result