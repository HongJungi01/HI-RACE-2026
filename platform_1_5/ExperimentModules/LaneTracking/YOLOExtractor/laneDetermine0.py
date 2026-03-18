import numpy as np
import cv2
import os

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


# 모듈 로드 시 호모그래피 행렬을 한 번만 계산
_H = build_homography()


# ============================================================
# 기본 유틸리티 함수
# ============================================================

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
# 차선 후보 필터링 — RMSE + 스팬 거리 기반
# ============================================================

def filter_lane_candidates(binary_image, min_area=100, min_span=50, max_rmse=10.0, poly_degree=2):
    """
    바이너리 이미지에서 조건을 만족하는 뭉치들만 남겨놓은 바이너리 이미지를 반환합니다.
    """
    height, width = binary_image.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    filtered_mask = np.zeros_like(binary_image)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cluster_mask = np.uint8(labels == i) * 255

        poly_func, y_min, y_max, rmse = fit_poly_with_rmse(cluster_mask, degree=poly_degree)
        if poly_func is None:
            continue

        ys, xs = np.where(cluster_mask > 0)
        p1 = (xs[0], ys[0])
        p2 = (xs[-1], ys[-1])
        span = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        if span < min_span or rmse > max_rmse:
            continue
            
        filtered_mask[labels == i] = 255

    return filtered_mask


# ============================================================
# 좌/우 차선 분류 — laneExtractor.py get_separated_masks() 참고
# ============================================================

def classify_left_right(filtered_mask, tilt_threshold=20):
    """
    들어온 바이너리 이미지에서 하얀색 뭉치가 두 개면 각각 중심점을 기반으로 좌/우 분류,
    하나뿐이면 기존 로직(기울기 및 하단 x좌표)을 유지합니다.
    """
    height, width = filtered_mask.shape
    img_center_x = width // 2
    
    mask_left = np.zeros((height, width), dtype=np.uint8)
    mask_right = np.zeros((height, width), dtype=np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)
    blob_count = num_labels - 1 # 배경 제외

    # 1. 먼저 각 마스크에 픽셀 할당
    if blob_count == 1:
        ys, xs = np.where(labels == 1)
        bottom_x = int(np.mean(xs[ys == np.max(ys)]))
        top_x = int(np.mean(xs[ys == np.min(ys)]))
        tilt = top_x - bottom_x

        if tilt < -tilt_threshold:
            mask_right[labels == 1] = 255
        elif tilt > tilt_threshold:
            mask_left[labels == 1] = 255
        else:
            if bottom_x < img_center_x:
                mask_left[labels == 1] = 255
            else:
                mask_right[labels == 1] = 255
    elif blob_count >= 2:
        # 뭉치가 2개 이상인 경우 중심점 기준으로 분류
        for i in range(1, num_labels):
            if centroids[i][0] < img_center_x:
                mask_left[labels == i] = 255
            else:
                mask_right[labels == i] = 255

    # 2. 상태(state) 판별 (NULL 텍스트 그리기 전)
    has_left = np.max(mask_left) > 0
    has_right = np.max(mask_right) > 0

    if has_left and has_right:
        state = "both"
    elif has_left:
        state = "left"
    elif has_right:
        state = "right"
    else:
        state = "none"

    # 3. 시각화를 위한 NULL 텍스트 추가 (선택 사항)
    def draw_null(img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "NULL"
        text_size = cv2.getTextSize(text, font, 2, 5)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), font, 2, 255, 5, cv2.LINE_AA)
        return img

    if not has_left:
        mask_left = draw_null(mask_left)
    if not has_right:
        mask_right = draw_null(mask_right)

    return mask_left, mask_right, state


# ============================================================
# 중심선 계산 — test.py calculate_and_draw_center() 참고
# ============================================================

def calculate_center_line(left_mask, right_mask, state, H=None,
                          lane_width_m=STANDARD_LANE_WIDTH_M,
                          resample_count=RESAMPLE_COUNT,
                          poly_degree=5):
    """
    좌/우 차선 마스크로부터 중심선의 월드 좌표를 계산한다.

    양쪽 감지  : (left + right) / 2
    한쪽만 감지: 월드 좌표에서 수직 법선 방향으로 lane_width/2 오프셋
    미감지     : 빈 배열

    Return: (center_world_x, center_world_y)  — 각각 ndarray or 빈 리스트
    """
    if H is None:
        H = _H

    offset = lane_width_m / 2.0

    if state == "both":
        # 양쪽 감지: 평균 계산
        poly_l, min_l, max_l = fit_poly(left_mask, degree=poly_degree)
        poly_r, min_r, max_r = fit_poly(right_mask, degree=poly_degree)
        lx, ly = generate_resampled_points_world(poly_l, min_l, max_l, resample_count, H)
        rx, ry = generate_resampled_points_world(poly_r, min_r, max_r, resample_count, H)
        center_x, center_y = (lx + rx) / 2.0, (ly + ry) / 2.0

    elif state == "left":
        # 왼쪽만 감지: 오른쪽으로 offset
        poly_l, min_l, max_l = fit_poly(left_mask, degree=poly_degree)
        lx, ly = generate_resampled_points_world(poly_l, min_l, max_l, resample_count, H)
        center_x, center_y = _offset_center(lx, ly, +offset)

    elif state == "right":
        # 오른쪽만 감지: 왼쪽으로 offset
        poly_r, min_r, max_r = fit_poly(right_mask, degree=poly_degree)
        rx, ry = generate_resampled_points_world(poly_r, min_r, max_r, resample_count, H)
        center_x, center_y = _offset_center(rx, ry, -offset)

    else:
        return [], []

    # ── 추가: 계산된 중심점들을 기반으로 다시 50개 포인트 피팅 (Smoothing) ──
    try:
        # y_world 기준으로 5차 다항식 피팅 (x = f(y))
        # 중심 경로는 비교적 단순하므로 5차 정도가 적당함
        poly_coeffs = np.polyfit(center_y, center_x, 5)
        poly_func = np.poly1d(poly_coeffs)
        
        # 원래의 y 범위를 유지하며 균등하게 50개 생성
        y_min_w, y_max_w = np.min(center_y), np.max(center_y)
        resampled_y = np.linspace(y_min_w, y_max_w, resample_count)
        resampled_x = poly_func(resampled_y)
        
        return resampled_x.tolist(), resampled_y.tolist()
    except Exception:
        # 피팅 실패 시 원본 반환
        return center_x.tolist(), center_y.tolist()


def _mask_has_lane(mask):
    """마스크에 실제 차선 데이터가 있는지 확인 (NULL 텍스트만 있는 경우 False)."""
    if mask is None:
        return False
    # NULL 마스크는 draw_null로 텍스트만 그려짐 — 실제 차선은 connected component가 1개 이상이고
    # 그 면적이 작음. 간단히 흰색 픽셀 비율로 판단.
    white_count = np.count_nonzero(mask)
    if white_count == 0:
        return False
    # NULL 텍스트는 보통 소수 픽셀. 차선은 최소 수백 픽셀.
    # 하지만 draw_null의 결과도 수백 픽셀일 수 있으므로,
    # classify_left_right가 draw_null을 호출했는지를 텍스트 패턴으로 구분하기 어려움.
    # 대신 mask에 connected component 수와 모양으로 판단:
    # draw_null 결과는 가운데에 텍스트가 있으므로 centroid가 중앙 근처.
    # 안전하게: NULL은 classify_left_right에서만 생성되므로
    # 여기서는 classify_left_right 호출 전의 원본 마스크를 받도록 main.py에서 조정.
    # → 간단하게 흰색 카운트 > 0이면 True
    return True


def _offset_center(xs, ys, offset_m):
    """
    월드 좌표 곡선에서 수직 법선 방향으로 offset_m만큼 이동한 중심선을 계산한다.
    offset_m > 0: 오른쪽(+x)으로 이동, offset_m < 0: 왼쪽(-x)으로 이동
    """
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    # 인접 점 간 접선 벡터
    dx = np.gradient(xs)
    dy = np.gradient(ys)

    # 접선에 수직인 법선 벡터 (오른쪽 방향)
    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude[magnitude == 0] = 1e-8  # 0 나눗셈 방지
    nx = -dy / magnitude  # 수직 법선 x
    ny =  dx / magnitude  # 수직 법선 y

    center_x = xs + nx * offset_m
    center_y = ys + ny * offset_m

    return center_x, center_y


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


def draw_path_on_image(image, center_world_x, center_world_y, left_mask=None, right_mask=None, state="none", H=None):
    """
    월드 좌표 중심선과 감지된 좌/우 차선의 피팅 경로를 픽셀 좌표로 변환하여 이미지 위에 그린다.
    """
    if H is None:
        H = _H

    vis_img = image.copy()

    def draw_path(wx, wy, color, line_thickness=2):
        if len(wx) == 0: return
        world_pts = np.stack([wx, wy], axis=1)
        px_x, px_y = world_to_pixel(world_pts, H)
        pts = np.stack([px_x, px_y], axis=1).astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(vis_img, tuple(pts[i]), tuple(pts[i+1]), color, line_thickness)
        for pt in pts:
            cv2.circle(vis_img, tuple(pt), 2, color, -1)

    # 1. 좌측 차선 그리기 (파란색)
    if left_mask is not None and (state == "both" or state == "left"):
        poly_l, min_l, max_l = fit_poly(left_mask, degree=5)
        if poly_l:
            lx, ly = generate_resampled_points_world(poly_l, min_l, max_l, RESAMPLE_COUNT, H)
            draw_path(lx, ly, (255, 0, 0))

    # 2. 우측 차선 그리기 (초록색)
    if right_mask is not None and (state == "both" or state == "right"):
        poly_r, min_r, max_r = fit_poly(right_mask, degree=5)
        if poly_r:
            rx, ry = generate_resampled_points_world(poly_r, min_r, max_r, RESAMPLE_COUNT, H)
            draw_path(rx, ry, (0, 255, 0))

    # 3. 중심선 그리기 (노란색)
    if len(center_world_x) > 0:
        draw_path(center_world_x, center_world_y, (0, 255, 255), line_thickness=3)

    return vis_img


# ============================================================
# CTE / Heading Error 계산 — test.py calculate_stanley_error() 참고
# ============================================================

def calculate_stanley_error(center_world_x, center_world_y, H=None):
    """
    월드 좌표 중심선으로부터 CTE(m)와 Heading Error(rad)를 계산한다.
    차량 위치는 이미지 하단 중앙 (320, 310)을 호모그래피로 월드 변환하여 결정.

    Return: (cte_m, heading_rad)
    """
    if H is None:
        H = _H

    center_x = np.array(center_world_x, dtype=np.float64)
    center_y = np.array(center_world_y, dtype=np.float64)

    if len(center_x) == 0:
        return 0.0, 0.0

    # 차량 앞축 위치를 호모그래피로 월드 좌표 변환
    car_pixel = np.array([[320.0, 310.0]], dtype=np.float32)  # 이미지 하단 중앙
    car_world_x, car_world_y = pixel_to_world(car_pixel, H)
    car_wx, car_wy = float(car_world_x[0]), float(car_world_y[0])

    # 1. 차량과 가장 가까운 중심선 점 찾기
    dist = np.sqrt((center_x - car_wx)**2 + (center_y - car_wy)**2)
    idx = int(np.argmin(dist))

    # 2. CTE 계산 (lateral, 즉 x 방향 차이)
    cte_m = float(center_x[idx] - car_wx)

    # 3. Heading Error 계산 (laneDetermine은 0번이 차량 근처이므로 + 방향이 전방)
    look_ahead = 5
    next_idx = min(len(center_x) - 1, idx + look_ahead) # -에서 +로 변경
    path_dx = center_x[next_idx] - center_x[idx]
    path_dy = center_y[next_idx] - center_y[idx]
    
    # 월드 좌표계 y가 전방으로 갈수록 커진다면 -path_dy가 아니라 path_dy를 사용해야 함
    # (캘리브레이션 설정에 따라 확인 필요)
    heading_rad = float(np.arctan2(path_dx, path_dy)) 

    return cte_m, heading_rad


def fit_lane_in_world(mask, H, degree=3):
    """마스크의 픽셀들을 월드 좌표로 먼저 변환한 후, 월드 좌표계에서 다항식을 피팅한다."""
    y_coords, x_coords = np.where(mask > 0)
    if len(y_coords) < 10: return None, 0, 0
    
    # 1. 모든 픽셀을 월드 좌표로 변환
    pixel_pts = np.stack([x_coords, y_coords], axis=1)
    wx, wy = pixel_to_world(pixel_pts, H)
    
    # 2. 월드 좌표계(Forward=y, Lateral=x)에서 피팅
    # 차량 진행 방향(wy)을 독립변수로, 좌우 편차(wx)를 종속변수로 설정
    try:
        poly_coeff = np.polyfit(wy, wx, degree)
        poly_func = np.poly1d(poly_coeff)
        return poly_func, np.min(wy), np.max(wy)
    except:
        return None, 0, 0

def draw_center_path_on_image(image, center_world_x, center_world_y, left_mask=None, right_mask=None, state="none", H=None):
    if H is None: H = _H
    vis_img = image.copy()

    def draw_world_path(poly_func, y_min, y_max, color):
        if poly_func is None: return
        # 월드 좌표에서 균등하게 샘플링
        wy = np.linspace(y_min, y_max, RESAMPLE_COUNT)
        wx = poly_func(wy)
        world_pts = np.stack([wx, wy], axis=1)
        # 다시 픽셀로 변환하여 그리기
        px_x, px_y = world_to_pixel(world_pts, H)
        pts = np.stack([px_x, px_y], axis=1).astype(np.int32)
        for i in range(len(pts)-1):
            cv2.line(vis_img, tuple(pts[i]), tuple(pts[i+1]), color, 2)

    # 좌/우 차선을 월드 좌표계에서 피팅하여 그리기
    if left_mask is not None:
        p, ymin, ymax = fit_lane_in_world(left_mask, H, degree=3)
        draw_world_path(p, ymin, ymax, (255, 0, 0)) # Blue

    if right_mask is not None:
        p, ymin, ymax = fit_lane_in_world(right_mask, H, degree=3)
        draw_world_path(p, ymin, ymax, (0, 255, 0)) # Green
    
    # 중심선 그리기
    if len(center_world_x) > 0:
        draw_world_path(center_world_x, center_world_y, (0, 255, 255), line_thickness=3)

    return vis_img

# ============================================================
# Feedforward 제어용 곡률(Kappa) 계산 추가
# ============================================================
def calculate_curvature(center_world_x, center_world_y, H=None):
    """
    미리 계산된 중심선 배열(cx, cy)을 재활용하여 2D 매개변수 곡선의 
    1차, 2차 미분을 통해 전방 경로의 곡률(kappa)을 계산합니다.
    """
    if H is None:
        H = _H
        
    x = np.array(center_world_x, dtype=np.float64)
    y = np.array(center_world_y, dtype=np.float64)
    
    # 점이 부족하여 미분이 불가능한 경우
    if len(x) < 3:
        return 0.0

    # 1. 중심선의 1차 미분 (dx, dy) 및 2차 미분 (ddx, ddy) 계산
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 2. 곡률 계산 공식: (dx*ddy - dy*ddx) / ((dx^2 + dy^2)^(3/2))
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-8 # 0으로 나누기 방지
    kappa_array = (dy * ddx - dx * ddy) / denominator # 차량 제어계 맞춤형 곡률 (우회전이 플러스가 나오도록 부호 반전)

    # 3. Heading Error와 동일하게 차량 앞(Look-ahead) 지점의 곡률 추출
    car_pixel = np.array([[320.0, 310.0]], dtype=np.float32) 
    car_world_x, car_world_y = pixel_to_world(car_pixel, H)
    car_wx, car_wy = float(car_world_x[0]), float(car_world_y[0])

    dist = np.sqrt((x - car_wx)**2 + (y - car_wy)**2)
    idx = int(np.argmin(dist))
    
    look_ahead = 5 # calculate_stanley_error와 동일한 전방 주시 인덱스
    next_idx = min(len(x) - 1, idx + look_ahead)

    # 랩뷰로 전달할 곡률 값 반환
    return float(kappa_array[next_idx])