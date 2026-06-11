import numpy as np
import cv2

# ============================================================
# 캘리브레이션 (640x300 해상도 기준)
# ============================================================
RESAMPLE_COUNT = 50

_CALIB_PIXEL_PTS = np.array([
    [89, 191],
    [522, 212],
    [458, 92],
    [171, 84],
], dtype=np.float32)

_CALIB_WORLD_PTS = np.array([
    [0.0,   0.0],
    [0.358, 0.0],
    [0.358, 0.252],
    [0.0,   0.252],
], dtype=np.float32)

WW_GAP = 0.45   # 흰선-흰선 간격 (m)
YW_GAP = 0.225   # 노란선-흰선 간격 (m)

# ============================================================
# 호모그래피 (모듈 로드 시 1회 계산)
# ============================================================

def _build_homography():
    H_raw, _ = cv2.findHomography(_CALIB_PIXEL_PTS, _CALIB_WORLD_PTS, cv2.RANSAC)
    # 차량 앞축중심 (320, 300) → 월드 원점 (0, 0)
    origin = np.array([[320, 300]], dtype=np.float32).reshape(-1, 1, 2)
    o_world = cv2.perspectiveTransform(origin, H_raw).reshape(2)
    T = np.array([
        [1, 0, -o_world[0]],
        [0, 1, -o_world[1]],
        [0, 0,            1],
    ], dtype=np.float64)
    return T @ H_raw

_H = _build_homography()


def pixel_to_world(pixel_points):
    pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, _H).reshape(-1, 2)
    return world[:, 0], world[:, 1]


# ============================================================
# HSV 색상 필터 (Phase 0 튜닝 결과)
# ============================================================
_Y_LOWER = np.array([10, 35, 107])
_Y_UPPER = np.array([31, 255, 255])
_W_LOWER = np.array([0,  0,  140])
_W_UPPER = np.array([180, 71, 255])

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
ROI_TOP = 90  # 상단 잘라내기 (px)


def extract_yellow_mask(hsv):
    return cv2.inRange(hsv, _Y_LOWER, _Y_UPPER)


def extract_white_mask(hsv):
    return cv2.inRange(hsv, _W_LOWER, _W_UPPER)


def preprocess_mask(mask, yellow_mask=None):
    """morphological open + ROI. yellow_mask가 주어지면 겹치는 영역 제거."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask[:ROI_TOP, :] = 0
    if yellow_mask is not None:
        mask[yellow_mask > 0] = 0
    return mask


# ============================================================
# 차선 추출 (Connected Component → Polyfit)
# ============================================================

def extract_lane_curves(binary_mask,
                        min_area=100,
                        min_span=50,
                        max_rmse=15.0,
                        fit_degree=3):
    """
    바이너리 마스크에서 차선 후보를 추출한다.
    반환: List[(poly_func, y_min, y_max, centroid_x)]
          centroid_x 기준 오름차순(왼→오) 정렬, 최대 3개.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels <= 1:
        return []

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == i)

        # span 필터 — 시작↔끝 직선 길이
        span = np.sqrt((xs[0] - xs[-1])**2 + (ys[0] - ys[-1])**2)
        if span < min_span:
            continue

        try:
            coeffs = np.polyfit(ys, xs, fit_degree)
        except Exception:
            continue
        poly_func = np.poly1d(coeffs)

        rmse = np.sqrt(np.mean((poly_func(ys) - xs) ** 2))
        if rmse > max_rmse:
            continue

        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        cx = centroids[i][0]
        candidates.append((poly_func, y_min, y_max, cx))

    candidates.sort(key=lambda c: c[3])
    return candidates[:3]


# ============================================================
# 월드 좌표 변환
# ============================================================

def generate_world_points(poly_func, y_min, y_max, count=RESAMPLE_COUNT):
    """
    다항식 곡선을 count개 포인트로 샘플링 후 월드 좌표로 변환.
    반환: np.ndarray shape (2, count) — [world_x, world_y]
    """
    y_pixel = np.linspace(y_max, y_min, count)
    x_pixel = poly_func(y_pixel)
    pixel_pts = np.stack([x_pixel, y_pixel], axis=1)
    wx, wy = pixel_to_world(pixel_pts)
    return np.array([wx, wy])


# ============================================================
# 차선 추론 (간격 기반)
# ============================================================

def _avg_world_x(lane):
    return float(np.mean(lane[0]))


GAP_THRESHOLD = (WW_GAP + YW_GAP) / 2  # 0.3375m — YW와 WW의 중간값


def _validate_white_lanes(white_lanes, yellow_lanes):
    """
    흰색 3개 이상 검출 시 간격 기반으로 오인식(노란→흰) 제거.
    트랙 구조상 흰색은 최대 2개. 인접 간격이 GAP_THRESHOLD 미만이면
    해당 쌍 중 노란색에 가까운 쪽을 제거.
    """
    if len(white_lanes) < 3:
        return white_lanes

    lanes = sorted(white_lanes, key=_avg_world_x)

    yellow_xs = [_avg_world_x(y) for y in yellow_lanes] if yellow_lanes else []

    to_remove = set()
    for i in range(len(lanes) - 1):
        gap = _avg_world_x(lanes[i + 1]) - _avg_world_x(lanes[i])
        if gap < GAP_THRESHOLD:
            # 쌍 중 하나가 오인식 — 노란색에 더 가까운 쪽 제거
            if yellow_xs:
                x_a = _avg_world_x(lanes[i])
                x_b = _avg_world_x(lanes[i + 1])
                dist_a = min(abs(x_a - yx) for yx in yellow_xs)
                dist_b = min(abs(x_b - yx) for yx in yellow_xs)
                to_remove.add(i if dist_a < dist_b else i + 1)
            else:
                # 노란색 없으면 바깥쪽(가장자리) 제거
                if i == 0:
                    to_remove.add(0)
                else:
                    to_remove.add(i + 1)

    return [lane for idx, lane in enumerate(lanes) if idx not in to_remove]


def _offset_lane(lane, offset_m):
    """
    월드 좌표 곡선을 법선 방향으로 offset_m만큼 이동.
    offset_m > 0: +x(오른쪽), offset_m < 0: -x(왼쪽).
    """
    xs = lane[0].astype(np.float64)
    ys = lane[1].astype(np.float64)
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    mag = np.sqrt(dx**2 + dy**2)
    mag[mag == 0] = 1e-8
    nx = dy / mag
    ny = -dx / mag
    return np.array([xs + nx * offset_m, ys + ny * offset_m])


def infer_missing_lanes(white_lanes, yellow_lanes):
    """
    보이는 흰색 차선이 부족할 때 간격 기반으로 추론.
    - 흰 2~3개: 그대로
    - 흰 1개: WW_GAP offset으로 1개 추론 (노란색으로 방향 결정)
    - 흰 0개, 노란 있음: YW_GAP offset
    반환: 흰색 차선 리스트 (왼→오 정렬, 최대 3개)
    """
    white_lanes = _validate_white_lanes(white_lanes, yellow_lanes)

    if len(white_lanes) >= 2:
        white_lanes.sort(key=_avg_world_x)
        return white_lanes[:3]

    if len(white_lanes) == 1:
        w = white_lanes[0]
        w_x = _avg_world_x(w)

        # 노란색 위치로 방향 판단
        direction = +1.0  # 기본: 오른쪽으로 추론
        if yellow_lanes:
            y_x = np.mean([_avg_world_x(y) for y in yellow_lanes])
            if y_x < w_x:
                direction = +1.0   # 노란이 왼쪽 → 누락은 오른쪽
            else:
                direction = -1.0   # 노란이 오른쪽 → 누락은 왼쪽

        inferred = _offset_lane(w, direction * WW_GAP)
        result = [w, inferred]
        result.sort(key=_avg_world_x)
        return result

    if len(white_lanes) == 0 and yellow_lanes:
        y = yellow_lanes[0]
        w1 = _offset_lane(y, +YW_GAP)
        w2 = _offset_lane(y, +(YW_GAP + WW_GAP))
        result = [w1, w2]
        result.sort(key=_avg_world_x)
        return result

    return []


# ============================================================
# 전체 파이프라인
# ============================================================

def process_frame(bgr_image, resample_count=RESAMPLE_COUNT):
    """
    BGR 이미지 → 흰색 차선 월드 좌표 (6, resample_count).
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    y_mask = extract_yellow_mask(hsv)
    w_mask = extract_white_mask(hsv)

    y_mask = preprocess_mask(y_mask)
    w_mask = preprocess_mask(w_mask, yellow_mask=y_mask)  # 노란 영역 제외

    white_curves = extract_lane_curves(w_mask)
    yellow_curves = extract_lane_curves(y_mask)

    white_world = [generate_world_points(pf, ymin, ymax, resample_count)
                   for pf, ymin, ymax, _ in white_curves]
    yellow_world = [generate_world_points(pf, ymin, ymax, resample_count)
                    for pf, ymin, ymax, _ in yellow_curves]

    all_white = infer_missing_lanes(white_world, yellow_world)

    # (6, resample_count) 포맷: 최대 3개, 부족하면 0 패딩
    final = []
    for lane in all_white[:3]:
        final.append(lane)
    while len(final) < 3:
        final.append(np.zeros((2, resample_count), dtype=np.float64))

    return np.vstack(final).astype(np.float64)
