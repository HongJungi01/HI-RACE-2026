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
# 차선 마스크 (밝기 임계값 + Canny 에지)
# ============================================================
ROI_TOP = 90          # 상단 잘라내기 (px)
BRIGHT_THRESH = 130   # 밝기 임계값 (0–255)
CANNY_LOW = 50        # Canny 하한 임계값
CANNY_HIGH = 150      # Canny 상한 임계값
CANNY_DILATE = 2      # Canny 에지 확장 횟수

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def extract_lane_mask(bgr_image):
    """
    BGR 이미지에서 색 구분 없이 모든 차선(흰+노란)을 추출.
    1) 그레이스케일 밝기 임계값 → 밝은 영역
    2) Canny 에지 → dilate → 에지 보조
    3) OR 결합 → morphological open → ROI 적용
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 밝기 임계값: 밝은 차선 vs 어두운 바닥
    _, bright_mask = cv2.threshold(blur, BRIGHT_THRESH, 255, cv2.THRESH_BINARY)

    # Canny 에지: 희미한 차선 보조
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    edges = cv2.dilate(edges, _DILATE_KERNEL, iterations=CANNY_DILATE)

    # 결합
    combined = cv2.bitwise_or(bright_mask, edges)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, _MORPH_KERNEL)
    combined[:ROI_TOP, :] = 0
    return combined


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
    반환: List[(poly_func, y_min, y_max)]
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels <= 1:
        return []

    candidates = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue

        ys, xs = np.where(labels == i)

        if np.sqrt((xs[0] - xs[-1])**2 + (ys[0] - ys[-1])**2) < min_span:
            continue

        try:
            coeffs = np.polyfit(ys, xs, fit_degree)
        except Exception:
            continue
        poly_func = np.poly1d(coeffs)

        if np.sqrt(np.mean((poly_func(ys) - xs) ** 2)) > max_rmse:
            continue

        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        candidates.append((poly_func, y_min, y_max))

    return candidates


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
# 전체 파이프라인
# ============================================================

def process_frame(bgr_image, resample_count=RESAMPLE_COUNT):
    """
    BGR 이미지 → 원점(0,0)에 가장 가까운 차선 1개 → (2, resample_count)
    """
    lane_mask = extract_lane_mask(bgr_image)
    curves = extract_lane_curves(lane_mask)

    if not curves:
        return np.zeros((2, resample_count), dtype=np.float64)

    world_lanes = [generate_world_points(pf, ymin, ymax, resample_count)
                   for pf, ymin, ymax in curves]

    best = min(world_lanes, key=lambda l: np.min(l[0]**2 + l[1]**2))
    return best.astype(np.float64)
