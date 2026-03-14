"""
PointCluster: LabVIEW Python 노드용 통합 진입점.

LabVIEW 호출 시그니처:
    track_objects(x_array, y_array, eps, min_samples, iterations)

반환: np.ndarray (5, N)
    [[id1,  id2,  ...],    ← 추적 ID
     [x1,   x2,   ...],    ← 추정 위치 X (mm)
     [y1,   y2,   ...],    ← 추정 위치 Y (mm)
     [vx1,  vx2,  ...],    ← 추정 속도 X (mm/s)
     [vy1,  vy2,  ...]]    ← 추정 속도 Y (mm/s)

iterations 인자는 LabVIEW 루프 카운터(0, 1, 2, ...)를 받습니다.
    - 0이 들어오면 트래커를 리셋합니다 (새 세션 시작).
    - ICP 반복 횟수는 내부 상수 _ICP_MAX_ITER로 고정됩니다.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from ObjectTracker import ObjectTracker

# ── 모듈 레벨 싱글턴: LabVIEW Python 세션 동안 상태 유지 ──
_tracker = ObjectTracker(max_miss=3, gate_distance=2000.0, use_icp=True)

# ── 상수 ──
_DT = 0.1              # 프레임 간 시간 간격 (초, 고정)
_ICP_MAX_ITER = 15     # ICP 최대 반복 횟수 (고정)


def _dbscan_with_points(x_array, y_array, eps, min_samples):
    """
    DBSCAN 클러스터링 → centroids (2, N) + 클러스터별 포인트 리스트 반환.
    """
    if not isinstance(x_array, np.ndarray):
        x_array = np.array(x_array)
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)

    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.array([[], []]), []

    points = np.column_stack((x_array, y_array))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    unique_labels = sorted(set(labels) - {-1})

    center_x = []
    center_y = []
    cluster_points_list = []

    for label in unique_labels:
        mask = labels == label
        cluster_pts = points[mask]
        center = cluster_pts.mean(axis=0)
        center_x.append(center[0])
        center_y.append(center[1])
        cluster_points_list.append(cluster_pts)

    return np.array([center_x, center_y]), cluster_points_list


def track_objects(x_array, y_array, eps, min_samples, iterations):
    """
    LabVIEW Python 노드에서 호출하는 메인 함수.

    Args:
        x_array: LiDAR X 좌표 배열 (mm)
        y_array: LiDAR Y 좌표 배열 (mm)
        eps: DBSCAN epsilon (mm)
        min_samples: DBSCAN 최소 샘플 수
        iterations: LabVIEW 루프 카운터 (0, 1, 2, ...)
                    0이면 트래커 리셋 (새 세션 시작)

    Returns:
        np.ndarray: (5, N) — [[id], [x], [y], [vx], [vy]]
                    추적 물체가 없으면 (5, 0) 빈 배열
    """
    # ── 루프 카운터 == 0 → 새 세션, 트래커 리셋 ──
    iterations = int(iterations)
    if iterations == 0:
        _tracker.reset()

    # ── ICP 설정은 내부 상수 사용 ──
    _tracker.use_icp = True
    _tracker._icp_max_iter = _ICP_MAX_ITER

    # ── 클러스터링 ──
    centroids, cluster_points_list = _dbscan_with_points(
        x_array, y_array, eps, min_samples
    )

    # ── 추적 + 속도 추정 (dt 고정 0.1s) ──
    results = _tracker.update(centroids, cluster_points_list, _DT)

    # ── (5, N) 배열로 패킹 ──
    if not results:
        return np.zeros((5, 0), dtype=np.float64)

    ids = [r['id'] for r in results]
    xs  = [r['centroid'][0] for r in results]
    ys  = [r['centroid'][1] for r in results]
    vxs = [r['velocity'][0] for r in results]
    vys = [r['velocity'][1] for r in results]

    return np.array([ids, xs, ys, vxs, vys], dtype=np.float64)


def reset_tracker():
    """추적 상태 초기화 (필요시 LabVIEW에서 호출)."""
    _tracker.reset()