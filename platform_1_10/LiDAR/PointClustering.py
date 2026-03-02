import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


def dbscancluster(x_array, y_array, eps, min_samples):
    """
    DBSCAN 클러스터링을 수행하고, 각 클러스터의 중심점과 포인트 배열을 반환한다.

    Args:
        x_array: X 좌표 배열 (mm)
        y_array: Y 좌표 배열 (mm)
        eps: DBSCAN epsilon (최대 이웃 거리, mm)
        min_samples: DBSCAN 최소 샘플 수

    Returns:
        tuple: (centroids, cluster_points_list)
            - centroids: np.ndarray, shape (2, N) — [[cx1, cx2, ...], [cy1, cy2, ...]]
            - cluster_points_list: list[np.ndarray] — 각 클러스터의 포인트 배열 (M_i, 2)
    """
    if not isinstance(x_array, np.ndarray):
        x_array = np.array(x_array)
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)

    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.array([[], []]), []

    points = np.column_stack((x_array, y_array))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)

    # 클러스터 중심점 계산 (노이즈 제외)
    unique_labels = sorted(set(labels) - {-1})

    center_x = []
    center_y = []
    cluster_points_list = []

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_pts = points[cluster_mask]

        center = cluster_pts.mean(axis=0)
        center_x.append(center[0])
        center_y.append(center[1])
        cluster_points_list.append(cluster_pts)

    centroids = np.array([center_x, center_y])

    return centroids, cluster_points_list