import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

def dbscancluster(x_array, y_array, eps, min_samples):
    """
    Args:
        x_array: X coordinates
        y_array: Y coordinates
        eps: DBSCAN epsilon (maximum distance)
        min_samples: DBSCAN minimum samples
    
    Returns:
        np.ndarray: [[center_x1, center_x2, ...], [center_y1, center_y2, ...]] 형태의 (2, N) 배열
    """
    if not isinstance(x_array, np.ndarray):
        x_array = np.array(x_array)
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)

    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.array([[], []])

    points = np.column_stack((x_array, y_array))
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    
    # 클러스터 중심점 계산 (노이즈 제외)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # 노이즈 레이블 제거
    
    center_x = []
    center_y = []
    
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_points = points[cluster_mask]
        
        # 클러스터의 중심점 (평균)
        center = cluster_points.mean(axis=0)
        center_x.append(center[0])
        center_y.append(center[1])
    
    # (2, N) 형태로 반환
    result = np.array([center_x, center_y])
    
    return result