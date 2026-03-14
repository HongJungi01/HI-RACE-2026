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

def interpolator(x_array, y_array, distance_threshold=1700, point_spacing=50, distance_ratio_threshold=4.0):
    """
    Args:
        x_array: X coordinates (mm)
        y_array: Y coordinates (mm)
        distance_threshold: 이 거리(mm) 이내의 포인트들을 연결
        point_spacing: 보간 포인트 간격(mm)
        distance_ratio_threshold: 이전 점보다 이 배수 이상 멀면 보간하지 않음 (기본값: 4.0)
    
    Returns:
        np.ndarray: [[x1, x2, ...], [y1, y2, ...]] 형태의 (2, N) 배열
    """
    if not isinstance(x_array, np.ndarray):
        x_array = np.array(x_array)
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)

    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.array([x_array, y_array])

    # 원본 포인트
    points = np.column_stack((x_array, y_array))
    
    # KDTree 사용
    tree = KDTree(points)
    
    # 각 포인트에서 가장 가까운 3개 점만 찾기 (자기 자신 포함 4개 중 자신 제외)
    all_interp_x = []
    all_interp_y = []
    
    processed_pairs = set()  # 중복 방지
    
    for i, point in enumerate(points):
        # 자기 자신 포함 4개 찾기 (k=4)
        distances, indices = tree.query(point, k=4)
        
        # 자기 자신 제외하고 가장 가까운 3개
        neighbor_distances = distances[1:4]
        neighbor_indices = indices[1:4]
        
        prev_dist = 0
        for dist, idx in zip(neighbor_distances, neighbor_indices):
            # 절대 거리 체크: distance_threshold보다 먼 점은 보간하지 않음
            if dist > distance_threshold:
                continue
            
            # 상대 거리 체크: 이전 점보다 distance_ratio_threshold배 이상 멀면 보간하지 않음
            if prev_dist > 0 and dist > prev_dist * distance_ratio_threshold:
                break  # 더 이상 처리하지 않음 (이후 점들은 더 멀기 때문)
            
            prev_dist = dist
            
            # 중복 방지 (양방향 체크)
            pair = tuple(sorted([i, idx]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)
            
            # 두 점 사이 보간
            p1 = points[i]
            p2 = points[idx]
            
            # 거리 계산
            segment_length = np.linalg.norm(p2 - p1)
            
            if segment_length < point_spacing:
                continue
            
            # 보간할 점의 개수
            num_points = int(segment_length / point_spacing)
            
            # 선형 보간
            t = np.linspace(0, 1, num_points + 2)[1:-1]  # 양 끝점 제외
            interp_points = p1 + np.outer(t, p2 - p1)
            
            all_interp_x.extend(interp_points[:, 0])
            all_interp_y.extend(interp_points[:, 1])
    
    # 원본 포인트와 보간 포인트 합치기
    if all_interp_x:
        final_x = np.concatenate([x_array, all_interp_x])
        final_y = np.concatenate([y_array, all_interp_y])
    else:
        final_x = x_array
        final_y = y_array
    
    return np.array([final_x, final_y])

def clustered_interpolator(x_array, y_array, eps=500, min_samples=5, 
                          distance_threshold=1700, point_spacing=50, 
                          distance_ratio_threshold=4.0):
    """
    DBSCAN 클러스터링 후 클러스터 중심점들을 보간하는 통합 함수
    
    Args:
        x_array: X coordinates (mm)
        y_array: Y coordinates (mm)
        eps: DBSCAN epsilon parameter (mm)
        min_samples: DBSCAN minimum samples parameter
        distance_threshold: 이 거리(mm) 이내의 포인트들을 연결
        point_spacing: 보간 포인트 간격(mm)
        distance_ratio_threshold: 이전 점보다 이 배수 이상 멀면 보간하지 않음
    
    Returns:
        np.ndarray: [[x1, x2, ...], [y1, y2, ...]] 형태의 (2, N) 배열
                   원본점 + 클러스터 중심점 + 보간점 모두 포함
    """
    # Step 1: DBSCAN 클러스터링으로 중심점 추출
    cluster_centers = dbscancluster(x_array, y_array, eps, min_samples)
    
    # Step 2: 클러스터 중심점들을 보간
    interpolated_result = interpolator(
        cluster_centers[0],  # x coordinates of cluster centers
        cluster_centers[1],  # y coordinates of cluster centers
        distance_threshold=distance_threshold,
        point_spacing=point_spacing,
        distance_ratio_threshold=distance_ratio_threshold
    )
    
    # Step 3: 원본점 + 클러스터 중심점 + 보간점 모두 합치기
    all_x = np.concatenate([x_array, interpolated_result[0]])
    all_y = np.concatenate([y_array, interpolated_result[1]])
    
    return np.array([all_x, all_y])