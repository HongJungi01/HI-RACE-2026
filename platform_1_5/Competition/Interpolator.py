import heapq
import math
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

def _angle_between(v1, v2):
    """두 벡터 사이의 각도(degree)를 반환 (0~180) — pure math, numpy 오버헤드 제거"""
    norm1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    norm2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    if norm1 == 0 or norm2 == 0:
        return 180.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    return math.degrees(math.acos(cos_angle))


def interpolator(x_array, y_array, distance_threshold=1700, point_spacing=50, angle_threshold=45.0, initial_angle_threshold=67.5, seed_indices=None, min_points_per_chain=3, force_distance=None):
    """
    체인 추적 방식 보간기: 가장 가까운 이웃 방향으로 시작해서,
    이전 연결 방향과 유사한 각도의 점을 순차적으로 연결.
    시드 체인은 Round-Robin(교대 성장) 방식으로 동시에 확장되며,
    너무 짧은 체인은 잘못된 연결로 간주하여 결과에서 제외.
    
    Args:
        x_array: X coordinates (mm)
        y_array: Y coordinates (mm)
        distance_threshold: 이 거리(mm) 이내의 포인트들만 연결 후보
        point_spacing: 보간 포인트 간격(mm)
        angle_threshold: 이전 연결 방향 대비 허용 각도 차이(degree, 기본 45°)
        initial_angle_threshold: 체인 첫 연결에 적용할 허용 각도 차이(degree, 기본 67.5°)
        seed_indices: 시드 포인트 인덱스 리스트. 각 시드는 독립 체인을 시작하며,
                     다른 시드 포인트로는 연결되지 않음. None이면 기존 Y-heap 로직 사용.
        min_points_per_chain: 체인으로 인정할 최소 점 개수 (기본 3).
                             크로스 방향성 등으로 짧게 끝난 체인 제거용.
        force_distance: 이 거리(mm) 이내의 후보는 각도 제한 무시하고 연결 허용 (None이면 비활성).
    
    Returns:
        np.ndarray: [[x1, x2, ...], [y1, y2, ...]] 형태의 (2, N) 배열
    """
    if not isinstance(x_array, np.ndarray):
        x_array = np.array(x_array)
    if not isinstance(y_array, np.ndarray):
        y_array = np.array(y_array)

    if x_array.size == 0 or x_array.shape != y_array.shape:
        return np.array([x_array, y_array])

    points = np.column_stack((x_array, y_array))
    n = len(points)

    if n <= 1:  # 1개 이하면 보간 불가 → 바로 반환
        return np.array([x_array, y_array])

    tree = KDTree(points)
    k_neighbors = min(n, 10)

    seed_set = set(seed_indices) if seed_indices is not None else set()
    seed_order = list(seed_indices) if seed_indices is not None else []

    # Y값 기준 정렬된 인덱스 힙 (min-heap)
    y_heap = [(points[i, 1], i) for i in range(n)]
    heapq.heapify(y_heap)

    visited = set()
    chains = []  # 체인별 인덱스 리스트

    def _find_best_candidate(current, direction, blocked, eff_angle_threshold):
        """현재 점에서 각도 차이가 가장 작은 후보를 찾는다.
        force_distance 이내의 후보는 각도 제한을 무시한다.
        Returns: (best_idx, best_angle_diff) or (-1, inf) if no candidate."""
        distances, indices = tree.query(points[current], k=k_neighbors)
        best_idx = -1
        best_angle_diff = float('inf')

        for dist, idx in zip(distances, indices):
            if idx == current or idx in visited or idx in blocked:
                continue
            if dist > distance_threshold:
                continue

            candidate_vec = points[idx] - points[current]
            angle_diff = _angle_between(direction, candidate_vec)

            # force_distance 이내면 각도 무시, 아니면 기존 각도 체크
            if force_distance is not None and dist <= force_distance:
                pass  # 각도 무관 — 무조건 후보
            elif angle_diff > eff_angle_threshold:
                continue

            if angle_diff < best_angle_diff:
                best_angle_diff = angle_diff
                best_idx = idx

        return best_idx, best_angle_diff

    def _build_chain(start_idx, blocked, use_fixed_direction=False):
        """start_idx에서 시작하여 체인 구축 (Phase 2 비시드용). blocked 인덱스는 연결 후보에서 제외."""
        chain = [start_idx]
        visited.add(start_idx)

        direction = np.array([0.0, 1.0])  # 기본 방향: +Y
        if not use_fixed_direction:
            init_dists, init_idxs = tree.query(points[start_idx], k=k_neighbors)
            for d, idx in zip(init_dists, init_idxs):
                if idx != start_idx and idx not in visited and idx not in blocked and d <= distance_threshold:
                    direction = points[idx] - points[start_idx]
                    break
        is_first_step = True

        while True:
            current = chain[-1]
            eff_threshold = initial_angle_threshold if is_first_step else angle_threshold
            best_idx, _ = _find_best_candidate(current, direction, blocked, eff_threshold)

            if best_idx == -1:
                break
            is_first_step = False

            direction = points[best_idx] - points[current]
            chain.append(best_idx)
            visited.add(best_idx)

        return chain

    # Phase 1: 시드 체인 Round-Robin 교대 성장
    active_seeds = []
    for start_idx in seed_order:
        if start_idx in visited:
            continue
        visited.add(start_idx)
        blocked = seed_set - {start_idx}
        active_seeds.append({
            'chain': [start_idx],
            'direction': np.array([0.0, 1.0]),
            'seed_steps_remaining': 2,  # 시드 + 직후 1스텝: +Y 방향 + initial_angle_threshold 유지
            'blocked': blocked,
        })

    while active_seeds:
        finished = []
        for i, state in enumerate(active_seeds):
            chain = state['chain']
            current = chain[-1]
            eff_threshold = initial_angle_threshold if state['seed_steps_remaining'] > 0 else angle_threshold

            best_idx, _ = _find_best_candidate(current, state['direction'], state['blocked'], eff_threshold)

            if best_idx == -1:
                finished.append(i)
                continue

            if state['seed_steps_remaining'] > 0:
                state['seed_steps_remaining'] -= 1
                if state['seed_steps_remaining'] == 0:
                    # 시드 구간 종료: 실제 이동 방향으로 전환
                    state['direction'] = points[best_idx] - points[current]
                # else: direction은 [0, 1] (+Y) 유지
            else:
                state['direction'] = points[best_idx] - points[current]
            chain.append(best_idx)
            visited.add(best_idx)

        for i in reversed(finished):
            # 최소 길이 조건 만족 시에만 최종 chains에 추가
            if len(active_seeds[i]['chain']) >= min_points_per_chain:
                chains.append(active_seeds[i]['chain'])
            del active_seeds[i]

    # Round-Robin 종료 후 아직 active인 체인도 길이 조건 확인 후 추가
    for state in active_seeds:
        if len(state['chain']) >= min_points_per_chain:
            chains.append(state['chain'])

    # Phase 2: 나머지 미방문 점들은 visited만 업데이트 (시드 체인만 유지하기 위함)
    while len(visited) < n:
        while y_heap and y_heap[0][1] in visited:
            heapq.heappop(y_heap)
        if not y_heap:
            break
        _, start_idx = heapq.heappop(y_heap)

        _build_chain(start_idx, set())

    # 체인 내 인접 점 쌍 사이 보간
    interp_segments = []

    for chain in chains:
        for j in range(len(chain) - 1):
            p1 = points[chain[j]]
            p2 = points[chain[j + 1]]
            diff = p2 - p1
            segment_length = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])

            if segment_length < point_spacing:
                continue

            num_points = int(segment_length / point_spacing)
            t = np.linspace(0, 1, num_points + 2)[1:-1]
            interp_points = p1 + np.outer(t, diff)
            interp_segments.append(interp_points)

    if interp_segments:
        all_interp = np.vstack(interp_segments)
        final_x = np.concatenate([x_array, all_interp[:, 0]])
        final_y = np.concatenate([y_array, all_interp[:, 1]])
    else:
        final_x = x_array
        final_y = y_array

    return np.array([final_x, final_y])

def clustered_interpolator(x_array, y_array, eps=500, min_samples=5, 
                          distance_threshold=1700, point_spacing=50, 
                          angle_threshold=45.0, initial_angle_threshold=67.5,
                          min_points_per_chain=3, force_distance=None):
    """
    DBSCAN 클러스터링 후 클러스터 중심점들을 체인 추적 보간하는 통합 함수.
    시드 포인트 (1000,0), (-1000,0) 에서 각각 독립 체인을 시작하며,
    서로 다른 시드 체인끼리는 연결되지 않음.
    Round-Robin 교대 성장 + 최소 점 개수 필터링 적용.
    
    Args:
        x_array: X coordinates (mm)
        y_array: Y coordinates (mm)
        eps: DBSCAN epsilon parameter (mm)
        min_samples: DBSCAN minimum samples parameter
        distance_threshold: 이 거리(mm) 이내의 포인트들을 연결
        point_spacing: 보간 포인트 간격(mm)
        angle_threshold: 이전 연결 방향 대비 허용 각도 차이(degree)
        initial_angle_threshold: 체인 첫 연결에 적용할 허용 각도 차이(degree)
        min_points_per_chain: 체인으로 인정할 최소 점 개수 (기본 3)
        force_distance: 이 거리(mm) 이내의 후보는 각도 제한 무시하고 연결 허용 (None이면 비활성)
    
    Returns:
        np.ndarray: [[x1, x2, ...], [y1, y2, ...]] 형태의 (2, N) 배열
                   원본점 + 클러스터 중심점 + 보간점 모두 포함
    """
    # Step 1: DBSCAN 클러스터링으로 중심점 추출
    cluster_centers = dbscancluster(x_array, y_array, eps, min_samples)
    
    # Step 2: 고정 시드 포인트 (1000,0), (-1000,0)를 맨 앞에 추가
    center_x = np.concatenate([[1000.0, -1000.0], cluster_centers[0]])
    center_y = np.concatenate([[0.0, 0.0], cluster_centers[1]])
    seed_idx_list = [0, 1]
    
    # Step 3: 클러스터 중심점들을 보간 (시드 체인끼리 격리)
    interpolated_result = interpolator(
        center_x,
        center_y,
        distance_threshold=distance_threshold,
        point_spacing=point_spacing,
        angle_threshold=angle_threshold,
        initial_angle_threshold=initial_angle_threshold,
        seed_indices=seed_idx_list,
        min_points_per_chain=min_points_per_chain,
        force_distance=force_distance
    )
    
    # Step 3: 원본점 + 클러스터 중심점 + 보간점 모두 합치기
    all_x = np.concatenate([x_array, interpolated_result[0]])
    all_y = np.concatenate([y_array, interpolated_result[1]])
    
    return np.array([all_x, all_y])