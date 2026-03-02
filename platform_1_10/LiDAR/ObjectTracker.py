"""
ObjectTracker: 프레임 간 클러스터 연관(Data Association) + ICP 정밀 매칭 + 칼만 필터 속도 추정

사용법:
    from PointClustering import dbscancluster
    from ObjectTracker import ObjectTracker

    tracker = ObjectTracker()

    # 매 프레임 호출
    centroids, cluster_points_list = dbscancluster(x, y, eps, min_samples)
    tracked_objects = tracker.update(centroids, cluster_points_list, dt)

    # tracked_objects: list[dict]
    #   - 'id': int (고유 추적 ID)
    #   - 'centroid': np.ndarray (2,) — 현재 위치 [x, y] (mm)
    #   - 'velocity': np.ndarray (2,) — 상대 속도 [vx, vy] (mm/s)
    #   - 'speed': float — 속력 (mm/s)
    #   - 'age': int — 추적 유지 프레임 수
    #   - 'points': np.ndarray (M, 2) — 현재 클러스터 포인트
"""

import numpy as np
from scipy.spatial import KDTree


# ──────────────────────────────────────────────
#  ICP (Iterative Closest Point) — 2D 간소화 버전
# ──────────────────────────────────────────────

def _icp_translation(source: np.ndarray, target: np.ndarray,
                     max_iter: int = 15, tolerance: float = 0.5) -> np.ndarray:
    """
    Translation-only ICP: source를 target에 정합시키는 병진 벡터를 반환한다.
    회전은 고려하지 않는다 (프레임 간 회전이 작다고 가정).

    Args:
        source: (N, 2) — 이전 프레임 클러스터 포인트
        target: (M, 2) — 현재 프레임 클러스터 포인트
        max_iter: 최대 반복 횟수
        tolerance: 수렴 판정 기준 (mm)

    Returns:
        translation: (2,) — source → target 변환 벡터 [dx, dy]
    """
    src = source.copy()
    cumulative_t = np.zeros(2)

    for _ in range(max_iter):
        tree = KDTree(target)
        dists, indices = tree.query(src)

        # 대응점 쌍의 중앙값 변위 (이상치에 강건)
        matched_target = target[indices]
        diff = matched_target - src
        t = np.median(diff, axis=0)

        src += t
        cumulative_t += t

        if np.linalg.norm(t) < tolerance:
            break

    return cumulative_t


def _median_displacement(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    ICP 대신 사용 가능한 경량 대안: 최근접 점 쌍의 중앙값 변위.

    Args:
        source: (N, 2) — 이전 프레임 클러스터 포인트
        target: (M, 2) — 현재 프레임 클러스터 포인트

    Returns:
        displacement: (2,) — 중앙값 변위 벡터 [dx, dy]
    """
    tree = KDTree(target)
    _, indices = tree.query(source)
    diff = target[indices] - source
    return np.median(diff, axis=0)


# ──────────────────────────────────────────────
#  2D 칼만 필터 — 등속 모델 (상태: x, y, vx, vy)
# ──────────────────────────────────────────────

class KalmanFilter2D:
    """
    상태 벡터: [x, y, vx, vy]
    관측 벡터: [x, y]
    등속(Constant Velocity) 모델
    """

    def __init__(self, x0: float, y0: float,
                 process_noise: float = 50.0,
                 measurement_noise: float = 30.0):
        """
        Args:
            x0, y0: 초기 위치 (mm)
            process_noise: 프로세스 노이즈 표준편차 (mm/s²)
            measurement_noise: 관측 노이즈 표준편차 (mm)
        """
        # 상태: [x, y, vx, vy]
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)

        # 공분산 행렬 — 초기 불확실성
        self.P = np.diag([measurement_noise ** 2, measurement_noise ** 2,
                          500.0 ** 2, 500.0 ** 2])

        # 관측 행렬: 위치만 관측
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # 관측 노이즈 공분산
        self.R = np.diag([measurement_noise ** 2, measurement_noise ** 2])

        self._process_noise_std = process_noise

    def predict(self, dt: float):
        """시간 dt(초)만큼 예측 단계 수행."""
        # 상태 전이 행렬
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # 프로세스 노이즈 — 이산 백색 노이즈 가속도 모델
        q = self._process_noise_std
        Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ],
        ], dtype=np.float64) * q ** 2

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray):
        """
        관측값으로 보정 단계 수행.

        Args:
            z: (2,) — 관측 위치 [x, y]
        """
        y = z - self.H @ self.x                      # 잔차
        S = self.H @ self.P @ self.H.T + self.R       # 잔차 공분산
        K = self.P @ self.H.T @ np.linalg.inv(S)      # 칼만 이득
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        """현재 추정 위치 (2,)."""
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """현재 추정 속도 (2,) — mm/s."""
        return self.x[2:4].copy()


# ──────────────────────────────────────────────
#  추적 대상 (Tracked Object)
# ──────────────────────────────────────────────

class TrackedObject:
    """단일 추적 물체."""

    _next_id = 0

    def __init__(self, centroid: np.ndarray, points: np.ndarray,
                 process_noise: float, measurement_noise: float):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1

        self.kf = KalmanFilter2D(centroid[0], centroid[1],
                                 process_noise=process_noise,
                                 measurement_noise=measurement_noise)
        self.points = points
        self.age = 1
        self.misses = 0          # 연속 미매칭 프레임 수
        self._raw_centroid = centroid.copy()

    def predict(self, dt: float):
        self.kf.predict(dt)

    def correct(self, centroid: np.ndarray, points: np.ndarray,
                prev_points: np.ndarray, use_icp: bool):
        """
        관측값으로 보정. ICP를 사용하면 centroid 대신 정밀 변위를 관측에 반영.
        """
        if use_icp and prev_points.shape[0] >= 3 and points.shape[0] >= 3:
            # ICP로 정밀 변위 계산
            displacement = _icp_translation(prev_points, points)
            corrected_pos = self._raw_centroid + displacement
        else:
            corrected_pos = centroid

        self.kf.update(corrected_pos)
        self._raw_centroid = self.kf.position
        self.points = points
        self.age += 1
        self.misses = 0

    def mark_missed(self):
        self.misses += 1
        self.age += 1

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'centroid': self.kf.position,
            'velocity': self.kf.velocity,
            'speed': float(np.linalg.norm(self.kf.velocity)),
            'age': self.age,
            'points': self.points,
        }


# ──────────────────────────────────────────────
#  ObjectTracker — 메인 추적기
# ──────────────────────────────────────────────

class ObjectTracker:
    """
    프레임 간 클러스터 연관 + ICP 보정 + 칼만 필터 속도 추정.

    Args:
        max_miss: 연속 미매칭 이 횟수 초과 시 트래커 삭제 (기본 3)
        gate_distance: 연관 게이트 거리 (mm). 이 거리 초과면 매칭 불가 (기본 2000)
        use_icp: ICP 정밀 매칭 사용 여부 (기본 True)
        process_noise: 칼만 필터 프로세스 노이즈 (mm/s²)
        measurement_noise: 칼만 필터 관측 노이즈 (mm)
    """

    def __init__(self, max_miss: int = 3, gate_distance: float = 2000.0,
                 use_icp: bool = True,
                 process_noise: float = 50.0, measurement_noise: float = 30.0):
        self.max_miss = max_miss
        self.gate_distance = gate_distance
        self.use_icp = use_icp
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self._tracks: list[TrackedObject] = []

    # ── 헝가리안 알고리즘 대체: 물체 수 ≤5 이므로 Greedy 최근접 매칭 ──

    def _associate(self, centroids: np.ndarray) -> tuple[list[tuple[int, int]],
                                                          list[int],
                                                          list[int]]:
        """
        기존 트랙과 새 관측 간 매칭.

        Args:
            centroids: (2, N) — 현재 프레임 클러스터 중심점

        Returns:
            matches: list[(track_idx, detection_idx)]
            unmatched_tracks: list[track_idx]
            unmatched_detections: list[detection_idx]
        """
        n_tracks = len(self._tracks)
        n_dets = centroids.shape[1] if centroids.size > 0 else 0

        if n_tracks == 0 or n_dets == 0:
            return [], list(range(n_tracks)), list(range(n_dets))

        # 비용 행렬: 트랙 예측 위치 ↔ 관측 centroid 간 유클리드 거리
        pred_positions = np.array([t.kf.position for t in self._tracks])   # (T, 2)
        det_positions = centroids.T                                         # (D, 2)

        cost = np.linalg.norm(
            pred_positions[:, np.newaxis, :] - det_positions[np.newaxis, :, :],
            axis=2,
        )  # (T, D)

        matches = []
        used_tracks = set()
        used_dets = set()

        # Greedy: 비용이 작은 순서대로 매칭
        flat_indices = np.argsort(cost, axis=None)
        for flat_idx in flat_indices:
            t_idx = int(flat_idx // n_dets)
            d_idx = int(flat_idx % n_dets)

            if t_idx in used_tracks or d_idx in used_dets:
                continue
            if cost[t_idx, d_idx] > self.gate_distance:
                break  # 이후는 전부 gate 초과

            matches.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

            if len(used_tracks) == n_tracks or len(used_dets) == n_dets:
                break

        unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in used_dets]

        return matches, unmatched_tracks, unmatched_dets

    # ── 메인 업데이트 ──

    def update(self, centroids: np.ndarray, cluster_points_list: list[np.ndarray],
               dt: float) -> list[dict]:
        """
        한 프레임의 클러스터링 결과를 입력받아 추적 및 속도 추정.

        Args:
            centroids: (2, N) — dbscancluster()에서 반환된 클러스터 중심점 배열
            cluster_points_list: list[np.ndarray] — 각 클러스터의 포인트 배열 (M_i, 2)
            dt: 이전 프레임으로부터의 경과 시간 (초)

        Returns:
            list[dict]: 추적 결과 리스트 (to_dict() 참고)
        """
        # 1) 기존 트랙 예측
        for track in self._tracks:
            track.predict(dt)

        # 2) 데이터 연관
        matches, unmatched_tracks, unmatched_dets = self._associate(centroids)

        # 3) 매칭된 트랙 보정
        n_dets = centroids.shape[1] if centroids.size > 0 else 0
        for t_idx, d_idx in matches:
            track = self._tracks[t_idx]
            det_centroid = centroids[:, d_idx]
            det_points = cluster_points_list[d_idx]
            prev_points = track.points
            track.correct(det_centroid, det_points, prev_points, self.use_icp)

        # 4) 미매칭 트랙 처리
        for t_idx in unmatched_tracks:
            self._tracks[t_idx].mark_missed()

        # 5) 새 관측으로 트랙 생성
        for d_idx in unmatched_dets:
            det_centroid = centroids[:, d_idx]
            det_points = cluster_points_list[d_idx]
            new_track = TrackedObject(
                centroid=det_centroid,
                points=det_points,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            self._tracks.append(new_track)

        # 6) 수명 초과 트랙 삭제
        self._tracks = [t for t in self._tracks if t.misses <= self.max_miss]

        # 7) 결과 반환
        return [t.to_dict() for t in self._tracks]

    def reset(self):
        """모든 트랙 초기화."""
        self._tracks.clear()
        TrackedObject._next_id = 0

    @property
    def active_tracks(self) -> int:
        """현재 활성 트랙 수."""
        return len(self._tracks)
