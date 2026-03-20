"""
laneInference.py — 비가시 차선 추론 및 주행 경로(흰색 차선) 생성 모듈

대회 규정:
  - 바닥: 검정, 차선: 흰색(주행 경로), 경계선: 노란색
  - 가능한 코스 형태:
      2차로: Y─0.225m─W─0.45m─W─0.225m─Y   (총 선 4개, 주행 경로 2개)
      3차로: Y─0.225m─W─0.45m─W─0.45m─W─0.225m─Y  (총 선 5개, 주행 경로 3개)
  - 카메라 최대 3개 선 감지 → 항상 일부 누락 가능

주행 경로 = 흰색 차선 위 (차선 사이가 아님)
노란색 차선은 코스 구조 판별 단서로만 사용, 출력에 포함하지 않음
"""

import numpy as np
from typing import List, Optional, Tuple

# ============================================================
# 상수
# ============================================================
YW_GAP = 0.225       # 노란선 ↔ 흰선 간격 (m)
WW_GAP = 0.45        # 흰선 ↔ 흰선 간격 (m)
GAP_TOL = 0.10       # 간격 허용 오차 (m) — 커브 등 고려

DEBOUNCE_COUNT = 3   # 코스 타입 변경에 필요한 연속 프레임 수
PERSIST_FRAMES = 5   # 차선 미감지 시 이전 결과 유지 프레임 수
EMA_ALPHA = 0.3      # EMA 스무딩 계수 (0=이전값 유지, 1=새값만)

# 코스 타입
TRACK_2LANE = "YWWY"    # 2차로 — 주행 경로 2개
TRACK_3LANE = "YWWWY"   # 3차로 — 주행 경로 3개


# ============================================================
# 유틸리티 함수
# ============================================================

def _avg_world_x(lane: np.ndarray) -> float:
    """차선 좌표 배열 (2, N)에서 world_x의 평균값 반환."""
    return float(np.mean(lane[0]))


def _offset_lane(xs: np.ndarray, ys: np.ndarray, offset_m: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    월드 좌표 곡선에서 수직 법선 방향으로 offset_m만큼 이동한 평행선을 생성.
    
    offset_m > 0: 오른쪽(+x) 방향
    offset_m < 0: 왼쪽(-x) 방향
    
    곡선의 접선 벡터에 수직인 법선 벡터를 사용하므로 커브에서도 곡률을 자연스럽게 반영.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    dx = np.gradient(xs)
    dy = np.gradient(ys)

    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude[magnitude == 0] = 1e-8

    # 접선에 수직인 법선 벡터 (오른쪽 방향이 +)
    nx = -dy / magnitude
    ny = dx / magnitude

    new_xs = xs + nx * offset_m
    new_ys = ys + ny * offset_m

    return new_xs, new_ys


def _make_lane_array(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """(world_x, world_y) 배열을 표준 (2, N) ndarray로 조립."""
    return np.array([xs, ys])


def _gap_matches(measured: float, expected: float, tol: float = GAP_TOL) -> bool:
    """측정된 간격이 기대값 ± 허용오차 범위 내인지 확인."""
    return abs(measured - expected) <= tol


def _gap_is_multiple_ww(measured: float, tol: float = GAP_TOL) -> int:
    """
    측정된 간격이 WW_GAP의 정수배인지 확인.
    반환값: 정수배 횟수 (1, 2, 3...) 또는 0 (일치하지 않음)
    """
    if measured < WW_GAP - tol:
        return 0
    ratio = measured / WW_GAP
    nearest_int = round(ratio)
    if nearest_int >= 1 and abs(measured - nearest_int * WW_GAP) <= tol:
        return nearest_int
    return 0


# ============================================================
# 관측 데이터 구조
# ============================================================

class ObservedLine:
    """감지된 단일 차선 정보."""
    __slots__ = ('avg_x', 'color', 'coords')

    def __init__(self, avg_x: float, color: str, coords: np.ndarray):
        self.avg_x = avg_x       # 평균 world_x
        self.color = color       # 'Y' 또는 'W'
        self.coords = coords     # shape (2, N)

    def __repr__(self):
        return f"ObservedLine({self.color}, avg_x={self.avg_x:.3f})"


# ============================================================
# LaneInferenceEngine
# ============================================================

class LaneInferenceEngine:
    """
    프레임 간 상태를 유지하며 비가시 차선을 추론하고,
    모든 주행 가능 경로(흰색 차선)의 월드 좌표를 반환하는 엔진.
    """

    def __init__(self):
        # Temporal state
        self.confirmed_track_type: Optional[str] = None   # TRACK_2LANE / TRACK_3LANE
        self._candidate_type: Optional[str] = None        # 디바운싱 대기 중인 타입
        self._debounce_counter: int = 0                    # 연속 동일 관측 횟수

        # Detection persistence
        self._last_result: Optional[List[np.ndarray]] = None
        self._no_detection_count: int = 0

        # EMA 스무딩 상태
        self._ema_lanes: Optional[List[np.ndarray]] = None

    # ────────────────────────────────────────────
    # 메인 인터페이스
    # ────────────────────────────────────────────
    def infer(self,
              yellow_lanes: List[np.ndarray],
              white_lanes: List[np.ndarray]
              ) -> List[np.ndarray]:
        """
        감지된 노란/흰색 차선으로부터 모든 주행 가능 경로(흰색 차선)를 추론.

        Args:
            yellow_lanes: 노란색 차선 리스트, 각 원소 shape (2, N), 좌→우 정렬
            white_lanes:  흰색 차선 리스트, 각 원소 shape (2, N), 좌→우 정렬

        Returns:
            list[np.ndarray]: 모든 주행 가능 경로(흰색 차선), 각 (2, N), 좌→우 정렬.
                              2차로면 2개, 3차로면 3개.
        """
        # Step 1: 통합 & 정렬
        observed = self._build_observed_lines(yellow_lanes, white_lanes)

        # 감지 없음 → persistence
        if len(observed) == 0:
            return self._handle_no_detection()

        self._no_detection_count = 0

        # Step 2: 패턴 분류 & 코스 타입 판별
        pattern = "".join(line.color for line in observed)
        gaps = self._measure_gaps(observed)
        track_type = self._determine_track_type(pattern, gaps, observed)

        # Step 3: 누락 차선 생성
        all_white_lanes = self._generate_all_white_lanes(
            observed, track_type, pattern, gaps
        )

        # Step 4: 정렬 & 스무딩
        all_white_lanes.sort(key=lambda lane: _avg_world_x(lane))
        all_white_lanes = self._apply_ema(all_white_lanes)

        self._last_result = all_white_lanes
        return all_white_lanes

    # ────────────────────────────────────────────
    # Step 1: 관측 데이터 구성
    # ────────────────────────────────────────────
    def _build_observed_lines(self,
                              yellow_lanes: List[np.ndarray],
                              white_lanes: List[np.ndarray]
                              ) -> List[ObservedLine]:
        """모든 감지 차선을 통합하여 avg_world_x 기준 좌→우 정렬."""
        lines = []
        for lane in yellow_lanes:
            if lane is not None and lane.size > 0:
                lines.append(ObservedLine(_avg_world_x(lane), 'Y', lane))
        for lane in white_lanes:
            if lane is not None and lane.size > 0:
                lines.append(ObservedLine(_avg_world_x(lane), 'W', lane))
        lines.sort(key=lambda l: l.avg_x)
        return lines

    # ────────────────────────────────────────────
    # Step 2: 간격 측정 & 코스 타입 판별
    # ────────────────────────────────────────────
    def _measure_gaps(self, observed: List[ObservedLine]) -> List[float]:
        """인접 선들 사이의 avg_world_x 간격 리스트."""
        gaps = []
        for i in range(len(observed) - 1):
            gaps.append(observed[i + 1].avg_x - observed[i].avg_x)
        return gaps

    def _determine_track_type(self,
                              pattern: str,
                              gaps: List[float],
                              observed: List[ObservedLine]
                              ) -> str:
        """
        관측 패턴과 간격으로 코스 타입을 판별.
        확정 가능하면 바로 확정, 모호하면 temporal state 사용.
        """
        detected_type = self._classify_pattern(pattern, gaps)

        if detected_type is not None:
            # 확정 패턴 → 디바운싱
            self._update_debounce(detected_type)
        else:
            # 모호 → 기존 상태 유지, 없으면 보수적 2차로
            pass

        if self.confirmed_track_type is not None:
            return self.confirmed_track_type
        else:
            # 아직 한번도 확정 안된 경우 → 보수적 2차로
            return TRACK_2LANE

    def _classify_pattern(self, pattern: str, gaps: List[float]) -> Optional[str]:
        """
        패턴과 간격 분석으로 코스 타입을 확정 가능한 경우 반환.
        확정 불가하면 None 반환.
        
        확정 가능한 패턴:
          WWW  → 3차로 (양쪽 Y 누락, W만 3개 보임)
          YWWY → 2차로 전부 보임 (드물지만 가능)
          YWY  → 2차로 (W 하나 미감지 케이스)
          YWW  → 간격 분석으로 판별 가능할 수 있음
          WWY  → 간격 분석으로 판별 가능할 수 있음
        """
        n_y = pattern.count('Y')
        n_w = pattern.count('W')

        # ── 확정 케이스 ──

        # WWW: 흰선 3개, 노란선 없음 → 무조건 3차로
        if pattern == "WWW":
            return TRACK_3LANE

        # YWWY: 4개 전부 보임 → 2차로
        if pattern == "YWWY":
            return TRACK_2LANE

        # YWWWY: 5개 전부 보임 (이론적) → 3차로
        if pattern == "YWWWY":
            return TRACK_3LANE

        # YWY: 양쪽 Y 사이에 W 하나 → 2차로 (나머지 W 하나 미감지)
        if pattern == "YWY":
            return TRACK_2LANE

        # ── 간격 기반 판별 ──

        # YWW: 오른쪽에 뭐가 누락되었는지
        if pattern == "YWW" and len(gaps) == 2:
            # gaps[0] = Y↔W₁, gaps[1] = W₁↔W₂
            # 정상 YWWY: Y↔W=0.225, W↔W=0.45
            # 정상 YWWWY: Y↔W=0.225, W₁↔W₂=0.45 (오른쪽에 W₃,Y 더 있음)
            # Y↔W₁ 간격이 0.225보다 훨씬 크면 → 사이에 누락 W 존재
            if _gap_matches(gaps[0], YW_GAP + WW_GAP):
                # Y↔W₁=0.675 → 사이에 W 하나 누락 → Y□WW = Y(W)WW → YWWWY 확정
                return TRACK_3LANE
            # 일반적인 경우: 간격만으로는 2차로/3차로 구분 불가 → None (모호)
            return None

        # WWY: 왼쪽에 뭐가 누락되었는지 (YWW의 대칭)
        if pattern == "WWY" and len(gaps) == 2:
            if _gap_matches(gaps[1], YW_GAP + WW_GAP):
                return TRACK_3LANE
            return None

        # WW: 양쪽 Y 모두 누락
        if pattern == "WW" and len(gaps) == 1:
            # W₁↔W₂ 간격이 0.45 → 인접한 두 W (2차로 또는 3차로 일부)
            # W₁↔W₂ 간격이 0.90 → 사이에 W 누락 → 3차로에서 중간 W 미감지
            if _gap_matches(gaps[0], 2 * WW_GAP):
                return TRACK_3LANE
            return None

        # YW: 한쪽만 보임
        if pattern == "YW" and len(gaps) == 1:
            # Y↔W=0.225 → 경계 바로 옆 W (2차로든 3차로든 첫 번째 W)
            # Y↔W=0.675 → 사이에 W 하나 누락 → 3차로에서 첫 W 미감지
            if _gap_matches(gaps[0], YW_GAP + WW_GAP):
                return TRACK_3LANE
            return None

        # WY: YW의 대칭
        if pattern == "WY" and len(gaps) == 1:
            if _gap_matches(gaps[0], YW_GAP + WW_GAP):
                return TRACK_3LANE
            return None

        # 단일 선 (W 또는 Y만 하나)
        if len(pattern) <= 1:
            return None

        # 그 외
        return None

    def _update_debounce(self, detected_type: str):
        """디바운싱 로직: 연속 N프레임 동일 타입이면 confirmed 변경."""
        if detected_type == self._candidate_type:
            self._debounce_counter += 1
        else:
            self._candidate_type = detected_type
            self._debounce_counter = 1

        if self._debounce_counter >= DEBOUNCE_COUNT:
            self.confirmed_track_type = detected_type

    # ────────────────────────────────────────────
    # Step 3: 누락 차선 생성
    # ────────────────────────────────────────────
    def _generate_all_white_lanes(self,
                                  observed: List[ObservedLine],
                                  track_type: str,
                                  pattern: str,
                                  gaps: List[float]
                                  ) -> List[np.ndarray]:
        """
        관측된 차선과 코스 타입에 따라 모든 흰색 차선(주행 경로)을 생성.
        보이는 흰색 차선은 그대로, 누락된 것은 normal-vector offset으로 추론.
        """
        # 보이는 W 차선 수집
        visible_whites = [l for l in observed if l.color == 'W']
        visible_yellows = [l for l in observed if l.color == 'Y']

        expected_w_count = 2 if track_type == TRACK_2LANE else 3

        # 이미 충분히 보이면 그대로 반환
        if len(visible_whites) >= expected_w_count:
            return [l.coords.copy() for l in visible_whites[:expected_w_count]]

        # 부족한 W를 추론해야 함
        missing_count = expected_w_count - len(visible_whites)

        # 참조선(offset 기준) 선택: W 우선, 없으면 Y
        result_whites = [l.coords.copy() for l in visible_whites]

        if len(visible_whites) > 0:
            # W가 최소 하나는 있음 → W 기준 offset
            inferred = self._infer_from_whites(
                visible_whites, visible_yellows, track_type,
                pattern, gaps, missing_count
            )
            result_whites.extend(inferred)
        elif len(visible_yellows) > 0:
            # W 없음, Y만 있음 → Y 기준 offset
            inferred = self._infer_from_yellows(
                visible_yellows, track_type
            )
            result_whites.extend(inferred)

        return result_whites

    def _infer_from_whites(self,
                           whites: List[ObservedLine],
                           yellows: List[ObservedLine],
                           track_type: str,
                           pattern: str,
                           gaps: List[float],
                           missing_count: int
                           ) -> List[np.ndarray]:
        """보이는 W를 기준으로 누락된 W를 offset 생성."""
        inferred = []

        if track_type == TRACK_2LANE:
            # 2차로: W 2개 필요, 1개 보임
            if len(whites) == 1:
                w = whites[0]
                # 보이는 Y로 방향 판단
                direction = self._determine_missing_direction_2lane(
                    w, yellows, pattern
                )
                new_xs, new_ys = _offset_lane(
                    w.coords[0], w.coords[1], direction * WW_GAP
                )
                inferred.append(_make_lane_array(new_xs, new_ys))

        elif track_type == TRACK_3LANE:
            # 3차로: W 3개 필요
            if len(whites) == 2:
                # 1개 누락 → 어느 쪽이 누락인지 판단
                inferred.extend(
                    self._infer_one_missing_3lane(whites, yellows, pattern, gaps)
                )
            elif len(whites) == 1:
                # 2개 누락 → 양쪽으로 생성
                inferred.extend(
                    self._infer_two_missing_3lane(whites, yellows, pattern)
                )

        return inferred

    def _determine_missing_direction_2lane(self,
                                           white: ObservedLine,
                                           yellows: List[ObservedLine],
                                           pattern: str
                                           ) -> float:
        """
        2차로에서 W가 1개만 보일 때, 누락된 W가 어느 쪽인지 판단.
        +1.0 = 오른쪽에 누락, -1.0 = 왼쪽에 누락.
        """
        if len(yellows) > 0:
            # Y가 보이면: Y가 W 왼쪽에 있으면 → 누락 W는 오른쪽
            #             Y가 W 오른쪽에 있으면 → 누락 W는 왼쪽
            y_avg = np.mean([y.avg_x for y in yellows])
            if y_avg < white.avg_x:
                return +1.0   # Y가 왼쪽 → 오른쪽에 W 누락
            else:
                return -1.0   # Y가 오른쪽 → 왼쪽에 W 누락

        # 패턴만으로 판단 (Y 없이)
        # 보이는 W의 위치가 코스 중앙보다 왼쪽이면 오른쪽에 누락, 반대도 마찬가지
        # → 보수적으로 오른쪽 가정
        return +1.0

    def _infer_one_missing_3lane(self,
                                 whites: List[ObservedLine],
                                 yellows: List[ObservedLine],
                                 pattern: str,
                                 gaps: List[float]
                                 ) -> List[np.ndarray]:
        """3차로에서 W 2개 보이고 1개 누락 시 추론."""
        inferred = []
        w_left = whites[0]   # avg_x가 작은 쪽
        w_right = whites[1]  # avg_x가 큰 쪽

        # 두 보이는 W 사이 간격으로 판단
        w_gap = w_right.avg_x - w_left.avg_x

        if _gap_matches(w_gap, WW_GAP):
            # 인접한 두 W → 누락된 W는 한쪽 끝에 있음
            # Y 위치로 어느 쪽인지 판단
            has_left_y = any(y.avg_x < w_left.avg_x for y in yellows)
            has_right_y = any(y.avg_x > w_right.avg_x for y in yellows)

            if has_left_y and not has_right_y:
                # 왼쪽에 Y 보임 → 오른쪽 끝에 W 누락
                new_xs, new_ys = _offset_lane(
                    w_right.coords[0], w_right.coords[1], +WW_GAP
                )
                inferred.append(_make_lane_array(new_xs, new_ys))
            elif has_right_y and not has_left_y:
                # 오른쪽에 Y 보임 → 왼쪽 끝에 W 누락
                new_xs, new_ys = _offset_lane(
                    w_left.coords[0], w_left.coords[1], -WW_GAP
                )
                inferred.append(_make_lane_array(new_xs, new_ys))
            else:
                # Y 정보 불충분 → 보이는 W 중 더 넓은 쪽으로 추론
                # 기본: 오른쪽에 누락 가정
                new_xs, new_ys = _offset_lane(
                    w_right.coords[0], w_right.coords[1], +WW_GAP
                )
                inferred.append(_make_lane_array(new_xs, new_ys))

        elif _gap_matches(w_gap, 2 * WW_GAP):
            # 두 W 사이에 0.9m → 중간에 W 하나 누락
            new_xs, new_ys = _offset_lane(
                w_left.coords[0], w_left.coords[1], +WW_GAP
            )
            inferred.append(_make_lane_array(new_xs, new_ys))

        else:
            # 간격이 예상과 다름 → 오른쪽에 추론 (보수적)
            new_xs, new_ys = _offset_lane(
                w_right.coords[0], w_right.coords[1], +WW_GAP
            )
            inferred.append(_make_lane_array(new_xs, new_ys))

        return inferred

    def _infer_two_missing_3lane(self,
                                 whites: List[ObservedLine],
                                 yellows: List[ObservedLine],
                                 pattern: str
                                 ) -> List[np.ndarray]:
        """3차로에서 W 1개만 보이고 2개 누락 시 추론."""
        inferred = []
        w = whites[0]

        # Y 위치로 보이는 W가 3개 중 어디인지 판단
        has_left_y = any(y.avg_x < w.avg_x for y in yellows)
        has_right_y = any(y.avg_x > w.avg_x for y in yellows)

        if has_left_y and not has_right_y:
            # 왼쪽에 Y → 보이는 W는 가장 왼쪽 W₁ → 오른쪽으로 2개 생성
            xs2, ys2 = _offset_lane(w.coords[0], w.coords[1], +WW_GAP)
            xs3, ys3 = _offset_lane(w.coords[0], w.coords[1], +2 * WW_GAP)
            inferred.append(_make_lane_array(xs2, ys2))
            inferred.append(_make_lane_array(xs3, ys3))

        elif has_right_y and not has_left_y:
            # 오른쪽에 Y → 보이는 W는 가장 오른쪽 W₃ → 왼쪽으로 2개 생성
            xs2, ys2 = _offset_lane(w.coords[0], w.coords[1], -WW_GAP)
            xs1, ys1 = _offset_lane(w.coords[0], w.coords[1], -2 * WW_GAP)
            inferred.append(_make_lane_array(xs1, ys1))
            inferred.append(_make_lane_array(xs2, ys2))

        elif has_left_y and has_right_y:
            # 양쪽 Y → 보이는 W는 중앙 W₂ → 양쪽으로 1개씩 생성
            xs1, ys1 = _offset_lane(w.coords[0], w.coords[1], -WW_GAP)
            xs3, ys3 = _offset_lane(w.coords[0], w.coords[1], +WW_GAP)
            inferred.append(_make_lane_array(xs1, ys1))
            inferred.append(_make_lane_array(xs3, ys3))

        else:
            # Y 정보 없음 → 보수적으로 중앙 W₂로 가정
            xs1, ys1 = _offset_lane(w.coords[0], w.coords[1], -WW_GAP)
            xs3, ys3 = _offset_lane(w.coords[0], w.coords[1], +WW_GAP)
            inferred.append(_make_lane_array(xs1, ys1))
            inferred.append(_make_lane_array(xs3, ys3))

        return inferred

    def _infer_from_yellows(self,
                            yellows: List[ObservedLine],
                            track_type: str
                            ) -> List[np.ndarray]:
        """W가 전혀 안 보이고 Y만 보일 때, Y 기준으로 W를 추론."""
        inferred = []

        if track_type == TRACK_2LANE:
            # 2차로: Y에서 안쪽으로 YW_GAP, YW_GAP + WW_GAP 만큼 offset
            if len(yellows) >= 2:
                y_left = yellows[0]
                y_right = yellows[-1]
                # 왼쪽 Y → 오른쪽으로 W₁
                xs1, ys1 = _offset_lane(
                    y_left.coords[0], y_left.coords[1], +YW_GAP
                )
                # 오른쪽 Y → 왼쪽으로 W₂
                xs2, ys2 = _offset_lane(
                    y_right.coords[0], y_right.coords[1], -YW_GAP
                )
                inferred.append(_make_lane_array(xs1, ys1))
                inferred.append(_make_lane_array(xs2, ys2))
            elif len(yellows) == 1:
                y = yellows[0]
                # 하나의 Y에서 안쪽으로 두 개 생성
                # Y가 왼쪽인지 오른쪽인지 모름 → 오른쪽 가정
                xs1, ys1 = _offset_lane(
                    y.coords[0], y.coords[1], +YW_GAP
                )
                xs2, ys2 = _offset_lane(
                    y.coords[0], y.coords[1], +(YW_GAP + WW_GAP)
                )
                inferred.append(_make_lane_array(xs1, ys1))
                inferred.append(_make_lane_array(xs2, ys2))

        elif track_type == TRACK_3LANE:
            if len(yellows) >= 2:
                y_left = yellows[0]
                y_right = yellows[-1]
                # 왼쪽 Y → W₁
                xs1, ys1 = _offset_lane(
                    y_left.coords[0], y_left.coords[1], +YW_GAP
                )
                # 왼쪽 Y → W₂
                xs2, ys2 = _offset_lane(
                    y_left.coords[0], y_left.coords[1], +(YW_GAP + WW_GAP)
                )
                # 오른쪽 Y → W₃
                xs3, ys3 = _offset_lane(
                    y_right.coords[0], y_right.coords[1], -YW_GAP
                )
                inferred.append(_make_lane_array(xs1, ys1))
                inferred.append(_make_lane_array(xs2, ys2))
                inferred.append(_make_lane_array(xs3, ys3))
            elif len(yellows) == 1:
                y = yellows[0]
                xs1, ys1 = _offset_lane(
                    y.coords[0], y.coords[1], +YW_GAP
                )
                xs2, ys2 = _offset_lane(
                    y.coords[0], y.coords[1], +(YW_GAP + WW_GAP)
                )
                xs3, ys3 = _offset_lane(
                    y.coords[0], y.coords[1], +(YW_GAP + 2 * WW_GAP)
                )
                inferred.append(_make_lane_array(xs1, ys1))
                inferred.append(_make_lane_array(xs2, ys2))
                inferred.append(_make_lane_array(xs3, ys3))

        return inferred

    # ────────────────────────────────────────────
    # Temporal 안전장치
    # ────────────────────────────────────────────
    def _handle_no_detection(self) -> List[np.ndarray]:
        """차선 미감지 시 이전 결과를 일정 프레임 동안 유지."""
        self._no_detection_count += 1
        if (self._last_result is not None
                and self._no_detection_count <= PERSIST_FRAMES):
            return self._last_result
        return []

    def _apply_ema(self, lanes: List[np.ndarray]) -> List[np.ndarray]:
        """
        EMA 스무딩: 이전 프레임과 동일 개수의 차선이면
        각 차선의 world_x에 지수이동평균 적용하여 떨림 감소.
        """
        if self._ema_lanes is None or len(self._ema_lanes) != len(lanes):
            # 차선 개수 변경 → EMA 리셋
            self._ema_lanes = [lane.copy() for lane in lanes]
            return lanes

        smoothed = []
        for prev, curr in zip(self._ema_lanes, lanes):
            if prev.shape == curr.shape:
                new = curr.copy()
                # world_x (row 0)만 스무딩 — world_y는 forward 방향이라 변동이 적음
                new[0] = EMA_ALPHA * curr[0] + (1 - EMA_ALPHA) * prev[0]
                smoothed.append(new)
            else:
                smoothed.append(curr.copy())

        self._ema_lanes = [s.copy() for s in smoothed]
        return smoothed

    # ────────────────────────────────────────────
    # 상태 리셋
    # ────────────────────────────────────────────
    def reset(self):
        """엔진 상태 초기화."""
        self.confirmed_track_type = None
        self._candidate_type = None
        self._debounce_counter = 0
        self._last_result = None
        self._no_detection_count = 0
        self._ema_lanes = None
