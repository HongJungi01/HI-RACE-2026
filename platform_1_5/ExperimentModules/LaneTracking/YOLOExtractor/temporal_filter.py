import numpy as np


class LaneTracker:
    """
    프레임 간 상태를 유지하여 차선 감지를 안정화하는 클래스.

    3가지 안정화 레이어:
      1. Detection Persistence — 감지 실패 시 최근 유효 마스크 재사용
      2. State Debouncing — 상태 전환이 연속 N프레임 유지돼야 실제 전환
      3. CTE/Heading EMA — 지수 이동 평균으로 출력값 스무딩
    """

    def __init__(self,
                 max_persist_frames=5,
                 debounce_frames=3,
                 ema_alpha=0.5):
        """
        Args:
            max_persist_frames: 감지 실패 시 이전 마스크를 재사용할 최대 프레임 수 (10fps 기준 0.5초)
            debounce_frames: 상태 전환에 필요한 연속 동일 프레임 수
            ema_alpha: EMA 가중치 (0~1, 클수록 현재값 비중 높음)
        """
        # --- Detection Persistence ---
        self._max_persist = max_persist_frames
        self._last_good_binary = None   # 마지막 유효 바이너리 마스크 (filter 전)
        self._frames_since_detection = 0

        # --- State Debouncing ---
        self._debounce_frames = debounce_frames
        self._confirmed_state = "none"
        self._pending_state = "none"
        self._pending_count = 0

        # --- CTE/Heading EMA ---
        self._ema_alpha = ema_alpha
        self._prev_cte = None
        self._prev_heading = None

    # =========================================================
    # 1. Detection Persistence
    # =========================================================

    def persist_detection(self, binary_mask):
        """
        YOLO 출력(바이너리 마스크)이 비어있으면 이전 유효 마스크를 반환한다.
        유효한 마스크가 들어오면 캐시를 갱신한다.

        Args:
            binary_mask: YOLO → ROI crop → resize → opening 후의 바이너리 마스크

        Returns:
            사용할 바이너리 마스크 (현재 또는 캐시된 것)
        """
        has_detection = np.any(binary_mask > 0)

        if has_detection:
            self._last_good_binary = binary_mask.copy()
            self._frames_since_detection = 0
            return binary_mask

        # 감지 실패 — 캐시된 마스크가 있고 제한 내이면 재사용
        if (self._last_good_binary is not None
                and self._frames_since_detection < self._max_persist):
            self._frames_since_detection += 1
            return self._last_good_binary

        # 캐시 소진 — 빈 마스크 그대로 반환
        self._frames_since_detection += 1
        return binary_mask

    # =========================================================
    # 2. State Debouncing
    # =========================================================

    def debounce_state(self, new_state, left_mask, right_mask):
        """
        좌/우 분류 상태 전환을 디바운싱한다.
        연속 debounce_frames 만큼 동일한 새 상태가 유지돼야 실제 전환.

        Args:
            new_state: 현재 프레임의 classify_left_right() 결과 ("both"/"left"/"right"/"none")
            left_mask: 현재 프레임의 좌측 마스크
            right_mask: 현재 프레임의 우측 마스크

        Returns:
            (state, left_mask, right_mask) — 디바운싱된 상태와 해당 마스크
        """
        if new_state == self._confirmed_state:
            # 기존 상태와 동일 — 카운터 리셋, 그대로 통과
            self._pending_state = new_state
            self._pending_count = 0
            return new_state, left_mask, right_mask

        # 새로운 상태 감지
        if new_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = new_state
            self._pending_count = 1

        if self._pending_count >= self._debounce_frames:
            # 충분히 연속되었으므로 상태 전환 확정
            self._confirmed_state = new_state
            self._pending_count = 0
            return new_state, left_mask, right_mask

        # 아직 전환 조건 불충분 — 이전 확정 상태 유지
        # 마스크는 현재 프레임 것을 그대로 사용 (시각적으로는 현재 감지된 것을 보여줌)
        return self._confirmed_state, left_mask, right_mask

    # =========================================================
    # 3. CTE / Heading EMA
    # =========================================================

    def smooth_output(self, cte, heading):
        """
        CTE와 heading에 지수 이동 평균(EMA)을 적용한다.

        Args:
            cte: 현재 프레임의 CTE (m)
            heading: 현재 프레임의 heading error (rad)

        Returns:
            (smoothed_cte, smoothed_heading)
        """
        alpha = self._ema_alpha

        if self._prev_cte is None:
            # 첫 프레임 — 초기화
            self._prev_cte = cte
            self._prev_heading = heading
            return cte, heading

        smoothed_cte = alpha * cte + (1 - alpha) * self._prev_cte
        smoothed_heading = alpha * heading + (1 - alpha) * self._prev_heading

        self._prev_cte = smoothed_cte
        self._prev_heading = smoothed_heading

        return smoothed_cte, smoothed_heading

    # =========================================================
    # 유틸리티
    # =========================================================

    def reset(self):
        """모든 내부 상태를 초기화한다."""
        self._last_good_binary = None
        self._frames_since_detection = 0
        self._confirmed_state = "none"
        self._pending_state = "none"
        self._pending_count = 0
        self._prev_cte = None
        self._prev_heading = None
