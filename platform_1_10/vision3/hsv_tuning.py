"""
HSV 임계값 튜닝 스크립트 — 샘플 이미지에서 차선 영역의 HSV 분포를 분석한다.
실행 후 결과를 기반으로 imgPreprocess.py의 임계값을 결정.
"""
import cv2
import numpy as np
import glob
import os

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")

def analyze_images():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
    print(f"Found {len(paths)} images")

    # 넓은 범위로 초기 마스크를 만들어 차선 픽셀만 추출
    # 검정 바닥(V<80)을 제외하고 밝은 픽셀만 모으기
    white_h, white_s, white_v = [], [], []
    yellow_h, yellow_s, yellow_v = [], [], []

    # 대표 이미지 몇 장만 분석 (전체 분포 파악에 충분)
    sample_indices = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    sample_paths = [paths[min(i, len(paths)-1)] for i in sample_indices]

    for p in sample_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        # ROI: 하단 60%만 (상단은 배경 노이즈)
        roi_top = int(img.shape[0] * 0.3)
        h_roi = h[roi_top:, :]
        s_roi = s[roi_top:, :]
        v_roi = v[roi_top:, :]

        # 밝은 픽셀 (V > 120) — 차선 후보
        bright = v_roi > 120

        # 노란색 후보: H가 대략 10-40 범위이고 S가 높은 (채도 있는) 밝은 픽셀
        yellow_cand = bright & (h_roi >= 10) & (h_roi <= 40) & (s_roi > 60)
        # 흰색 후보: 채도가 낮고 밝은 픽셀
        white_cand = bright & (s_roi < 80) & (v_roi > 150)

        if np.any(yellow_cand):
            yellow_h.extend(h_roi[yellow_cand].tolist())
            yellow_s.extend(s_roi[yellow_cand].tolist())
            yellow_v.extend(v_roi[yellow_cand].tolist())

        if np.any(white_cand):
            white_h.extend(h_roi[white_cand].tolist())
            white_s.extend(s_roi[white_cand].tolist())
            white_v.extend(v_roi[white_cand].tolist())

    # 통계 출력
    print("\n=== YELLOW LANE HSV DISTRIBUTION ===")
    if yellow_h:
        yh, ys, yv = np.array(yellow_h), np.array(yellow_s), np.array(yellow_v)
        for name, arr in [("H", yh), ("S", ys), ("V", yv)]:
            print(f"  {name}: min={arr.min()}, p5={np.percentile(arr,5):.0f}, "
                  f"p25={np.percentile(arr,25):.0f}, median={np.median(arr):.0f}, "
                  f"p75={np.percentile(arr,75):.0f}, p95={np.percentile(arr,95):.0f}, max={arr.max()}")
        print(f"  Total pixels: {len(yellow_h)}")
    else:
        print("  No yellow pixels found!")

    print("\n=== WHITE LANE HSV DISTRIBUTION ===")
    if white_h:
        wh, ws, wv = np.array(white_h), np.array(white_s), np.array(white_v)
        for name, arr in [("H", wh), ("S", ws), ("V", wv)]:
            print(f"  {name}: min={arr.min()}, p5={np.percentile(arr,5):.0f}, "
                  f"p25={np.percentile(arr,25):.0f}, median={np.median(arr):.0f}, "
                  f"p75={np.percentile(arr,75):.0f}, p95={np.percentile(arr,95):.0f}, max={arr.max()}")
        print(f"  Total pixels: {len(white_h)}")
    else:
        print("  No white pixels found!")

    # 추천 임계값 출력
    print("\n=== RECOMMENDED THRESHOLDS ===")
    if yellow_h:
        yh = np.array(yellow_h)
        ys = np.array(yellow_s)
        yv = np.array(yellow_v)
        print(f"  Yellow lower: ({max(0, int(np.percentile(yh,2))-2)}, "
              f"{max(0, int(np.percentile(ys,2))-10)}, "
              f"{max(0, int(np.percentile(yv,2))-10)})")
        print(f"  Yellow upper: ({min(179, int(np.percentile(yh,98))+2)}, "
              f"255, 255)")
    if white_h:
        ws = np.array(white_s)
        wv = np.array(white_v)
        print(f"  White lower: (0, 0, {max(0, int(np.percentile(wv,2))-10)})")
        print(f"  White upper: (180, {min(255, int(np.percentile(ws,98))+10)}, 255)")


def visualize_masks():
    """추천 임계값으로 마스크를 생성하고 몇 장 저장해서 확인."""
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpeg")))

    # 추천 임계값
    y_lower = np.array([8, 53, 122])
    y_upper = np.array([32, 255, 255])
    w_lower = np.array([0, 0, 147])
    w_upper = np.array([180, 87, 255])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_top = 90  # 상단 노이즈 영역 제거

    # 대표 이미지: 직선, 완만커브, 급커브, S커브, 밝은장면
    test_indices = {"02": 0, "15": 13, "30": 28, "45": 43, "60": 58,
                    "75": 73, "90": 88, "100": 98, "110": 108, "120": 118}

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_masks")
    os.makedirs(out_dir, exist_ok=True)

    for label, idx in test_indices.items():
        if idx >= len(paths):
            continue
        img = cv2.imread(paths[idx])
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        y_mask = cv2.inRange(hsv, y_lower, y_upper)
        w_mask = cv2.inRange(hsv, w_lower, w_upper)

        # morphological open
        y_mask = cv2.morphologyEx(y_mask, cv2.MORPH_OPEN, kernel)
        w_mask = cv2.morphologyEx(w_mask, cv2.MORPH_OPEN, kernel)

        # ROI
        y_mask[:roi_top, :] = 0
        w_mask[:roi_top, :] = 0

        # 시각화: 원본 위에 마스크 오버레이 (노란=Yellow, 초록=White)
        overlay = img.copy()
        overlay[y_mask > 0] = [0, 255, 255]   # Yellow for yellow lanes
        overlay[w_mask > 0] = [0, 255, 0]     # Green for white lanes
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        cv2.imwrite(os.path.join(out_dir, f"mask_{label}.png"), blended)
        print(f"  Saved mask_{label}.png — yellow_px={np.sum(y_mask>0)}, white_px={np.sum(w_mask>0)}")

    print(f"\nVisualizations saved to {out_dir}")


if __name__ == "__main__":
    analyze_images()
    print("\n--- Generating mask visualizations ---")
    visualize_masks()
