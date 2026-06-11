"""HSV 임계값 튜닝 스크립트 — 새 카메라 위치(앞쪽)에 맞춰 재분석."""
import cv2
import numpy as np
import glob
import os

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")

def analyze_images():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
    print(f"Found {len(paths)} images")

    white_h, white_s, white_v = [], [], []
    yellow_h, yellow_s, yellow_v = [], [], []

    # 159장 중 20장 고르게 샘플링
    n = len(paths)
    sample_indices = [int(i * (n - 1) / 19) for i in range(20)]
    sample_paths = [paths[i] for i in sample_indices]
    print(f"Sampling {len(sample_paths)} images: indices {sample_indices}")

    for p in sample_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

        # ROI: 하단 70%
        roi_top = int(img.shape[0] * 0.3)
        h_roi = h[roi_top:, :]
        s_roi = s[roi_top:, :]
        v_roi = v[roi_top:, :]

        # 밝은 픽셀 (V > 100) — 어두운 장면 대비
        bright = v_roi > 100

        # 노란색 후보: H 10-45, S>40인 밝은 픽셀
        yellow_cand = bright & (h_roi >= 10) & (h_roi <= 45) & (s_roi > 40)
        # 흰색 후보: 채도 낮고 밝은 픽셀 (노란색과 겹치면 노란 우선)
        white_cand = bright & (s_roi < 100) & (v_roi > 140) & ~yellow_cand

        if np.any(yellow_cand):
            yellow_h.extend(h_roi[yellow_cand].tolist())
            yellow_s.extend(s_roi[yellow_cand].tolist())
            yellow_v.extend(v_roi[yellow_cand].tolist())
        if np.any(white_cand):
            white_h.extend(h_roi[white_cand].tolist())
            white_s.extend(s_roi[white_cand].tolist())
            white_v.extend(v_roi[white_cand].tolist())

    print("\n=== YELLOW LANE HSV DISTRIBUTION ===")
    if yellow_h:
        yh, ys, yv = np.array(yellow_h), np.array(yellow_s), np.array(yellow_v)
        for name, arr in [("H", yh), ("S", ys), ("V", yv)]:
            print(f"  {name}: min={arr.min()}, p2={np.percentile(arr,2):.0f}, "
                  f"p5={np.percentile(arr,5):.0f}, p25={np.percentile(arr,25):.0f}, "
                  f"median={np.median(arr):.0f}, p75={np.percentile(arr,75):.0f}, "
                  f"p95={np.percentile(arr,95):.0f}, p98={np.percentile(arr,98):.0f}, max={arr.max()}")
        print(f"  Total pixels: {len(yellow_h)}")
    else:
        print("  No yellow pixels found!")

    print("\n=== WHITE LANE HSV DISTRIBUTION ===")
    if white_h:
        wh, ws, wv = np.array(white_h), np.array(white_s), np.array(white_v)
        for name, arr in [("H", wh), ("S", ws), ("V", wv)]:
            print(f"  {name}: min={arr.min()}, p2={np.percentile(arr,2):.0f}, "
                  f"p5={np.percentile(arr,5):.0f}, p25={np.percentile(arr,25):.0f}, "
                  f"median={np.median(arr):.0f}, p75={np.percentile(arr,75):.0f}, "
                  f"p95={np.percentile(arr,95):.0f}, p98={np.percentile(arr,98):.0f}, max={arr.max()}")
        print(f"  Total pixels: {len(white_h)}")
    else:
        print("  No white pixels found!")

    print("\n=== RECOMMENDED THRESHOLDS ===")
    if yellow_h:
        yh, ys, yv = np.array(yellow_h), np.array(yellow_s), np.array(yellow_v)
        y_h_lo = max(0, int(np.percentile(yh, 2)) - 2)
        y_h_hi = min(179, int(np.percentile(yh, 98)) + 2)
        y_s_lo = max(0, int(np.percentile(ys, 2)) - 10)
        y_v_lo = max(0, int(np.percentile(yv, 2)) - 10)
        print(f"  Yellow lower: ({y_h_lo}, {y_s_lo}, {y_v_lo})")
        print(f"  Yellow upper: ({y_h_hi}, 255, 255)")
    if white_h:
        ws, wv = np.array(white_s), np.array(white_v)
        w_s_hi = min(255, int(np.percentile(ws, 98)) + 10)
        w_v_lo = max(0, int(np.percentile(wv, 2)) - 10)
        print(f"  White lower: (0, 0, {w_v_lo})")
        print(f"  White upper: (180, {w_s_hi}, 255)")


def visualize_masks():
    """현재 imgPreprocess.py 임계값으로 마스크 시각화."""
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
    from imgPreprocess import _Y_LOWER, _Y_UPPER, _W_LOWER, _W_UPPER, _MORPH_KERNEL, ROI_TOP

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_masks")
    os.makedirs(out_dir, exist_ok=True)

    n = len(paths)
    test_indices = {str(i): i-1 for i in [1, 15, 30, 45, 60, 75, 90, 100, 120, 130, 150, 159]}

    for label, idx in test_indices.items():
        if idx >= n or idx < 0:
            continue
        img = cv2.imread(paths[idx])
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        y_mask = cv2.inRange(hsv, _Y_LOWER, _Y_UPPER)
        w_mask = cv2.inRange(hsv, _W_LOWER, _W_UPPER)
        y_mask = cv2.morphologyEx(y_mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        w_mask = cv2.morphologyEx(w_mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
        y_mask[:ROI_TOP, :] = 0
        w_mask[:ROI_TOP, :] = 0
        w_mask[y_mask > 0] = 0

        overlay = img.copy()
        overlay[y_mask > 0] = [0, 255, 255]
        overlay[w_mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        cv2.imwrite(os.path.join(out_dir, f"mask_{label}.png"), blended)
        print(f"  mask_{label}.png — yellow_px={np.sum(y_mask>0)}, white_px={np.sum(w_mask>0)}")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    analyze_images()
    print("\n--- Generating mask visualizations ---")
    visualize_masks()
