"""
검증 스크립트 — 121장 전체에 대해 파이프라인 테스트.
1. FPS 벤치마크
2. 차선 검출 통계
3. 대표 이미지에 polyfit 오버레이 저장
"""
import cv2
import numpy as np
import glob
import os
import time

from imgPreprocess import (
    process_frame, extract_lane_mask, extract_lane_curves,
    generate_world_points, _H, pixel_to_world, RESAMPLE_COUNT
)

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_overlay")


def world_to_pixel(world_points):
    """월드 좌표 → 픽셀 좌표 (시각화용)."""
    H_inv = np.linalg.inv(_H)
    pts = np.array(world_points, dtype=np.float32).reshape(-1, 1, 2)
    pixel = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return pixel[:, 0], pixel[:, 1]


def draw_lane_overlay(img, result):
    """(2, N) 결과를 원본 이미지 위에 오버레이."""
    overlay = img.copy()
    wx, wy = result[0], result[1]
    if np.all(wx == 0) and np.all(wy == 0):
        return overlay

    world_pts = np.stack([wx, wy], axis=1)
    px, py = world_to_pixel(world_pts)

    for j in range(len(px) - 1):
        x1, y1 = int(px[j]), int(py[j])
        x2, y2 = int(px[j+1]), int(py[j+1])
        if 0 <= x1 < img.shape[1] and 0 <= y1 < img.shape[0]:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return cv2.addWeighted(img, 0.6, overlay, 0.4, 0)


def run_validation():
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpeg")))
    print(f"Images: {len(paths)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 벤치마크
    times = []
    lane_counts = []
    errors = 0

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            errors += 1
            continue

        t0 = time.perf_counter()
        result = process_frame(img)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

        # 검출 여부 (1 or 0)
        detected = 0 if (np.all(result[0] == 0) and np.all(result[1] == 0)) else 1
        lane_counts.append(detected)

    times = np.array(times)
    print(f"\n=== FPS BENCHMARK ===")
    print(f"  Mean:   {times.mean():.2f} ms ({1000/times.mean():.0f} FPS)")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  p95:    {np.percentile(times, 95):.2f} ms")
    print(f"  Errors: {errors}")

    counts = np.array(lane_counts)
    print(f"\n=== LANE DETECTION STATS ===")
    print(f"  detected: {np.sum(counts == 1)} frames")
    print(f"  missed:   {np.sum(counts == 0)} frames")

    # 대표 이미지 오버레이 저장
    test_indices = {"02": 0, "15": 13, "30": 28, "45": 43, "60": 58,
                    "75": 73, "90": 88, "100": 98, "110": 108, "120": 118}

    for label, idx in test_indices.items():
        if idx >= len(paths):
            continue
        img = cv2.imread(paths[idx])
        if img is None:
            continue
        result = process_frame(img)
        overlay = draw_lane_overlay(img, result)

        detected = 0 if (np.all(result[0] == 0) and np.all(result[1] == 0)) else 1
        cv2.imwrite(os.path.join(OUT_DIR, f"overlay_{label}.png"), overlay)
        print(f"  overlay_{label}.png — {'detected' if detected else 'missed'}")


if __name__ == "__main__":
    run_validation()
