import array2Image as a2i
import colorExtractor as cE
import laneExtractor as lE
import laneDetermine as lD
import cv2
import numpy as np
import time
import os

# DebugImageFolder 경로 설정
debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DebugImageFolder"))

if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
    
def main(Red_2D_array, Green_2D_array, Blue_2D_array, min_area=800, min_span=250, max_rmse=10.0, filter_poly_degree=2, tilt_threshold=20, lower_level=220):
    start_total = time.time()

    # Step 1: Merge RGB channels
    t1 = time.time()
    bgr_image = a2i.merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array)
    dt_merge = (time.time() - t1) * 1000

    # Step 2: Color Extraction
    t2 = time.time()
    white_mask = cE.white_color_extractor(bgr_image, lower_level=lower_level)
    dt_color = (time.time() - t2) * 1000

    # Step 3: Lane Candidates Filtering
    t3 = time.time()
    candidates, labels = lD.filter_lane_candidates(white_mask, min_area=min_area, min_span=min_span, max_rmse=max_rmse)
    dt_filter = (time.time() - t3) * 1000

    # Step 4: Left/Right Classification
    t4 = time.time()
    mask_left, mask_right = lD.classify_left_right(candidates, labels, white_mask.shape, tilt_threshold=tilt_threshold)
    dt_classify = (time.time() - t4) * 1000

    total_time = (time.time() - start_total) * 1000

    # Draw timing info on mask_left
    # 텍스트 정보 리스트
    timings = [
        f"merge_rgb: {dt_merge:.2f}ms",
        f"color_extractor: {dt_color:.2f}ms",
        f"filter_candidates: {dt_filter:.2f}ms",
        f"classify_lr: {dt_classify:.2f}ms",
        f"TOTAL: {total_time:.2f}ms"
    ]

    # 흰색 글씨(255)로 mask_left 상단에 작성
    for i, text in enumerate(timings):
        cv2.putText(mask_left, text, (10, 30 + (i * 30)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255), 2)

    # Debug: Save masks
    left_mask_path = os.path.join(debug_dir, "left_lane_mask.png")
    right_mask_path = os.path.join(debug_dir, "right_lane_mask.png")
    cv2.imwrite(left_mask_path, mask_left)
    cv2.imwrite(right_mask_path, mask_right)

    return [left_mask_path, right_mask_path]