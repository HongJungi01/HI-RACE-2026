from array2Image import merge_rgb_to_bgr
from colorExtractor import yellow_color_extractor, white_color_extractor, black_color_extractor
from laneExtractor import lane_extractor, get_separated_masks, morphology_thinning, post_process_skeleton, fit_poly, generate_resampled_points
import cv2
import numpy as np
import time
import os

def main(Red_2D_array, Green_2D_array, Blue_2D_array):
    # Step 1: Merge RGB channels into a BGR image
    bgr_image = merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array)

    black_mask = black_color_extractor(bgr_image)

    # DebugImageFolder 경로 설정 (상위 폴더의 DebugImageFolder)
    debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DebugImageFolder"))
    
    # 폴더가 없으면 생성
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    temp_image_path = os.path.join(debug_dir, "temp_black_mask.png")
    cv2.imwrite(temp_image_path, black_mask)

    left_lane_mask, right_lane_mask = get_separated_masks(black_mask, min_area=1)

    left_image_path = os.path.join(debug_dir, "temp_left_lane_mask.png")
    right_image_path = os.path.join(debug_dir, "temp_right_lane_mask.png")
    
    cv2.imwrite(left_image_path, left_lane_mask)
    cv2.imwrite(right_image_path, right_lane_mask)

    # Step 4: return the absolute path of the saved image
    return os.path.abspath(temp_image_path)