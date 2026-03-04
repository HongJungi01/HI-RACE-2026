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


    temp_image_path = "temp_black_mask.png"
    cv2.imwrite(temp_image_path, black_mask)

    left_lane_mask, right_lane_mask = get_separated_masks(black_mask, min_area=100)

    left_image_path = "temp_left_lane_mask.png"
    right_image_path = "temp_right_lane_mask.png"
    cv2.imwrite(left_image_path, left_lane_mask)
    cv2.imwrite(right_image_path, right_lane_mask)

    # Step 4: return the absolute path of the saved image
    return os.path.abspath(temp_image_path)