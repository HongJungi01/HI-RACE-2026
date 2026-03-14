from array2Image import merge_rgb_to_bgr
from colorExtractor import yellow_color_extractor, white_color_extractor
from laneExtractor import lane_extractor
import cv2
import numpy as np
import time
import os

def main(Red_2D_array, Green_2D_array, Blue_2D_array, min_area=100, resample_count=50, meters_per_pixel=0.01):
    t0 = time.time()    # Start time measurement
    # Step 1: Merge RGB channels into a BGR image
    bgr_image = merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array)

    # Step 2: Extract yellow and white color masks
    yellow_mask = yellow_color_extractor(bgr_image)
    white_mask = white_color_extractor(bgr_image)
    
    # Step 3 & 4: Extract lane line points
    white_left, white_right = lane_extractor(white_mask, min_area=min_area, resample_count=resample_count, meters_per_pixel=meters_per_pixel)
    yellow_left, yellow_right = lane_extractor(yellow_mask, min_area=min_area, resample_count=resample_count, meters_per_pixel=meters_per_pixel)

    # Data that is empty (detection failure) is replaced with a zero array of size resample_count
    empty_lane = np.zeros((2, resample_count))

    def ensure_shape(lane):
        # If the array is empty or has no elements, return the empty lane
        if lane is None or lane.size == 0:
            return empty_lane
        return lane

    # Combine the extracted points into a 3D array: (4, 2, resample_count)
    # 0: white_left, 1: white_right, 2: yellow_left, 3: yellow_right
    t1 = time.time()    # End time measurement
    processing_time = (t1 - t0) * 1000  # calculate processing time as ms
    
    # Create a processing time array with the same width as the lane data
    proc_time_array = np.zeros((1, resample_count))
    proc_time_array[0, 0] = processing_time

    result = np.vstack([
        ensure_shape(white_left),   # Row 0(World X), 1(World Y)
        ensure_shape(white_right),  # Row 2(World X), 3(World Y)
        ensure_shape(yellow_left),  # Row 4(World X), 5(World Y)
        ensure_shape(yellow_right), # Row 6(World X), 7(World Y)
        proc_time_array            # Row 8(Processing Time in ms)
    ])

    return result