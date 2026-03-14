import numpy as np
import cv2

def merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array):
    # 랩뷰 데이터를 NumPy 배열로 변환
    b = np.asanyarray(Blue_2D_array, dtype=np.uint8)
    g = np.asanyarray(Green_2D_array, dtype=np.uint8)
    r = np.asanyarray(Red_2D_array, dtype=np.uint8)
    
    # 1. 먼저 RGB 채널을 병합하여 BGR 이미지를 생성합니다. (640x360 예상)
    bgr_img = cv2.merge((b, g, r))
    
    return bgr_img