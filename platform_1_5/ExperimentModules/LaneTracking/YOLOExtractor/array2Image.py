import numpy as np
import cv2

def merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array):
    # 랩뷰 데이터를 NumPy 배열로 변환
    b = np.asanyarray(Blue_2D_array, dtype=np.uint8)
    g = np.asanyarray(Green_2D_array, dtype=np.uint8)
    r = np.asanyarray(Red_2D_array, dtype=np.uint8)
    
    # 1. 먼저 RGB 채널을 병합하여 BGR 이미지를 생성합니다. (640x360 예상)
    full_img = cv2.merge((b, g, r))
    
    # 2. 상단 50픽셀을 크롭합니다. [시작y:끝y, 시작x:끝x]
    # 결과 이미지 크기: 360 - 50 = 310 (세로), 640 (가로)
    cropped_img = full_img[50:, :]
    
    return cropped_img