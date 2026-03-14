import numpy as np
import cv2

def merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array):
    # 랩뷰 데이터를 NumPy 배열로 변환 (이미 배열이면 복사 없이 참조만 함)
    # dtype=np.uint8을 명시하여 OpenCV가 처리 가능한 타입으로 확정합니다.
    b = np.asanyarray(Blue_2D_array, dtype=np.uint8)
    g = np.asanyarray(Green_2D_array, dtype=np.uint8)
    r = np.asanyarray(Red_2D_array, dtype=np.uint8)
    
    # OpenCV 최적화된 merge 함수 호출
    return cv2.merge((b, g, r))