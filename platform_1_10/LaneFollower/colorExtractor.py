import cv2
import numpy as np

def yellow_color_extractor(hsv_image):
    """
    HSV 이미지에서 노란색 영역을 추출한다.
    Parameters:
    - hsv_image: HSV 색공간 이미지
    Returns:
    - yellow_mask: 노란색 영역이 흰색인 바이너리 마스크
    """
    lower_yellow = np.array([15, 120, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    return yellow_mask

def white_color_extractor(hsv_image):
    """
    HSV 이미지에서 흰색 영역을 추출한다.
    Parameters:
    - hsv_image: HSV 색공간 이미지
    Returns:
    - white_mask: 흰색 영역이 흰색인 바이너리 마스크
    """
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 80, 255])
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    return white_mask