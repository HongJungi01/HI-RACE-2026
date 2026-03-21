import numpy as np
import cv2

def merge_rgb_to_bgr(raw_bytes):
    # LabVIEW에서 U8 Array를 넘겨주면 파이썬에서는 'list' 타입이 됩니다.
    # list를 np.uint8 타입의 넘파이 배열로 변환합니다.
    if isinstance(raw_bytes, list):
        nparr = np.array(raw_bytes, dtype=np.uint8)
    else:
        # 혹시 나중에 bytes 타입으로 들어올 경우를 대비한 예외 처리
        nparr = np.frombuffer(raw_bytes, np.uint8)
    
    # 이미지 디코딩 (JPEG 바이너리를 BGR 배열로 복원)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        # 디코딩 실패 시 (데이터가 비었거나 깨졌을 때)
        return None
        
    return img