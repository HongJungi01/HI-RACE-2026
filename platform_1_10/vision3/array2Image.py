import numpy as np
import cv2


def merge_rgb_to_bgr(raw_bytes):
    """LabVIEW U8 Array(JPEG 바이너리)를 BGR 이미지로 디코딩."""
    if isinstance(raw_bytes, list):
        nparr = np.array(raw_bytes, dtype=np.uint8)
    else:
        nparr = np.frombuffer(raw_bytes, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img