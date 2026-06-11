from array2Image import merge_rgb_to_bgr
from imgPreprocess import process_frame


def main(image_string, resample_count=10):
    """
    LabVIEW 진입점.
    image_string: JPEG 바이너리 (list of U8)
    반환: (2, resample_count) np.float64 — 원점에 가장 가까운 차선 월드 좌표
    """
    bgr_image = merge_rgb_to_bgr(image_string)
    if bgr_image is None:
        import numpy as np
        return np.zeros((2, resample_count), dtype=np.float64)
    return process_frame(bgr_image, resample_count)
