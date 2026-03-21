import array2Image as a2i
import YOLOInference
import laneDetermine as lD
from temporal_filter import LaneTracker
import cv2
import numpy as np
import os
import time
import base64

# 프레임 간 상태를 유지하는 트래커 (모듈 레벨 인스턴스)
tracker = LaneTracker()

def encode_image_to_base64(image):
    """이미지를 JPEG 형식의 Base64 문자열로 변환"""
    if image is None:
        return ""
    # JPEG 품질 80으로 설정 (속도와 화질의 타협점)
    success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        return ""
    # 바이너리를 Base64 문자열로 변환
    return base64.b64encode(buffer).decode('utf-8')

def main(Image_String, min_area, min_span, max_rmse, morph_kernel_size=3):
    poly_degree=2   # 다항식 차수는 고정 (현재 2차) - 필요 시 매개변수로 확장 가능
    # 1. BGR 이미지 통합
    imagedecode_start_time = time.time()
    bgr_image = a2i.merge_rgb_to_bgr(Image_String)
    imagedecode_end_time = time.time()
    
    # 2. YOLO 추론
    inferece_start_time = time.time()
    binary_mask = YOLOInference.extract_lane_binary(bgr_image)
    inferece_end_time = time.time()

    # 3. ROI 크롭 및 리사이즈
    roi_img = binary_mask[79:230, :] 
    binary_mask = cv2.resize(roi_img, (640, 310), interpolation=cv2.INTER_LINEAR)

    # 3.5 얇은 연결 제거 (Morphological Opening)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)

    # 3.6 Detection Persistence — 빈 마스크일 때 이전 유효 마스크 재사용
    binary_mask = tracker.persist_detection(binary_mask)

    # 4. 차선 검증
    lanedetermine_start_time = time.time()
    filtered_mask = lD.filter_lane_candidates(binary_mask, min_area, min_span, max_rmse, poly_degree)

    # 5. 좌/우 차선 분류
    left_mask, right_mask, state = lD.classify_left_right(filtered_mask)

    # 5.5 State Debouncing — 급격한 상태 전환 방지
    state, left_mask, right_mask = tracker.debounce_state(state, left_mask, right_mask)
    lanedetermine_end_time = time.time()

    # 6. 중심선 계산
    calculation_start_time = time.time()
    cx, cy = lD.calculate_center_line(left_mask, right_mask, state)
    calculation_end_time = time.time()

    # 7. CTE / Heading Error 계산
    stanley_start_time = time.time()
    cte, heading = lD.calculate_stanley_error(cx, cy)

    # 7.5 CTE/Heading EMA 스무딩
    cte, heading = tracker.smooth_output(cte, heading)
    stanley_end_time = time.time()

    # 8. 디버그 이미지 생성 (그리기 연산)
    debug_image_start_time = time.time()
    debug_img = lD.draw_path_on_image(bgr_image, cx, cy, left_mask, right_mask, state)
    debug_image_end_time = time.time()

    # 9. 이미지 메모리 인코딩 (디스크 저장 대신 Base64 변환)
    # 랩뷰로 보낼 이미지 3종
    encoding_start_time = time.time()
    binary_mask_base64 = encode_image_to_base64(binary_mask)
    debug_img_base64 = encode_image_to_base64(debug_img)
    encoding_end_time = time.time()

    # 연산 시간 계산
    total_time_ms = (time.time() - imagedecode_start_time) * 1000
    
    # 리턴 리스트 조립
    # 랩뷰에서 구분자 '|'로 잘라서 사용
    return [
        str(heading),           # [0] 헤딩 에러
        str(cte),               # [1] CTE
        binary_mask_base64,     # [2] 이진 마스크 이미지 (Base64)
        debug_img_base64,       # [3] 전체 디버그 이미지 (Base64)
        total_time_ms # [4] 시간 정보
    ]