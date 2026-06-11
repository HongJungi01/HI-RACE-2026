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

def region_of_interest(img):
    # 1. 이미지의 높이와 너비 가져오기
    height, width = img.shape[:2]

    # 2. 남기고자 하는 다각형(사다리꼴)의 꼭짓점 좌표 설정
    # 사진과 비슷한 비율로 임의 설정했습니다. 실제 이미지에 맞게 조절해야 합니다.
    # 왼쪽 아래, 왼쪽 위, 오른쪽 위, 오른쪽 아래 순서
    pt1 = (0, 180)                   # 좌측 하단
    pt2 = (int(width * 0.3), 70)        # 좌측 상단 20% 지점
    pt3 = (int(width * 0.7), 70)        # 우측 상단 80% 지점
    pt4 = (640, 180)               # 우측 하단

    # cv2.fillPoly 함수는 3차원 배열을 요구하므로 형태를 맞춰줍니다.
    vertices = np.array([[pt1, pt2, pt3, pt4]], dtype=np.int32)

    # 3. 원본 이미지와 동일한 크기의 검은색(0) 마스크 생성 (흑백 이미지 형태)
    mask = np.zeros((height, width), dtype=np.uint8)

    # 4. 설정한 다각형 영역을 마스크 위에 하얀색(255)으로 채우기
    cv2.fillPoly(mask, vertices, 255)

    # 5. 원본 이미지와 마스크를 비트 연산(AND)하여 ROI 영역만 추출
    # 마스크가 하얀색(255)인 부분의 원본 픽셀만 남고, 검은색(0)인 부분은 검게 변합니다.
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img

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

    binary_mask = region_of_interest(binary_mask)

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
    if state == 'none':
        lane_count = 0
    elif state == 'left':
        lane_count = 1
    elif state == 'right':
        lane_count = 1
    elif state == 'both':
        lane_count = 2
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
        total_time_ms,          # [4] 시간 정보
        str(lane_count)         # [5] 검출된 차선 수
    ]