from array2Image import merge_rgb_to_bgr
from colorExtractor import yellow_color_extractor, white_color_extractor
from laneInference import LaneInferenceEngine
import laneDetermine as lD
import cv2
import numpy as np
import time

# 모듈 레벨에서 추론 엔진 생성 — LabVIEW Python Node 반복 호출 시 상태 유지
_inference_engine = LaneInferenceEngine()

def preprocess_image(binary_mask, len=3):
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (len, len))
    preprocessed_image = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)
    return preprocessed_image

def main(image_string, min_area=100, resample_count=50, len=3):
    """
    메인 파이프라인: 이미지 → 차선 감지 → 추론 → 전체 흰색 차선(주행 경로) 좌표 반환.
    
    * 주행 가능 차선(흰색)만 반환하며, 노란색 경계선은 추론용으로만 사용됩니다. *
    
    Returns:
        np.ndarray: (2 * lane_count, resample_count) 형태의 2D Float 배열.
            - 차선 1개당 2행(X, Y)을 차지하며, 좌→우 순서로 쌓여 있습니다.
            - 2차로 코스일 때: (4, resample_count) 크기 배열 반환
            - 3차로 코스일 때: (6, resample_count) 크기 배열 반환
            
            LabVIEW에서 'Index Array'로 데이터 추출 시:
              [행 인덱스] = 데이터 내용
                0  = Lane 1 (가장 왼쪽) X 리스트
                1  = Lane 1 (가장 왼쪽) Y 리스트
                2  = Lane 2 X 리스트
                3  = Lane 2 Y 리스트
                ...
    """
    t0 = time.time()
    # Step 1: Merge RGB channels into a BGR image
    bgr_image = merge_rgb_to_bgr(image_string)

    # Step 2: Extract yellow and white color masks
    yellow_mask = yellow_color_extractor(bgr_image)
    white_mask = white_color_extractor(bgr_image)

    yellow_mask = preprocess_image(yellow_mask, len=len)
    white_mask = preprocess_image(white_mask, len=len)
    
    # Step 3: Extract lane world points (최대 3개씩)
    white_lanes = lD.extract_lane_world_points(white_mask, min_area=min_area, resample_count=resample_count)
    yellow_lanes = lD.extract_lane_world_points(yellow_mask, min_area=min_area, resample_count=resample_count)

    # Step 4: 비가시 차선 추론 — 번호 부여된 전체 차선 생성
    # 반환: list[ndarray(2, resample_count)], 좌→우 정렬(인덱스+1 = 차선번호)
    all_lanes = _inference_engine.infer(yellow_lanes, white_lanes)

    elapsed = time.time() - t0

    if not all_lanes:
        return np.zeros((0, 0)) # 빈 배열 반환
    
    # 모든 차선을 세로로 쌓음 (Vertical Stack)
    # 결과 모양: (2 * 차선수, resample_count) -> 예: 2차로면 (4, 50) 배열
    # 행 0: Lane 1 X
    # 행 1: Lane 1 Y
    # 행 2: Lane 2 X
    # 행 3: Lane 2 Y
    combined = np.vstack(all_lanes).astype(np.float64)
    
    return combined


def get_lane_count():
    """현재 코스 타입 기반 차선 수 반환. LabVIEW에서 상태 조회용."""
    if _inference_engine.confirmed_track_type == "YWWWY":
        return 3
    return 2