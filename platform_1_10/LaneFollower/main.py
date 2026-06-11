from array2Image import merge_rgb_to_bgr
from colorExtractor import yellow_color_extractor, white_color_extractor
from laneInference import LaneInferenceEngine
import laneDetermine as lD
import cv2
import numpy as np
import os

# 추론 엔진 생성 (내부 상태가 제거된 상태를 전제로 함)
_inference_engine = LaneInferenceEngine()

# 상단 ROI 마스킹 높이 (px) — 이 값 위쪽은 노이즈 영역으로 무시
ROI_TOP = 100

def preprocess_image(binary_mask, side=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    preprocessed_image = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return preprocessed_image

def main(image_string, min_area=100, min_span=50, max_rmse=100.0, resample_count=50, side=3):
    """
    메인 파이프라인: 매 프레임 독립적인 차선 추론 수행.
    """
    # Step 1: 이미지 디코딩
    bgr_image = merge_rgb_to_bgr(image_string)

    # Step 2: HSV 변환 1회 → 마스크 추출 및 전처리
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    yellow_mask = yellow_color_extractor(hsv_image)
    white_mask = white_color_extractor(hsv_image)

    # 처리 전 이미지 현재 위치 절대경로 저장 (디버깅용)

    yellow_mask = preprocess_image(yellow_mask, side=side)
    white_mask = preprocess_image(white_mask, side=side)

    # 현재 파일의 디렉토리를 기준으로 절대 경로 설정
    save_dir = os.path.dirname(os.path.abspath(__file__))
    # 처리된 마스크 이미지 저장 (디버깅용)
    cv2.imwrite(os.path.join(save_dir, "processed_yellow.png"), yellow_mask)
    cv2.imwrite(os.path.join(save_dir, "processed_white.png"), white_mask)


    # 상단 ROI 마스킹 (원거리 노이즈 제거)
    yellow_mask[:ROI_TOP, :] = 0
    white_mask[:ROI_TOP, :] = 0

    # Step 3: 월드 좌표 추출
    white_lanes = lD.extract_lane_world_points(white_mask, min_area=min_area, min_span=min_span, max_rmse=max_rmse, fit_degree=5, resample_count=resample_count)
    yellow_lanes = lD.extract_lane_world_points(yellow_mask, min_area=min_area, min_span=min_span, max_rmse=max_rmse, fit_degree=5, resample_count=resample_count)

    # Step 4: Stateless 추론 (과거 프레임 참조 없음)
    all_lanes = _inference_engine.infer(yellow_lanes, white_lanes)

    # Step 5: LabVIEW 반환용 데이터 포맷팅 (항상 6행 고정)
    # 추론 결과가 없거나 부족하더라도 0으로 채워서 반환
    final_lanes = []
    if all_lanes:
        final_lanes = all_lanes[:3] # 최대 3개까지만 취함

    # 3개가 안 될 경우 빈 배열(2xN)로 패딩
    while len(final_lanes) < 3:
        final_lanes.append(np.zeros((2, resample_count), dtype=np.float64))
    
    # 수직으로 쌓아서 (6, resample_count) 배열 생성
    combined = np.vstack(final_lanes).astype(np.float64)
    
    return combined