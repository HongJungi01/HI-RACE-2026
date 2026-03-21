import cv2
import numpy as np
import os
from ultralytics import YOLO

# 현재 파일의 위치를 기준으로 모델 폴더의 절대 경로 생성
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'best_openvino_model')

# 모델 로드 (절대 경로 사용)
lane_model = YOLO(model_path, task='segment')

def extract_lane_binary(bgr_img):
    """
    BGR 이미지를 입력받아 가장 큰 차선 뭉치 2개만 255(흰색), 
    나머지는 0(검은색)인 바이너리 마스크 이미지를 반환합니다.
    """
    results = lane_model.predict(bgr_img, imgsz=320, conf=0.4, verbose=False)
    h, w = bgr_img.shape[:2]
    
    if results[0].masks is not None:
        # 1. 모든 검출된 마스크 가져오기
        masks = results[0].masks.data.cpu().numpy() # [N, H_model, W_model]
        
        # 2. 각 마스크의 면적(흰색 픽셀 수) 계산 및 정렬
        # (N, 면적) 튜플 리스트 생성 후 면적 기준 내림차순 정렬
        mask_areas = []
        for i in range(len(masks)):
            area = np.sum(masks[i])
            mask_areas.append((i, area))
        
        mask_areas.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 상위 2개 인덱스만 선택
        top_indices = [idx for idx, area in mask_areas[:2]]
        
        # 4. 선택된 마스크들만 합치기
        if top_indices:
            selected_masks = masks[top_indices]
            combined_mask = np.any(selected_masks, axis=0).astype(np.uint8)
        else:
            return np.zeros((h, w), dtype=np.uint8)
        
        # 5. 모델 출력 크기를 원본 이미지 크기로 복원 및 255 변환
        binary_output = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_output = (binary_output * 255).astype(np.uint8)
        
        return binary_output
    else:
        return np.zeros((h, w), dtype=np.uint8)

# --- 실제 사용 예시 (다른 CV 로직과의 연결) ---
# img = cv2.imread('road.jpg') # BGR 이미지 읽기
# lane_mask = extract_lane_binary(img) # 바이너리 마스크 획득

# 이제 lane_mask를 가지고 다른 CV 처리를 수행:
# 예: 차선 부분만 컬러로 추출하기
# color_lane = cv2.bitwise_and(img, img, mask=lane_mask)