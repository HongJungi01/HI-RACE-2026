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
    BGR 이미지를 입력받아 차선 부분만 255(흰색), 나머지는 0(검은색)인 
    바이너리 마스크 이미지를 반환합니다.
    """
    # 1. YOLO 추론 (BGR 이미지를 넣으면 내부에서 RGB로 자동 변환하여 처리함)
    # imgsz는 학습과 동일하게 320, conf는 상황에 맞게 조절
    results = lane_model.predict(bgr_img, imgsz=320, conf=0.4, verbose=False)
    
    # 원본 이미지 크기
    h, w = bgr_img.shape[:2]
    
    # 2. 결과 마스크가 있는지 확인
    if results[0].masks is not None:
        # 모든 검출된 객체(차선들)의 마스크를 하나로 합침 (Logical OR 연산)
        # results[0].masks.data는 [N, 80, 80] 형태 (imgsz=320인 경우 출력 해상도)
        masks = results[0].masks.data.cpu().numpy()
        combined_mask = np.any(masks, axis=0).astype(np.uint8)
        
        # 3. 모델 출력 크기(80x80 등)를 원본 이미지 크기로 복원
        binary_output = cv2.resize(combined_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 4. 0과 1 상태를 0과 255로 변환
        binary_output = (binary_output * 255).astype(np.uint8)
        
        return binary_output
    else:
        # 검출된 차선이 없으면 빈 검정 이미지 반환
        return np.zeros((h, w), dtype=np.uint8)

# --- 실제 사용 예시 (다른 CV 로직과의 연결) ---
# img = cv2.imread('road.jpg') # BGR 이미지 읽기
# lane_mask = extract_lane_binary(img) # 바이너리 마스크 획득

# 이제 lane_mask를 가지고 다른 CV 처리를 수행:
# 예: 차선 부분만 컬러로 추출하기
# color_lane = cv2.bitwise_and(img, img, mask=lane_mask)