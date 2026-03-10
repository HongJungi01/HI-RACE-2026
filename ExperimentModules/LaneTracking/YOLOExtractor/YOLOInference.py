# inference.py
import cv2
import numpy as np
from ultralytics import YOLO

# 1. OpenVINO로 변환된 모델 로드
model = YOLO('best_openvino_model/') 

def get_binary_lane(frame):
    # 2. 추론
    results = model.predict(frame, imgsz=320, conf=0.3, verbose=False)
    
    # 3. 빈 검정색 이미지 생성 (바이너리 마스크용)
    h, w = frame.shape[:2]
    lane_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 4. 차선 검출 결과가 있다면 마스킹 수행
    if results[0].masks is not None:
        # 모든 차선 마스크를 하나로 합침
        for mask in results[0].masks.data:
            m = mask.cpu().numpy()
            m = cv2.resize(m, (w, h))
            lane_mask[m > 0.5] = 255 # 차선 부분만 흰색(255)으로 채움
            
    return lane_mask # <-- 최종 바이너리 이미지 리턴

# OpenCV 메인 루프
cap = cv2.VideoCapture('road_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    binary_img = get_binary_lane(frame) # 목표 달성
    
    cv2.imshow('Binary Lane', binary_img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break