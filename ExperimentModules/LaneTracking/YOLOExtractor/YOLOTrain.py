# train.py 예시
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # 가장 가벼운 Nano 모델로 시작
model.train(
    data='lane_data.yaml',    # 데이터셋 경로 설정 파일
    epochs=100,               # 학습 횟수
    imgsz=320,                # 중요: i5를 위해 해상도를 320~416으로 낮춰 학습
    device=0                  # RTX 4080(GPU) 사용
)