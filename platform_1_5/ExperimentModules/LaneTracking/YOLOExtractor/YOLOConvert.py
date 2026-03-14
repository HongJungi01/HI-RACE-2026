# export.py
from ultralytics import YOLO

model = YOLO('best.pt') # 학습 완료된 모델
model.export(format='openvino', imgsz=320, half=True) # FP16 혹은 INT8로 변환