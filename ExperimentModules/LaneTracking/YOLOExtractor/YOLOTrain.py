from ultralytics import YOLO
import os

def main():
    # 1. 모델 로드 (가벼운 Nano 모델)
    model = YOLO('yolov8n-seg.pt')

    # 2. 학습 시작 (4080 성능을 활용!)
    model.train(
        data='C:\\Users\\JungiHong\\OneDrive\\Hongik\\HI-RACE\\2026-01\\ExperimentModules\\LaneTracking\\YOLOExtractor\\LaneProject\\data.yaml', # yaml 파일 경로
        epochs=100,                      # 학습 횟수
        imgsz=320,                       # i5를 위한 해상도
        device=0,                        # 4080 GPU 사용
        workers=8                        # CPU 코어 수에 맞춰 조절
    )

    # 3. 학습 완료 후 최신 모델(.pt) 찾기
    # 보통 runs/segment/trainX/weights/best.pt에 저장됨
    best_model_path = model.trainer.best
    print(f"학습 완료! 모델 위치: {best_model_path}")

    # 4. OpenVINO로 변환 (i5 CPU용)
    print("OpenVINO 변환 중...")
    model.export(format='openvino', imgsz=320, half=True)
    print("변환 완료!")

if __name__ == '__main__':
    main()