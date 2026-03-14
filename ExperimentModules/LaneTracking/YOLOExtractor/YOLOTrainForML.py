from ultralytics import YOLO
import os

def main():
    # 1. 고정밀 모델 로드 (가장 강력한 x 모델 사용)
    model = YOLO('yolov8x-seg.pt')

    # 2. 고정밀 학습 시작 (RTX 4080 성능 활용)
    model.train(
        data=r'C:\Users\JungiHong\OneDrive\Hongik\HI-RACE\2026-01\ExperimentModules\LaneTracking\YOLOExtractor\LaneProject\data.yaml',
        epochs=300,
        imgsz=640,          # 데이터셋의 실제 해상도가 더 높다면 1024도 고려해 보세요.
        batch=8,            # VRAM 16GB 기준, 메모리 부족 시 4로 낮추세요.
        device=0,
        workers=8,
        patience=50,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        overlap_mask=True,
        mask_ratio=1,
        project='LaneMLProject', # 결과 저장을 위한 프로젝트 이름
        name='high_precision_x'  # 실행 시도 이름
    )

    # 3. 학습 완료 후 최상의 모델 경로 출력
    # 학습이 정상 종료되면 model.trainer.best에 경로가 담깁니다.
    try:
        best_model_path = model.trainer.best
        print(f"정밀 학습 완료! 모델 위치: {best_model_path}")
    except Exception as e:
        print(f"학습 완료 후 경로를 찾을 수 없습니다: {e}")

if __name__ == '__main__':
    main()