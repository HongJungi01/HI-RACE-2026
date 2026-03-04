import cv2
import numpy as np

def inverse_birdseye_from_ni_data(bev_image_path):
    # 1. 일반 뷰(원본 카메라)의 픽셀 좌표 (6개의 점)
    # 줄바꿈이 정상적으로 적용된 좌표 배열입니다.
    pts_normal = np.array([[166, 172],  # Point 1
        [480, 175],  # Point 2
        [448, 130],  # Point 3[426, 100],  # Point 4
        [193, 129],  # Point 5
        [320, 101]   # Point 6
    ], dtype=np.float32)

    # 2. 실제 세계의 미터(m) 좌표 (6개의 점)
    pts_world = np.array([
        [0.0,  1.0],   # Point 1
        [1.5,  1.0],   # Point 2
        [1.5,  0.5],   # Point 3[1.5,  0.0],   # Point 4
        [0.0,  0.5],   # Point 5
        [0.75, 0.0]    # Point 6
    ], dtype=np.float32)

    # 3. 세계 좌표(m)를 버드뷰 이미지 픽셀 좌표로 변환
    scale = 200      
    offset_x = 100   
    offset_y = 100   

    pts_bev = np.zeros_like(pts_world)
    pts_bev[:, 0] = (pts_world[:, 0] * scale) + offset_x
    pts_bev[:, 1] = (pts_world[:, 1] * scale) + offset_y

    # 4. 역변환 행렬(Homography) 계산
    # 이제 양쪽 모두 정확히 6개의 점이 매칭되므로 에러가 발생하지 않습니다.
    M_inverse, status = cv2.findHomography(pts_bev, pts_normal, cv2.RANSAC)
    
    print("--- 계산된 역투시 변환 행렬 ---")
    print(M_inverse)

    # 5. 이미지 불러오기
    bev_img = cv2.imread(bev_image_path)
    if bev_img is None:
        print(f"[{bev_image_path}] 이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
        return

    # 6. 역 원근 변환 적용 (원본 해상도 640x310으로 복원)
    original_width = 2560
    original_height = 1440
    restored_img = cv2.warpPerspective(bev_img, M_inverse, (original_width, original_height))

    # 7. 결과 출력
    cv2.imshow("Input: Bird's Eye View", bev_img)
    cv2.imshow("Output: Restored Normal View", restored_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# 실행 부분: 테스트할 버드뷰 이미지 경로를 입력하세요.
# ==========================================
image_file_name = r"C:\Users\JungiHong\OneDrive\Hongik\HI-RACE\2026-01\platform_1_5\Competition\LaneExtractor\sample_image.png"  # 여기에 실제 파일명(예: "my_image.png")을 적어주세요.
inverse_birdseye_from_ni_data(image_file_name)