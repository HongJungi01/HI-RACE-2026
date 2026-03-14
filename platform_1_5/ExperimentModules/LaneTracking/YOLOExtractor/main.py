import array2Image as a2i
import YOLOInference
import laneDetermine as lD
import cv2
import numpy as np
import os

# DebugImageFolder 경로 설정
debug_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DebugImageFolder"))

if not os.path.exists(debug_dir):
    os.makedirs(debug_dir)
    
def main(Red_2D_array, Green_2D_array, Blue_2D_array, min_area, min_span, max_rmse, poly_degree=2):
        # 1. BGR 이미지 통합 (원본 해상도)
        bgr_image = a2i.merge_rgb_to_bgr(Red_2D_array, Green_2D_array, Blue_2D_array)
        
        # 2. YOLO 추론
        binary_mask_before_preprocessing = YOLOInference.extract_lane_binary(bgr_image)

        # 3. ROI 크롭 및 리사이즈 (y=79 ~ y=230 영역을 310px로 확장)
        # 차선이 집중된 영역을 캘리브레이션 기준인 310 높이에 맞춤
        roi_img = binary_mask_before_preprocessing[79:230, :] 
        binary_mask = cv2.resize(roi_img, (640, 310), interpolation=cv2.INTER_LINEAR)
        
        # 4. 결과 저장
        YOLO_lane_binary_path = os.path.join(debug_dir, "YOLO_lane_binary.png")
        cv2.imwrite(YOLO_lane_binary_path, binary_mask)

        # 5. 차선 검증 (바이너리 마스크 반환으로 변경)
        filtered_mask = lD.filter_lane_candidates(binary_mask, min_area, min_span, max_rmse, poly_degree)

        # 6. 좌/우 차선 분류
        left_mask, right_mask, state = lD.classify_left_right(filtered_mask)

        # 7. 디버그 이미지 저장
        left_lane_path = os.path.join(debug_dir, "left_lane.png")
        right_lane_path = os.path.join(debug_dir, "right_lane.png")
        cv2.imwrite(left_lane_path, left_mask)
        cv2.imwrite(right_lane_path, right_mask)

        # 8. 중심선 계산 (호모그래피 기반 월드 좌표)
        cx, cy = lD.calculate_center_line(left_mask, right_mask, state)

        # 9. CTE / Heading Error 계산
        cte, heading = lD.calculate_stanley_error(cx, cy)

        # 10. 디버그 이미지 저장
        debug_img = lD.draw_path_on_image(bgr_image, cx, cy, left_mask, right_mask, state)
        debug_img_path = os.path.join(debug_dir, "debug_image.png")
        cv2.imwrite(debug_img_path, debug_img)

        # [CTE(m), HeadingError(rad), left_lane_path, right_lane_path, debug_img_path]
        return [str(heading), str(cte), left_lane_path, right_lane_path, debug_img_path]