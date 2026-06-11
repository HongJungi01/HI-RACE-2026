"""
캘리브레이션 포인트 클릭 도구
- 이미지 위에서 노트의 4 꼭짓점을 순서대로 클릭
- 순서: ① 좌하 ② 우하 ③ 우상 ④ 좌상
- 4점 선택 후 좌표를 출력하고 ESC로 종료
"""
import cv2
import sys
import numpy as np

points = []
img = None
display = None

def draw(img_base):
    vis = img_base.copy()
    labels = ["BL(좌하)", "BR(우하)", "TR(우상)", "TL(좌상)"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    for i, pt in enumerate(points):
        cv2.circle(vis, pt, 5, colors[i], -1)
        cv2.putText(vis, f"{labels[i]} {pt}", (pt[0]+8, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i], 1)
    if len(points) > 1:
        for i in range(len(points)):
            j = (i + 1) % len(points)
            if j < len(points):
                cv2.line(vis, points[i], points[j], (200, 200, 200), 1)
    return vis

def on_mouse(event, x, y, flags, param):
    global display
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        display = draw(img)
        cv2.imshow("Calibration - Click 4 corners", display)
        if len(points) == 4:
            print("\n=== 선택된 캘리브레이션 포인트 ===")
            labels = ["좌하(BL)", "우하(BR)", "우상(TR)", "좌상(TL)"]
            for i, pt in enumerate(points):
                print(f"  {labels[i]}: [{pt[0]}, {pt[1]}]")
            print("\n_CALIB_PIXEL_PTS 형식:")
            print(f"_CALIB_PIXEL_PTS = np.array([")
            for pt in points:
                print(f"    [{pt[0]}, {pt[1]}],")
            print(f"], dtype=np.float32)")
            print("\n아무 키를 누르면 종료합니다. 다시 하려면 'r'을 누르세요.")
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()
        display = draw(img)
        cv2.imshow("Calibration - Click 4 corners", display)

def main():
    global img, display, points

    if len(sys.argv) < 2:
        print("사용법: python calibrate_click.py <이미지 경로>")
        print("예: python calibrate_click.py calib_image.jpg")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"이미지를 열 수 없습니다: {path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"이미지 크기: {w}x{h}")
    print("순서: ① 좌하 → ② 우하 → ③ 우상 → ④ 좌상")
    print("좌클릭: 포인트 추가 | 우클릭: 마지막 포인트 삭제 | r: 리셋 | ESC: 종료")

    display = img.copy()
    cv2.imshow("Calibration - Click 4 corners", display)
    cv2.setMouseCallback("Calibration - Click 4 corners", on_mouse)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            points.clear()
            display = img.copy()
            cv2.imshow("Calibration - Click 4 corners", display)
            print("리셋됨. 다시 클릭하세요.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
