# 구현 목표: 
- 기존 Stanley Method에서, 곡선 주행 시 경로 이탈 현상을 개선하기 위해 경로를 예측하여 주행하는 feedforward 제어를 구현하고, feedforward 제어 시 발생하는 노이즈를 줄이기 위해 LPF를 도입한다.
## 목표 계산식: 
$$\delta_{final}[t] = \underbrace{(\psi_{ref} - \psi)}_{\text{1. Heading Error}} + \underbrace{\arctan\left(\frac{k \cdot e}{v}\right)}_{\text{2. Cross-Track Error}} + \underbrace{\Big[ \alpha \cdot \arctan(L \cdot \kappa) + (1 - \alpha) \cdot \delta_{ff}[t-1] \Big]}_{\text{3. Filtered Feedforward}}$$
## Heading Error
기존 코드로 구현 완료
## Cross-Track Error
기존 코드로 구현 완료
## Filtered Feedforward
### 필요 파라미터: 
- 1. $\alpha$: 필터 계수($0<\alpha<=1$)
	- 작을수록 핸들이 묵직하고 부드러워짐
	- 보통 0.1~0.3(제미나이 피셜)
- 2. L: 차량의 휠베이스
	- HENES: 약 0.7m
- 3. $\kappa$: 목표 지점의 곡률
	- 현재 경로의 $L_d$만큼 앞 부분의 곡률을 계산
	- 이 때 $L_d = idx 5로 지정(기존 Stanley method와 통일해서 일관성 유지)
	- $t_{lookahead}$: 튜닝 파라미터
	- 계산식: 
- $$\kappa = \frac{|x'y'' - y'x''|}{(x'^2 + y'^2)^{\frac{3}{2}}}$$
- 실제 계산식: 매개변수 이용한 변환(더 쉬움)
- 4. $\delta_{ff}[t-1]$: 이전 프레임의 피드포워드 조향각
	- 즉, 이전 프레임 $\delta$ 값을 기억할 수 있도록 해야 함
## 코드 수정 사항: 
### main.py
```python title:"수정사항 1"
    # 7. CTE / Heading Error 계산
    stanley_start_time = time.time()
    cte, heading = lD.calculate_stanley_error(cx, cy)
    stanley_end_time = time.time()

    # --- [20260316추가] 곡률(kappa) 계산 ---
    kappa = lD.calculate_curvature(cx, cy)
```

```python title:"수정사항 2"
   # 20260316수정: kappa 추가 및 인덱스 조정
    return [
        str(heading),           # [0] 헤딩 에러
        str(cte),               # [1] CTE
        str(kappa),             # [2] 곡률 (kappa)
        left_lane_base64,       # [3] 왼쪽 차선 이미지 (Base64)
        right_lane_base64,      # [4] 오른쪽 차선 이미지 (Base64)
        debug_img_base64,       # [5] 전체 디버그 이미지 (Base64)
        f"Total: {total_time_ms:.1f}ms" # [6] 시간 정보
    ]
```
### laneDetermine.py
```python title:"수정사항"
# ============================================================
# Feedforward 제어용 곡률(Kappa) 계산 추가
# ============================================================
def calculate_curvature(center_world_x, center_world_y, H=None):
    """
    미리 계산된 중심선 배열(cx, cy)을 재활용하여 2D 매개변수 곡선의 
    1차, 2차 미분을 통해 전방 경로의 곡률(kappa)을 계산합니다.
    """
    if H is None:
        H = _H
        
    x = np.array(center_world_x, dtype=np.float64)
    y = np.array(center_world_y, dtype=np.float64)
    
    # 점이 부족하여 미분이 불가능한 경우
    if len(x) < 3:
        return 0.0

    # 1. 중심선의 1차 미분 (dx, dy) 및 2차 미분 (ddx, ddy) 계산
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 2. 곡률 계산 공식: (dx*ddy - dy*ddx) / ((dx^2 + dy^2)^(3/2))
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-8 # 0으로 나누기 방지
    kappa_array = (dy * ddx - dx * ddy) / denominator # 차량 제어계 맞춤형 곡률 (우회전이 플러스가 나오도록 부호 반전)

    # 3. Heading Error와 동일하게 차량 앞(Look-ahead) 지점의 곡률 추출
    car_pixel = np.array([[320.0, 310.0]], dtype=np.float32) 
    car_world_x, car_world_y = pixel_to_world(car_pixel, H)
    car_wx, car_wy = float(car_world_x[0]), float(car_world_y[0])

    dist = np.sqrt((x - car_wx)**2 + (y - car_wy)**2)
    idx = int(np.argmin(dist))
    
    look_ahead = 5 # calculate_stanley_error와 동일한 전방 주시 인덱스
    next_idx = min(len(x) - 1, idx + look_ahead)

    # 랩뷰로 전달할 곡률 값 반환
    return float(kappa_array[next_idx])
```
### LaneExtractorHenesUseNetwork.vi
VisionSteering module parameter 추가
### StanleyMethod.vi
parameter 추가