// ================================================================
// 구동 + 조향 + 브레이크 통합 제어 (타이밍 기반 하이브리드 AUTO / RC 수동 겸용)
//
// combined.ino 기반 변형. AUTO 모드 제어 로직만 교체:
//   - 방향: 어느 센서가 켜졌나(위치) → 영역별 베이스 조향각으로 커브 유지
//   - 크기: 중앙→옆 센서 전환 시간(미감지 갭 통과 시간) → 짧을수록 크게 꺾음
//   - 속도 플래닝: 직선(에러無) 가속 50→150(2s), 코너(에러有) 감속 →50(1s)
//
// 설계 가정: 라인 물리폭 30mm이나 센서 유효 감지폭 ≈15mm → A3↔A2(18mm) 사이
//   ~3mm, A2↔A1(29mm) 사이 ~14mm의 미감지 갭이 전환 시 발생. 이 갭 통과 시간으로
//   라인의 횡방향 속도를 측정한다.
//
// 핀 배치 (combined.ino와 동일):
//   조향 서보   → Pin 9  (Servo, 중립 110°, 범위 75~145°)
//   브레이크 서보 → Pin 10 (Servo, 해제 0°, 제동 70°)
//   구동 PWM    → Pin 11 (analogWrite)
//   구동 DIR    → Pin 12 (digital, 0=전진, 1=후진)
//   엔코더 A/B  → Pin 3 / Pin 2 (인터럽트)
//   RC 조향     → Pin 4 (PCINT, 1000~2000μs → 서보 75~145°)
//   RC 구동     → Pin 5 (PCINT)
//   RC auto     → Pin 6 (PCINT, >1500μs=AUTO, ≤1500μs=RC수동)
//
// 안전 알고리즘:
//   브레이크 >= 20 → 구동 PWM 강제 0
//   라인 완전 로스트(미감지 GAP_TIMEOUT 초과) → 마지막 방향으로 최대 조향 홀딩
//   AUTO 모드 중에도 D5(스로틀) < 1300μs → 강제 브레이크 모드(제동, PWM 0)
// ================================================================

#include <Servo.h>

// --- 핀 번호 ---
const int STEER_PIN = 9;
const int BRAKE_PIN = 10;
const int RMS = 11;   // 구동 모터 PWM
const int RMD = 12;   // 구동 모터 DIR

// --- 엔코더 핀 ---
const int ENC_A = 3;
const int ENC_B = 2;
volatile long EncoderCount = 0;
long EncoderCount_prev = 0;

// --- RC 수신기 핀 (PCINT) ---
const int RC_PINS[3]  = {4, 5, 6};  // 0=조향, 1=구동, 2=auto
volatile unsigned long rc_timer_start[3] = {0, 0, 0};
volatile int           rc_pulse[3]       = {1500, 1500, 1000};
volatile byte          rc_last_state[3]  = {0, 0, 0};

// --- 모드 플래그 ---
bool autoMode = false;

// --- 서보 ---
Servo steerServo;
Servo brakeServo;

// --- 제어 입력 ---
int steerAngle = 110;   // 중립
int brakeAngle = 0;     // 해제 상태
int pwmCmd = 0;         // 0 ~ 255 (전진)

// --- 타이밍 ---
unsigned long lastFeedback = 0;
const unsigned long FEEDBACK_INTERVAL = 50;

// ================================================================
// 라인 트레이서 파라미터
// ================================================================
// 센서 위치 가중치 [mm, design.md 실측, 우측 양수]
// 인덱스: 0=A1(우 +47), 1=A2(+18), 2=A3(중앙 0), 3=A4(-18), 4=A5(좌 -47)
const float SENSOR_W[5] = { 47.0, 18.0, 0.0, -18.0, -47.0 };
const int   IR_ACTIVE_MIN = 100;   // (1023-raw) 이 값 이상이어야 "라인 감지"
const int   SUM_MIN       = 150;   // 합이 이보다 작으면 미감지(갭)

// 조향 범위
const int STEER_NEUTRAL = 110;
const int STEER_MIN     = 75;
const int STEER_MAX     = 145;

// 영역(region) 분류 경계 [mm] (centroid error 절댓값 기준)
const float TH_CENTER = 9.0;    // |e| <= 9     → CENTER (A3)
const float TH_R1     = 32.0;   // 9 < |e| <=32 → R1 (A2/A4), >32 → R2 (A1/A5)

// 영역별 위치 베이스 조향 오프셋 [deg] (정상상태 커브 유지용)
const int OFF_R1 = 15;   // R1: 110±15
const int OFF_R2 = 30;   // R2: 110±30 (75~145 내)

// 타이밍항 (전환 속도 → 조향 크기)
const float Kv          = 0.10;   // [deg per (mm/s)] 횡속도 → 조향 보정
const float DEADBAND_01 = 3.0;    // CENTER↔R1 미감지 갭 폭 [mm] (18 - 15)
const float DEADBAND_12 = 14.0;   // R1↔R2 미감지 갭 폭 [mm] (29 - 15)
const float MIN_GAP_SEC = 0.002;  // 갭 시간 하한(=속도 상한 방지) [s]
const float MAX_TIMING  = 35.0;   // timingMag 클램프 [deg]
const float TIMING_DECAY = 60.0;  // 같은 영역 체류 시 timingMag decay [deg/s]

// 라인 로스트 판정
const unsigned long GAP_TIMEOUT_US = 80000UL;  // 80ms 이상 미감지 → 로스트
const int MAX_LOCK = 35;   // 로스트 시 최대 조향 오프셋 [deg]

// 속도 플래닝
const float PWM_MIN    = 50.0;    // 코너/출발
const float PWM_MAX    = 150.0;   // 직선 최고속
const float ACCEL_RATE = (PWM_MAX - PWM_MIN) / 2.0;  // 50→150을 2초 = 50 PWM/s
const float DECEL_RATE = (PWM_MAX - PWM_MIN) / 1.0;  // 150→50을 1초 = 100 PWM/s

// 오실레이션 억제 (방향 반전 감지 → 감쇠 + 속도 즉시 드롭)
const unsigned long OSC_WINDOW_US = 250000UL;  // 반대 방향 전환이 이 안에 오면 wobble
const unsigned long OSC_HOLD_US   = 400000UL;  // 반전 감지 후 감쇠/속도드롭 유지 시간
const float         OSC_DAMP      = 0.25;      // wobble 시 조향 보정 감쇠 계수

// --- 영역 정의 ---
enum Region { R_CENTER, R_R1, R_R2, R_L1, R_L2, R_GAP, R_LOST };

// --- 라인 트레이서 상태 ---
float         currentPwm     = PWM_MIN;   // 램핑되는 구동 PWM
float         timingMag      = 0.0;       // 현재 타이밍 보정 크기 [deg]
int           dirLast        = 0;         // 마지막 비중립 방향 (+1 우 / -1 좌)
Region        lastSolidRegion = R_CENTER; // 마지막으로 센서가 감지된 영역
bool          inGap          = false;     // 현재 미감지(갭) 중인가
unsigned long tEnterGap      = 0;         // 갭 진입 시각 [us]
Region        regionBeforeGap = R_CENTER; // 갭 직전 영역
unsigned long lastUpdateUs   = 0;         // dt 계산용

// 오실레이션 억제 상태
int           lastOutwardDir  = 0;        // 직전 바깥쪽 전환 방향 (+1/-1)
unsigned long lastOutwardTime = 0;        // 직전 바깥쪽 전환 시각 [us]
int           oscCount        = 0;        // 최근 방향 반전 누적 횟수
unsigned long lastReversalTime = 0;       // 마지막 방향 반전 시각 [us]

// 디버그 출력용
Region dbgRegion  = R_CENTER;
float  dbgDtMs    = 0.0;
float  dbgSpeed   = 0.0;
bool   dbgOsc     = false;

// 영역 → 레벨 (부호=방향, 절댓값=중앙에서의 거리 단계)
int regionLevel(Region r) {
  switch (r) {
    case R_R1: return  1;
    case R_R2: return  2;
    case R_L1: return -1;
    case R_L2: return -2;
    default:   return  0;   // CENTER
  }
}

// 영역 → 위치 베이스 조향 오프셋 [deg]
int regionOffset(Region r) {
  switch (r) {
    case R_R1: return  OFF_R1;
    case R_R2: return  OFF_R2;
    case R_L1: return -OFF_R1;
    case R_L2: return -OFF_R2;
    default:   return  0;
  }
}

void setup() {
  Serial.begin(115200);

  // 구동 모터
  pinMode(RMS, OUTPUT);
  pinMode(RMD, OUTPUT);
  analogWrite(RMS, 0);
  digitalWrite(RMD, 0);

  // 서보 초기화
  steerServo.attach(STEER_PIN);
  brakeServo.attach(BRAKE_PIN);
  steerServo.write(steerAngle);
  brakeServo.write(brakeAngle);

  // 엔코더
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), ISR_Encoder, RISING);

  // RC 수신기 (PCINT, Port D: D4=PCINT20, D5=PCINT21, D6=PCINT22)
  for (int i = 0; i < 3; i++) {
    pinMode(RC_PINS[i], INPUT_PULLUP);
    rc_last_state[i] = digitalRead(RC_PINS[i]);
  }
  noInterrupts();
  PCICR  |= (1 << PCIE2);
  PCMSK2 |= (1 << PCINT20) | (1 << PCINT21) | (1 << PCINT22);
  interrupts();

  lastUpdateUs = micros();
  delay(300);
  Serial.println("COMBINED_TIMING READY");
}

void loop() {
  // --- RC auto 채널로 모드 결정 ---
  noInterrupts();
  int autoPulse  = rc_pulse[2];
  int drivePulse = rc_pulse[1];   // D5 스로틀 채널 (AUTO 브레이크 오버라이드용)
  interrupts();
  autoMode = (autoPulse > 1500);

  if (autoMode) {
    if (drivePulse < 1300) {
      autoBrake();            // AUTO 중에도 D5<1300 → 강제 브레이크 모드
    } else {
      runLineTracer();        // 타이밍 기반 하이브리드 라인 트레이싱
    }
  } else {
    applyRCInput();           // RC 입력 직접 제어
  }

  // --- 브레이크 인터록: brake >= 20 → PWM 강제 0 ---
  int effectivePwm = pwmCmd;
  if (brakeAngle >= 20) {
    effectivePwm = 0;
  }

  // --- 모터 구동 ---
  driveMotor(effectivePwm);

  // --- 피드백 전송 (50ms 주기) ---
  if (millis() - lastFeedback >= FEEDBACK_INTERVAL) {
    lastFeedback = millis();
    sendFeedback();
  }

  delay(1);
}

// ================================================================
// 라인 트레이서: 타이밍 기반 하이브리드
//   1) 5개 IR → centroid error → 영역(region) 분류
//   2) 미감지 갭 통과 시간으로 횡속도 측정 → timingMag (크기)
//   3) steerAngle = 중립 + 위치베이스(방향/유지) + dir·timingMag(진입 가속)
//      좌우 번갈아 흔들림(오실레이션) 감지 시 보정 전체를 OSC_DAMP로 감쇠
//   4) 속도 플래닝: 오실레이션 → 즉시 50 / 감지 → 가속 / 미감지 → 서서히 감속
// ================================================================
void runLineTracer() {
  // dt (실시간, 초)
  unsigned long now = micros();
  float dt = (now - lastUpdateUs) * 1e-6f;
  if (dt < 0 || dt > 0.05f) dt = 0.001f;   // 비정상값 방어
  lastUpdateUs = now;

  // 1) 센서 읽기 → centroid error
  int raw[5] = {
    analogRead(A1), analogRead(A2), analogRead(A3),
    analogRead(A4), analogRead(A5)
  };
  long weightedSum = 0;
  long sum = 0;
  for (int i = 0; i < 5; i++) {
    int s = 1023 - raw[i];
    if (s < IR_ACTIVE_MIN) s = 0;          // 약한 반사 컷오프
    weightedSum += (long)(SENSOR_W[i] * s);
    sum += s;
  }
  bool detected = (sum >= SUM_MIN);
  float error = detected ? ((float)weightedSum / (float)sum) : 0.0f;

  // 2) 영역 분류 + 갭/로스트 처리
  Region region;
  if (detected) {
    float ae = fabs(error);
    Region solid;
    if (ae <= TH_CENTER)      solid = R_CENTER;
    else if (ae <= TH_R1)     solid = (error > 0) ? R_R1 : R_L1;
    else                      solid = (error > 0) ? R_R2 : R_L2;

    // 갭에서 막 빠져나옴 → 전환 완료: 갭 통과 시간으로 횡속도 산출
    if (inGap) {
      handleTransition(regionBeforeGap, solid, now - tEnterGap, now);
      inGap = false;
    }
    region = solid;
    lastSolidRegion = solid;
  } else {
    // 미감지: 갭 진입/지속
    if (!inGap) {
      inGap = true;
      tEnterGap = now;
      regionBeforeGap = lastSolidRegion;
    }
    region = (now - tEnterGap >= GAP_TIMEOUT_US) ? R_LOST : R_GAP;
  }
  dbgRegion = region;

  // 오실레이션(좌우 번갈아 흔들림) 활성 여부 — 최근 방향 반전 후 OSC_HOLD 동안 유지
  bool oscActive = (lastReversalTime != 0) && (now - lastReversalTime < OSC_HOLD_US);
  dbgOsc = oscActive;

  // 3) 조향 결정
  if (region == R_LOST) {
    // 완전 로스트: 마지막 방향으로 최대 조향 홀딩 (복귀 유도)
    int dir = (dirLast != 0) ? dirLast : 1;
    steerAngle = constrain(STEER_NEUTRAL + dir * MAX_LOCK, STEER_MIN, STEER_MAX);
  } else if (region == R_GAP) {
    // 전환 중(갭): 직전 steerAngle 그대로 홀딩
    // (steerAngle 유지)
  } else {
    int lvl = regionLevel(region);
    int dir = (lvl > 0) ? 1 : (lvl < 0 ? -1 : 0);
    if (dir != 0) dirLast = dir;

    int   posBase    = regionOffset(region);             // 위치(방향+유지)
    float correction = posBase + dir * timingMag;        // +타이밍(크기)
    // wobble 감지 시 조향 보정 전체를 감쇠 → 완만하게 (진짜 코너는 oscActive=false)
    float steerDamp  = oscActive ? OSC_DAMP : 1.0f;
    float angle      = STEER_NEUTRAL + steerDamp * correction;
    steerAngle = constrain((int)angle, STEER_MIN, STEER_MAX);

    // 같은 영역 체류 시 타이밍 보정은 서서히 소멸 → 위치베이스가 커브 유지
    timingMag -= TIMING_DECAY * dt;
    if (timingMag < 0) timingMag = 0;
  }
  steerServo.write(steerAngle);

  // 4) 속도 플래닝 (우선순위: 오실레이션 > 감지 > 미감지)
  if (oscActive) {
    // 오실레이션 감지 시에만 즉시 50으로 드롭·유지
    currentPwm = PWM_MIN;
  } else {
    // 그 외: 감지 중이면 가속, 일반 미감지(GAP/LOST)면 서서히 감속
    bool detectedRegion = (region != R_GAP && region != R_LOST);
    if (detectedRegion) {
      currentPwm += ACCEL_RATE * dt;
    } else {
      currentPwm -= DECEL_RATE * dt;
    }
  }
  if (currentPwm > PWM_MAX) currentPwm = PWM_MAX;
  if (currentPwm < PWM_MIN) currentPwm = PWM_MIN;
  pwmCmd = (int)currentPwm;

  // 브레이크 해제
  brakeAngle = 0;
  brakeServo.write(brakeAngle);
}

// ================================================================
// AUTO 브레이크 오버라이드 (D5 스로틀 채널 < 1300)
//   AUTO 모드 중에도 RC 스로틀을 후방으로 내리면 즉시 제동.
//   조향은 마지막 값 유지. 해제(>=1300) 시 라인트레이서가 50에서 부드럽게 재개되도록
//   속도/타이밍/오실레이션 상태를 초기화한다.
// ================================================================
void autoBrake() {
  pwmCmd = 0;
  brakeAngle = 70;
  brakeServo.write(brakeAngle);

  currentPwm = PWM_MIN;
  timingMag = 0;
  inGap = false;
  lastOutwardDir = 0;
  oscCount = 0;
  lastReversalTime = 0;
}

// ================================================================
// 전환 완료 처리: 갭 직전 영역(src) → 갭 직후 영역(dst), 갭 통과 시간(us)
//   바깥쪽으로 한 단계 이상 이동한 경우에만 횡속도 기반 timingMag 갱신
// ================================================================
void handleTransition(Region src, Region dst, unsigned long gapUs, unsigned long now) {
  int srcLvl = regionLevel(src);
  int dstLvl = regionLevel(dst);

  // 바깥쪽 이동(중앙에서 멀어짐)일 때만 타이밍 보정 산출
  if (abs(dstLvl) > abs(srcLvl)) {
    int d = (dstLvl > 0) ? 1 : -1;

    // --- 방향 반전(오실레이션) 감지 ---
    unsigned long since = now - lastOutwardTime;
    if (since >= OSC_WINDOW_US) {
      oscCount = 0;                 // 조용한 구간 경과 → 새로 시작
    }
    if (lastOutwardDir != 0 && d != lastOutwardDir && since < OSC_WINDOW_US) {
      if (oscCount < 4) oscCount++; // 반대 방향 전환 = wobble
      lastReversalTime = now;
    } else if (d == lastOutwardDir) {
      oscCount = 0;                 // 같은 방향 연속 = 진짜 코너
    }
    lastOutwardDir  = d;
    lastOutwardTime = now;

    // --- 갭 통과 시간 → 횡속도 → timingMag (감쇠는 출력단에서 oscActive로 적용) ---
    // 통과한 갭 폭: 0↔1 구간이면 DEADBAND_01, 1↔2 구간이면 DEADBAND_12
    float deadband = (abs(dstLvl) >= 2) ? DEADBAND_12 : DEADBAND_01;
    float gapSec = gapUs * 1e-6f;
    if (gapSec < MIN_GAP_SEC) gapSec = MIN_GAP_SEC;   // 속도 상한 방지

    float speed = deadband / gapSec;                  // 횡속도 [mm/s]
    float mag = Kv * speed;
    if (mag > MAX_TIMING) mag = MAX_TIMING;
    timingMag = mag;

    dbgDtMs = gapUs / 1000.0f;
    dbgSpeed = speed;
  }
  // 안쪽으로 복귀하는 전환은 timingMag를 건드리지 않고 decay에 맡김
}

// ================================================================
// RC 입력 → 제어값 변환 (RC 수동 모드 전용)
//   조향: 1000~2000μs → 서보 75~145°
//   구동: 1600~2000μs → PWM   0~255  (전진)
//         1550~1600μs → 정지
//         1450~1550μs → 브레이크 ON  (70° 고정)
//         1400~1450μs → 정지
//         1000~1400μs → PWM -255~0   (후진)
// ================================================================
void applyRCInput() {
  noInterrupts();
  int steerPulse = rc_pulse[0];
  int drivePulse = rc_pulse[1];
  interrupts();

  steerAngle = (int)map(constrain(steerPulse, 1000, 2000), 1000, 2000, 75, 145);
  steerServo.write(steerAngle);

  if (drivePulse >= 1600) {
    pwmCmd     = (int)map(constrain(drivePulse, 1600, 2000), 1600, 2000, 0, 255);
    brakeAngle = 0;
    brakeServo.write(brakeAngle);
  } else if (drivePulse >= 1550) {
    pwmCmd     = 0;
    brakeAngle = 0;
    brakeServo.write(brakeAngle);
  } else if (drivePulse >= 1450) {
    pwmCmd     = 0;
    brakeAngle = 70;
    brakeServo.write(brakeAngle);
  } else if (drivePulse >= 1400) {
    pwmCmd     = 0;
    brakeAngle = 0;
    brakeServo.write(brakeAngle);
  } else {
    pwmCmd     = (int)map(constrain(drivePulse, 1000, 1400), 1000, 1400, -255, 0);
    brakeAngle = 0;
    brakeServo.write(brakeAngle);
  }

  // RC 수동에서는 속도 플래닝/오실레이션 상태 초기화(다음 AUTO 진입 시 부드럽게 시작)
  currentPwm = PWM_MIN;
  timingMag = 0;
  inGap = false;
  lastOutwardDir = 0;
  oscCount = 0;
  lastReversalTime = 0;
}

// ================================================================
// 단순 PWM 모터 구동
// ================================================================
void driveMotor(int pwm) {
  if (pwm > 0) {
    analogWrite(RMS, pwm);
    digitalWrite(RMD, 0);   // 전진
  } else if (pwm < 0) {
    analogWrite(RMS, -pwm);
    digitalWrite(RMD, 1);   // 후진
  } else {
    analogWrite(RMS, 0);
    digitalWrite(RMD, 0);
  }
}

// ================================================================
// 시리얼 피드백 (튜닝용)
//   R=<region> dt=<전환시간ms> v=<횡속도mm/s> T=<timingMag> S=<조향> P=<PWM>
// ================================================================
void sendFeedback() {
  noInterrupts();
  long enc = EncoderCount;
  interrupts();
  EncoderCount_prev = enc;

  Serial.print("R=");
  Serial.print(regionName(dbgRegion));
  Serial.print(" dt=");
  Serial.print(dbgDtMs, 1);
  Serial.print(" v=");
  Serial.print(dbgSpeed, 0);
  Serial.print(" T=");
  Serial.print(timingMag, 1);
  Serial.print(" S=");
  Serial.print(steerAngle);
  Serial.print(" P=");
  Serial.print(pwmCmd);
  Serial.print(" O=");
  Serial.println(dbgOsc ? 1 : 0);
}

const char* regionName(Region r) {
  switch (r) {
    case R_CENTER: return "C";
    case R_R1:     return "R1";
    case R_R2:     return "R2";
    case R_L1:     return "L1";
    case R_L2:     return "L2";
    case R_GAP:    return "GAP";
    case R_LOST:   return "LOST";
    default:       return "?";
  }
}

// ================================================================
// 엔코더 ISR (A채널 RISING, B로 방향 판별)
// ================================================================
void ISR_Encoder() {
  if (digitalRead(ENC_B) == LOW) {
    EncoderCount++;
  } else {
    EncoderCount--;
  }
}

// ================================================================
// RC 수신기 PCINT ISR (D4=조향, D5=구동, D6=auto)
// ================================================================
ISR(PCINT2_vect) {
  unsigned long now = micros();
  for (int i = 0; i < 3; i++) {
    byte state = digitalRead(RC_PINS[i]);
    if (state != rc_last_state[i]) {
      if (state == HIGH) {
        rc_timer_start[i] = now;
      } else {
        if (rc_timer_start[i] != 0) {
          rc_pulse[i] = (int)(now - rc_timer_start[i]);
        }
      }
      rc_last_state[i] = state;
    }
  }
}
