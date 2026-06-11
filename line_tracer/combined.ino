// ================================================================
// 구동 + 조향 + 브레이크 통합 제어 (시리얼 AUTO / RC 수동 겸용)
//
// 핀 배치:
//   조향 서보   → Pin 9  (Servo, 중립 110°, 범위 75~145°)
//   브레이크 서보 → Pin 10 (Servo, 해제 0°, 제동 70°)
//   구동 PWM    → Pin 11 (analogWrite)
//   구동 DIR    → Pin 12 (digital, 0=전진, 1=후진)
//   엔코더 A    → Pin 3 (인터럽트)
//   엔코더 B    → Pin 2 (인터럽트)
//   RC 조향     → Pin 4 (PCINT, 1000~2000μs → 서보 75~145°)
//   RC 구동     → Pin 5 (PCINT, 1600~2000μs → PWM 0~255 전진
//                              1550~1600μs → 데드존(정지)
//                              1450~1550μs → 브레이크 ON(70°)
//                              1400~1450μs → 데드존(정지)
//                              1000~1400μs → PWM -255~0 후진)
//   RC auto     → Pin 6 (PCINT, >1500μs=AUTO, ≤1500μs=RC수동)
//
// 동작 모드:
//   AUTO 모드 (핀6 > 1500μs): 시리얼 명령(S/B/V)으로 제어
//   RC 수동 모드 (핀6 ≤ 1500μs): RC 수신기 입력으로 직접 제어
//
// 시리얼 프로토콜 (115200 baud):
//   수신 (AUTO 모드 전용):
//         S<angle>\n  조향 (75~145)
//         B<angle>\n  브레이크 (0~180)
//         V<pwm>\n    구동 (-255~255)
//   송신 (50ms 주기, 양 모드 공통):
//         E<delta>,C<cumulative>,S<steerAngle>,B<brakeAngle>\n
//
// 안전 알고리즘:
//   브레이크 >= 20 → 구동 PWM 강제 0
//   AUTO 모드에서 명령 타임아웃 500ms → PWM=0, 조향=110(중립)
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
int pwmCmd = 0;         // -255 ~ 255

// --- 타이밍 ---
unsigned long lastFeedback = 0;
const unsigned long FEEDBACK_INTERVAL = 50;
unsigned long lastCommand = 0;
const unsigned long CMD_TIMEOUT = 500;

// --- 시리얼 파싱 ---
char cmdBuf[32];
uint8_t cmdIdx = 0;

// --- 라인 트레이서 ---
const int BASE_PWM     = 40;    // 정속 구동 PWM

// PID 제어 (Pololu QTR 방식: 가중평균 error → PID → 조향 보정)
// 게인 0 → 해당 항 비활성화 (기본 Ki=0이므로 PD 동작)
float Kp = 0.6;
float Ki = 0.0;                 // windup 위험. 사용 시 0.001~0.01부터
float Kd = 2.0;
const float I_LIMIT      = 50.0;   // 적분 windup 한계 (±)
const int   STEER_NEUTRAL = 110;
const int   STEER_MIN     = 75;
const int   STEER_MAX     = 145;

// 센서 위치 가중치 [mm, design.md 실측, 우측 양수]
// 인덱스: 0=A1(우), 1=A2, 2=A3(중앙), 3=A4, 4=A5(좌)
const float SENSOR_W[5] = {  47.0,  18.0,  0.0, -18.0, -47.0 };
const int   IR_ACTIVE_MIN = 100;   // (1023-raw) 이 값 이상이어야 "라인 감지"로 인정
const int   SUM_MIN       = 150;   // 합이 이보다 작으면 미감지(이전 error 유지)

// PID 상태
float pidErrorPrev = 0.0;
float pidIntegral  = 0.0;
float pidErrorDbg  = 0.0;       // 디버그 출력용

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

  delay(300);
  Serial.println("COMBINED READY");
}

void loop() {
  // --- RC auto 채널로 모드 결정 ---
  noInterrupts();
  int autoPulse = rc_pulse[2];
  interrupts();
  autoMode = (autoPulse > 1500);

  if (autoMode) {
    // --- AUTO 모드: IR 센서 기반 라인 트레이싱 ---
    runLineTracer();
    lastCommand = millis();   // CMD_TIMEOUT 방지
  } else {
    // --- RC 수동 모드: RC 입력 적용 ---
    applyRCInput();
    lastCommand = millis();  // 타임아웃 방지
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

    noInterrupts();
    long enc = EncoderCount;
    interrupts();

    int encDelta = (int)(enc - EncoderCount_prev);
    EncoderCount_prev = enc;

    Serial.print("A1=");
    Serial.print(analogRead(A1));
    Serial.print(" A2=");
    Serial.print(analogRead(A2));
    Serial.print(" A3=");
    Serial.print(analogRead(A3));
    Serial.print(" A4=");
    Serial.print(analogRead(A4));
    Serial.print(" A5=");
    Serial.print(analogRead(A5));
    Serial.print(" | S=");
    Serial.print(steerAngle);
    Serial.print(" E=");
    Serial.println(pidErrorDbg, 2);
  }

  delay(1);
}

// ================================================================
// 라인 트레이서: PID 제어 (Pololu QTR 표준 방식)
//   1) 5개 IR 센서를 아날로그값으로 읽어 라인 강도 산출
//   2) 센서 위치(mm) 가중평균으로 연속 error 계산 (우측 양수)
//   3) PID 보정량 = Kp·e + Ki·∫e + Kd·de
//   4) steerAngle = 110(중립) + 보정량, 75~145로 clamp
//
//   미감지(전 센서 흰색): 이전 error 유지 → 적분 동결, derivative=0
// ================================================================
void runLineTracer() {
  // 1) 아날로그 → 라인 강도 (검정에 가까울수록 큰 값)
  int raw[5] = {
    analogRead(A1), analogRead(A2), analogRead(A3),
    analogRead(A4), analogRead(A5)
  };
  long weightedSum = 0;
  long sum = 0;
  for (int i = 0; i < 5; i++) {
    int s = 1023 - raw[i];
    if (s < IR_ACTIVE_MIN) s = 0;   // 약한 반사 컷오프 (노이즈 억제)
    weightedSum += (long)(SENSOR_W[i] * s);
    sum += s;
  }

  // 2) error 산출 (미감지면 이전 error 유지)
  float error;
  bool detected = (sum >= SUM_MIN);
  if (detected) {
    error = (float)weightedSum / (float)sum;  // 범위 약 -47 ~ +47
  } else {
    error = pidErrorPrev;
  }
  pidErrorDbg = error;

  // 3) PID 항 계산
  float dError = detected ? (error - pidErrorPrev) : 0.0;  // 미감지 시 미분 0
  if (detected) {
    pidIntegral += error;
    if (pidIntegral >  I_LIMIT) pidIntegral =  I_LIMIT;    // anti-windup
    if (pidIntegral < -I_LIMIT) pidIntegral = -I_LIMIT;
  }
  float correction = Kp * error + Ki * pidIntegral + Kd * dError;
  pidErrorPrev = error;

  // 4) 조향 적용
  int angle = STEER_NEUTRAL + (int)correction;
  if (angle < STEER_MIN) angle = STEER_MIN;
  if (angle > STEER_MAX) angle = STEER_MAX;
  steerAngle = angle;
  steerServo.write(steerAngle);

  // 5) 브레이크 해제 + 정속 구동
  brakeAngle = 0;
  brakeServo.write(brakeAngle);
  pwmCmd = BASE_PWM;
}

// ================================================================
// 명령 파싱: S<angle>, B<angle>, V<pwm>
// ================================================================
void parseCommand(const char* cmd) {
  char prefix = cmd[0];
  int value = atoi(cmd + 1);

  switch (prefix) {
    case 'S':
    case 's':
      steerAngle = constrain(value, 75, 145);
      steerServo.write(steerAngle);
      break;
    case 'B':
    case 'b':
      brakeAngle = constrain(value, 0, 180);
      brakeServo.write(brakeAngle);
      break;
    case 'V':
    case 'v':
      pwmCmd = constrain(value, -255, 255);
      break;
    default:
      break;
  }
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
// RC 입력 → 제어값 변환 (RC 수동 모드 전용)
//   조향: 1000~2000μs → 서보 75~145°
//   구동: 1600~2000μs → PWM   0~255  (전진)
//         1550~1600μs → PWM   0      (데드존/정지)
//         1450~1550μs → 브레이크 ON  (70° 고정)
//         1400~1450μs → PWM   0      (데드존/정지)
//         1000~1400μs → PWM -255~0   (후진)
// ================================================================
void applyRCInput() {
  noInterrupts();
  int steerPulse = rc_pulse[0];
  int drivePulse = rc_pulse[1];
  interrupts();

  // 조향 매핑
  steerAngle = (int)map(constrain(steerPulse, 1000, 2000), 1000, 2000, 75, 145);
  steerServo.write(steerAngle);

  // 구동 / 브레이크 매핑
  if (drivePulse >= 1600) {
    // 전진: 1600~2000 → PWM 0~255
    pwmCmd     = (int)map(constrain(drivePulse, 1600, 2000), 1600, 2000, 0, 255);
    brakeAngle = 0;
    brakeServo.write(brakeAngle);

  } else if (drivePulse >= 1550) {
    // 데드존 상단: 정지
    pwmCmd     = 0;
    brakeAngle = 0;
    brakeServo.write(brakeAngle);

  } else if (drivePulse >= 1450) {
    // 브레이크 구간: 고정값 제동
    pwmCmd     = 0;
    brakeAngle = 70;
    brakeServo.write(brakeAngle);

  } else if (drivePulse >= 1400) {
    // 데드존 하단: 정지
    pwmCmd     = 0;
    brakeAngle = 0;
    brakeServo.write(brakeAngle);

  } else {
    // 후진: 1000~1400 → PWM -255~0
    pwmCmd     = (int)map(constrain(drivePulse, 1000, 1400), 1000, 1400, -255, 0);
    brakeAngle = 0;
    brakeServo.write(brakeAngle);
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