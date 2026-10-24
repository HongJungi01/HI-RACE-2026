// ================================================================
// 라인트레이서 PID 샘플: 앞 IR 11ch + 뒤 IR 5ch, 조향 PID + 속도 PID, 가변 속도
// (시뮬레이터 탭 1 기본 배치·배선 기준)
//
// 핀 배치:
//   조향 서보   → D9  (Servo, 중립 110°, 범위 75~145°, 각도↑ = 우회전)
//   구동 PWM    → D6  (analogWrite)
//   구동 DIR    → D7  (LOW = 전진, HIGH = 후진)
//   엔코더 A/B  → D2 (INT4) / D3 (INT5), CHANGE 인터럽트 4체배
//   앞 IR 11ch  → A0~A10 (센서 0~10, 0번 중앙 · 홀수 왼쪽 · 짝수 오른쪽)
//   뒤 IR 5ch   → A11~A15 (센서 1~5, 모듈을 뒤로 돌려 달아서 1번이 차의 오른쪽)
//
// 제어 구조 (10 ms 주기):
//   1) 앞 11ch 가중평균 → eF [mm, 오른쪽 +]
//      뒤 5ch 가중평균  → eR [mm, 오른쪽 +]
//   2) 차체 방향 오차 head = atan2(eF − eR, 앞뒤 센서 간격) [deg]
//      → 선이 차체와 얼마나 비스듬한지 (곡선 진입·출구 판단)
//   3) 조향 PID: 보정 = Kp·eF + Kh·head + Ki·∫eF + Kd·d(eF)/dt
//   4) 목표 속도: 직선(작은 |eF|·|head|)은 V_MAX, 곡선은 V_MIN 쪽으로 감속
//                 선을 놓치면 V_LOST로 줄이고 마지막 방향으로 최대 조향
//   5) 속도 PID: 엔코더 속도 → PWM = 피드포워드 + Kp·e + Ki·∫e
//
// 시리얼 (115200, 100 ms 주기):
//   eF=<mm> eR=<mm> hd=<deg> sv=<servo> vr=<cm/s> v=<cm/s> pwm=<0~255>
//
// 게인은 시뮬레이터 기본 파라미터(PWM 255 = 200 cm/s 등)에서 잡은 값.
// 실차에서는 다시 튜닝해야 함.
// ================================================================

#include <Servo.h>

// --- 핀 ---
const int STEER_PIN = 9;
const int PWM_PIN   = 6;
const int DIR_PIN   = 7;
const int ENC_A     = 2;
const int ENC_B     = 3;

// --- 조향 ---
const int STEER_NEUTRAL = 110;
const int STEER_MIN     = 75;
const int STEER_MAX     = 145;

// --- 라인 센서 ---
#define LINE_IS_DARK 1            // 흰 바닥 + 검은 선. 반대 코스면 0
// 앞 11ch: 센서 번호 0~10의 좌우 위치 [mm, 오른쪽 +]
const float FRONT_X[11] = {0, -12.5, 12.5, -37.5, 37.5, -62.5, 62.5, -87.5, 87.5, -112.5, 112.5};
// 뒤 5ch: A11~A15 = 센서 1~5. 뒤를 향해 달려 있어 1번이 차의 오른쪽
const float REAR_X[5]   = {47.0, 18.0, 0.0, -18.0, -47.0};
const float SENSOR_GAP  = 380.0;  // 앞 센서열 ~ 뒤 센서열 거리 [mm]
const int   IR_ACTIVE_MIN = 250;  // (선 강도) 이 값 미만은 0으로 (바닥 얼룩·노이즈 컷)
const int   SUM_MIN       = 300;  // 강도 합이 이보다 작으면 미감지

// --- 조향 PID (보정 단위 = 서보 각도°) ---
float Kp_s = 0.32;                // °/mm
float Kh_s = 0.9;                 // °/deg (방향 오차)
float Ki_s = 0.0;                 // °/(mm·s)  적분은 기본 꺼 둠 (트림 오차가 크면 0.05~0.2)
float Kd_s = 0.012;               // °/(mm/s)
const float I_LIMIT_S = 60.0;     // 적분 상한 [mm·s]

// --- 속도 PID (보정 단위 = PWM) ---
const float V_MAX  = 140.0;       // 직선 목표 속도 [cm/s]
const float V_MIN  = 70.0;        // 곡선 목표 속도 [cm/s]
const float V_LOST = 40.0;        // 선 놓쳤을 때 [cm/s]
const float ACCEL  = 250.0;       // 목표 속도 증가 한계 [cm/s²] (감속은 즉시)
float Kp_v = 0.8;                 // PWM/(cm/s)
float Ki_v = 2.0;                 // PWM/(cm/s·s)
const float FF_V  = 255.0 / 200.0;// 피드포워드 PWM/(cm/s)  (PWM 255 ≈ 200 cm/s)
const float PWM_DB = 10;          // 모터 드라이버 데드밴드
const float I_LIMIT_V = 60.0;     // 속도 적분 상한 [PWM]

// --- 엔코더 ---
const float COUNTS_PER_CM = 11.0 * 4 * 8 / (3.14159 * 6.5);   // 11 PPR × 4체배 × 기어비 8 / 바퀴 둘레
volatile long encCount = 0;
volatile uint8_t prevAB = 0;

// --- 상태 ---
Servo steer;
const unsigned long CTRL_MS = 10;
unsigned long lastCtrl = 0, lastLog = 0;
float eF = 0, eR = 0, head = 0;
float eFPrev = 0, iSteer = 0;
float vRef = 0, vMeas = 0, iSpeed = 0;
long  encPrev = 0;
int   servoOut = STEER_NEUTRAL, pwmOut = 0;
bool  frontSeen = false, rearSeen = false;
float lastSide = 0;               // 마지막으로 선이 있던 쪽 (오른쪽 +)

// ================================================================
void encISR() {                   // A·B 양 엣지, 상태표로 ±1
  static const int8_t T[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
  uint8_t ab = (digitalRead(ENC_A) << 1) | digitalRead(ENC_B);
  encCount += T[(prevAB << 2) | ab];
  prevAB = ab;
}

void setup() {
  Serial.begin(115200);
  pinMode(DIR_PIN, OUTPUT);
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, 0);

  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  prevAB = (digitalRead(ENC_A) << 1) | digitalRead(ENC_B);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), encISR, CHANGE);

  steer.attach(STEER_PIN);
  steer.write(STEER_NEUTRAL);

  delay(300);
  Serial.println("PID DUAL READY");
  lastCtrl = millis();
}

void loop() {
  unsigned long now = millis();
  if (now - lastCtrl >= CTRL_MS) {
    float dt = (now - lastCtrl) / 1000.0;
    lastCtrl = now;
    readLines();
    steeringControl(dt);
    speedControl(dt);
  }
  if (now - lastLog >= 100) {
    lastLog = now;
    Serial.print("eF="); Serial.print(eF, 1);
    Serial.print(" eR="); Serial.print(eR, 1);
    Serial.print(" hd="); Serial.print(head, 1);
    Serial.print(" sv="); Serial.print(servoOut);
    Serial.print(" vr="); Serial.print(vRef, 0);
    Serial.print(" v="); Serial.print(vMeas, 0);
    Serial.print(" pwm="); Serial.println(pwmOut);
  }
}

// ================================================================
// 1) 센서 → 선 위치 (가중평균)
// ================================================================
int lineStrength(int pin) {
  int raw = analogRead(pin);
  int s = LINE_IS_DARK ? 1023 - raw : raw;     // 선일수록 큰 값
  return s < IR_ACTIVE_MIN ? 0 : s;
}

void readLines() {
  long sum = 0; float wsum = 0;
  for (int i = 0; i < 11; i++) { int s = lineStrength(A0 + i); sum += s; wsum += FRONT_X[i] * s; }
  frontSeen = sum >= SUM_MIN;
  if (frontSeen) { eF = wsum / sum; lastSide = eF; }

  sum = 0; wsum = 0;
  for (int i = 0; i < 5; i++) { int s = lineStrength(A11 + i); sum += s; wsum += REAR_X[i] * s; }
  rearSeen = sum >= SUM_MIN;
  if (rearSeen) eR = wsum / sum;

  // 앞·뒤 모두 보일 때만 방향 오차 갱신
  if (frontSeen && rearSeen) head = atan2(eF - eR, SENSOR_GAP) * 57.2958;
}

// ================================================================
// 2) 조향 PID
// ================================================================
void steeringControl(float dt) {
  float corr;
  if (frontSeen) {
    float dE = (eF - eFPrev) / dt;
    if (Ki_s != 0) iSteer = constrain(iSteer + eF * dt, -I_LIMIT_S, I_LIMIT_S);
    corr = Kp_s * eF + Kh_s * head + Ki_s * iSteer + Kd_s * dE;
    eFPrev = eF;
  } else {
    // 선을 놓침: 마지막으로 본 쪽으로 최대 조향, 미분·적분 정지
    corr = lastSide >= 0 ? (STEER_MAX - STEER_NEUTRAL) : (STEER_MIN - STEER_NEUTRAL);
  }
  servoOut = constrain(STEER_NEUTRAL + (int)corr, STEER_MIN, STEER_MAX);
  steer.write(servoOut);
}

// ================================================================
// 3) 가변 목표 속도 + 속도 PID
// ================================================================
void speedControl(float dt) {
  // 엔코더 속도 (1차 저역통과)
  noInterrupts(); long c = encCount; interrupts();
  float v = (c - encPrev) / COUNTS_PER_CM / dt;
  encPrev = c;
  vMeas += 0.3 * (v - vMeas);

  // 곡선 정도: 앞 센서 위치와 방향 오차를 0~1로
  float bend = max(fabs(eF) / 60.0, fabs(head) / 12.0);
  if (bend > 1) bend = 1;
  float target = frontSeen ? V_MAX - (V_MAX - V_MIN) * bend : V_LOST;
  // 가속은 천천히, 감속은 즉시
  if (target > vRef) vRef = min(target, vRef + ACCEL * dt); else vRef = target;

  float e = vRef - vMeas;
  iSpeed = constrain(iSpeed + Ki_v * e * dt, -I_LIMIT_V, I_LIMIT_V);
  float u = (vRef > 1 ? PWM_DB + FF_V * vRef * (255 - PWM_DB) / 255.0 : 0) + Kp_v * e + iSpeed;
  pwmOut = constrain((int)u, 0, 255);       // 이 샘플은 후진 안 씀
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, pwmOut);
}
