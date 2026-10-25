// 샘플: 앞 IR 11ch + 뒤 IR 5ch 가중평균 + 비례(P) 조향 라인트레이서
// 탭 1 기본 배선 기준: 조향 서보 D9, 모터 PWM D6 / DIR D7 (HIGH = 후진)
//   앞 11ch 아날로그 A0~A10 (센서 0~10), 뒤 5ch 아날로그 A11~A15 (센서 1~5)
// 흰 바닥 + 검은 선 코스(N1~N5)용. 검은 바닥 + 흰 선 코스에서는 LINE_IS_DARK를 0으로.
//
// 조향 = 중립 + KP·eF + KR·(eF − eR)
//   eF: 앞 센서열에서 본 선 위치 [mm, 오른쪽 +]
//   eR: 뒤 센서열에서 본 선 위치 [mm, 오른쪽 +]
//   eF − eR: 차체가 선에 비해 얼마나 비스듬한지 (둘 다 비례항)
#include <Servo.h>

#define LINE_IS_DARK 1
const int STEER_PIN = 9, PWM_PIN = 6, DIR_PIN = 7;
const int NEUTRAL = 110, S_MIN = 75, S_MAX = 145;
const int BASE_PWM = 70;          // 약 53 cm/s (시뮬레이터 기본 모터 파라미터)
const float KP = 0.30;            // 서보° / mm (앞 위치 오차)
const float KR = 0.15;            // 서보° / mm (앞뒤 차이 = 방향 오차)

// 앞 11ch: 센서 번호 0~10 → A0~A10, 좌우 위치 [mm, 오른쪽 +] (0 중앙, 홀수 왼쪽, 짝수 오른쪽)
const float FX[11] = {0, -12.5, 12.5, -37.5, 37.5, -62.5, 62.5, -87.5, 87.5, -112.5, 112.5};
// 뒤 5ch: 센서 1~5 → A11~A15. 모듈을 뒤로 돌려 달아서 1번이 차의 오른쪽
const float RX[5] = {47, 18, 0, -18, -47};

Servo steer;
float lastF = 0, lastR = 0;

// 가중평균 선 위치. 선을 못 보면 false
bool linePos(int firstPin, int n, const float *x, float &pos) {
  long sum = 0; float wsum = 0;
  for (int i = 0; i < n; i++) {
    int raw = analogRead(firstPin + i);
    int s = LINE_IS_DARK ? 1023 - raw : raw;   // 선일수록 큰 값
    if (s < 300) s = 0;
    sum += s; wsum += x[i] * s;
  }
  if (sum <= 200) return false;
  pos = wsum / sum;
  return true;
}

void setup() {
  Serial.begin(115200);
  pinMode(DIR_PIN, OUTPUT);
  steer.attach(STEER_PIN);
  steer.write(NEUTRAL);
  Serial.println("LINE TRACER READY");
}

void loop() {
  float eF = lastF, eR = lastR;
  bool seeF = linePos(A0, 11, FX, eF);    // 놓치면 마지막 값 유지
  bool seeR = linePos(A11, 5, RX, eR);
  lastF = eF; lastR = eR;

  float corr = KP * eF + (seeF && seeR ? KR * (eF - eR) : 0);
  int angle = constrain(NEUTRAL + (int)corr, S_MIN, S_MAX);
  steer.write(angle);
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, BASE_PWM);

  static unsigned long t0 = 0;
  if (millis() - t0 >= 200) {
    t0 = millis();
    Serial.print("eF="); Serial.print(eF, 1);
    Serial.print(" eR="); Serial.print(eR, 1);
    Serial.print(" servo="); Serial.println(angle);
  }
}
