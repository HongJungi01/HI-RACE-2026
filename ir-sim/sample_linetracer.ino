// 샘플: IR 11ch(아날로그 A0~A10) 가중평균 + 비례 조향 라인트레이서
// 탭 1 기본 배선 기준: 조향 서보 D9, 모터 PWM D6 / DIR D7 (HIGH = 후진)
// 흰 바닥 + 검은 선 코스(N1~N5)용. 검은 바닥 + 흰 선 코스에서는 LINE_IS_DARK를 0으로.
#include <Servo.h>

#define LINE_IS_DARK 1
const int STEER_PIN = 9, PWM_PIN = 6, DIR_PIN = 7;
const int NEUTRAL = 110, S_MIN = 75, S_MAX = 145;
const int BASE_PWM = 70;          // 약 50 cm/s
const float KP = 0.30;            // 서보° / mm

// 센서 번호 0~10 → A0~A10, 좌우 위치 [mm, 오른쪽 +] (0 중앙, 홀수 왼쪽, 짝수 오른쪽)
const float SX[11] = {0, -12.5, 12.5, -37.5, 37.5, -62.5, 62.5, -87.5, 87.5, -112.5, 112.5};

Servo steer;
float lastErr = 0;

void setup() {
  Serial.begin(115200);
  pinMode(DIR_PIN, OUTPUT);
  steer.attach(STEER_PIN);
  steer.write(NEUTRAL);
  Serial.println("LINE TRACER READY");
}

void loop() {
  long sum = 0; float wsum = 0;
  for (int i = 0; i < 11; i++) {
    int raw = analogRead(A0 + i);
    int s = LINE_IS_DARK ? 1023 - raw : raw;   // 선일수록 큰 값
    if (s < 300) s = 0;
    sum += s; wsum += SX[i] * s;
  }
  float err = sum > 200 ? wsum / sum : lastErr;  // 선을 놓치면 마지막 방향 유지
  lastErr = err;

  int angle = constrain(NEUTRAL + (int)(KP * err), S_MIN, S_MAX);
  steer.write(angle);
  digitalWrite(DIR_PIN, LOW);
  analogWrite(PWM_PIN, BASE_PWM);

  static unsigned long t0 = 0;
  if (millis() - t0 >= 200) {
    t0 = millis();
    Serial.print("err="); Serial.print(err, 1);
    Serial.print(" servo="); Serial.println(angle);
  }
}
