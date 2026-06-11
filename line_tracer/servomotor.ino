/*
 * Hitec HS-956MG 서보모터 제어
 * 배선: 5V(빨강), GND(검정), Signal(노랑) → Pin 10
 *
 * 프로토콜:
 *   수신: A<angle>\n   (angle: 0~180)
 *   송신: A<angle>\n   (에코백)
 */

#include <Servo.h>

static const int SERVO_PIN = 10;

Servo servo;
int currentAngle = 90;

void setup() {
  Serial.begin(115200);
  servo.attach(SERVO_PIN);
  servo.write(currentAngle);
  Serial.println("HS-956MG Ready");
}

void loop() {
  if (!Serial.available()) return;

  char c = Serial.peek();
  if (c == 'A') {
    Serial.read();                       // 'A' 소비
    int angle = Serial.parseInt();       // 숫자 파싱
    while (Serial.available() && Serial.peek() == '\n') Serial.read(); // '\n' 소비

    angle = constrain(angle, 0, 180);
    servo.write(angle);
    currentAngle = angle;

    // 에코백
    Serial.print("A");
    Serial.println(currentAngle);
  } else {
    Serial.read(); // 알 수 없는 문자 버림
  }
}