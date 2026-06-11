#include <Arduino.h>

// ================================================================
// [테스트용 단순 PWM 구동 + 엔코더 읽기]
//
// 시리얼 명령 (115200 baud):
//   V<pwm>\n   (pwm: -255 ~ 255, 양수=전진, 음수=후진)
//
// 시리얼 피드백 (50ms 주기):
//   E<delta>,C<cumulative>\n
// ================================================================

// --- 핀 번호 ---
const int RMS = 11;   // 구동 모터 PWM
const int RMD = 12;   // 구동 모터 DIR

// --- 엔코더 핀 ---
const int ENC_A = 3;
const int ENC_B = 2;
volatile long EncoderCount = 0;
long EncoderCount_prev = 0;
int encDelta = 50;

// --- 제어 입력 ---
int pwmCmd = 0;  // -255 ~ 255

// --- 타이밍 ---
unsigned long lastFeedback = 0;
const unsigned long FEEDBACK_INTERVAL = 50;
unsigned long lastCommand = 0;
const unsigned long CMD_TIMEOUT = 500;

// --- 시리얼 파싱 ---
char cmdBuf[32];
uint8_t cmdIdx = 0;

void setup() {
  Serial.begin(115200);

  pinMode(RMS, OUTPUT);
  pinMode(RMD, OUTPUT);
  analogWrite(RMS, 0);
  digitalWrite(RMD, 0);

  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), ISR_Encoder, RISING);

  delay(300);
  Serial.println("READY");
}

void loop() {
  // --- 시리얼 명령 수신 ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdIdx > 0) {
        cmdBuf[cmdIdx] = '\0';
        parseCommand(cmdBuf);
        cmdIdx = 0;
        lastCommand = millis();
      }
    } else {
      if (cmdIdx < sizeof(cmdBuf) - 1) {
        cmdBuf[cmdIdx++] = c;
      }
    }
  }

  // --- 타임아웃: 명령 없으면 정지 ---
  if (millis() - lastCommand > CMD_TIMEOUT) {
    pwmCmd = 0;
  }

  // --- 모터 구동 (단순 PWM) ---
  driveMotor(pwmCmd);

  // --- 피드백 전송 (50ms 주기) ---
  if (millis() - lastFeedback >= FEEDBACK_INTERVAL) {
    lastFeedback = millis();

    noInterrupts();
    long enc = EncoderCount;
    interrupts();

    encDelta = (int)(enc - EncoderCount_prev);
    EncoderCount_prev = enc;

    Serial.print("E");
    Serial.print(encDelta);
    Serial.print(",C");
    Serial.println(enc);
  }

  delay(1);
}

// ================================================================
// 명령 파싱: "V<pwm>"
// ================================================================
void parseCommand(const char* cmd) {
  for (const char* p = cmd; *p; p++) {
    if (*p == 'V' || *p == 'v') {
      pwmCmd = constrain(atoi(p + 1), -255, 255);
      return;
    }
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
// 엔코더 ISR (A채널 RISING만 사용, B로 방향 판별)
// ================================================================
void ISR_Encoder() {
  if (digitalRead(ENC_B) == LOW) {
    EncoderCount++;
  } else {
    EncoderCount--;
  }
}