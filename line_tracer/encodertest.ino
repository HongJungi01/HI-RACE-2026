// SZH-GNP196 엔코더 동작 확인 코드
// 엔코더 채널 A → 핀 2 (외부 인터럽트)
// 엔코더 채널 B → 핀 3 (외부 인터럽트)
// 엔코더 VCC → 5V, GND → GND
// 하얀 선 → VCC, 노란 선 → GND, 초록 선 → A(1상), 파란 선 → B(2상)

#define ENC_A 2
#define ENC_B 3
#define CHECK_TIMEOUT 50000   // 50초 안에 신호 감지 못하면 경고

volatile long count = 0;
volatile int lastA = LOW;
volatile bool signalDetected = false;

void onEncoderChange() {
  int A = digitalRead(ENC_A);
  int B = digitalRead(ENC_B);
  signalDetected = true;
  if (A != lastA) {
    if (A == HIGH) {
      count += (B == LOW) ? 1 : -1;
    }
    lastA = A;
  }
}

bool checkWiring() {
  Serial.println(">>> 엔코더 배선 확인 중...");
  Serial.println(">>> 샤프트를 손으로 돌려주세요 (5초)");

  // 풀업 상태에서 두 핀이 모두 HIGH에 고정되어 있으면 우선 경고
  int stateA = digitalRead(ENC_A);
  int stateB = digitalRead(ENC_B);
  if (stateA == LOW && stateB == LOW) {
    Serial.println("[경고] A/B 핀이 모두 LOW → GND 역결선 의심");
    return false;
  }

  // 5초간 신호 변화 감지 대기
  unsigned long start = millis();
  signalDetected = false;
  while (millis() - start < CHECK_TIMEOUT) {
    if (signalDetected) {
      Serial.println("[OK] 신호 감지 성공 → 배선 정상");
      return true;
    }
  }

  Serial.println("[경고] 신호 없음 → VCC/GND 역결선 또는 미연결 의심");
  return false;
}

void setup() {
  Serial.begin(115200);
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), onEncoderChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), onEncoderChange, CHANGE);

  if (checkWiring()) {
    Serial.println("=============================");
    Serial.println("   엔코더 테스트 시작");
    Serial.println("=============================");
  } else {
    Serial.println("=============================");
    Serial.println("   배선을 확인 후 리셋하세요");
    Serial.println("=============================");
    while (true);  // 배선 불량 시 이후 진행 중단
  }
}

long lastCount = 0;
unsigned long lastTime = 0;

void loop() {
  unsigned long now = millis();
  if (now - lastTime >= 200) {
    long c;
    noInterrupts();
    c = count;
    interrupts();

    long delta = c - lastCount;
    float rpm = (delta / 280.0) * (1000.0 / 200.0) * 60.0;

    Serial.print("카운트: ");
    Serial.print(c);
    Serial.print("  |  RPM: ");
    Serial.println(rpm, 1);

    lastCount = c;
    lastTime = now;
  }
}