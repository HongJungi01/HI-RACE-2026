/*
 * Radiomaster R88 + Arduino 인터럽트 방식 제어
 * 핀 연결: 조향(4), 구동(5), auto(6)
 */

// 채널별 값을 저장할 변수
volatile unsigned long timer_start[3];
volatile int pulse_time[3];
volatile byte last_state[3];

// 핀 번호 설정 (순서대로 조향, 구동, auto)
const int pins[] = {4, 5, 6}; 
const char* labels[] = {"조향", "구동", "auto"};

void setup() {
  // 시리얼 모니터 속도를 115200으로 설정하세요.
  Serial.begin(115200);

  for (int i = 0; i < 3; i++) {
    pinMode(pins[i], INPUT);
    last_state[i] = digitalRead(pins[i]);
  }

  // 핀 변화 인터럽트(PCINT) 설정
  // 아두이노 우노/나노의 4, 5, 6번 핀은 모두 Port D에 속합니다.
  noInterrupts();           // 설정 중 인터럽트 일시 정지
  PCICR |= (1 << PCIE2);    // Port D (D0~D7) 인터럽트 활성화
  PCMSK2 |= (1 << PCINT20); // D4 (조향) 활성화
  PCMSK2 |= (1 << PCINT21); // D5 (구동) 활성화
  PCMSK2 |= (1 << PCINT22); // D6 (auto) 활성화
  interrupts();             // 인터럽트 다시 시작

  Serial.println("--- 무선 조종 시스템 시작 (조향/구동/auto) ---");
}

void loop() {
  // 시리얼 모니터 출력
  for (int i = 0; i < 3; i++) {
    Serial.print(labels[i]);
    Serial.print(": ");
    Serial.print(pulse_time[i]);
    Serial.print(i == 2 ? "" : "  |  "); // 마지막 칸이 아니면 구분선 출력
  }
  Serial.println();

  delay(50); // 출력 속도 조절 (0.05초 간격)
}

// 핀의 상태가 변할 때마다 자동으로 실행되는 함수 (ISR)
ISR(PCINT2_vect) {
  unsigned long current_time = micros();
  
  for (int i = 0; i < 3; i++) {
    byte state = digitalRead(pins[i]);
    if (state != last_state[i]) { // 해당 핀의 상태가 변했다면
      if (state == HIGH) {        // 신호가 HIGH가 되는 순간 (시작)
        timer_start[i] = current_time;
      } else {                    // 신호가 LOW가 되는 순간 (끝)
        if (timer_start[i] != 0) {
          pulse_time[i] = (int)(current_time - timer_start[i]);
        }
      }
      last_state[i] = state;
    }
  }
}