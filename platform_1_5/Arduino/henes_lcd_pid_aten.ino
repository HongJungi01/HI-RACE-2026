#include <LiquidCrystal.h>

// ================================================================
// [변수 선언부]
// ================================================================

// --- LCD 화면 관련 변수 ---
LiquidCrystal lcd(30, 31, 32, 33, 34, 35); // RS, E, DB4, DB5, DB6, DB7

// --- 수신기 관련 변수 ---
int steer, throttle, elev, rudd, mode, gear; // 수신기 데이터
int res = 20;                                // 수신기 데이터의 안정화를 위한 상수
unsigned long i = 0;                         // 루프 타임을 줄이기 위해 수신기 데이터를 차례로 받도록 해주는 변수

// --- 플랫폼 제어 관련 변수 ---
int Speed;                                   // 플랫폼 속도 명령값
int Steer;                                   // 플랫폼 조향 명령값
int St_M = 512;                              // 플랫폼 직진 조향값 (200~800) [영진:500, 제주1:610, 제주2:450, 제주3:520]
int St_D = 370;                              // 플랫폼 조향 최대 중립 차이 [영진:300, 제주1:420, 제주2:420, 충남:420, 우리:380]
int Max_Speed = 250;                         // 플랫폼의 최대 속도값 (0~255)
byte Gear;                                   // 플랫폼 기어 명령값
byte AorM;                                   // 플랫폼 자율주행 모드(1), 수동주행 모드(0), 디버깅 모드(2) 명령값
byte ESTOP;                                  // 플랫폼 긴급정지 명령값

int lsp1, lsp2, lsp3, lsp4, lsp5 = 0;        // 모터 속도 이동 평균 제어
int Lsp, Lst, Hsp, Hst, Hsp_p, Hst_p;        // 상위 및 하위 제어의 속도, 조향 명령값
byte Lge, Les, Ham, Hge, Hes, Hal, Ham_p, Hge_p, Hes_p, Hal_p; // 제어 플래그들 (기어, 모드, 긴급정지 등)
byte Lal = 0;
int Steer_Feed, RPM;                         // 피드백 데이터
char Receivedata[13];                        // 수신 버퍼: S T X Ham Hes Hge Hsp Hst Hal 0D0A
char Senddata[13];                           // 송신 버퍼: S T X AorM Les Lge RPM St_feed Lal 0D0A
int Con_status = 0;                          // 통신 상태
byte rxBuf[13];
uint8_t rxIdx = 0;

// --- 구동 모터 드라이버 핀 번호 ---
const int RMS = 4;
const int RMD = 22;
const int LMS = 5;
const int LMD = 24;

// --- 구동 모터 엔코더 센서 핀 번호 ---
const int interruptPinA = 3;
const int interruptPinB = 2;
volatile long EncoderCount_R = 0;
long EncoderCount_prev_R = 0;

// --- 구동 모터 PID 변수 (R) ---
int E_R = 0;
int Sp_MAX_R = 15;
int Sp_MIN_R = -15;
int Sp_encoder_pot_R;
int Sp_val_R;
int Sp_encoder_val_R;

float Sp_kp_R = 0.1;
float Sp_ki_R = 0.000001;
float Sp_kd_R = 0.0000;

float Sp_Theta_R, Sp_Theta_d_R;
int Sp_dT_R;
unsigned long Sp_T_R;
unsigned long Sp_T_prev_R = 0;
int Sp_val_prev_R = 0;
float Sp_e_R, Sp_e_prev_R = 0, Sp_inte_R, Sp_inte_prev_R = 0, Sp_deri_R;

float Sp_Vmax_R = 24;
float Sp_Vmin_R = -24;
float Sp_V_R = 0.1;

// DC모터 PWM 변수
int Sp_PWMval_R;
int Sp_PWM_F_R = 0;

// --- RPM 측정을 위한 변수들 ---
volatile long EncoderCount = 0;         // 엔코더 값 변수 (회전당 280 count)
volatile long EncoderCount_prev = 0;
unsigned long rpmtime;
unsigned long rpmtime_prev = 0;
long E, R;
float rev, rps;

// --- 조향 모터 가변저항 및 PID 제어 변수 ---
int encoder_pot = A8;
int val;
int encoder_val;

float kp = 0.1;
float ki = 0.00000;
float kd = 2.00;

float Theta, Theta_d;
int dt;
unsigned long t;
unsigned long t_prev = 0;
int val_prev = 0;
float e, e_prev = 0, inte, inte_prev = 0;

float Vmax = 24;
float Vmin = -24;
float V = 0.1;
int PWMval;

// --- LCD 갱신 주기 변수 ---
unsigned long lcd_time_prev = 0;
const int LCD_INTERVAL = 300; // 300ms (0.3초) 마다 갱신

// ================================================================
// [Setup 함수]
// ================================================================
void setup() {
  // 시리얼 통신 초기화
  Serial.begin(115200);
  Serial2.begin(115200);
  Serial2.setTimeout(1);

  // 모터 핀 모드 설정
  pinMode(RMS, OUTPUT);
  pinMode(RMD, OUTPUT);
  pinMode(LMS, OUTPUT);
  pinMode(LMD, OUTPUT);

  // 엔코더 핀 설정 및 인터럽트
  pinMode(interruptPinA, INPUT_PULLUP);
  pinMode(interruptPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(interruptPinA), ISR_EncoderA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(interruptPinB), ISR_EncoderB, CHANGE);

  // 모터 초기화 (정지)
  analogWrite(RMS, 0);
  digitalWrite(RMD, 0);
  analogWrite(LMS, 0);
  digitalWrite(LMD, 0);

  // LCD 초기화 및 인트로 화면
  lcd.begin(16, 2);
  lcd.setCursor(0, 0);
  lcd.write("ROMOMO LAB");
  lcd.setCursor(0, 1);
  lcd.write("HENES_T870");
  delay(1000);

  // 카운트다운
  lcd.setCursor(0, 0); lcd.write("3               ");
  lcd.setCursor(0, 1); lcd.write("                ");
  delay(1000);
  lcd.setCursor(0, 0); lcd.write("2");
  delay(1000);
  lcd.setCursor(0, 0); lcd.write("1");
  delay(1000);
}

// ================================================================
// [Loop 함수]
// ================================================================
void loop() {
  unsigned long loopstart = millis();

  controller();           // 수신기 데이터 수신 및 처리
  platform_control();     // 제어 모드에 따른 명령값 결정
  SteerCon(Steer);        // 조향 모터 제어 (PID)
  SpeedCon_R(Speed, Gear);// 구동 모터 제어 (PID)
  
  // --- LCD 갱신 주기 제한 ---
  if (loopstart - lcd_time_prev >= LCD_INTERVAL) {
    LCD();                  // LCD 정보 표시
    lcd_time_prev = loopstart;
  }
  // -------------------------

  Send();                 // 상위 제어기로 데이터 전송
  
  int potpot = analogRead(A8);
  unsigned long loopend = millis();

  // 디버깅용 시리얼 출력
  Serial.print(Hsp);         Serial.print(", ");
  Serial.print(Lsp);         Serial.print(", ");
  Serial.print(steer);       Serial.print(", ");
  Serial.print(gear);        Serial.print(", ");
  Serial.print(Steer);       Serial.print(", ");
  Serial.print(encoder_val); Serial.print(", ");
  Serial.print(RPM);         Serial.print(", ");
  Serial.println(loopend - loopstart);

  delay(1);

  // 루프 주기 계산을 위한 업데이트
  E_R = EncoderCount_R - EncoderCount_prev_R;
  EncoderCount_prev_R = EncoderCount_R;
  i++;
}

// ================================================================
// [Controller : 수신기 데이터 처리]
// ================================================================
void controller() {
  noInterrupts();
  
  // 루프 부하 분산을 위해 순차적으로 수신
  if (i % 4 == 0) {
    throttle = pulseIn(A0, HIGH, 30000); // Throttle
    throttle = throttle / res * res;
    // 이동 평균 필터용 데이터 시프트
    lsp5 = lsp4; lsp4 = lsp3; lsp3 = lsp2; lsp2 = lsp1;
  }
  else if (i % 4 == 1) {
    steer = pulseIn(A1, HIGH, 30000);    // Steer
    steer = steer / res * res;
  }
  /* Elevation, Rudder (사용 안함)
  else if(i%6 == 2){ elev = pulseIn(A2,HIGH,30000); elev = elev/res*res; }
  else if(i%6 == 3){ rudd = pulseIn(A3,HIGH,30000); rudd = rudd/res*res; }
  */
  else if (i % 4 == 2) {
    mode = pulseIn(A4, HIGH, 30000);     // Debug / Manual / Auto
    mode = mode / res * res;
  }
  else if (i % 4 == 3) {
    gear = pulseIn(A5, HIGH, 30000);     // Drive / Neutral / Reverse
    gear = gear / res * res;
  }
  interrupts();

  // --- Throttle 신호 처리 ---
  if (throttle < 1600 && throttle > 1450) { // Deadzone
    throttle = 1500;
    Les = 0;
  }
  else if (throttle < 1200) {
    throttle = throttle; // 그대로 유지 (명시적)
    Les = 1;
  }
  else {
    throttle = throttle;
    Les = 0;
  }
  
  lsp1 = map(throttle, 1600, 1800, 0, Sp_MAX_R);
  lsp1 = constrain(lsp1, 0, Sp_MAX_R);
  Lsp = (lsp1 + lsp2 + lsp3 + lsp4 + lsp5) / 5;

  // --- Steer 신호 처리 ---
  if (steer < 1530 && steer > 1430) { // Deadzone
    steer = 1450;
  }
  else {
    steer = steer;
  }
  Lst = map(steer, 1100, 1860, St_M - St_D, St_M + St_D);
  Lst = constrain(Lst, St_M - St_D, St_M + St_D);

  // --- Gear 신호 처리 ---
  if (gear > 1600) {
    Lge = 0; // 전진
  }
  else if (gear <= 1600 && gear > 1400) {
    Lge = 1; // 중립
  }
  else {
    Lge = 2; // 후진
  }

  // --- Mode 신호 처리 ---
  if (mode < 1250) {
    AorM = 1; // 자율주행 모드
  }
  else if (mode > 1251 && mode < 1650) {
    AorM = 0; // 수동주행 모드
  }
  else {
    AorM = 2; // 디버깅 모드
  }
}

// ================================================================
// [Platform Control : 주행 모드별 제어값 설정]
// ================================================================
void platform_control() {
  if (AorM == 0) { // === 수동주행 모드 ===
    Speed = Lsp;
    Steer = Lst;
    Gear = Lge;
    Hes = 0;
  }
  else if (AorM == 1) { // === 자율주행 모드 ===
    bool packet_ok = false;

    // HLC(상위제어기) → LLC(하위제어기) 13바이트 패킷 파싱
    while (Serial2.available()) {
      byte b = Serial2.read();

      // STX 동기화: 첫 바이트는 반드시 'S'
      if (rxIdx == 0) {
        if (b != 'S') continue; 
      }

      rxBuf[rxIdx++] = b;

      if (rxIdx == 13) { // 패킷 길이 도달
        // STX(S,T,X) 및 종료 문자(0D, 0A) 확인
        if (rxBuf[0] == 'S' && rxBuf[1] == 'T' && rxBuf[2] == 'X' &&
            rxBuf[11] == 0x0D && rxBuf[12] == 0x0A) {
          packet_ok = true;
        }
        rxIdx = 0;  // 초기화
        break;      // 한 루프당 1패킷 처리
      }
    }

    if (packet_ok) {
      // 프로토콜: S, T, X, AorM, ESTOP, GEAR, SP_H, SP_L, ST_H, ST_L, ALIVE, 0D, 0A
      byte d = rxBuf[3];               // A or M
      byte e = rxBuf[4];               // ESTOP
      byte f = rxBuf[5];               // GEAR
      int  g = word(rxBuf[6], rxBuf[7]); // SPEED (0~1000)
      int  h = word(rxBuf[8], rxBuf[9]); // STEER (-2000~2000)
      byte i = rxBuf[10];              // ALIVE

      Ham = d;
      Hes = e;
      Hge = f;
      Hsp = g;
      Hst = h;
      Hal = i;

      // 이전 값 백업
      Ham_p = d;
      Hes_p = e;
      Hge_p = f;
      Hsp_p = g;
      Hst_p = h;
      Hal_p = i;

      Con_status = 1;
    }
    else {
      // 통신 패킷 없으면 이전 값 유지 혹은 안전 모드
      Ham = Ham_p;
      Hes = Hes_p;
      Hge = 1;   // N(중립)
      Hsp = 0;   // 정지
      Hst = Hst_p;
      Hal = Hal_p;
      Con_status = 0;
    }

    Hsp = map(Hsp, 0, 1000, 0, 15);
    Hst = map(Hst, -2000, 2000, St_M + St_D, St_M - St_D);

    if (Ham == 1) {
      Speed = Hsp;
      Steer = Hst;
      Gear = Hge;
    }
    else {
      Speed = 0;
      Steer = St_M;
      Gear = 0;
    }
  }
  else { // === 디버그 모드 ===
    Speed = 0;
    Steer = St_M;
    Gear = 1;
  }

  // --- 긴급정지 처리 ---
  if (Les == 1 || Hes == 1) {
    ESTOP = 1;
  }
  else {
    ESTOP = 0;
  }

  Steer = constrain(Steer, St_M - St_D, St_M + St_D);
  Speed = constrain(Speed, 0, 250);

  if (ESTOP == 1) {
    Steer = St_M;
    Speed = 0;
  }
}

// ================================================================
// [SteerCon : 조향 모터 PID 제어]
// ================================================================
void SteerCon(int Q) { // Q = 조향 목표값
  val = Q; // Steer 조향값 범위 (St_M-300 ~ St_M+300)
  
  // 조향 최대/최소값 물리적 제한
  if (val > St_M + 420) {
    val = St_M + 420;
  }
  else if (val < St_M - 420) {
    val = St_M - 420;
  }

  encoder_val = analogRead(A8);             // 가변저항 피드백 읽기
  t = millis();
  dt = (t - t_prev);                        // Time step

  Theta = encoder_val;                      // 현재 위치
  Theta_d = val;                            // 목표 위치
  e = Theta_d - Theta;                      // 오차
  inte = inte_prev + (dt * (e + e_prev) / 2); // 오차 적분
  V = kp * e + ki * inte + (kd * (e - e_prev) / dt); // PID 제어량 계산

  // 제어량 제한 (Saturation)
  if (V > Vmax) {
    V = Vmax;
    inte = inte_prev; // Anti-windup
  }
  if (V < Vmin) {
    V = Vmin;
    inte = inte_prev;
    val_prev = val;
  }

  PWMval = int(255 * abs(V) / Vmax);
  if (PWMval > 150) {
    PWMval = 150; // PWM 최대값 제한
  }

  // 모터 구동
  if (V > 0.2) {
    analogWrite(LMS, PWMval);
    digitalWrite(LMD, 1);
  }
  else if (V < -0.2) {
    analogWrite(LMS, PWMval);
    digitalWrite(LMD, 0);
  }
  else {
    digitalWrite(LMD, 0);
    analogWrite(LMS, 0);
  }
  
  t_prev = t;
  inte_prev = inte;
  e_prev = e;
}

// ================================================================
// [SpeedCon_R : 구동 모터 PID 제어]
// ================================================================
void SpeedCon_R(int Sp_R, int g) {
  if (g == 0) { // 전진
    Sp_val_R = Sp_R;
  }
  else if (g == 2) { // 후진
    Sp_val_R = Sp_R * -1;
  }
  else { // 중립
    Sp_val_R = 0;
  }

  // 속도 명령 제한
  if (Sp_val_R > Sp_MAX_R) {
    Sp_val_R = Sp_MAX_R;
  }
  else if (Sp_val_R < Sp_MIN_R) {
    Sp_val_R = Sp_MIN_R;
  }

  Sp_encoder_val_R = E_R;
  Sp_T_R = millis();
  Sp_dT_R = (Sp_T_R - Sp_T_prev_R);

  Sp_e_R = Sp_val_R - Sp_encoder_val_R;                                         // 오차
  Sp_inte_R = Sp_inte_prev_R + (Sp_dT_R * (Sp_e_R + Sp_e_prev_R) / 2);          // 적분
  Sp_deri_R = (Sp_e_R - Sp_e_prev_R) / Sp_dT_R;                                 // 미분
  Sp_V_R = Sp_kp_R * Sp_e_R + Sp_ki_R * Sp_inte_R + Sp_kd_R * Sp_deri_R;        // PID 계산

  // 제어량 제한 (Saturation)
  if (Sp_V_R > Sp_Vmax_R) {
    Sp_V_R = Sp_Vmax_R;
    Sp_inte_R = Sp_inte_prev_R;
  }
  else if (Sp_V_R < Sp_Vmin_R) {
    Sp_V_R = Sp_Vmin_R;
    Sp_inte_R = Sp_inte_prev_R;
  }

  Sp_PWMval_R = int(250 * Sp_V_R / Sp_Vmax_R);
  Sp_PWM_F_R = Sp_PWM_F_R + Sp_PWMval_R;

  // PWM 누적값 제한
  if (Sp_PWM_F_R > 250) {
    Sp_PWM_F_R = 250;
  }
  else if (Sp_PWM_F_R < -250) {
    Sp_PWM_F_R = -250;
  }

  // 모터 구동
  if (Sp_PWM_F_R > 0) {
    analogWrite(RMS, Sp_PWM_F_R);
    digitalWrite(RMD, 0);
  }
  else if (Sp_PWM_F_R < 0) {
    analogWrite(RMS, -1 * Sp_PWM_F_R);
    digitalWrite(RMD, 1);
  }
  else {
    analogWrite(RMS, 0);
    digitalWrite(RMD, 0);
  }

  Sp_T_prev_R = Sp_T_R;
  Sp_inte_prev_R = Sp_inte_R;
  Sp_e_prev_R = Sp_e_R;
}

// ================================================================
// [LCD : 상태 표시]
// ================================================================
void LCD() {
  if (AorM == 0) { // 매뉴얼 모드
    lcd.setCursor(0, 0);
    lcd.print("Manual Mode     ");
    lcd.setCursor(0, 1);
    lcd.print("Gear:");
    
    lcd.setCursor(5, 1);
    if (Lge == 0)      lcd.print("D ");
    else if (Lge == 1) lcd.print("N ");
    else               lcd.print("R ");

    lcd.setCursor(7, 1);
    lcd.print("ESTOP:");
    
    lcd.setCursor(13, 1);
    if (Les == 1 && Hes == 0)      lcd.print("R  ");
    else if (Les == 0 && Hes == 1) lcd.print("H  ");
    else if (Les == 1 && Hes == 1) lcd.print("A  ");
    else                           lcd.print("X  ");
  }
  else if (AorM == 2) { // 디버깅 모드
    lcd.setCursor(0, 0); lcd.print("Debugging Mode  ");
    lcd.setCursor(0, 1); lcd.print("thr:");
    lcd.setCursor(4, 1); lcd.print(throttle);
    lcd.setCursor(8, 1); lcd.print("str:");
    lcd.setCursor(12, 1); lcd.print(steer);
  }
  else { // 오토 모드
    lcd.setCursor(0, 0);
    lcd.print("Auto Mode       ");
    lcd.setCursor(0, 1);
    lcd.print("Con:");
    
    if (Con_status == 1) {
      lcd.setCursor(4, 1); lcd.print("O  ");
    } else {
      lcd.setCursor(4, 1); lcd.print("X  ");
    }
    
    lcd.setCursor(7, 1);
    lcd.print("ESTOP:");
    
    lcd.setCursor(13, 1);
    if (Les == 1 && Hes == 0)      lcd.print("R  ");
    else if (Les == 0 && Hes == 1) lcd.print("H  ");
    else if (Les == 1 && Hes == 1) lcd.print("A  ");
    else                           lcd.print("X  ");
  }
}

// ================================================================
// [Send : 상위 제어기로 데이터 전송]
// ================================================================
void Send() {
  Steer_Feed = map(encoder_val, St_M + St_D, St_M - St_D, -2000, 2000); // 피드백값 변환
  Steer_Feed = constrain(Steer_Feed, -2000, 2000);
  
  rpmtime = millis();
  E = E_R;
  R = (rpmtime - rpmtime_prev);
  rev = (1000000.00 * E) / 280.00;
  rps = rev / R;
  RPM = 6 * rps; // 플랫폼 바퀴 RPM

  rpmtime_prev = rpmtime;

  Senddata[0] = 'S';
  Senddata[1] = 'T';
  Senddata[2] = 'X';
  Senddata[3] = AorM | 0x00;
  Senddata[4] = Les | 0x00;
  Senddata[5] = Lge | 0x00;
  Senddata[6] = (RPM >> 8) | 0x00;
  Senddata[7] = RPM | 0x00;
  Senddata[8] = (Steer_Feed >> 8) | 0x00;
  Senddata[9] = Steer_Feed | 0x00;
  Senddata[10] = Lal | 0x00;
  Senddata[11] = 0x0D;
  Senddata[12] = 0x0A;
  
  Serial2.write(Senddata, 13);
  Lal = Lal + 1;
}

// ================================================================
// [ISR : 엔코더 인터럽트 서비스 루틴]
// ================================================================
void ISR_EncoderA() { 
  bool PinB = digitalRead(interruptPinB);
  bool PinA = digitalRead(interruptPinA);

  if (PinB == LOW) {
    if (PinA == HIGH) {
      EncoderCount_R++;
    } else {
      EncoderCount_R--;
    }
  } 
  else {
    if (PinA == HIGH) {
      EncoderCount_R--;
    } else {
      EncoderCount_R++;
    }
  }
}

void ISR_EncoderB() { 
  bool PinB = digitalRead(interruptPinA);
  bool PinA = digitalRead(interruptPinB);

  if (PinA == LOW) {
    if (PinB == HIGH) {
      EncoderCount_R--;
    } else {
      EncoderCount_R++;
    }
  } 
  else {
    if (PinB == HIGH) {
      EncoderCount_R++;
    } else {
      EncoderCount_R--;
    }
  }
}