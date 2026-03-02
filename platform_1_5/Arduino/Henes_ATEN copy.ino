#include <LiquidCrystal.h>

// lcd 화면 관련 변수
LiquidCrystal lcd(30,31,32,33,34,35);//RS,E,DB4,DB5,DB6,DB7
// 수신기 관련 변수
int steer , throttle , elev , rudd ,mode, gear; // 수신기 데이터
int res = 20; // 수신기 데이터의 안정화를 위한 상수
unsigned long i = 0; // 루프 타임을 줄이기 위해 수신기 데이터를 차례로 받도록 해주는 변수
// 플랫폼 제어 관련 변수
int Speed; // 플랫폼 속도 명령값
int Steer; // 플랫폼 조향 명령값
int St_M = 518; // 플랫폼 직진 조향값 (200~800) 영진전문대:500 제주도1호기:610 제주도2호기:450 제주도3호기:520
int St_D = 380; // 플랫폼 조향 최대 중립 차이  영진전문대:300 제주도 1호기:420  제주도 2호기:420
int Max_Speed = 250; // 플랫폼의 최대 속도값 (0~255)
byte Gear; // 플랫폼 기어 명령값
byte AorM; // 플랫폼 자율주행 모드, 수동주행 모드, 디버깅 모드 명령값
byte ESTOP; // 플랫폼 긴급정지 명령값
int lsp1,lsp2,lsp3,lsp4,lsp5 = 0;// 모터 속도 이동 평균 제어
int Lsp,Lst,Hsp,Hst,Hsp_p,Hst_p; // 상위 및 하위 제어의 속도, 조향 명령값
byte Lge,Les,Ham,Hge,Hes,Hal,Ham_p,Hge_p,Hes_p,Hal_p; // 상위 및 하위 제어의 기어, 모드, 긴급정지 명령값
byte Lal = 0;
int Steer_Feed,RPM;
char Receivedata[13]; // S T X Ham Hes Hge Hsp Hst Hal 0D0A
char Senddata[13]; // S T X AorM Les Lge RPM St_feed Lal 0D0A
int Con_status = 0;
byte rxBuf[13];
uint8_t rxIdx = 0;



//구동모터드라이버의 핀번호      
  const int RMS = 4;
  const int RMD = 22;
  const int LMS = 5;
  const int LMD = 24;
//구동모터의 엔코더센서 핀번호    
  const int interruptPinA = 3;  
  const int interruptPinB = 2;
  volatile long EncoderCount_R= 0; 
  long EncoderCount_prev_R = 0;
   //PID 변수_R
  int E_R = 0;
  int Sp_MAX_R = 15;
  int Sp_MIN_R = -15;
  int Sp_encoder_pot_R;    
  int Sp_val_R; 
  int Sp_encoder_val_R;
  float Sp_kp_R = 0.1;
  float Sp_ki_R = 0.000001 ;
  float Sp_kd_R = 0.0000;
  float Sp_Theta_R, Sp_Theta_d_R;
  int Sp_dT_R;
  unsigned long Sp_T_R;
  unsigned long Sp_T_prev_R = 0;
  int Sp_val_prev_R =0;
  float Sp_e_R, Sp_e_prev_R = 0, Sp_inte_R, Sp_inte_prev_R = 0, Sp_deri_R;
  float Sp_Vmax_R = 24;
  float Sp_Vmin_R = -24;
  float Sp_V_R = 0.1;
  //DC모터 PWM변수  
  int Sp_PWMval_R;
  int Sp_PWM_F_R = 0;
//RPM측정을 위한 변수들:NO
  volatile long EncoderCount = 0; // 엔코더 값 변수 회전당 280 count
  volatile long EncoderCount_prev = 0;
  unsigned long rpmtime;
  unsigned long rpmtime_prev = 0;
  long E,R;                 
  float rev, rps;

//조향모터 가변저항 및 조향모터의 PID제어를 위한 변수 
  int encoder_pot = A8;    
  int val; 
  int encoder_val;
  float kp = 0.1;
  float ki = 0.00000 ;
  float kd = 2.00;
  float Theta, Theta_d;
  int dt;
  unsigned long t;
  unsigned long t_prev = 0;
  int val_prev =0;
  float e, e_prev = 0, inte, inte_prev = 0;
  float Vmax = 24;
  float Vmin = -24;
  float V = 0.1;
  int PWMval;


void setup() {
  Serial.begin(115200);
  Serial2.begin(115200);
  Serial2.setTimeout(1);

  pinMode(RMS,OUTPUT);
  pinMode(RMD,OUTPUT);
  pinMode(LMS,OUTPUT);
  pinMode(LMD,OUTPUT);
  
  pinMode(interruptPinA, INPUT_PULLUP); // 엔코더 핀 
  pinMode(interruptPinB, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(interruptPinA), ISR_EncoderA, CHANGE); // 엔코더 값을 읽기 위한 인터럽트 서비스 루틴 
  attachInterrupt(digitalPinToInterrupt(interruptPinB), ISR_EncoderB, CHANGE);

  analogWrite(RMS,0);
  digitalWrite(RMD,0);
  analogWrite(LMS,0);
  digitalWrite(LMD,0);

  lcd.begin(16,2);
  lcd.setCursor(0,0);
  lcd.write("ROMOMO LAB");
  lcd.setCursor(0,1);
  lcd.write("HENES_T870");
  delay(1000);
  lcd.setCursor(0,0);
  lcd.write("3               ");
  lcd.setCursor(0,1);  
  lcd.write("                ");
  delay(1000);
  lcd.setCursor(0,0);
  lcd.write("2");
  delay(1000);
  lcd.setCursor(0,0);
  lcd.write("1");
  delay(1000);
}

void loop() {
unsigned long loopstart = millis();
controller();
platform_control();
SteerCon(Steer);
SpeedCon_R(Speed,Gear);
LCD();
Send();
int potpot = analogRead(A8);
unsigned long loopend = millis();
// Serial.print(Lsp); Serial.print(", ");
// Serial.print(steer); Serial.print(", ");
// Serial.print(gear); Serial.print(", ");
// Serial.print(mode); Serial.print(", ");
// Serial.print(encoder_val); Serial.print(", ");
// Serial.print(RPM); Serial.print(", "); 
// Serial.println(loopend-loopstart);
delay(1);
E_R = EncoderCount_R-EncoderCount_prev_R;
EncoderCount_prev_R = EncoderCount_R;
i++;
}

void controller(){ // 수신기 데이터를 받는 함수
  noInterrupts();
  
  if(i%6 == 0){
    throttle = pulseIn(A0,HIGH,30000); // throttle
    throttle = throttle/res*res;
    lsp5 = lsp4;
    lsp4 = lsp3;
    lsp3 = lsp2;
    lsp2 = lsp1;
    // Serial.print(throttle); Serial.print(", ");
  }
  else if(i%6 == 1){
    elev = pulseIn(A1,HIGH,30000); // elevation 
    elev = elev/res*res;
    // Serial.print(elev); Serial.print(", ");
  }
  
  else if(i%6 == 2){
    steer = pulseIn(A1,HIGH,30000); // steer
    steer = steer/res*res;
    // Serial.print(steer); Serial.print(", ");
  }
  else if(i%6 == 3){
    rudd = pulseIn(A2,HIGH,30000); // rudder
    rudd = rudd/res*res;
    // Serial.print(rudd); Serial.print(", ");
  }
  
  else if(i%6 == 4){
    mode = pulseIn(A4,HIGH,30000); // Debugging Manual Auto
    mode = mode/res*res;
    // Serial.print(mode); Serial.print(", ");


    
  }
  else if(i%6 == 5){
    gear = pulseIn(A5,HIGH,30000); // Drive Neutral Reverse
    gear = gear/res*res;
    // Serial.print(gear); Serial.println(", ");
  }
  interrupts();
  // throttle 신호 처리
  if(throttle < 1600 and throttle > 1450){ // throttle 신호의 DEADZONE
    throttle = 1500;
    Les = 0;
  }
  else if(throttle < 1200){
    throttle = throttle;
    Les = 1;
  }
  else{
    throttle = throttle;
    Les = 0;
  }
  lsp1 = map(throttle,1600,1800,0,Sp_MAX_R);
  lsp1 = constrain(lsp1,0,Sp_MAX_R);
  Lsp = (lsp1+lsp2+lsp3+lsp4+lsp5)/5;

  // steer 신호 처리
  if(steer < 1550 and steer > 1450){ // steer 신호의 DEADZONE
    steer = 1500;
  }
  else{
    steer = steer;
  }
  Lst = map(steer,1200,1800,St_M-St_D,St_M+St_D);
  Lst = constrain(Lst,St_M-St_D,St_M+St_D);
  // gear 신호 처리
  if(gear > 1600){
    Lge = 0; //전진
    
  }
  else if(gear <= 1600 and gear > 1400){
    Lge = 1; //중립
    
  }
  else{
    Lge = 2; //후진
    
  }
  // mode 신호 처리
  if(mode < 1250){
    AorM = 1; // 자율주행 모드
  }
  else if(mode > 1251 and mode < 1650){
    AorM = 0; // 수동주행 모드
  }
  else{
    AorM = 2; // 디버깅 모드
  }
}

void platform_control(){ // 플랫폼 제어값 결정
  if(AorM == 0){ // 수동주행 모드일때
    Speed = Lsp;
    Steer = Lst;
    Gear = Lge;
    Hes = 0;
  }
  else if(AorM == 1){ // 자율주행 모드일때
    bool packet_ok = false;

    // ====== HLC → LLC 13바이트 패킷 파싱 ======
    while (Serial2.available()) {
      byte b = Serial2.read();

      // STX 동기화: 첫 바이트는 반드시 'S'
      if (rxIdx == 0) {
        if (b != 'S') {
          continue; // 'S' 나올 때까지 버림
        }
      }

      rxBuf[rxIdx++] = b;

      if (rxIdx == 13) { // 풀 패킷 도착
        // S, T, X, ..., 0x0D, 0x0A 확인
        if (rxBuf[0] == 'S' && rxBuf[1] == 'T' && rxBuf[2] == 'X' &&
            rxBuf[11] == 0x0D && rxBuf[12] == 0x0A) {
          packet_ok = true;
        }
        rxIdx = 0;  // 다음 패킷 대비
        break;      // 한 루프당 최대 1패킷만 처리
      }
    }

    if (packet_ok) {
      // 프로토콜: S,T,X,AorM,ESTOP,GEAR,SPEED_H,SPEED_L,STEER_H,STEER_L,ALIVE,0D,0A
      byte a = rxBuf[0];
      byte b = rxBuf[1];
      byte c = rxBuf[2];
      byte d = rxBuf[3];               // A or M
      byte e = rxBuf[4];               // ESTOP
      byte f = rxBuf[5];               // GEAR
      int  g = word(rxBuf[6],rxBuf[7]); // SPEED (0~1000)
      int  h = word(rxBuf[8],rxBuf[9]); // STEER (-2000~2000)
      byte i = rxBuf[10];              // ALIVE

      // a,b,c는 이미 STX 확인됨
      Ham = d;
      Hes = e;
      Hge = f;
      Hsp = g;
      Hst = h;
      Hal = i;

      Ham_p = d;
      Hes_p = e;
      Hge_p = f;
      Hsp_p = g;
      Hst_p = h;
      Hal_p = i;

      Con_status = 1;
    }
    else {
      // 새 유효 패킷이 없으면 이전 값 기반으로 안전 동작
      Ham = Ham_p;
      Hes = Hes_p;
      Hge = 1;   // N
      Hsp = 0;   // 속도 0
      Hst = Hst_p;
      Hal = Hal_p;
      Con_status = 0;
    }
    Hsp = map(Hsp,0,1000,0,15);
    Hst = map(Hst,-2000,2000,St_M+St_D,St_M-St_D);
    if(Ham == 1){
      Speed = Hsp;
      Steer = Hst;
      Gear = Hge;       
    }
    else{
      Speed = 0;
      Steer = St_M;
      Gear = 0;
    }
  }
  else{ //디버그 모드
    Speed = 0;
    Steer = St_M;
    Gear = 1;
  }
  if(Les == 1 or Hes == 1){
    ESTOP = 1;
  }
  else{
    ESTOP = 0;
  }
  Steer = constrain(Steer,St_M-St_D,St_M+St_D);
  Speed = constrain(Speed,0,250); 
  if(ESTOP == 1){
    Steer = St_M;
    Speed = 0;
  }
  else{
    Steer = Steer;
    Speed = Speed;
  }
}

void SteerCon(int Q) { // 조향모터의 제어를 위한 함수, Q = 조향 목표값  
  val = Q; //Q = steer 조향값 범위(St_M-300~St_M+300)                           
  if(val>St_M+420){  // 조향 최대값 제한
    val=St_M+420;
  }
  else if(val<St_M-420){ // 조향 최소값 제한 
    val= St_M-420;
  }
  encoder_val =analogRead(A8);               // Read V_out from Feedback Pot
  t = millis();
  dt = (t - t_prev);                                  // Time step
  Theta = encoder_val;                                        // Theta= Actual Angular Position of the Motor
  Theta_d = val;                              // Theta_d= Desired Angular Position of the Motor 조종기값
  e = Theta_d - Theta;                                // Error
  inte = inte_prev + (dt * (e + e_prev) / 2);         // Integration of Error
  V = kp * e + ki * inte + (kd * (e - e_prev) / dt) ; // Controlling Function
  if (V > Vmax) {
    V = Vmax;
    inte = inte_prev;
  }
  if (V < Vmin) {
    V = Vmin;
    inte = inte_prev;
    val_prev= val;
  }
  PWMval = int(255 * abs(V) / Vmax);
  if (PWMval > 150) {
    PWMval = 150;
  }
  if (V > 0.5) {
    analogWrite(LMS, PWMval);
    digitalWrite(LMD, 1);
  }
  else if (V < -0.5) {
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


void SpeedCon_R(int Sp_R, int g) { 
  if(g==0){
  Sp_val_R = Sp_R;
  }
  else if(g ==2){
    Sp_val_R = Sp_R*-1;
  }
  else{
    Sp_val_R = 0;
  }
  if(Sp_val_R>Sp_MAX_R){  
    Sp_val_R = Sp_MAX_R;
  }
  else if(Sp_val_R<Sp_MIN_R){ 
    Sp_val_R = Sp_MIN_R;
  }
  else{
    Sp_val_R = Sp_val_R;
  }

  Sp_encoder_val_R = E_R;          
  Sp_T_R = millis();
  Sp_dT_R = (Sp_T_R - Sp_T_prev_R);                          
  Sp_e_R = Sp_val_R - Sp_encoder_val_R;                                // Error
  Sp_inte_R = Sp_inte_prev_R + (Sp_dT_R * (Sp_e_R + Sp_e_prev_R) / 2);         // Integration of Error
  Sp_deri_R = (Sp_e_R - Sp_e_prev_R)/Sp_dT_R;	// Derivation of Error
  Sp_V_R = Sp_kp_R * Sp_e_R + Sp_ki_R * Sp_inte_R + Sp_kd_R * Sp_deri_R ; // Controlling Function

  if (Sp_V_R > Sp_Vmax_R) {
    Sp_V_R = Sp_Vmax_R;
    Sp_inte_R = Sp_inte_prev_R;
  }
  else if (Sp_V_R < Sp_Vmin_R) {
    Sp_V_R = Sp_Vmin_R;
    Sp_inte_R = Sp_inte_prev_R;
  } 
  else{
    Sp_V_R = Sp_V_R;
    Sp_inte_R = Sp_inte_prev_R;    
  }

  Sp_PWMval_R = int(250*Sp_V_R/Sp_Vmax_R);
  Sp_PWM_F_R = Sp_PWM_F_R + Sp_PWMval_R;

  if(Sp_PWM_F_R > 250){
    Sp_PWM_F_R = 250;
  }
  else if(Sp_PWM_F_R < -250){
    Sp_PWM_F_R = -250;
  }
  else{
    Sp_PWM_F_R = Sp_PWM_F_R;
  }

  if(Sp_PWM_F_R>1){
    analogWrite(RMS,Sp_PWM_F_R);
    digitalWrite(RMD,0);  
  }
  else if(Sp_PWM_F_R<0){
    analogWrite(RMS,-1*Sp_PWM_F_R);
    digitalWrite(RMD,1); 
  }
  else{
    analogWrite(RMS,0);
    digitalWrite(RMD,0);     
  }
  Sp_T_prev_R = Sp_T_R;
  Sp_inte_prev_R = Sp_inte_R;
  Sp_e_prev_R = Sp_e_R;

}
void LCD(){ // LCD 화면
  if(AorM == 0){
    lcd.setCursor(0,0);
    lcd.print("Manual Mode     ");
    lcd.setCursor(0,1);
    lcd.print("Gear:");
    lcd.setCursor(5,1);
    if(Lge == 0){
      lcd.print("D ");
    }
    else if(Lge == 1){
      lcd.print("N ");
    }
    else{
      lcd.print("R ");
    }
    lcd.setCursor(7,1);
    lcd.print("ESTOP:");
    lcd.setCursor(13,1);
    if(Les == 1 and Hes == 0){
      lcd.print("R  ");
    }
    else if(Les == 0 and Hes == 1){
      lcd.print("H  ");
    }
    else if( Les == 1 and Hes == 1){
      lcd.print("A  ");
    }
    else{
      lcd.print("X  ");
    }
  }
  else if(AorM == 2){
    lcd.setCursor(0,0);
    lcd.print("Debugging Mode  ");
    lcd.setCursor(0,1);
    lcd.print("thr:");
    lcd.setCursor(4,1);
    lcd.print(throttle);
    lcd.setCursor(8,1);
    lcd.print("str:");
    lcd.setCursor(12,1);
    lcd.print(steer);
  }
  else{
    lcd.setCursor(0,0);
    lcd.print("Auto Mode       ");
    lcd.setCursor(0,1);
    lcd.print("Con:");
    if(Con_status == 1){
      lcd.setCursor(4,1);
      lcd.print("O  ");
    }
    else{
      lcd.setCursor(4,1);
      lcd.print("X  ");      
    }
    lcd.setCursor(7,1);
    lcd.print("ESTOP:");
    lcd.setCursor(13,1);
    if(Les == 1 and Hes == 0){
      lcd.print("R  ");
    }
    else if(Les == 0 and Hes == 1){
      lcd.print("H  ");
    }
    else if( Les == 1 and Hes == 1){
      lcd.print("A  ");
    }
    else{
      lcd.print("X  ");
    }
  }
}

void Send(){
  Steer_Feed = map(encoder_val,St_M+St_D,St_M-St_D,-2000,2000); //조향모터의 피드백값 저장
  Steer_Feed = constrain(Steer_Feed,-2000,2000);
  rpmtime = millis(); 
  E = E_R;
  R = (rpmtime - rpmtime_prev);
  rev = (1000000.00*E)/280.00;
  rps = rev/R;
  RPM = 6*rps; //플랫폼의 바퀴 rpm
  
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
  Serial2.write(Senddata,13);
  Lal=Lal+1;
}

void ISR_EncoderA() { // 엔코더 값을 저장하기 위한 내부 인터럽트 서비스 함수
  bool PinB = digitalRead(interruptPinB);
  bool PinA = digitalRead(interruptPinA);

  if (PinB == LOW) {
    if (PinA == HIGH) {
      EncoderCount_R++;
    }
    else {
      EncoderCount_R--;
    }
  }

  else {
    if (PinA == HIGH) {
      EncoderCount_R--;
    }
    else {
      EncoderCount_R++;
    }
  }
}

void ISR_EncoderB() { // 엔코더 값을 저장하기 위한 내부 인터럽트 서비스 함수
  bool PinB = digitalRead(interruptPinA);
  bool PinA = digitalRead(interruptPinB);

  if (PinA == LOW) {
    if (PinB == HIGH) {
      EncoderCount_R--;
    }
    else {
      EncoderCount_R++;
    }
  }

  else {
    if (PinB == HIGH) {
      EncoderCount_R++;
    }
    else {
      EncoderCount_R--;
    }
  }
}