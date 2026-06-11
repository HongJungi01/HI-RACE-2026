#include <Arduino.h>
#include <Servo.h>
#include <math.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

/* ==========================================================================
 * [System Configuration]
 * Board: YD-RP2040 (VCC-GND Studio) - 3.3V Logic
 * Motor Driver: Traxxas XL-5
 * RC System: Spektrum AR620 (Outputs 6V -> Level Shift Required!)
 * Display: I2C LCD 1602
 * Debugging: Serial Monitor @ 115200 baud
 * ==========================================================================
 */

/* ===== Board check: RP2040 (YD-RP2040 / Raspberry Pi Pico) ===== */
#if !defined(ARDUINO_ARCH_RP2040)
# error "Select a RP2040 board (Raspberry Pi Pico, Arduino-pico core)."
#endif

/* ===== Pins (YD-RP2040 / Pico GP Pin Numbers) ===== */
static const uint8_t SERVO_PIN   = 6;   // SERVO : GP6
static const uint8_t RC_THR_PIN  = 7;   // THR   : GP7
static const uint8_t RC_STR_PIN  = 8;   // STR   : GP8
static const uint8_t RC_MODE_PIN = 9;   // MODE  : GP9
static const uint8_t RC_GEAR_PIN = 10;  // GEAR  : GP10
static const uint8_t PIN_PWM     = 11;  // ESC PWM : GP11
static const uint8_t ENC_A       = 12;  // ENC_A   : GP12
static const uint8_t ENC_B       = 13;  // ENC_B   : GP13

/* ===== I2C LCD (PCF8574) ===== */
static const uint8_t I2C_SDA_PIN  = 4;     // GP4
static const uint8_t I2C_SCL_PIN  = 5;     // GP5
static const uint8_t LCD_I2C_ADDR = 0x27;  // 0x27 or 0x3F

LiquidCrystal_I2C lcd(LCD_I2C_ADDR, 16, 2);

/* ===== Vehicle Specs ===== */
static const float    WHEEL_D_M   = 0.066f;
static const float    GEAR_RATIO  = 9.49f;
static const float    CPR         = 17.0f * 4.0f * GEAR_RATIO;
static inline float   CIRC_M() { return PI * WHEEL_D_M; }

/* ===== Loop/RC timing ===== */
static const uint32_t RC_TIMEOUT_US = 80000;   // 80 ms
static const uint32_t LOOP_DT_MS    = 20;      // 50 Hz Control Loop
static const uint16_t DBG_MS        = 100;     // 10 Hz Debug

/* ===== [CALIBRATION] RC PWM Settings (User Measured) ===== */
static const int16_t RC_NEU_US  = 1540;  // 실제 중립값
static const int16_t RC_RANGE   = 350;   // 중립에서 최대/최소까지의 변위량
static const int16_t ESC_DB     = 30;    // 데드밴드

/* ===== RC E-STOP threshold ===== */
static const uint16_t RC_ESTOP_US = 1150;

/* Servo Limits */
static const int16_t SERVO_MIN = 1000;
static const int16_t SERVO_MAX = 2000;

/* ===== Steering bands ===== */
static const int16_t STR_MIN_US     = 1150;
static const int16_t STR_MAX_US     = 1850;

static const int16_t STEER_RC_TRIM_US = -1;
static const int16_t STEER_ZERO_US    = (1500 + STEER_RC_TRIM_US);
static const int16_t STEER_DB_HALF_US = 20;
static const int8_t  STEER_DIR        = -1;

/* ===== ESC/Servo ===== */
Servo Esc, Str;
static const int16_t ESC_US_NEU  = 1500;
static const int16_t ESC_US_MIN  = 1000;
static const int16_t ESC_US_MAX  = 2000;

/* ===== [TUNING] Motor Actuation Thresholds ===== */
static const int16_t PHYSICAL_START_THRESHOLD = 180;
static const int16_t LOGICAL_START_THRESHOLD  = 50;

/* ===== [E-STOP ACTIVE BRAKE TUNING] ===== */
enum { ESTOP_BRAKE_IDLE=0, ESTOP_BRAKE_ACTIVE, ESTOP_BRAKE_HOLD };
static uint8_t  estop_brake_state = ESTOP_BRAKE_IDLE;
static uint32_t estop_brake_start_ms = 0;
static uint32_t estop_hold_start_ms  = 0;
static uint8_t  estop_prev = 0;
static int8_t   last_motion_sign = 0;   // +1 forward, -1 reverse, 0 unknown

static const int16_t  ESTOP_BRAKE_CMD      = 450;   // 0~1000
static const uint32_t ESTOP_BRAKE_TIME_MS  = 150;   // 반대 PWM 인가 시간
static const uint32_t ESTOP_HOLD_TIME_MS   = 120;   // neutral hold
static const float    ESTOP_STOP_MPS_TH    = 0.05f; // 거의 정지 판정

/* ===== Runtime ===== */
volatile int32_t  enc_cnt=0;
volatile uint32_t thr_rise=0, str_rise=0, mode_rise=0, gear_rise=0;
volatile uint16_t thr_us=1500, str_us=1500, mode_us=1500, gear_us=1500;
volatile uint32_t thr_fall_ts=0, str_fall_ts=0, mode_fall_ts=0, gear_fall_ts=0;

int16_t Steer1730_cmd = 0;
int16_t Steer_us_cmd  = 1500;
int16_t SpeedCmd = 0;

/* ===== HLC (host link) ===== */
static const uint8_t RX_LEN=13;
uint8_t rx_buf[RX_LEN], rx_idx=0;
inline void rx_reset(){ rx_idx=0; }

int16_t Hsp=0, Hst=0;
uint8_t Hgear=1, Ham=0, Hal=0, Hestop=0;

static uint32_t hlc_last_rx_ms = 0;
static const uint32_t HLC_TIMEOUT_MS = 300;

/* Modes */
enum { MODE_MAN=0, MODE_AUTO=1, MODE_AF=2 };
uint8_t ModeFB=MODE_MAN, EstopFB=0;
uint8_t Estop_RC = 0, Estop_HLC = 0;
uint8_t RC_gear=1, RC_auto=0;

/* Telemetry */
int16_t rps_i16=0, mps_centi_i16=0;

/* Timers */
uint32_t t_loop=0, t_dbg=0;

/* ===== Gear prime state ===== */
enum { PRIME_IDLE=0, PRIME_STEP1, PRIME_STEP2 };
static uint8_t gear_prime_state = PRIME_IDLE;
static int8_t  last_dir = 0;

/* ===== RP2040 fast read ===== */
#include "hardware/structs/sio.h"
static inline uint8_t fastRead(uint8_t pin){ return (sio_hw->gpio_in >> pin) & 1u; }

/* ===== RC pulse capture (glitch-robust) ===== */
volatile uint8_t mode_rise_valid = 0;

static inline void rc_isr_update(uint8_t pin,
                                 volatile uint32_t &t_rise,
                                 volatile uint16_t &pulse_us,
                                 volatile uint32_t &t_fall_ts,
                                 volatile uint8_t  &rise_valid)
{
  uint32_t t = micros();
  uint8_t  s = fastRead(pin);

  if (s) {
    t_rise = t;
    rise_valid = 1;
  } else {
    if (!rise_valid) return;
    rise_valid = 0;

    uint32_t w = t - t_rise;
    if (w < 900 || w > 2100) return;

    pulse_us  = (uint16_t)w;
    t_fall_ts = t;
  }
}

/* ===== Utils ===== */
static inline int16_t  clamp16 (int16_t  v, int16_t  lo, int16_t  hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }
static inline float    clampf  (float    v, float    lo, float    hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }
static inline int16_t  clamp_us(int16_t us){ return clamp16(us, ESC_US_MIN, ESC_US_MAX); }
static inline bool     is_neu_us(int16_t us){ return abs(us-ESC_US_NEU)<=30; }

static inline uint16_t med3(uint16_t a,uint16_t b,uint16_t c){
  if(a>b){uint16_t t=a;a=b;b=t;}
  if(b>c){uint16_t t=b;b=c;c=t;}
  if(a>b){uint16_t t=a;a=b;b=t;}
  return b;
}
static inline uint8_t map_rc_gear(uint16_t us){ if(us<1300) return 2; if(us>1700) return 0; return 1; }
static inline int8_t  gear_sign(uint8_t g){ return g==0?+1:(g==2?-1:0); }
static inline int16_t apply_steer_dir_us(int16_t us){ return (STEER_DIR >= 0) ? us : (3000 - us); }

static inline int32_t encoder_snapshot(){
  int32_t v; noInterrupts(); v = enc_cnt; interrupts(); return v;
}

static inline int16_t apply_gear_prime_to_cmd(int16_t cmd){
  int8_t new_dir = gear_sign(Hgear);
  if(new_dir != last_dir){
    if(last_dir >= 0 && new_dir < 0){
      gear_prime_state = PRIME_STEP1;
    }
    last_dir = new_dir;
  }
  switch(gear_prime_state){
    case PRIME_STEP1: gear_prime_state = PRIME_STEP2; return -100;
    case PRIME_STEP2: gear_prime_state = PRIME_IDLE;  return 0;
    default: return cmd;
  }
}

/* ===== RC MODE debounce ===== */
static inline void rc_auto_update_debounce(uint16_t md_us){
  uint8_t auto_raw = (md_us <= 1300) ? 1 : 0;

  static uint8_t auto_cnt = 0;
  static uint8_t man_cnt  = 0;

  if(auto_raw){
    if(auto_cnt < 10) auto_cnt++;
    man_cnt = 0;
  }else{
    if(man_cnt < 10) man_cnt++;
    auto_cnt = 0;
  }

  static const uint8_t N = 5; // 100 ms
  if(auto_cnt >= N) RC_auto = 1;
  if(man_cnt  >= N) RC_auto = 0;
}

/* ===== Steering conversions ===== */
static inline int16_t steer_us_to_1730(int16_t us){
  if(us < STR_MIN_US) us = STR_MIN_US;
  if(us > STR_MAX_US) us = STR_MAX_US;

  const int16_t db_lo = STEER_ZERO_US - STEER_DB_HALF_US;
  const int16_t db_hi = STEER_ZERO_US + STEER_DB_HALF_US;

  if(us >= db_lo && us <= db_hi) return 0;

  if(us < db_lo){
    float r = (float)(us - STR_MIN_US) / (float)(db_lo - STR_MIN_US);
    int16_t v = (int16_t)lroundf(1730.0f * (1.0f - r));
    return (v > 1730) ? 1730 : v;
  }else{
    float r = (float)(us - db_hi) / (float)(STR_MAX_US - db_hi);
    int16_t v = (int16_t)lroundf(-1730.0f * r);
    return (v < -1730) ? -1730 : v;
  }
}

static inline int16_t steer_1730_to_us(int16_t v1730){
  if(v1730 > 1730)  v1730 = 1730;
  if(v1730 < -1730) v1730 = -1730;

  const int16_t db_lo = STEER_ZERO_US - STEER_DB_HALF_US;
  const int16_t db_hi = STEER_ZERO_US + STEER_DB_HALF_US;

  if(v1730 == 0) return STEER_ZERO_US;

  if(v1730 > 0){
    float r = (float)v1730 / 1730.0f;
    int16_t us = db_lo - (int16_t)lroundf(r * (db_lo - STR_MIN_US));
    return clamp16(us, STR_MIN_US, db_lo);
  }else{
    float r = (float)(-v1730) / 1730.0f;
    int16_t us = db_hi + (int16_t)lroundf(r * (STR_MAX_US - db_hi));
    return clamp16(us, db_hi, STR_MAX_US);
  }
}

/* ===== Encoder ISR ===== */
inline __attribute__((always_inline)) void enc_from(uint8_t va,uint8_t vb){
  enc_cnt += (vb==0) ? (va?+1:-1) : (va?-1:+1);
}
void ISR_EncA(){ enc_from(fastRead(ENC_A), fastRead(ENC_B)); }
void ISR_EncB(){
  uint8_t va=fastRead(ENC_A), vb=fastRead(ENC_B);
  enc_cnt += (va==0) ? (vb?-1:+1) : (vb?+1:-1);
}

/* ===== RC capture ISR ===== */
void ISR_RC_THR(){
  uint8_t s=digitalRead(RC_THR_PIN);
  uint32_t t=micros();
  if(s){thr_rise=t;}
  else {
    uint32_t w=t-thr_rise;
    thr_us=(uint16_t)constrain((int32_t)w,800,2200);
    thr_fall_ts=t;
  }
}
void ISR_RC_STR(){
  uint8_t s=digitalRead(RC_STR_PIN);
  uint32_t t=micros();
  if(s){str_rise=t;}
  else {
    uint32_t w=t-str_rise;
    str_us=(uint16_t)constrain((int32_t)w,800,2200);
    str_fall_ts=t;
  }
}
void ISR_RC_MODE(){
  rc_isr_update(RC_MODE_PIN, mode_rise, mode_us, mode_fall_ts, mode_rise_valid);
}
void ISR_RC_GEAR(){
  uint8_t s=digitalRead(RC_GEAR_PIN);
  uint32_t t=micros();
  if(s){gear_rise=t;}
  else{
    uint32_t w=t-gear_rise;
    gear_us=(uint16_t)constrain((int32_t)w,800,2200);
    gear_fall_ts=t;
  }
}

/* ===== HLC parser ===== */
inline void rx_reset_and_sync(uint8_t b0){ rx_idx=0; if(b0=='S') rx_buf[rx_idx++]=b0; }

inline void hlc_parse_stream(){
  while(Serial1.available()){
    uint8_t b=(uint8_t)Serial1.read();
    if(rx_idx==0 && b!='S') continue;
    if(rx_idx==1 && b!='T'){ rx_reset_and_sync(b); continue; }
    if(rx_idx==2 && b!='X'){ rx_reset_and_sync(b); continue; }
    rx_buf[rx_idx++]=b;

    if(rx_idx == RX_LEN){
      if(rx_buf[RX_LEN-2]==0x0D && rx_buf[RX_LEN-1]==0x0A){
        Ham   = rx_buf[3];
        Hestop= rx_buf[4];
        Hgear = rx_buf[5]; if(Hgear>2) Hgear=1;
        Hsp   = (int16_t)((uint16_t)rx_buf[6]<<8 | rx_buf[7]); Hsp=clamp16(Hsp,0,1000);
        Hst   = (int16_t)((uint16_t)rx_buf[8]<<8 | rx_buf[9]); Hst=clamp16(Hst,-1730,1730);
        Hal   = rx_buf[10];
        hlc_last_rx_ms = millis();
      }
      rx_reset();
    }
  }
}

/* ===== Resolve mode / estop / build commands ===== */
inline void resolve_mode_estop(){
  noInterrupts();
  uint32_t tthr=thr_fall_ts,tstr=str_fall_ts,tmd=mode_fall_ts,tgr=gear_fall_ts;
  uint16_t thr=thr_us, str=str_us, md=mode_us, gr=gear_us;
  interrupts();

  uint32_t nowu=micros();
  if(nowu-tthr>RC_TIMEOUT_US) thr=1500;
  if(nowu-tstr>RC_TIMEOUT_US) str=1500;
  if(nowu-tmd >RC_TIMEOUT_US) md =1500;
  if(nowu-tgr >RC_TIMEOUT_US) gr =1500;

  static uint16_t thrf[3]={1500,1500,1500}, strf[3]={1500,1500,1500}, mdf[3]={1500,1500,1500}, grf[3]={1500,1500,1500};
  thrf[0]=thrf[1]; thrf[1]=thrf[2]; thrf[2]=thr;
  strf[0]=strf[1]; strf[1]=strf[2]; strf[2]=str;
  mdf [0]=mdf [1]; mdf [1]=mdf [2]; mdf [2]=md;
  grf [0]=grf[1]; grf[1]=grf[2]; grf[2]=gr;

  uint16_t thr_m=med3(thrf[0],thrf[1],thrf[2]);
  uint16_t str_m=med3(strf[0],strf[1],strf[2]);
  uint16_t md_m =med3(mdf [0],mdf [1],mdf [2]);
  uint16_t gr_m =med3(grf[0],grf[1],grf[2]);

  rc_auto_update_debounce(md_m);

  uint8_t g_new=map_rc_gear(gr_m);

  static uint8_t rc_gear_raw=1, rc_gear_lat=1;
  static uint32_t rc_gear_edge_ms=0;
  if(g_new!=rc_gear_raw){ rc_gear_raw=g_new; rc_gear_edge_ms=millis(); }
  if(millis()-rc_gear_edge_ms>=30) rc_gear_lat=rc_gear_raw;
  RC_gear=rc_gear_lat;

  bool comm_alive = (millis() - hlc_last_rx_ms) <= HLC_TIMEOUT_MS;

  if(RC_auto && Ham==1 && comm_alive) ModeFB=MODE_AUTO;
  else if(RC_auto && (Ham==0 || !comm_alive)) ModeFB=MODE_AF;
  else ModeFB=MODE_MAN;

  int16_t steer_c = clamp16((int16_t)(str_m + STEER_RC_TRIM_US), 800, 2200);
  steer_c = clamp16(steer_c, STR_MIN_US, STR_MAX_US);
  int16_t steer1730_manual = steer_us_to_1730(steer_c);

  /* ===== E-STOP logic ===== */
  Estop_RC  = (thr_m <= RC_ESTOP_US) ? 1 : 0;   // 수신기 throttle cut
  Estop_HLC = (Hestop != 0) ? 1 : 0;            // 상위제어 estop
  EstopFB   = Estop_RC || ((ModeFB == MODE_AUTO) && Estop_HLC);

  /* ===== SpeedCmd Calc ===== */
  int16_t man_cmd = 0;
  {
    int16_t stick = (int16_t)thr_m - RC_NEU_US;

    if(abs(stick) < ESC_DB) {
      stick = 0;
    }

    if (stick != 0) {
      man_cmd = map(stick, -RC_RANGE, RC_RANGE, -1000, 1000);
      man_cmd = clamp16(man_cmd, -1000, 1000);
    }

    int8_t sgn = gear_sign(RC_gear);
    man_cmd = man_cmd * sgn;
  }

  int16_t auto_cmd = 0;
  {
    int16_t mag = clamp16(Hsp, 0, 1000);
    int8_t sgn  = gear_sign(Hgear);
    auto_cmd = mag * sgn;
  }

  if(ModeFB==MODE_AUTO){
    Steer1730_cmd = clamp16(Hst, -1730, 1730);
    SpeedCmd      = auto_cmd;
  }else{
    Steer1730_cmd = steer1730_manual;
    SpeedCmd      = man_cmd;
    Hgear         = RC_gear;
  }

  SpeedCmd = apply_gear_prime_to_cmd(SpeedCmd);

  if(EstopFB || ModeFB==MODE_AF){
    SpeedCmd      = 0;
    Steer1730_cmd = 0;
  }

  int16_t us_cmd = steer_1730_to_us(Steer1730_cmd);
  Steer_us_cmd   = clamp16(us_cmd, SERVO_MIN, SERVO_MAX);
}

/* ===== ESC APPLY with Active Braking ===== */
inline void esc_apply(float dt){
  static int32_t enc_prev=0;
  int32_t now_enc;
  noInterrupts();
  now_enc=enc_cnt;
  interrupts();

  int32_t d = now_enc - enc_prev;
  enc_prev=now_enc;

  float rps = ((float)d / CPR)/dt;
  rps_i16   = (int16_t)constrain(rps, -32768, 32767);

  float mps = rps*CIRC_M();
  float cms = mps*100.0f;
  mps_centi_i16 = (int16_t)constrain(cms, -32768, 32767);

  if(mps > ESTOP_STOP_MPS_TH)       last_motion_sign = +1;
  else if(mps < -ESTOP_STOP_MPS_TH) last_motion_sign = -1;

  /* ===== E-STOP or AUTO FAIL ===== */
  if(EstopFB || ModeFB==MODE_AF){
    uint32_t now_ms = millis();
    int16_t us = ESC_US_NEU;

    switch(estop_brake_state){
      case ESTOP_BRAKE_ACTIVE:
      {
        int16_t brake_cmd = 0;

        if(mps > ESTOP_STOP_MPS_TH){
          brake_cmd = -ESTOP_BRAKE_CMD;
        }else if(mps < -ESTOP_STOP_MPS_TH){
          brake_cmd = +ESTOP_BRAKE_CMD;
        }else{
          if(last_motion_sign > 0)      brake_cmd = -ESTOP_BRAKE_CMD;
          else if(last_motion_sign < 0) brake_cmd = +ESTOP_BRAKE_CMD;
          else                          brake_cmd = 0;
        }

        float u = clampf((float)brake_cmd / 1000.0f, -1.0f, 1.0f);
        us = clamp_us((int16_t)(ESC_US_NEU + lroundf(u * 500.0f)));

        if(fabs(mps) <= ESTOP_STOP_MPS_TH){
          estop_brake_state = ESTOP_BRAKE_HOLD;
          estop_hold_start_ms = now_ms;
          us = ESC_US_NEU;
        }

        if(now_ms - estop_brake_start_ms >= ESTOP_BRAKE_TIME_MS){
          estop_brake_state = ESTOP_BRAKE_HOLD;
          estop_hold_start_ms = now_ms;
          us = ESC_US_NEU;
        }
      }
      break;

      case ESTOP_BRAKE_HOLD:
      default:
      {
        us = ESC_US_NEU;

        if(now_ms - estop_hold_start_ms >= ESTOP_HOLD_TIME_MS){
          us = ESC_US_NEU;
        }
      }
      break;
    }

    Esc.writeMicroseconds(us);
    return;
  }

  estop_brake_state = ESTOP_BRAKE_IDLE;

  /* ===== Normal drive ===== */
  int16_t final_cmd = 0;
  if (abs(SpeedCmd) < LOGICAL_START_THRESHOLD) {
    final_cmd = 0;
  } else {
    if (SpeedCmd > 0) {
      final_cmd = map(SpeedCmd, LOGICAL_START_THRESHOLD, 1000, PHYSICAL_START_THRESHOLD, 1000);
    } else {
      final_cmd = map(SpeedCmd, -LOGICAL_START_THRESHOLD, -1000, -PHYSICAL_START_THRESHOLD, -1000);
    }
  }

  float u  = clampf((float)final_cmd / 1000.0f, -1.0f, 1.0f);
  int16_t us = clamp_us((int16_t)(ESC_US_NEU + lroundf(u*500.0f)));

  if(is_neu_us(us)) us=ESC_US_NEU;

  uint8_t gear_sel = Hgear;
  int8_t cmd_sign  = (us>ESC_US_NEU+ESC_DB)?+1:(us<ESC_US_NEU-ESC_DB)?-1:0;

  if(gear_sel==1) {
    us = ESC_US_NEU;
  }else if(gear_sel==0 && cmd_sign<0){
    us = ESC_US_NEU;
  }else if(gear_sel==2 && cmd_sign>0){
    us = ESC_US_NEU;
  }

  Esc.writeMicroseconds(us);
}

inline void send_up(){
  int16_t v_out     = mps_centi_i16;
  int16_t steer_fb  = Steer1730_cmd;
  int32_t enc       = encoder_snapshot();

  uint8_t b[17];
  b[0]='S'; b[1]='T'; b[2]='X';
  b[3]=ModeFB;
  b[4]=EstopFB?1:0;
  uint8_t gear_fb = (EstopFB||ModeFB==MODE_AF)? 1 : Hgear;
  b[5]=gear_fb;
  b[6]=(uint8_t)((v_out>>8)&0xFF);
  b[7]=(uint8_t)( v_out     &0xFF);
  b[8]=(uint8_t)((steer_fb>>8)&0xFF);
  b[9]=(uint8_t)( steer_fb     &0xFF);
  b[10]=(uint8_t)((enc>>24)&0xFF);
  b[11]=(uint8_t)((enc>>16)&0xFF);
  b[12]=(uint8_t)((enc>> 8)&0xFF);
  b[13]=(uint8_t)( enc      &0xFF);
  b[14]=(uint8_t)(millis()&0xFF);
  b[15]=0x0D; b[16]=0x0A;
  Serial1.write(b, sizeof(b));
}

inline void lcd_update(){
  uint32_t now_ms = millis();
  bool comm_alive = (now_ms - hlc_last_rx_ms) <= HLC_TIMEOUT_MS;

  const char *comm_str = comm_alive ? "HLV CON" : "DIS-CON";
  const char *estop_str;
  if(Estop_RC && Estop_HLC)      estop_str = "E-STOP  ";
  else if(Estop_RC)              estop_str = "Re_E-ST ";
  else if(Estop_HLC)             estop_str = "Hi_E-ST ";
  else                           estop_str = "        ";

  lcd.setCursor(0,0);
  lcd.print(comm_str);
  lcd.setCursor(7,0); lcd.print(" ");
  lcd.setCursor(8,0); lcd.print(estop_str);

  const char *mode_label;
  switch(ModeFB){
    case MODE_MAN:  mode_label = "MANUAL";    break;
    case MODE_AUTO: mode_label = "AUTO";      break;
    case MODE_AF:   mode_label = "AUTO FAIL"; break;
    default:        mode_label = "UNKNOWN";   break;
  }

  uint8_t gear_fb = (EstopFB || ModeFB==MODE_AF) ? 1 : Hgear;
  char gear_ch;
  if(gear_fb == 0)      gear_ch = 'D';
  else if(gear_fb == 2) gear_ch = 'R';
  else                  gear_ch = 'N';

  char line2[17];
  uint8_t pos = 0;
  for(const char *p = mode_label; *p && pos < 16; ++p) line2[pos++] = *p;
  if(pos < 16) line2[pos++] = ' ';
  if(pos < 16) line2[pos++] = '/';
  if(pos < 16) line2[pos++] = ' ';
  if(pos < 16) line2[pos++] = gear_ch;
  while(pos < 16) line2[pos++] = ' ';
  line2[16] = '\0';

  lcd.setCursor(0,1);
  lcd.print(line2);
}

void setup(){
  Serial.begin(115200);
  Serial1.setRX(1);
  Serial1.setTX(0);
  Serial1.begin(115200);

  Wire.setSDA(I2C_SDA_PIN);
  Wire.setSCL(I2C_SCL_PIN);
  Wire.begin();
  lcd.init();
  lcd.backlight();

  lcd.clear();
  lcd.setCursor(0,0); lcd.print("Hello! RMM_mini");
  lcd.setCursor(0,1); lcd.print("by RMMLab.");
  delay(3000);

  lcd.clear();
  lcd.setCursor(0,0); lcd.print("DIS-CON");
  lcd.setCursor(0,1); lcd.print("MANUAL / N");

  Str.attach(SERVO_PIN, SERVO_MIN, SERVO_MAX);
  Str.writeMicroseconds(1500);

  Esc.attach(PIN_PWM);
  Esc.writeMicroseconds(ESC_US_NEU);
  delay(300);

  pinMode(ENC_A,INPUT_PULLUP);
  pinMode(ENC_B,INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A), ISR_EncA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_B), ISR_EncB, CHANGE);

  pinMode(RC_THR_PIN,INPUT);
  pinMode(RC_STR_PIN,INPUT);
  pinMode(RC_MODE_PIN,INPUT);
  pinMode(RC_GEAR_PIN,INPUT);
  attachInterrupt(digitalPinToInterrupt(RC_THR_PIN),  ISR_RC_THR,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(RC_STR_PIN),  ISR_RC_STR,  CHANGE);
  attachInterrupt(digitalPinToInterrupt(RC_MODE_PIN), ISR_RC_MODE, CHANGE);
  attachInterrupt(digitalPinToInterrupt(RC_GEAR_PIN), ISR_RC_GEAR, CHANGE);

  uint32_t nowu=micros();
  thr_fall_ts=str_fall_ts=mode_fall_ts=gear_fall_ts=nowu;

  t_loop=t_dbg=millis();

  ModeFB   = MODE_MAN;
  EstopFB  = 0;
  Hgear    = 1;
  Estop_RC = 0;
  Estop_HLC= 0;
  SpeedCmd = 0;

  gear_prime_state = PRIME_IDLE;
  last_dir         = gear_sign(Hgear);

  estop_brake_state    = ESTOP_BRAKE_IDLE;
  estop_brake_start_ms = 0;
  estop_hold_start_ms  = 0;
  estop_prev           = 0;
  last_motion_sign     = 0;

  Serial.println(F("# RP2040 ready: UART0 GP0/1, ESC GP11, ENC GP12/13, SERVO GP6, RC GP7/8/9/10, LCD I2C GP4/5"));
}

void loop(){
  uint32_t now=millis();
  if(now-t_loop>=LOOP_DT_MS){
    float dt=(now-t_loop)/1000.0f;
    t_loop=now;

    hlc_parse_stream();
    resolve_mode_estop();

    /* ===== E-stop rising edge detect ===== */
    if(EstopFB && !estop_prev){
      estop_brake_state = ESTOP_BRAKE_ACTIVE;
      estop_brake_start_ms = millis();
    }
    if(!EstopFB && estop_prev){
      estop_brake_state = ESTOP_BRAKE_IDLE;
    }
    estop_prev = EstopFB;

    int16_t us_out = apply_steer_dir_us(Steer_us_cmd);
    Str.writeMicroseconds(us_out);

    esc_apply(dt);
    send_up();
  }

  if(now-t_dbg>=DBG_MS){
    t_dbg=now;

    Serial.print("Mode=");        Serial.print(ModeFB);
    Serial.print(" Estop=");      Serial.print(EstopFB);
    Serial.print(" EstopRC=");    Serial.print(Estop_RC);
    Serial.print(" EstopHLC=");   Serial.print(Estop_HLC);
    Serial.print(" Gear=");       Serial.print((int)Hgear);
    Serial.print(" Thr_us=");     Serial.print(thr_us);
    Serial.print(" Str_us=");     Serial.print(str_us);
    Serial.print(" RPS=");        Serial.print((float)rps_i16,3);
    Serial.print(" MPS=");        Serial.print((float)mps_centi_i16/100.0f,3);
    Serial.print(" Steer_us_cmd=");   Serial.print(Steer_us_cmd);
    Serial.print(" Steer1730_cmd=");  Serial.print(Steer1730_cmd);
    Serial.print(" SpeedCmd=");       Serial.print(SpeedCmd);
    Serial.print(" RC_auto=");        Serial.print(RC_auto);
    Serial.print(" mode_us=");        Serial.print(mode_us);
    Serial.print(" BrakeState=");     Serial.print(estop_brake_state);
    Serial.print(" LastMotion=");     Serial.println(last_motion_sign);

    lcd_update();
  }
}