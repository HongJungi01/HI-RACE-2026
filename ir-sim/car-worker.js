// 차량 1대 시뮬레이션 워커 (모듈 워커)
// ATmega2560 에뮬레이션(avr8js) + 차량 모델(기구학 자전거) + 엔코더 + IR(맵 선 벡터 샘플링) + 초음파(레이 모델, 상대 차량 외곽)
// + LIDAR-Lite v3HP (I2C 슬레이브: 하드웨어 TWI D20/D21 또는 아무 핀 비트뱅, PWM 모드 핀, 전원 EN 핀)
// 단위: 위치 cm, 각도 rad (메시지의 yaw도 rad), 맵 좌표 cm
import * as A from "./avr8js.js";

// ---------------- Mega 2560 설정 (avr8js 기본값은 ATmega328P) ----------------
const PORTS = { A: [0x20,0x21,0x22], B: [0x23,0x24,0x25], C: [0x26,0x27,0x28], D: [0x29,0x2a,0x2b], E: [0x2c,0x2d,0x2e], F: [0x2f,0x30,0x31], G: [0x32,0x33,0x34], H: [0x100,0x101,0x102], J: [0x103,0x104,0x105], K: [0x106,0x107,0x108], L: [0x109,0x10a,0x10b] };
const PINMAP = "E0 E1 E4 E5 G5 E3 H3 H4 H5 H6 B4 B5 B6 B7 J1 J0 H1 H0 D3 D2 D1 D0 A0 A1 A2 A3 A4 A5 A6 A7 C7 C6 C5 C4 C3 C2 C1 C0 D7 G2 G1 G0 L7 L6 L5 L4 L3 L2 L1 L0 B3 B2 B1 B0 F0 F1 F2 F3 F4 F5 F6 F7 K0 K1 K2 K3 K4 K5 K6 K7".split(" ").map(s => [s[0], +s[1]]);
function megaConfig() {
  const v = n => n * 2;
  const INT = [0,1,2,3,4,5,6,7].map(i => ({ EICR: i < 4 ? 0x69 : 0x6a, EIMSK: 0x3d, EIFR: 0x3c, index: i, iscOffset: (i % 4) * 2, interrupt: v(1 + i) }));
  const pc = (n, PCMSK, mask, offset) => ({ PCIE: n, PCICR: 0x68, PCIFR: 0x3b, PCMSK, pinChangeInterrupt: v(9 + n), mask, offset });
  const ports = {};
  for (const [k, [PIN, DDR, PORT]] of Object.entries(PORTS)) ports[k] = { PIN, DDR, PORT, externalInterrupts: [] };
  ports.B.pinChange = pc(0, 0x6b, 0xff, 0); ports.E.pinChange = pc(1, 0x6c, 0x01, 0); ports.J.pinChange = pc(1, 0x6c, 0x7f, 1); ports.K.pinChange = pc(2, 0x6d, 0xff, 0);
  ports.D.externalInterrupts = [INT[0], INT[1], INT[2], INT[3]];
  ports.E.externalInterrupts = [null, null, null, null, INT[4], INT[5], INT[6], INT[7]];
  const P = k => PORTS[k][2];
  const bits = { TOV: 1, OCFA: 2, OCFB: 4, OCFC: 8, TOIE: 1, OCIEA: 2, OCIEB: 4, OCIEC: 8 };
  const t16 = (base, TIFR, TIMSK, vec, oc, ext) => ({ bits: 16, captureInterrupt: v(vec), compAInterrupt: v(vec + 1), compBInterrupt: v(vec + 2), compCInterrupt: v(vec + 3), ovfInterrupt: v(vec + 4),
    TIFR, TIMSK, TCCRA: base, TCCRB: base + 1, TCCRC: base + 2, TCNT: base + 4, ICR: base + 6, OCRA: base + 8, OCRB: base + 10, OCRC: base + 12, dividers: A.timer1Config.dividers,
    compPortA: P(oc[0][0]), compPinA: oc[0][1], compPortB: P(oc[1][0]), compPinB: oc[1][1], compPortC: P(oc[2][0]), compPinC: oc[2][1], externalClockPort: P(ext[0]), externalClockPin: ext[1], ...bits });
  const timers = [
    { ...A.timer0Config, compAInterrupt: v(21), compBInterrupt: v(22), ovfInterrupt: v(23), compPortA: P("B"), compPinA: 7, compPortB: P("G"), compPinB: 5, externalClockPort: P("D"), externalClockPin: 7 },
    t16(0x80, 0x36, 0x6f, 16, [["B",5],["B",6],["B",7]], ["D",6]),
    { ...A.timer2Config, compAInterrupt: v(13), compBInterrupt: v(14), ovfInterrupt: v(15), compPortA: P("B"), compPinA: 4, compPortB: P("H"), compPinB: 6 },
    t16(0x90, 0x38, 0x71, 31, [["E",3],["E",4],["E",5]], ["E",6]),
    t16(0xa0, 0x39, 0x72, 41, [["H",3],["H",4],["H",5]], ["H",7]),
    t16(0x120, 0x3a, 0x73, 46, [["L",3],["L",4],["L",5]], ["L",2]),
  ];
  const ch = {};
  for (let i = 0; i < 8; i++) { ch[i] = { type: A.ADCMuxInputType.SingleEnded, channel: i }; ch[0x20 + i] = { type: A.ADCMuxInputType.SingleEnded, channel: 8 + i }; }
  ch[0x1e] = { type: A.ADCMuxInputType.Constant, voltage: 1.1 }; ch[0x1f] = { type: A.ADCMuxInputType.Constant, voltage: 0 };
  const adc = { ...A.adcConfig, adcInterrupt: v(29), numChannels: 16, muxInputMask: 0x3f, muxChannels: ch,
    adcReferences: [A.ADCReference.AREF, A.ADCReference.AVCC, A.ADCReference.Internal1V1, A.ADCReference.Internal2V56] };
  const usart = { ...A.usart0Config, rxCompleteInterrupt: v(25), dataRegisterEmptyInterrupt: v(26), txCompleteInterrupt: v(27) };
  const twi = { ...A.twiConfig, twiInterrupt: v(39) };                   // Mega TWI_vect = 39 (레지스터 주소는 328P와 같음)
  return { ports, timers, adc, usart, twi };
}
function loadHex(text, out) {
  let base = 0, n = 0;
  for (const line of text.split(/\r?\n/)) {
    if (line[0] !== ":") continue;
    const len = parseInt(line.substr(1, 2), 16), addr = parseInt(line.substr(3, 4), 16), type = parseInt(line.substr(7, 2), 16);
    if (type === 0) { for (let i = 0; i < len; i++) out[base + addr + i] = parseInt(line.substr(9 + i * 2, 2), 16); n += len; }
    else if (type === 2) base = parseInt(line.substr(9, 4), 16) << 4;
    else if (type === 4) base = parseInt(line.substr(9, 4), 16) << 16;
  }
  return n;
}
class Mega {
  constructor(hex) {
    const C = megaConfig();
    this.prog = new Uint16Array(0x20000);                        // 256 KB → 22bit PC
    if (!loadHex(hex, new Uint8Array(this.prog.buffer))) throw new Error("hex 데이터가 없습니다");
    this.cpu = new A.CPU(this.prog, 0x2100);                      // SRAM 0x200~0x21FF
    this.timers = C.timers.map(c => new A.AVRTimer(this.cpu, c));
    this.port = {}; for (const k in C.ports) this.port[k] = new A.AVRIOPort(this.cpu, C.ports[k]);
    this.usart = new A.AVRUSART(this.cpu, C.usart, 16e6);
    this.adc = new A.AVRADC(this.cpu, C.adc);
    this.twi = new A.AVRTWI(this.cpu, C.twi, 16e6);              // Wire 라이브러리(D20 SDA / D21 SCL)
    this.events = [];
    // 타이머는 다음 비교 일치·오버플로 시점에만 깨어나므로(avr8js.js 패치), 타이머 설정·OCR을 쓰기 전에 지금까지의 틱을 먼저 세고
    // 쓴 뒤에 다음 깨어날 시점을 다시 잡음
    const cpu = this.cpu;
    for (const t of this.timers) {
      const c = t.config;
      for (const a of [c.TCCRA, c.OCRA, c.OCRB, c.OCRC, c.ICR]) {
        const h = a > 0 && cpu.writeHooks[a]; if (!h) continue;
        cpu.writeHooks[a] = (v, o, addr, m) => {
          if (!t.updateDivider && t.divider) t.count(false);
          const r = h(v, o, addr, m);
          if (!t.updateDivider && t.divider) cpu.updateClockEvent(t.count, 0);
          return r;
        };
      }
      const hb = cpu.writeHooks[c.TCCRB];                        // 분주 변경 전에 이전 분주로 센 틱 반영
      cpu.writeHooks[c.TCCRB] = (v, o, addr, m) => { if (!t.updateDivider && t.divider) t.count(false); return hb(v, o, addr, m); };
    }
  }
  setPin(n, hi) { const [p, b] = PINMAP[n]; this.port[p].setPin(b, hi); }
  getPin(n) { const [p, b] = PINMAP[n]; return this.port[p].pinState(b) === A.PinState.High; }
  drivenLow(n) { const [p, b] = PINMAP[n]; return this.port[p].pinState(b) === A.PinState.Low; }   // Mega가 출력 LOW로 끌어내림
  at(cycle, fn) { this.events.push({ cycle, fn }); }
  runTo(end) {
    const cpu = this.cpu;
    while (cpu.cycles < end) {
      let next = end, idx = -1;
      for (let i = 0; i < this.events.length; i++) if (this.events[i].cycle < next) { next = this.events[i].cycle; idx = i; }
      while (cpu.cycles < next) { A.avrInstruction(cpu); cpu.tick(); }
      if (idx >= 0) { const ev = this.events.splice(idx, 1)[0]; ev.fn(cpu.cycles); }
    }
  }
}

// ---------------- 맵: 선 폴리라인 → 격자 색인 (IR 샘플링) ----------------
let MAP = null;          // { floorIR, halfW, cell, grid: Map(key → [seg...]) }
function buildMap(m) {
  const cell = 5, halfW = m.lineWidth / 2, grid = new Map();       // cm
  const add = (k, s) => { let a = grid.get(k); if (!a) grid.set(k, a = []); a.push(s); };
  for (const ln of m.lines) {
    const p = ln.pts, n = p.length, segs = n - (ln.closed ? 0 : 1);
    for (let i = 0; i < segs; i++) {
      const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % n];
      const s = [x1, y1, x2, y2, ln.ir];
      const i0 = Math.floor((Math.min(x1, x2) - halfW) / cell), i1 = Math.floor((Math.max(x1, x2) + halfW) / cell);
      const j0 = Math.floor((Math.min(y1, y2) - halfW) / cell), j1 = Math.floor((Math.max(y1, y2) + halfW) / cell);
      for (let a = i0; a <= i1; a++) for (let b = j0; b <= j1; b++) add(a * 100003 + b, s);
    }
  }
  MAP = { floorIR: m.floorIR, halfW, cell, grid };
}
function reflAt(x, y) {
  if (!MAP) return 0.9;
  const segs = MAP.grid.get(Math.floor(x / MAP.cell) * 100003 + Math.floor(y / MAP.cell));
  if (!segs) return MAP.floorIR;
  let best = -1;
  for (const [x1, y1, x2, y2, ir] of segs) {
    const dx = x2 - x1, dy = y2 - y1, L2 = dx * dx + dy * dy;
    const t = L2 ? Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / L2)) : 0;
    const ex = x - x1 - t * dx, ey = y - y1 - t * dy;
    if (ex * ex + ey * ey <= MAP.halfW * MAP.halfW && ir > best) best = ir;
  }
  return best >= 0 ? best : MAP.floorIR;
}

// ---------------- 초음파 2D 모델 (ultrasonic_sensor.py 포팅) ----------------
const US = { beamHalf: 15, specW: 10, diffuse: 0.02, threshold: 0.3, span: 60, step: 0.5, rMin: 2, rMax: 400, window: 5, noiseBase: 0.2, noiseGain: 0.003 };
const LN2 = Math.log(2);
const randn = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
function usMeasure(sx, sy, heading, segs) {
  if (!segs.length) return null;
  const nb = US.rMax + 20, bins = new Float64Array(nb);
  for (let phi = -US.span; phi <= US.span + 1e-9; phi += US.step) {
    const a = heading + phi * Math.PI / 180, dx = Math.cos(a), dy = Math.sin(a);
    let tb = Infinity, nx = 0, ny = 0;
    for (const [ax, ay, bx, by] of segs) {
      const ex = bx - ax, ey = by - ay, den = dx * ey - dy * ex;
      if (Math.abs(den) < 1e-9) continue;
      const t = ((ax - sx) * ey - (ay - sy) * ex) / den, u = ((ax - sx) * dy - (ay - sy) * dx) / den;
      if (t > 0.5 && u >= 0 && u <= 1 && t < tb) {
        const L = Math.hypot(ex, ey); let snx = ey / L, sny = -ex / L;
        if (snx * dx + sny * dy > 0) { snx = -snx; sny = -sny; }
        tb = t; nx = snx; ny = sny;
      }
    }
    if (tb > US.rMax + 10) continue;
    const alpha = Math.acos(Math.max(-1, Math.min(1, -(dx * nx + dy * ny)))) * 180 / Math.PI;
    const c = Math.exp(-LN2 * (phi / US.beamHalf) ** 2) * (Math.exp(-((alpha / US.specW) ** 2)) + US.diffuse) / (Math.max(tb, 5) / 50) ** 2;
    const k = Math.round(tb); if (k < nb) bins[k] += c;
  }
  let win = 0;
  for (let i = 0; i <= US.rMax; i++) {
    win += bins[i]; if (i - US.window >= 0) win -= bins[i - US.window];
    if (i >= 2 && win > US.threshold) {
      let first = Math.max(0, i - US.window + 1); while (first < i && bins[first] === 0) first++;
      if (first < US.rMin) return null;
      return first + randn() * (US.noiseBase + US.noiseGain * first);
    }
  }
  return null;
}


// ---------------- LIDAR-Lite v3HP 모델 ----------------
// 사양 (Garmin LIDAR-Lite v3HP Operation Manual): 40 m (70 % 반사 타깃), 분해능 1 cm, 정확도 2 m 미만 ±5 cm (1 m 미만 비선형) /
// 2 m 이상 ±2.5 cm, 평균 오차 최대 ±1 % 거리, 빔 발산 8 mrad, 레이저 출구 12 × 2 mm, 1 kHz 이상 (획득 횟수 작으면 1.5 kHz),
// 신호 없으면 1 cm 보고, 최대 거리 ∝ n^(1/4) (n = SIG_COUNT_VAL), 전원 인가 후 약 22 ms 뒤부터 측정 가능.
// I2C 0x62, 레지스터 포인터 자동 증가. PWM: MODE 선을 LOW로 당기면 측정, 끝나면 10 µs/cm HIGH 펄스, 계속 LOW면 반복
const LL_ADDR = 0x62, LL_BOOT_MS = 22, LL_RAYS = 7;
function rayHit(ox, oy, a, segs) {                           // 반직선과 선분들의 첫 교점 → [거리, 입사각 cos]
  const dx = Math.cos(a), dy = Math.sin(a); let tb = Infinity, cs = 0;
  for (const [ax, ay, bx, by] of segs) {
    const ex = bx - ax, ey = by - ay, den = dx * ey - dy * ex; if (Math.abs(den) < 1e-9) continue;
    const t = ((ax - ox) * ey - (ay - oy) * ex) / den, u = ((ax - ox) * dy - (ay - oy) * dx) / den;
    if (t > 0.05 && u >= 0 && u <= 1 && t < tb) { tb = t; cs = Math.abs(dx * ey - dy * ex) / Math.hypot(ex, ey); }
  }
  return tb < Infinity ? [tb, cs] : null;
}
// 빔 = 출구 폭 1.2 cm에 걸친 레이 7개, 바깥 레이는 발산 ±4 mrad. 반환 세기 ∝ 반사율·cos(입사각)/거리², 레이 평균
// 거리 = 깊이 8 cm 안에서 묶은 반사 중 세기 합이 가장 큰 묶음의 세기 가중 평균 (모서리에 걸치면 섞인 값)
function lidarRaw(sx, sy, h, segs, n, refl) {
  if (!segs.length) return null;
  const hits = [], nx = -Math.sin(h), ny = Math.cos(h);
  for (let i = 0; i < LL_RAYS; i++) { const f = i / (LL_RAYS - 1) * 2 - 1, r = rayHit(sx + nx * f * 0.6, sy + ny * f * 0.6, h + f * 0.004, segs);
    if (r) hits.push({ t: r[0], s: refl * r[1] / Math.max(r[0] / 100, 0.05) ** 2 }); }
  if (!hits.length) return null;
  hits.sort((a, b) => a.t - b.t);
  let best = null;
  for (let i = 0; i < hits.length;) { let j = i, s = 0, ts = 0; while (j < hits.length && hits[j].t - hits[i].t < 8) { s += hits[j].s; ts += hits[j].t * hits[j].s; j++; }
    if (!best || s > best.s) best = { s, d: ts / s }; i = j; }
  const S = best.s / LL_RAYS, rmax = 40 * Math.pow(Math.max(1, n) / 255, 0.25), thr = 0.7 / (rmax * rmax);
  return S >= thr ? { d: best.d, S, thr } : null;
}
// 센서별 오차 (실행마다 새로 뽑음): 오프셋, 거리 비례 오차(최대 ±1 %), 1 m 미만 비선형 방향
function lidarErr(L, d, P) {
  const near = d < 100 ? L.nearK * (P.llNear ?? 5) * (1 - d / 100) ** 2 : 0;
  const sig = (P.llNoise ?? 1) * (d < 200 ? 1.5 : 1);
  return Math.max(1, Math.round(d * (1 + L.scale) + L.bias + near + randn() * sig));
}
function llDefaults(L) {
  L.regs.fill(0); L.regs[0x02] = 0xff; L.regs[0x04] = 0x08; L.regs[0x12] = 0x03;
  L.dist = 1; L.peak = false; L.sig = 0; L.ptr = 0; L.first = false; L.busyUntil = 0; L.pend = null;
}
function newLidar(u) {
  const P = cfg.params, uni = () => Math.random() * 2 - 1;
  const L = { u, regs: new Uint8Array(128), unit: (Math.random() * 65536) | 0, power: true, bootUntil: LL_BOOT_MS * 16000,
    bias: uni() * (P.llBias ?? 2), scale: uni() * 0.005, nearK: uni(), pulse: false, mon: false };
  llDefaults(L); return L;
}
function lidarPose(ui) {
  const u = cfg.us[ui], co = Math.cos(veh.yaw), si = Math.sin(veh.yaw);
  const h = veh.yaw + u.yaw + (usServo[ui] ? (usServo[ui].ang - 90) * D2R : 0);
  return [veh.x + u.x * co - u.y * si + (u.off || 0) * Math.cos(h), veh.y + u.x * si + u.y * co + (u.off || 0) * Math.sin(h), h];
}
const llReady = L => L.power && mega.cpu.cycles >= L.bootUntil;
function llBusy(L) { llFinish(L); return mega.cpu.cycles < L.busyUntil; }
function llFinish(L) {                                       // 측정 결과는 끝난 시각에 레지스터로 (그 전에는 이전 값 유지)
  if (!L.pend || mega.cpu.cycles < L.busyUntil) return;
  const r = L.pend; L.pend = null; L.dist = r.dist; L.peak = r.peak; L.sig = r.sig;
  usLast[L.idx] = { d: r.peak ? r.dist : null, c: mega.cpu.cycles, lidar: true };
}
// 측정 시작: 지금 자세로 거리 계산, 결과는 측정 시간 뒤 반영. 측정 시간 = 기본 + 획득 1회 × 사용한 획득 횟수
// (강한 반사는 상관 피크가 일찍 포화돼 횟수가 줄고, 빠른 종료(0x04 bit3 = 0)면 더 줄어듦. 1 kHz 이상 / 최대 1.5 kHz 사양에 맞춘 가정값)
function llStart(L, onDone) {
  llFinish(L); if (!llReady(L) || mega.cpu.cycles < L.busyUntil) return false;
  const P = cfg.params, n = L.regs[0x02] || 1, [sx, sy, h] = lidarPose(L.idx);
  const rx = !(L.regs[0x65] & 1) ? lidarRaw(sx, sy, h, others, n, (P.llRefl ?? 70) / 100) : null;
  let used = n;
  if (rx) { used = Math.min(n, Math.max(8, Math.round(n * rx.thr * 400 / rx.S))); if (!(L.regs[0x04] & 0x08)) used = Math.max(4, used >> 1); }
  const ms = (P.llT0 ?? 0.6) + used * (P.llTacq ?? 2) / 1000;
  L.pend = rx ? { dist: Math.min(4000, lidarErr(L, rx.d, P)), peak: true, sig: Math.min(255, Math.round(30 + 40 * Math.log10(rx.S / rx.thr))) } : { dist: 1, peak: false, sig: 0 };
  L.busyUntil = mega.cpu.cycles + Math.round(ms * 16000);
  mega.at(L.busyUntil, () => { llFinish(L); onDone?.(); });
  return true;
}
// I2C 슬레이브 동작 (하드웨어 TWI와 비트뱅이 같이 씀)
function llAddr(L, a7, rd) {
  if (!llReady(L)) return false;
  const c = L.regs[0x1e], hit = (!(c & 8) && a7 === LL_ADDR) || ((c & 16) && a7 === (L.regs[0x1a] >> 1) && a7 !== 0);
  if (hit && !rd) L.first = true;
  return hit;
}
function llWrite(L, b) {
  if (L.first) { L.ptr = b & 0x7f; L.first = false; return; }
  const a = L.ptr; L.ptr = (L.ptr + 1) & 0x7f;
  if (a === 0x00) {
    if (b === 0) {                                            // 0 = 하드 리셋 (0x06 bit0로 허용한 경우만), 0이 아니면 측정 시작
      if (L.regs[0x06] & 1) { llDefaults(L); L.bootUntil = mega.cpu.cycles + LL_BOOT_MS * 16000; }
      return;
    }
    llStart(L, () => pwmAfter(L)); return;
  }
  if (a === 0x1e && (b & 16) && !((L.regs[0x18] << 8 | L.regs[0x19]) === L.unit)) b &= ~16;   // 시리얼 번호 확인 후에만 보조 주소 허용
  if (![0x01, 0x0e, 0x0f, 0x10, 0x16, 0x17, 0x48].includes(a)) L.regs[a] = b;
}
function llRead(L) {
  const a = L.ptr; L.ptr = (L.ptr + 1) & 0x7f; llFinish(L);
  switch (a) {
    case 0x01: return (mega.cpu.cycles < L.busyUntil ? 1 : 0) | (L.peak ? 8 : 0) | 0x30;    // bit0 busy, bit3 피크, bit4 DC 정상, bit5 정상
    case 0x0e: return L.sig;
    case 0x0f: return L.dist >> 8;
    case 0x10: return L.dist & 0xff;
    case 0x16: return L.unit >> 8;
    case 0x17: return L.unit & 0xff;
    default: return L.regs[a];
  }
}
// 하드웨어 TWI (Wire): D20 SDA / D21 SCL에 붙은 센서. 바이트당 9클럭(SCL 주파수 = TWBR 설정)만큼 시간이 걸림
function attachTwi(L) {
  const twi = mega.twi, cpu = mega.cpu, later = (bits, fn) => cpu.addClockEvent(fn, Math.max(1, Math.round(bits * 16e6 / twi.sclFrequency)));
  twi.eventHandler = {
    start() { later(1, () => twi.completeStart()); },
    stop() { later(1, () => twi.completeStop()); },
    connectToSlave(addr, write) { const ack = !!L && llAddr(L, addr, !write); later(9, () => twi.completeConnect(ack)); },
    writeByte(v) { if (L) llWrite(L, v); later(9, () => twi.completeWrite(!!L)); },
    readByte() { const v = L ? llRead(L) : 255; later(9, () => twi.completeRead(v)); },
  };
}
// 비트뱅 I2C (SoftwareWire 등): 핀 레벨에서 START/STOP·클럭 엣지를 보고 슬레이브처럼 SDA를 끌어내림. 풀업 저항은 있다고 가정
function attachBitbang(L, sdaN, sclN) {
  const [sp, sb] = PINMAP[sdaN], [cp, cb] = PINMAP[sclN];
  const st = { sda: true, scl: true, low: false, ph: "idle", cnt: 0, sh: 0, isAddr: false, next: "", byte: 0, bit: 0, mAck: false };
  const bus = () => { const sda = !(mega.drivenLow(sdaN) || st.low), scl = !mega.drivenLow(sclN);
    mega.port[sp].setPin(sb, sda); mega.port[cp].setPin(cb, scl); return [sda, scl]; };
  const drive = low => { st.low = low; st.sda = bus()[0]; };
  const tx = () => { st.byte = llRead(L); st.bit = 7; drive(!((st.byte >> 7) & 1)); st.ph = "tx"; };
  const onEdge = () => {
    const [sda, scl] = bus(), pS = st.sda, pC = st.scl;
    if (sda === pS && scl === pC) return;
    st.sda = sda; st.scl = scl;
    if (scl && pC) {                                          // SCL HIGH 중 SDA 변화 = START / STOP
      if (!sda) { st.low = false; bus(); st.ph = "rx"; st.cnt = 0; st.sh = 0; st.isAddr = true; } else { drive(false); st.ph = "idle"; }
      return;
    }
    if (scl && !pC) {                                         // 상승 엣지: 비트 샘플
      if (st.ph === "rx" && st.cnt < 8) { st.sh = (st.sh << 1) | (sda ? 1 : 0); st.cnt++; }
      else if (st.ph === "ackIn") st.mAck = !sda;
      return;
    }
    if (!scl && pC) {                                         // 하강 엣지: 다음 비트 준비
      if (st.ph === "rx" && st.cnt === 8) {
        if (st.isAddr) { st.isAddr = false; const rd = !!(st.sh & 1);
          if (!llAddr(L, st.sh >> 1, rd)) { st.ph = "ignore"; return; }
          st.next = rd ? "tx" : "rx"; }
        else { llWrite(L, st.sh); st.next = "rx"; }
        drive(true); st.ph = "ackOut";
      } else if (st.ph === "ackOut") {
        drive(false);
        if (st.next === "tx") tx(); else { st.ph = "rx"; st.cnt = 0; st.sh = 0; }
      } else if (st.ph === "tx") {
        st.bit--; if (st.bit >= 0) drive(!((st.byte >> st.bit) & 1)); else { drive(false); st.ph = "ackIn"; }
      } else if (st.ph === "ackIn") { if (st.mAck) tx(); else st.ph = "ignore"; }
    }
  };
  mega.port[sp].addListener(onEdge); if (cp !== sp) mega.port[cp].addListener(onEdge);
  bus();
}
// PWM 모드 (MODE 선): MTRIG(1 kΩ 경유) 핀이 출력 LOW면 측정 → 끝나면 MODE 선에 10 µs/cm HIGH 펄스 → 계속 LOW면 반복
// MTRIG를 연결 안 하면 1 kΩ로 GND에 고정한 것으로 보고 계속 측정. ACQ_CONFIG 0x04 bit1:0 = 01이면 측정 중 HIGH(상태 출력), 10이면 핀 트리거 안 함
function pwmOn(L) { return L.u.mode >= 0 && llReady(L); }
function pwmTrigLow(L) { return L.u.mtrig < 0 || mega.drivenLow(L.u.mtrig); }
function setMon(L, hi) { if (L.mon === hi) return; L.mon = hi; mega.setPin(L.u.mode, hi); }
function pwmPoll(L) {
  if (!pwmOn(L) || L.pulse || llBusy(L)) return;
  const m = L.regs[0x04] & 3; if (m === 2 || m === 3 || !pwmTrigLow(L)) return;
  if (llStart(L, () => pwmAfter(L)) && m === 1) setMon(L, true);
}
function pwmAfter(L) {
  if (L.u.mode < 0) return;
  const m = L.regs[0x04] & 3;
  if (m === 1) { setMon(L, false); pwmPoll(L); return; }
  if (m !== 0 || !pwmTrigLow(L)) return;                     // I2C로 건 측정도 MODE 선이 LOW로 잡혀 있을 때만 펄스
  L.pulse = true; setMon(L, true);
  mega.at(mega.cpu.cycles + L.dist * 10 * 16, () => { setMon(L, false); L.pulse = false; pwmPoll(L); });
}
// 전원 EN(주황) 핀: 내부 풀업 → Mega가 출력 LOW로 당길 때만 꺼짐. 다시 켜지면 레지스터 초기화 + 22 ms 부팅
function powerPoll(L) {
  const on = !(L.u.en >= 0 && mega.drivenLow(L.u.en));
  if (on === L.power) return;
  L.power = on;
  if (on) { llDefaults(L); L.bootUntil = mega.cpu.cycles + LL_BOOT_MS * 16000; mega.at(L.bootUntil + 1, () => pwmPoll(L)); }
  else { L.pend = null; L.busyUntil = 0; L.pulse = false; if (L.u.mode >= 0) setMon(L, false); }
}

// ---------------- 상태 ----------------
let irVar = [], floorSeed = 0, servoQ = [], usServo = [], usLast = [], lidars = [];
let cfg = null, hexText = "", mega = null, veh = null, meas = {}, others = [], serial = "", trail = [], pinState = {};
const D2R = Math.PI / 180, clamp = (a, lo, hi) => Math.min(hi, Math.max(lo, a));
const servoDeg = us => (us - 544) / (2400 - 544) * 180;
const fresh = (m, c) => m && m.last && c - m.last < 16000 * 50;

function boot() {
  mega = new Mega(hexText);
  meas = {}; serial = ""; pinState = {}; usLast = [];
  const act = cfg.act;
  const dec = new TextDecoder("utf-8");                       // Serial 바이트 → UTF-8 (한글 출력 깨짐 방지)
  mega.usart.onByteTransmit = b => { serial += dec.decode(new Uint8Array([b]), { stream: true }); };
  for (let n = 0; n < 16; n++) mega.adc.channelValues[n] = 0;
  if (act.encA >= 0) mega.setPin(act.encA, false); if (act.encB >= 0) mega.setPin(act.encB, false);
  const watch = (n, key) => { if (n < 0) return; const [p] = PINMAP[n]; const m = meas[key] = { hi: 0, per: 0, rise: 0, state: false };
    mega.port[p].addListener(() => { const s = mega.getPin(n), c = mega.cpu.cycles; if (s === m.state) return;
      if (s) { if (m.rise) m.per = (c - m.rise) / 16; m.rise = c; } else if (m.rise) m.hi = (c - m.rise) / 16; m.state = s; m.last = c; }); };
  watch(act.steer, "steer"); watch(act.brake, "brake"); watch(act.pwm, "pwm");
  setDip(cfg.dip || []);
  // 초음파: TRIG 하강 엣지 → 그 순간 자세로 거리 계산 → ECHO 펄스 예약
  // 서보 + 초음파 모듈: 서보 SIG 펄스 → 각도(90° = 정면, + = 반시계), 속도 제한
  usServo = cfg.us.map((u, i) => { if (u.servo == null || u.servo < 0) return null; watch(u.servo, "us" + i); return { ang: 90 }; });
  for (const [ui, u] of cfg.us.entries()) { if (u.kind === "lidar" || u.trig < 0 || u.echo < 0) continue; const [p] = PINMAP[u.trig]; let prev = false;
    mega.port[p].addListener(() => { const s = mega.getPin(u.trig); if (prev && !s) {
      const c = mega.cpu.cycles, co = Math.cos(veh.yaw), si = Math.sin(veh.yaw);
      const sx = veh.x + u.x * co - u.y * si, sy = veh.y + u.x * si + u.y * co;
      const d = usMeasure(sx, sy, veh.yaw + u.yaw + (usServo[ui] ? (usServo[ui].ang - 90) * D2R : 0), others);
      usLast[ui] = { d, c };                                  // 화면 표시용: 코드가 마지막으로 받은 거리
      const w = d == null ? 38000 : 2 * d / 0.0343;          // 왕복 [µs], 에코 없으면 38 ms
      mega.at(c + 460 * 16, () => mega.setPin(u.echo, true)); mega.at(c + (460 + w) * 16, () => mega.setPin(u.echo, false)); } prev = s; }); }
  // LIDAR-Lite v3HP: SDA/SCL = D20/D21이면 하드웨어 TWI(Wire), 그 밖의 핀이면 비트뱅 슬레이브. MODE·MTRIG·EN은 연결했을 때만
  lidars = [];
  let twiDev = null;
  for (const [ui, u] of cfg.us.entries()) { if (u.kind !== "lidar") continue;
    const L = newLidar(u); L.idx = ui; lidars.push(L);
    if (u.sda >= 0 && u.scl >= 0) { if (u.sda === 20 && u.scl === 21) twiDev = L; attachBitbang(L, u.sda, u.scl); }
    if (u.mode >= 0) { mega.setPin(u.mode, false); if (u.mtrig >= 0) mega.port[PINMAP[u.mtrig][0]].addListener(() => pwmPoll(L)); mega.at(L.bootUntil + 1, () => pwmPoll(L)); }
    if (u.en >= 0) mega.port[PINMAP[u.en][0]].addListener(() => powerPoll(L));
  }
  attachTwi(twiDev);
  const P0 = cfg.params, uni = () => Math.random() * 2 - 1;
  irVar = cfg.ir.map(() => ({ g: 1 + (P0.irVar || 0) / 100 * uni(), o: (P0.irVar || 0) / 100 * 0.5 * (P0.irWhite - P0.irBlack) * uni() }));
  floorSeed = Math.random() * 1000;
  veh = { x: cfg.start.x, y: cfg.start.y, yaw: cfg.start.yaw, v: 0, i: 0, vy: 0, r: 0, servo: cfg.params.sNeutral, wheel: cfg.params.sNeutral, q: 0, delta: 0, pwm: 0, dir: 0 };
  servoQ = [];
  trail = [[veh.x, veh.y]];
}

// 매끄러운 2D 값 노이즈 [-1, 1] (바닥 얼룩)
const hash = (i, j) => { const s = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; };
function valueNoise(x, y) { const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash(i, j), b = hash(i + 1, j), c = hash(i, j + 1), d = hash(i + 1, j + 1); return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy; }
// DIP 스위치: ON = GND에 붙어 LOW, OFF = 풀업으로 HIGH
function setDip(list) { for (const d of list) if (d.pin >= 0) mega.setPin(d.pin, !d.on); }
// IR: 센서 지름 irSpot 원 안 7점 반사율 평균 → ADC, 디지털은 임계값 비교
const SPOT = [[0, 0], ...[0, 1, 2, 3, 4, 5].map(i => [Math.cos(i * Math.PI / 3) * 0.7, Math.sin(i * Math.PI / 3) * 0.7])];
function updateIR() {
  const P = cfg.params, co = Math.cos(veh.yaw), si = Math.sin(veh.yaw), r = P.irSpot / 20;   // mm 지름 → cm 반경
  for (let k = 0; k < cfg.ir.length; k++) {
    const s = cfg.ir[k], wx = veh.x + s.x * co - s.y * si, wy = veh.y + s.x * si + s.y * co;
    let refl = 0; for (const [a, b] of SPOT) refl += reflAt(wx + a * r, wy + b * r); refl /= SPOT.length;
    if (P.irFloorVar) refl *= 1 + P.irFloorVar / 100 * valueNoise(wx / 8 + floorSeed, wy / 8);
    const V = irVar[k] || { g: 1, o: 0 };
    const adc = clamp(P.irBlack + V.o + (P.irWhite - P.irBlack) * V.g * (refl - 0.05) / (0.9 - 0.05) + randn() * (P.irNoise ?? 4), 0, 1023);
    if (s.analog >= 54) mega.adc.channelValues[s.analog - 54] = adc / 1023 * 5;
    if (s.digital >= 0) { const hi = (adc >= P.irTh) === !!P.irWhiteHigh; if (pinState[s.digital] !== hi) { pinState[s.digital] = hi; mega.setPin(s.digital, hi); } }
  }
}

// 구동: DC 모터(역기전력·저항·인덕턴스·마찰) + 감속기 + 차량 질량·구름저항
//   V = ±duty·Vbat, L di/dt = V − R i − Ke ω, F = gear·(Kt i − T_f)/r − Crr m g
//   Ke(=Kt)는 "PWM 255 이론 최고속도"(마찰 무시)에서 역산
function driveStep(P, duty, rev, brake, dt) {
  const r = P.wheelD / 200, gear = P.gear, m = P.mass || 1.5, g = 9.81;
  const vbat = P.vbat || 7.4, R = Math.max(0.05, P.mR || 1.5), Lm = Math.max(1e-6, (P.mL ?? 0.1) / 1000);
  const Ke = vbat / (gear * (P.vmax / 100) / r), Tf = (P.mFric || 0) / 1000;
  const meq = m + (P.jEq || 0), Fmax = (P.mu || 1.0) * m * g * 0.5;   // 구동륜 견인 한계 (하중 절반 가정)
  const V = (rev ? -1 : 1) * duty * vbat, N = 10, h = dt / N, te = Lm / R;
  let v = veh.v / 100, i = veh.i || 0;
  for (let k = 0; k < N; k++) {
    const w = gear * v / r;
    const coast = !P.slowDecay && duty === 0;
    const iss = coast ? 0 : (V - Ke * w) / R;
    i = iss + (i - iss) * Math.exp(-h / te);
    const Fm = gear * Ke * i / r;                              // 모터 힘
    const Fres = gear * Tf / r + (P.crr ?? 0.03) * m * g + (brake ? 0.6 * m * g : 0);   // 마찰·구름·브레이크 (속도 반대)
    let F = clamp(Fm, -Fmax, Fmax);
    if (Math.abs(v) < 1e-4) {                                  // 정지: 정지마찰을 넘을 때만 출발
      if (Math.abs(F) <= Fres) { v = 0; continue; }
      F -= Math.sign(F) * Fres;
    } else F -= Math.sign(v) * Fres;
    const v1 = v + F / meq * h;
    v = (v !== 0 && Math.sign(v1) !== Math.sign(v) && Math.abs(Fm) <= Fres) ? 0 : v1;
  }
  veh.v = v * 100; veh.i = i;
}

function vehicleStep(dt, cycle0) {
  const P = cfg.params, act = cfg.act, c = mega.cpu.cycles;
  usServo.forEach((sv, i) => { if (!sv) return; const m = meas["us" + i];             // 초음파 서보: 약 500 °/s (SG90 급)
    const tgt = fresh(m, c) && m.hi > 400 ? clamp(servoDeg(m.hi), 0, 180) : sv.ang; sv.ang += clamp(tgt - sv.ang, -500 * dt, 500 * dt); });
  const ms = meas.steer;
  let cmd = fresh(ms, c) && ms.hi > 400 ? servoDeg(ms.hi) : null;
  // 서보: 응답 지연(데드타임) → 속도 제한 → 기구 한계 → 백래시(유격) → 바퀴
  servoQ.push(cmd); const lag = Math.max(0, Math.round((P.sDelay || 0) / (dt * 1000)));
  while (servoQ.length > lag + 1) servoQ.shift();
  let target = servoQ[0] ?? veh.servo;
  target = clamp(target, P.sMin, P.sMax);
  veh.servo = P.sRate > 0 ? veh.servo + clamp(target - veh.servo, -P.sRate * dt, P.sRate * dt) : target;
  const bl = (P.sBacklash || 0) / 2;
  if (veh.servo - veh.wheel > bl) veh.wheel = veh.servo - bl; else if (veh.wheel - veh.servo > bl) veh.wheel = veh.servo + bl;
  const mp = meas.pwm;
  let duty = 0; if (mp) duty = fresh(mp, c) && mp.per > 0 ? mp.hi / mp.per : (mp.state ? 1 : 0);
  const pwm = Math.round(duty * 255), rev = act.dir >= 0 && (mega.getPin(act.dir) === !!P.dirRev);
  const mb = meas.brake, brake = !!(mb && fresh(mb, c) && servoDeg(mb.hi) >= 20);
  driveStep(P, duty, rev, brake, dt);
  veh.pwm = pwm; veh.dir = rev ? 1 : 0;
  const d = P.sSign * P.sGain * (veh.wheel - P.sNeutral - (P.sTrim || 0)) * D2R;
  veh.delta = d;
  const L = cfg.body.L / 100, vx = veh.v / 100;             // SI
  const b = clamp((P.cgX ?? L * 50) / 100, 0.01, L - 0.01), a = L - b;
  if (P.dyn && Math.abs(vx) > 0.15) {
    // 동역학 자전거 모델 (무게중심 기준 vy, r)
    // 타이어 횡력: 정지마찰 구간은 매우 단단한 선형(접지면 변형만, 실제 미끄러짐 거의 없음) → μs·Fz에서 최대 →
    // 넘으면 미끄러지며 운동마찰 μk·Fz로 떨어짐 (꼭짓점 뒤 약 3° 동안 감소)
    const m = P.mass || 1.5, g = 9.81, Iz = m * (a * a + b * b) / 2, muS = P.mu || 1.0, muK = Math.min(P.muK ?? 0.8, muS);
    const Fzf = m * g * b / L, Fzr = m * g * a / L, sx = Math.sign(vx), ax = Math.abs(vx);
    const tire = (C, Fz, al) => { const ap = muS * Fz / C, aa = Math.abs(al);
      if (aa <= ap) return C * al;                                      // 접지 유지 (정지마찰)
      return Math.sign(al) * Fz * (muK + (muS - muK) * Math.exp(-(aa - ap) / 0.05));  // 미끄러짐 (운동마찰로 감소)
    };
    const N = 20, h = dt / N;                                  // 단단한 타이어라 0.05 ms로 잘게 적분
    let slipF = false, slipR = false;
    for (let i = 0; i < N; i++) {
      const af = d - Math.atan2(veh.vy + a * veh.r, ax) * sx, ar = -Math.atan2(veh.vy - b * veh.r, ax) * sx;
      const Fyf = tire(P.cf, Fzf, af), Fyr = tire(P.cr, Fzr, ar);
      slipF ||= Math.abs(af) > muS * Fzf / P.cf; slipR ||= Math.abs(ar) > muS * Fzr / P.cr;
      const dvy = (Fyf * Math.cos(d) + Fyr) / m - vx * veh.r, dr = (a * Fyf * Math.cos(d) - b * Fyr) / Iz;
      veh.vy += dvy * h; veh.r += dr * h;
    }
    veh.slip = slipF || slipR ? (slipR ? 2 : 1) : 0;
  } else {                                                   // 저속: 미끄러짐 없는 기구학
    veh.r = vx * Math.tan(d) / L; veh.vy = b * veh.r; veh.slip = 0;
  }
  // 후축 중심 속도 = (vx, vy − b·r)
  const vRy = veh.vy - b * veh.r, co = Math.cos(veh.yaw), si = Math.sin(veh.yaw), ds = veh.v * dt;
  veh.x += (vx * co - vRy * si) * 100 * dt; veh.y += (vx * si + vRy * co) * 100 * dt;
  veh.yaw = Math.atan2(Math.sin(veh.yaw + veh.r * dt), Math.cos(veh.yaw + veh.r * dt));
  if (act.encA >= 0 || act.encB >= 0) {                    // 엔코더 쿼드러처 엣지를 이번 1 ms 안 시각에 예약
    const cpc = P.ppr * 4 * P.gear / (Math.PI * P.wheelD);
    const dq = ds * cpc * (P.aLeads ? 1 : -1), q0 = veh.q, q1 = q0 + dq;
    const setAB = n => { const m = ((n % 4) + 4) % 4; if (act.encA >= 0) mega.setPin(act.encA, m === 1 || m === 2); if (act.encB >= 0) mega.setPin(act.encB, m === 2 || m === 3); };
    if (dq > 0) for (let n = Math.floor(q0) + 1; n <= Math.floor(q1); n++) mega.at(cycle0 + (n - q0) / dq * 16000, () => setAB(n));
    else if (dq < 0) for (let n = Math.floor(q0); n > Math.floor(q1); n--) mega.at(cycle0 + (q0 - n) / -dq * 16000, () => setAB(n - 1));
    veh.q = q1;
  }
}

// 화면 표시용 초음파 거리: 코드가 0.3 s 안에 쟀으면 그 값(src 1), 아니면 지금 자세로 한 번 재 본 값(src 0). null = 에코 없음
let probe = true;
function usView() {
  if (!cfg || !veh) return [];
  const now = mega ? mega.cpu.cycles : 0, co = Math.cos(veh.yaw), si = Math.sin(veh.yaw);
  return cfg.us.map((u, ui) => {
    const L = usLast[ui];
    if (L && now - L.c < 0.3 * 16e6) return [L.d == null ? null : +L.d.toFixed(1), 1];
    if (u.kind === "lidar") {                                 // 라이다: 코드가 안 쟀으면 지금 자세로 노이즈 없이 한 번 계산
      if (u.sda < 0 && u.mode < 0) return [null, 0];
      if (!probe) return null;
      const [lx, ly, lh] = lidarPose(ui), L = lidars.find(q => q.idx === ui);
      const r = lidarRaw(lx, ly, lh, others, L ? L.regs[0x02] || 1 : 255, (cfg.params.llRefl ?? 70) / 100);
      return [r ? +r.d.toFixed(1) : null, 0];
    }
    if (u.trig < 0 && u.echo < 0) return [null, 0];
    if (!probe) return null;                                  // 선택 안 된 차는 화면용 측정 생략 (CPU 절약)
    const sx = veh.x + u.x * co - u.y * si, sy = veh.y + u.x * si + u.y * co;
    const d = usMeasure(sx, sy, veh.yaw + u.yaw + (usServo[ui] ? (usServo[ui].ang - 90) * D2R : 0), others);
    return [d == null ? null : +d.toFixed(1), 0];
  });
}

function state() {
  const out = { type: "state", usD: usView(), x: veh.x, y: veh.y, yaw: veh.yaw, v: veh.v, slip: veh.slip || 0, usAng: usServo.map(s => s ? (s.ang - 90) * D2R : 0), delta: veh.delta, servo: veh.servo, pwm: veh.pwm, dir: veh.dir, t: mega.cpu.cycles / 16e6, serial, trail };
  serial = ""; trail = [];
  return out;
}

self.onmessage = ({ data: m }) => {
  try {
    if (m.type === "map") { buildMap(m.map); return; }
    if (m.type === "dip") { if (cfg) cfg.dip = m.dip; if (mega) setDip(m.dip); return; }
    if (m.type === "init") { cfg = m.cfg; hexText = m.hex; boot(); self.postMessage({ type: "ready" }); self.postMessage(state()); return; }
    if (m.type === "reset") { if (m.cfg) cfg = m.cfg; boot(); self.postMessage(state()); return; }
    if (m.type === "step") {
      others = m.others || []; probe = m.probe !== false;
      if (m.pose) { veh.x = m.pose.x; veh.y = m.pose.y; veh.yaw = m.pose.yaw; veh.v = 0; veh.i = 0; veh.vy = 0; veh.r = 0; }
      for (let i = 0; i < m.n; i++) {
        const c0 = mega.cpu.cycles;
        vehicleStep(0.001, c0); updateIR(); mega.runTo(c0 + 16000);
        const lp = trail.length ? trail[trail.length - 1] : null;
        if (!lp || Math.hypot(veh.x - lp[0], veh.y - lp[1]) > 1) trail.push([veh.x, veh.y]);
      }
      self.postMessage(state());
    }
  } catch (e) { self.postMessage({ type: "error", message: String(e.message || e) }); }
};
