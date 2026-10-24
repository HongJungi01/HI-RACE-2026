// 차량 1대 시뮬레이션 워커 (모듈 워커)
// ATmega2560 에뮬레이션(avr8js) + 차량 모델(기구학 자전거) + 엔코더 + IR(맵 선 벡터 샘플링) + 초음파(레이 모델, 상대 차량 외곽)
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
  return { ports, timers, adc, usart };
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
    this.events = [];
  }
  setPin(n, hi) { const [p, b] = PINMAP[n]; this.port[p].setPin(b, hi); }
  getPin(n) { const [p, b] = PINMAP[n]; return this.port[p].pinState(b) === A.PinState.High; }
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

// ---------------- 상태 ----------------
let irVar = [], floorSeed = 0, servoQ = [];
let cfg = null, hexText = "", mega = null, veh = null, meas = {}, others = [], serial = "", trail = [], pinState = {};
const D2R = Math.PI / 180, clamp = (a, lo, hi) => Math.min(hi, Math.max(lo, a));
const servoDeg = us => (us - 544) / (2400 - 544) * 180;
const fresh = (m, c) => m && m.last && c - m.last < 16000 * 50;

function boot() {
  mega = new Mega(hexText);
  meas = {}; serial = ""; pinState = {};
  const act = cfg.act;
  mega.usart.onByteTransmit = b => { serial += String.fromCharCode(b); };
  for (let n = 0; n < 16; n++) mega.adc.channelValues[n] = 0;
  if (act.encA >= 0) mega.setPin(act.encA, false); if (act.encB >= 0) mega.setPin(act.encB, false);
  const watch = (n, key) => { if (n < 0) return; const [p] = PINMAP[n]; const m = meas[key] = { hi: 0, per: 0, rise: 0, state: false };
    mega.port[p].addListener(() => { const s = mega.getPin(n), c = mega.cpu.cycles; if (s === m.state) return;
      if (s) { if (m.rise) m.per = (c - m.rise) / 16; m.rise = c; } else if (m.rise) m.hi = (c - m.rise) / 16; m.state = s; m.last = c; }); };
  watch(act.steer, "steer"); watch(act.brake, "brake"); watch(act.pwm, "pwm");
  // 초음파: TRIG 하강 엣지 → 그 순간 자세로 거리 계산 → ECHO 펄스 예약
  for (const u of cfg.us) { if (u.trig < 0 || u.echo < 0) continue; const [p] = PINMAP[u.trig]; let prev = false;
    mega.port[p].addListener(() => { const s = mega.getPin(u.trig); if (prev && !s) {
      const c = mega.cpu.cycles, co = Math.cos(veh.yaw), si = Math.sin(veh.yaw);
      const sx = veh.x + u.x * co - u.y * si, sy = veh.y + u.x * si + u.y * co;
      const d = usMeasure(sx, sy, veh.yaw + u.yaw, others);
      const w = d == null ? 38000 : 2 * d / 0.0343;          // 왕복 [µs], 에코 없으면 38 ms
      mega.at(c + 460 * 16, () => mega.setPin(u.echo, true)); mega.at(c + (460 + w) * 16, () => mega.setPin(u.echo, false)); } prev = s; }); }
  const P0 = cfg.params, uni = () => Math.random() * 2 - 1;
  irVar = cfg.ir.map(() => ({ g: 1 + (P0.irVar || 0) / 100 * uni(), o: (P0.irVar || 0) / 100 * 0.5 * (P0.irWhite - P0.irBlack) * uni() }));
  floorSeed = Math.random() * 1000;
  veh = { x: cfg.start.x, y: cfg.start.y, yaw: cfg.start.yaw, v: 0, vy: 0, r: 0, servo: cfg.params.sNeutral, wheel: cfg.params.sNeutral, q: 0, delta: 0, pwm: 0, dir: 0 };
  servoQ = [];
  trail = [[veh.x, veh.y]];
}

// 매끄러운 2D 값 노이즈 [-1, 1] (바닥 얼룩)
const hash = (i, j) => { const s = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; };
function valueNoise(x, y) { const i = Math.floor(x), j = Math.floor(y), fx = x - i, fy = y - j, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash(i, j), b = hash(i + 1, j), c = hash(i, j + 1), d = hash(i + 1, j + 1); return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy; }
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

function vehicleStep(dt, cycle0) {
  const P = cfg.params, act = cfg.act, c = mega.cpu.cycles;
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
  const mag = pwm <= P.db ? 0 : P.vmax * (pwm - P.db) / (255 - P.db);
  const vCmd = brake ? 0 : (rev ? -mag : mag), tau = brake ? P.brakeTau : P.tau;
  veh.v = tau > 0 ? vCmd + (veh.v - vCmd) * Math.exp(-dt / tau) : vCmd;
  veh.pwm = pwm; veh.dir = rev ? 1 : 0;
  const d = P.sSign * P.sGain * (veh.wheel - P.sNeutral - (P.sTrim || 0)) * D2R;
  veh.delta = d;
  const L = cfg.body.L / 100, vx = veh.v / 100;             // SI
  const b = clamp((P.cgX ?? L * 50) / 100, 0.01, L - 0.01), a = L - b;
  if (P.dyn && Math.abs(vx) > 0.15) {
    // 동역학 자전거 모델 (무게중심 기준 vy, r), 타이어 횡력 = μFz·tanh(Cα/μFz)
    const m = P.mass || 1.5, g = 9.81, Iz = m * (a * a + b * b) / 2, mu = P.mu || 0.8;
    const Fzf = m * g * b / L, Fzr = m * g * a / L;
    const af = d - Math.atan2(veh.vy + a * veh.r, Math.abs(vx)) * Math.sign(vx), ar = -Math.atan2(veh.vy - b * veh.r, Math.abs(vx)) * Math.sign(vx);
    const Fyf = mu * Fzf * Math.tanh(P.cf * af / (mu * Fzf)), Fyr = mu * Fzr * Math.tanh(P.cr * ar / (mu * Fzr));
    // 1 ms를 잘게 나눠 안정 적분
    const N = 4, h = dt / N;
    for (let i = 0; i < N; i++) {
      const dvy = (Fyf * Math.cos(d) + Fyr) / m - vx * veh.r, dr = (a * Fyf * Math.cos(d) - b * Fyr) / Iz;
      veh.vy += dvy * h; veh.r += dr * h;
    }
  } else {                                                   // 저속: 미끄러짐 없는 기구학
    veh.r = vx * Math.tan(d) / L; veh.vy = b * veh.r;
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

function state() {
  const out = { type: "state", x: veh.x, y: veh.y, yaw: veh.yaw, v: veh.v, delta: veh.delta, servo: veh.servo, pwm: veh.pwm, dir: veh.dir, t: mega.cpu.cycles / 16e6, serial, trail };
  serial = ""; trail = [];
  return out;
}

self.onmessage = ({ data: m }) => {
  try {
    if (m.type === "map") { buildMap(m.map); return; }
    if (m.type === "init") { cfg = m.cfg; hexText = m.hex; boot(); self.postMessage({ type: "ready" }); self.postMessage(state()); return; }
    if (m.type === "reset") { if (m.cfg) cfg = m.cfg; boot(); self.postMessage(state()); return; }
    if (m.type === "step") {
      others = m.others || [];
      if (m.pose) { veh.x = m.pose.x; veh.y = m.pose.y; veh.yaw = m.pose.yaw; veh.v = 0; veh.vy = 0; veh.r = 0; }
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
