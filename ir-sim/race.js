// 기록모드 판정 로직 (DOM 없음, 단위: cm · ms · rad). 설계: claude/race_record_mode.md
const Race = (() => {
  const LANE_W = 45;
  const isWhite = l => (l.ir ?? 0) >= 0.85;

  // ---------- 코스 좌표 ----------
  // 1차로 선(맵 start에 가장 가까운 닫힌 흰 선)을 기준 경로로, 진행 거리 s ∈ [0, L), 바깥쪽 + 횡거리 d
  function buildTrack(map) {
    const st = map.start, whites = map.lines.filter(isWhite);
    const nearestIdx = pts => { let bi = 0, bd = Infinity; pts.forEach((p, i) => { const d = (p[0] - st.x) ** 2 + (p[1] - st.y) ** 2; if (d < bd) { bd = d; bi = i; } }); return [bi, bd]; };
    let base = null, bestD = Infinity;
    for (const l of whites) if (l.closed) { const [i, d] = nearestIdx(l.pts); if (d < bestD) { bestD = d; base = { l, i }; } }
    let P = base.l.pts.slice(base.i).concat(base.l.pts.slice(0, base.i));
    const tx = P[1][0] - P[0][0], ty = P[1][1] - P[0][1];
    if (tx * Math.cos(st.yaw) + ty * Math.sin(st.yaw) < 0) P = [P[0], ...P.slice(1).reverse()];   // 진행 방향으로 점 순서 맞춤
    const n = P.length, S = new Float64Array(n), TX = new Float64Array(n), TY = new Float64Array(n);
    for (let i = 0; i < n; i++) { const a = P[i], b = P[(i + 1) % n], dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1e-9;
      TX[i] = dx / len; TY[i] = dy / len; if (i + 1 < n) S[i + 1] = S[i] + len; }
    const L = S[n - 1] + Math.hypot(P[0][0] - P[n - 1][0], P[0][1] - P[n - 1][1]);

    // 흰 선 점 격자 (20 cm 칸)
    const CELL = 20, grid = new Map(), key = (cx, cy) => cx * 100003 + cy;
    whites.forEach((l, li) => l.pts.forEach((p, pi) => { const k = key(Math.floor(p[0] / CELL), Math.floor(p[1] / CELL)); if (!grid.has(k)) grid.set(k, []); grid.get(k).push(li, pi); }));
    const around = (x, y, r, fn) => { const c0 = Math.floor((x - r) / CELL), c1 = Math.floor((x + r) / CELL), r0 = Math.floor((y - r) / CELL), r1 = Math.floor((y + r) / CELL);
      for (let cx = c0; cx <= c1; cx++) for (let cy = r0; cy <= r1; cy++) { const a = grid.get(key(cx, cy)); if (a) for (let k = 0; k < a.length; k += 2) fn(a[k], a[k + 1]); } };

    // 바깥쪽 = 다른 흰 선이 있는 쪽 (출발점에서 판정)
    let side = 1; { let bd = Infinity; around(P[0][0], P[0][1], 70, (li, pi) => { const q = whites[li].pts[pi], dx = q[0] - P[0][0], dy = q[1] - P[0][1], dd = dx * dx + dy * dy;
      if (dd > 400 && dd < bd && Math.abs(dx * TX[0] + dy * TY[0]) < 3) { bd = dd; side = (dx * TY[0] - dy * TX[0]) >= 0 ? 1 : -1; } }); }
    const NX = i => side * TY[i], NY = i => -side * TX[i];   // 바깥 법선

    // 1차로 점마다 가장 바깥 흰 선의 d (법선 위에 있는 선 점만)
    const outer = new Float32Array(n);
    for (let i = 0; i < n; i++) { let o = 0; const p = P[i];
      around(p[0], p[1], 2 * LANE_W + 30, (li, pi) => { const q = whites[li].pts[pi], dx = q[0] - p[0], dy = q[1] - p[1];
        const along = dx * TX[i] + dy * TY[i], d = dx * NX(i) + dy * NY(i); if (Math.abs(along) < 2 && d > o) o = d; });
      outer[i] = o; }
    for (let k = 0; k < 2; k++) for (let i = 0; i < n; i++) if (outer[i] < LANE_W - 3) outer[i] = Math.max(outer[(i + n - 1) % n], outer[(i + 1) % n], outer[i]);   // 선 점 틈 메움

    const idxAt = s => { s = ((s % L) + L) % L; let lo = 0, hi = n - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (S[m] <= s) lo = m; else hi = m - 1; } return lo; };
    function project(x, y, hint) {
      let bi = -1, bd = Infinity;
      const scan = (a, b) => { for (let k = a; k <= b; k++) { const i = ((k % n) + n) % n, d = (P[i][0] - x) ** 2 + (P[i][1] - y) ** 2; if (d < bd) { bd = d; bi = i; } } };
      if (hint != null && hint >= 0) scan(hint - 80, hint + 80);
      if (bi < 0 || bd > 150 * 150) { bd = Infinity; scan(0, n - 1); }
      const dx = x - P[bi][0], dy = y - P[bi][1];
      let s = S[bi] + dx * TX[bi] + dy * TY[bi]; s = ((s % L) + L) % L;
      return { s, d: dx * NX(bi) + dy * NY(bi), i: bi, yaw: Math.atan2(TY[bi], TX[bi]) };
    }
    const outerAt = s => outer[idxAt(s)];
    const normalAt = s => { const i = idxAt(s); return [NX(i), NY(i)]; };
    function nearestWhite(x, y) {
      let best = null, bd = Infinity;
      for (const r of [40, 120, 400]) { around(x, y, r, (li, pi) => { const q = whites[li].pts[pi], d = (q[0] - x) ** 2 + (q[1] - y) ** 2; if (d < bd) { bd = d; best = q; } }); if (best) break; }
      if (!best) return null;
      const p = project(best[0], best[1]);
      return { x: best[0], y: best[1], yaw: p.yaw, s: p.s, lane: Math.max(1, Math.round(p.d / LANE_W) + 1) };
    }
    function pointAt(s, lane = 1) {
      const i = idxAt(s), t = ((s % L) + L) % L - S[i], off = (lane - 1) * LANE_W;
      const x = P[i][0] + TX[i] * t + NX(i) * off, y = P[i][1] + TY[i] * t + NY(i) * off, yaw = Math.atan2(TY[i], TX[i]);
      if (lane === 1) return { x, y, yaw };
      const w = nearestWhite(x, y); return w && Math.hypot(w.x - x, w.y - y) < 10 ? { x: w.x, y: w.y, yaw } : { x, y, yaw };
    }
    return { L, project, outerAt, normalAt, nearestWhite, pointAt, laneW: LANE_W };
  }

  // ---------- 이탈·멈춤 ----------
  function makeOffTrack(track, { margin = 10, holdMs = 1000, stallMs = 5000, stallV = 2 } = {}) {
    let outSince = null, stillSince = null, hint = -1;
    return {
      update(t, x, y, v) {
        const p = track.project(x, y, hint); hint = p.i;
        const out = p.d < -margin || p.d > track.outerAt(p.s) + margin;
        if (out) { stillSince = null; if (outSince == null) outSince = t; return t - outSince >= holdMs ? "off" : "candidate"; }
        outSince = null;
        if (Math.abs(v) < stallV) { if (stillSince == null) stillSince = t; if (t - stillSince >= stallMs) return "stall"; } else stillSince = null;
        return "ok";
      },
      reset() { outSince = stillSince = null; hint = -1; },
    };
  }

  // ---------- 랩·구간 ----------
  // 구간 경계 = L·k/nSec (k = nSec는 출발선). 다음 경계를 앞으로 지날 때만 인정 → 건너뛰기·역주행·왕복은 무시
  function makeLapCounter(L, nSec = 3) {
    const segOf = s => Math.min(nSec - 1, Math.floor((((s % L) + L) % L) / (L / nSec)));
    let prev = null, armed = false, expect = 1, lapStart = 0, secStart = 0;
    const c = { laps: 0, lapTimes: [], sectorTimes: [], best: null, update };
    function update(t, s) {
      const ev = [], cur = segOf(s);
      if (prev == null) { prev = cur; armed = cur !== nSec - 1; expect = armed ? 1 : nSec; return ev; }   // 출발선 뒤(그리드)에서 시작하면 첫 통과는 출발
      if (cur === (prev + 1) % nSec) {                                   // 경계 하나를 앞으로 지남
        const k = cur === 0 ? nSec : cur;
        if (!armed) { if (k === nSec) { armed = true; expect = 1; } }
        else if (k === expect) {
          c.sectorTimes.push(t - secStart); secStart = t; ev.push({ type: "sector", t, lap: c.laps + 1, sec: k, time: c.sectorTimes.at(-1) });
          if (k === nSec) { c.laps++; const lt = t - lapStart; lapStart = t; c.lapTimes.push(lt); if (c.best == null || lt < c.best) c.best = lt; ev.push({ type: "lap", t, lap: c.laps, time: lt }); expect = 1; }
          else expect++;
        }
      }
      prev = cur; return ev;
    }
    return c;
  }

  // ---------- 충돌 판정 ----------
  function makeLaneHistory({ winMs = 500, dTh = 6, headTh = 8 * Math.PI / 180 } = {}) {
    const H = [];
    return {
      push(t, dLine, headErr) { H.push([t, Math.abs(dLine), Math.abs(headErr)]); while (H.length && H[0][0] < t - winMs) H.shift(); },
      changing(t) { return H.some(([ht, d, h]) => ht >= t - winMs && (d > dTh || h > headTh)); },
    };
  }
  // a, b = { id, dummy, s, d, v, lane, changing, contactFront }
  function judgeHit(a, b, { L }) {
    const pen = ids => ({ penalized: ids.filter(x => x), reason: null });
    if (a.changing || b.changing) { const r = pen([!a.dummy && a.id, !b.dummy && b.id]); r.reason = "change"; return r; }
    if (a.lane !== b.lane) return { penalized: [], reason: "side" };
    let ds = (b.s - a.s) % L; if (ds > L / 2) ds -= L; if (ds < -L / 2) ds += L;
    const [rear, front] = ds >= 0 ? [a, b] : [b, a];
    if (!rear.dummy && rear.contactFront && !front.contactFront && rear.v > front.v) return { penalized: [rear.id], reason: "rear" };
    return { penalized: [], reason: "none" };
  }

  // ---------- 순위 ----------
  function classify(entries) {
    const maxLaps = Math.max(0, ...entries.map(e => e.laps));
    return entries.map(e => ({ ...e, total: (e.finishT ?? Infinity) + (e.penalty || 0), lapsDown: maxLaps - e.laps }))
      .sort((x, y) => y.laps - x.laps || x.total - y.total).map((e, i) => ({ ...e, pos: i + 1 }));
  }

  return { buildTrack, makeOffTrack, makeLapCounter, makeLaneHistory, judgeHit, classify, LANE_W };
})();
if (typeof module !== "undefined") module.exports = Race;
