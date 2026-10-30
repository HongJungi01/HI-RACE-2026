// Mega 2560 브라우저 컴파일러 워커: .ino 텍스트 → Intel HEX
// 도구: @horang-corp/avr-gcc-wasm 0.2.0 의 cc1plus / avr-as / avr-ld / avr-objcopy (WebAssembly)
// 자산: Arduino AVR core 1.8.x mega variant, Servo, Wire, SPI, EEPROM, SoftwareWire 1.6.0(GPLv3), Garmin LIDARLite_v3HP(Apache 2.0) 헤더와 ATmega2560용 미리 빌드한 오브젝트, avr6 libc/libm/libgcc
const base = new URL("./", self.location.href);
let assets = null; const wasm = {};
async function loadAssets() {
  if (!assets) {
    const j = await (await fetch(new URL("mega_assets.json", base))).json();
    assets = Object.entries(j).map(([p, b64]) => [p, Uint8Array.from(atob(b64), c => c.charCodeAt(0))]);
  }
  return assets;
}
async function tool(name, args, files, out, log) {
  if (!wasm[name]) wasm[name] = new Uint8Array(await (await fetch(new URL(`tools/${name}.wasm`, base))).arrayBuffer());
  const factory = (await import(new URL(`tools/${name}.js`, base).href)).default;
  const err = [];
  const mod = await factory({ noInitialRun: true, wasmBinary: wasm[name], print() {}, printErr: l => l && err.push(l) });
  for (const [vp, data] of files) { let c = ""; for (const p of vp.split("/").slice(1, -1)) { c += "/" + p; try { mod.FS.mkdir(c); } catch {} } mod.FS.writeFile(vp, data); }
  let failed = null;
  try { mod.callMain(args); } catch (e) { if (e?.status !== 0 && !/exit\(0\)/.test(String(e?.message))) failed = e; }
  log.push(...err.map(l => `[${name}] ${l}`));
  let o; try { o = mod.FS.readFile(out); } catch { failed = failed || new Error(`${name} 출력 없음`); }
  if (failed) throw Object.assign(new Error(`${name} 실패`), { log });
  return o;
}
// Arduino IDE처럼: Arduino.h 포함 + 함수 원형 자동 선언. 원형은 첫 함수 정의 바로 앞에 넣음
// (그래야 #include한 라이브러리 타입이나 그 위에서 정의한 구조체를 매개변수로 쓰는 함수도 원형이 컴파일됨)
function preprocess(src) {
  const protos = [], seen = new Set();
  const noComments = src.replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, " ")).replace(/\/\/.*$/gm, m => " ".repeat(m.length));   // 길이 유지 → 위치가 원본과 같음
  const re = /^[ \t]*((?:static\s+|inline\s+)*(?:unsigned\s+|signed\s+|const\s+)*[A-Za-z_][\w:<>]*[\s\*&]+)([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*\{/gm;
  let m, depth = 0, last = 0, first = -1;
  while ((m = re.exec(noComments))) {
    const before = noComments.slice(last, m.index); depth += (before.match(/\{/g) || []).length - (before.match(/\}/g) || []).length; last = m.index;
    if (depth !== 0) continue;
    const name = m[2]; if (["if", "for", "while", "switch", "return", "else"].includes(name) || /^(?:else|return)\b/.test(m[1].trim())) continue;
    if (first < 0) first = m.index;
    const sig = `${m[1].trim()} ${name}(${m[3].trim()});`; if (!seen.has(sig)) { seen.add(sig); protos.push(sig); }
  }
  if (first < 0) return `#include <Arduino.h>\n#line 1 "sketch.ino"\n${src}`;
  const line = src.slice(0, first).split("\n").length;
  return `#include <Arduino.h>\n#line 1 "sketch.ino"\n${src.slice(0, first)}${protos.join("\n")}\n#line ${line} "sketch.ino"\n${src.slice(first)}`;
}
self.onmessage = async ({ data }) => {
  const log = [], t0 = performance.now();
  try {
    const A = await loadAssets(); const tLoad = performance.now();
    const headers = A.filter(([p]) => !p.startsWith("/objects/") && !p.startsWith("/libs/") && !p.startsWith("/ldscripts/"));
    const asm = await tool("cc1plus", ["-quiet", "-imultilib", "avr6", "-D__AVR_ATmega2560__", "-D__AVR_DEVICE_NAME__=atmega2560", "-DF_CPU=16000000L", "-DARDUINO=10819",
      "-DARDUINO_AVR_MEGA2560", "-DARDUINO_ARCH_AVR", "-isystem", "/sysroot/gcc/include", "-isystem", "/sysroot/avr/include", "-I", "/arduino/core", "-I", "/arduino/variant",
      "-I", "/libraries/Servo/src", "-I", "/arduino/libraries/Wire/src", "-I", "/arduino/libraries/SPI/src", "-I", "/arduino/libraries/EEPROM/src", "-I", "/libraries/SoftwareWire/src", "-I", "/libraries/LIDARLite/src", "/build/sketch.cpp",
      "-mn-flash=4", "-mno-skip-bug", "-dumpbase", "sketch.cpp", "-mmcu=avr6", "-auxbase-strip", "/build/sketch.s", "-Os", "-std=gnu++11", "-fpermissive", "-fno-exceptions",
      "-fno-threadsafe-statics", "-fno-rtti", "-fno-enforce-eh-specs", "-ffunction-sections", "-fdata-sections", "-o", "/build/sketch.s"],
      [...headers, ["/build/sketch.cpp", new TextEncoder().encode(preprocess(data.source))]], "/build/sketch.s", log);
    if (log.some(l => /: error:/.test(l))) throw Object.assign(new Error("컴파일 오류"), { log });
    const obj = await tool("avr-as", ["-mmcu=atmega2560", "-o", "/build/sketch.o", "/build/sketch.s"], [["/build/sketch.s", asm]], "/build/sketch.o", log);
    const objs = A.filter(([p]) => p.startsWith("/objects/")), libs = A.filter(([p]) => p.startsWith("/libs/") || p.startsWith("/ldscripts/"));
    const elf = await tool("avr-ld", ["-m", "avr6", "-Tdata=0x800200", "--gc-sections", "-o", "/build/sketch.elf", "/libs/crtatmega2560.o", "/build/sketch.o", ...objs.map(o => o[0]),
      "-L/libs", "-lm", "-lc", "-lgcc"], [["/build/sketch.o", obj], ...objs, ...libs], "/build/sketch.elf", log);
    const hex = await tool("avr-objcopy", ["-O", "ihex", "-R", ".eeprom", "/build/sketch.elf", "/build/sketch.hex"], [["/build/sketch.elf", elf]], "/build/sketch.hex", log);
    const text = new TextDecoder().decode(hex);
    let flash = 0; for (const l of text.split(/\r?\n/)) if (l.startsWith(":") && l.slice(7, 9) === "00") flash += parseInt(l.slice(1, 3), 16);
    self.postMessage({ ok: true, hex: text, flash, ms: { load: tLoad - t0, total: performance.now() - t0 }, log });
  } catch (e) {
    self.postMessage({ ok: false, error: String(e.message || e), log: e.log || log });
  }
};
