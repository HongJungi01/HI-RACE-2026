# IR 라인트레이서 시뮬레이터

1/10 IR 라인트레이서용 브라우저 시뮬레이터. Arduino Mega 2560 코드를 브라우저 안에서 컴파일(avr-gcc wasm)하고 avr8js로 에뮬레이션해서, 차량 모델과 라인 맵 위에서 최대 4대를 동시에 돌려 본다.

- 바로 쓰기: https://hongjungi01.github.io/HI-RACE-2026/ir-sim/
- 로컬 실행: 이 폴더에서 `python -m http.server 8000` → http://localhost:8000/ (Windows는 `run.bat`). `index.html`을 직접 열면(file://) 동작하지 않음

## 구성
- 탭 1 차량 모델: 부품 배치·배선, 차량 파라미터, JSON 저장/불러오기
- 탭 2 시뮬레이션: 맵 선택, 차량별 실행/일시정지/정지, 출발 위치, 차량별 코드 에디터·Serial
- 샘플 코드: `sample_linetracer.ino`(앞 11ch + 뒤 5ch, P 제어)

## 주의
- 차량·코드는 브라우저(localStorage)에 저장된다. 공유는 차량 JSON과 .ino 파일로.
- 차량 동작 파라미터(속도, 서보, IR, 타이어 등)는 실측 전 가정값.

## 파일
| 파일 | 내용 |
|---|---|
| index.html | 메인 페이지 |
| car-worker.js | 차량 1대 = 워커 1개: Mega 에뮬레이션 + 차량 모델 + IR·초음파 |
| avr8js.js | avr8js 0.21.1 번들 |
| compiler-worker.js, tools/, mega_assets.json | 브라우저 avr-gcc (cc1plus/as/ld/objcopy wasm) + Mega 코어·라이브러리 |
| maps.json | 샘플 라인 맵 11개 (cm) |

## 출처·라이선스
- avr8js (MIT, Wokwi)
- avr-gcc/binutils wasm: npm `@horang-corp/avr-gcc-wasm` (GPL, 원본 소스는 GNU 프로젝트)
- Arduino AVR 코어·Servo 등 라이브러리 (LGPL), avr-libc (BSD)
- CodeMirror 5 (MIT, CDN에서 로드)
