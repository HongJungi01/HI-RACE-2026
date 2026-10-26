# IR 라인트레이서 시뮬레이터

1/10 IR 라인트레이서용 브라우저 시뮬레이터. Arduino Mega 2560 코드를 브라우저 안에서 컴파일(avr-gcc wasm)하고 avr8js로 에뮬레이션해서, 차량 모델과 라인 맵 위에서 최대 4대를 동시에 돌려 본다.

- 바로 쓰기: https://hongjungi01.github.io/HI-RACE-2026/ir-sim/
- 로컬 실행: 이 폴더에서 `python -m http.server 8000` → http://localhost:8000/ (Windows는 `run.bat`). `index.html`을 직접 열면(file://) 동작하지 않음

## 구성
- 탭 1 차량 모델: 부품 배치·배선, 차량 파라미터, JSON 저장/불러오기
- 탭 2 시뮬레이션: 맵 선택, 차량별 실행/일시정지/정지, 출발 위치, 차량별 코드 에디터·Serial
- 샘플 코드: `sample_linetracer.ino`(앞 11ch + 뒤 5ch, P 제어)

## 공유 차고
- 시뮬레이션 탭 오른쪽 "공유 차고": 목록은 누구나 보고 "맵에 올리기"로 바로 불러옴. 올리기·삭제는 Google 로그인 후 자기 차만 (한 사람당 8대)
- 서버: Supabase (`garage-config.js`에 Project URL과 publishable 키. 공개 키라 저장소에 있어도 됨, 권한은 테이블 RLS가 막음)
- 코드 비공개: 올릴 때 체크하면 코드 대신 컴파일된 펌웨어(hex)만 공개, 다른 사람은 주행만 가능. 원본 코드는 본인만 읽는 `garage_code` 테이블
- 테이블·정책: `garage_setup.sql`, `garage_private.sql`(비공개 기능, 이어서 실행). 개인정보처리방침: `privacy.html`

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
- supabase-js 2.117.3 UMD 번들 `supabase.js` (MIT)
