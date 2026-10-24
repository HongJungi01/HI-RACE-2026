@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo IR 라인트레이서 시뮬레이터: http://localhost:8000/
echo 이 창을 닫으면 시뮬레이터도 꺼집니다.
start "" http://localhost:8000/
python -m http.server 8000 || py -m http.server 8000
pause