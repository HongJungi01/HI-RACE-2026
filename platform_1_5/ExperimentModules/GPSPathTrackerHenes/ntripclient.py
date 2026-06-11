import serial
import socket
import threading
import base64
import time

# --- [설정 부분] ---
COM_PORT = 'COM5'           # F9P 포트 확인
BAUD_RATE = 115200          # F9P 속도
TCP_OUT_PORT = 5000         # 외부 프로그램용 포트

NTRIP_HOST = 'RTS2.ngii.go.kr' 
NTRIP_PORT = 2101
NTRIP_USER = 'james6137' 
NTRIP_PW = 'ngii'     
NTRIP_MOUNT = 'VRS-RTCM32'

# F9P가 위치를 잡기 전, 서버를 깨우기 위한 초기 위치 (서울 시청 근처)
INITIAL_GGA = "$GPGGA,000001,3733.90,N,12658.50,E,1,12,1.0,0.0,M,0.0,M,,*60\r\n"
# ------------------

def create_ntrip_auth(user, pw):
    auth = f"{user}:{pw}"
    return base64.b64encode(auth.encode()).decode()

def run_bridge():
    # 1. 시리얼 포트 연결
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"[*] F9P 연결 성공: {COM_PORT}")
    except Exception as e:
        print(f"[!] 시리얼 연결 실패: {e}")
        return

    # 2. TCP 서버 설정
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_server.bind(('0.0.0.0', TCP_OUT_PORT))
    tcp_server.listen(5)
    clients = []

    def handle_tcp_clients():
        while True:
            try:
                client_sock, addr = tcp_server.accept()
                print(f"[+] 외부 프로그램 접속됨: {addr}")
                clients.append(client_sock)
            except: break

    threading.Thread(target=handle_tcp_clients, daemon=True).start()

    # 3. NTRIP 서버 연결 시도
    try:
        ntrip_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ntrip_sock.settimeout(10) # 10초 타임아웃 설정
        print(f"[*] NTRIP 서버 접속 중... ({NTRIP_HOST})")
        ntrip_sock.connect((NTRIP_HOST, NTRIP_PORT))
        
        auth_b64 = create_ntrip_auth(NTRIP_USER, NTRIP_PW)
        headers = (
            f"GET /{NTRIP_MOUNT} HTTP/1.0\r\n"
            f"User-Agent: NTRIP PythonClient\r\n"
            f"Authorization: Basic {auth_b64}\r\n"
            f"Connection: close\r\n\r\n"
        )
        ntrip_sock.sendall(headers.encode())

        # 서버 응답 확인 (ICY 200 OK 등)
        response = ntrip_sock.recv(1024).decode(errors='ignore')
        print(f"[*] 서버 응답: {response.strip()}")
        
        if "200 OK" not in response and "ICY" not in response:
            print("[!] 인증 실패 또는 마운트포트 오류")
            return

        # 서버에 초기 GGA 전송 (데이터 송신 유도)
        ntrip_sock.sendall(INITIAL_GGA.encode())

    except socket.timeout:
        print("[!] 타임아웃: 학교 네트워크에서 2101 포트를 막았을 가능성이 높습니다.")
        return
    except Exception as e:
        print(f"[!] NTRIP 서버 연결 실패: {e}")
        return

    # 4. 데이터 중계 로직
    def ntrip_to_f9p():
        while True:
            try:
                data = ntrip_sock.recv(4096)
                if data:
                    ser.write(data) # F9P로 RTCM 주입
            except: 
                print("[!] NTRIP 연결 끊김")
                break

    threading.Thread(target=ntrip_to_f9p, daemon=True).start()

    print(f"[*] RTK 가동 시작 (포트: {TCP_OUT_PORT})")

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline()
                if not line: continue
                
                # NMEA 처리 및 서버 피드백
                if b'$GNGGA' in line or b'$GPGGA' in line:
                    try:
                        ntrip_sock.sendall(line) # 실시간 위치 서버 전송
                        # 상태 확인용 출력
                        print(f"GPS Data: {line.decode(errors='ignore').strip()}")
                    except: pass

                # 외부 프로그램으로 전송
                for client in clients[:]:
                    try:
                        client.sendall(line)
                    except:
                        clients.remove(client)
            time.sleep(0.01) # CPU 점유율 방지
    except KeyboardInterrupt:
        print("\n[!] 종료 중...")
    finally:
        ser.close()
        ntrip_sock.close()
        tcp_server.close()

if __name__ == "__main__":
    run_bridge()