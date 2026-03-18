import socket
import struct

def start_test_server():
    # 서버 설정 (모든 인터페이스 허용)
    IP = '0.0.0.0'
    PORT = 5555

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 포트 재사용 및 지연 시간 방지 설정
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    server.bind((IP, PORT))
    server.listen(1)
    
    print(f"--- 원격 테스트 서버 가동 중 (Port: {PORT}) ---")
    print("외부 접속을 기다리고 있습니다...")

    try:
        while True:
            conn, addr = server.accept()
            print(f"\n[접속 성공] 클라이언트 주소: {addr}")
            
            try:
                while True:
                    # 1바이트(U8) 데이터 수신
                    data = conn.recv(1)
                    if not data:
                        print("클라이언트가 연결을 끊었습니다.")
                        break
                    
                    # 수신된 바이트를 숫자로 변환
                    val = struct.unpack('B', data)[0]
                    
                    # 간단한 연산 (+6)
                    result = (val + 6) % 256
                    print(f"수신 데이터: {val} -> 처리 결과: {result}")
                    
                    # 1바이트(U8) 데이터 송신
                    conn.sendall(struct.pack('B', result))
            
            except Exception as e:
                print(f"통신 중 오류 발생: {e}")
            finally:
                conn.close()
                print("세션이 종료되었습니다. 다음 접속을 기다립니다...")

    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.close()

if __name__ == "__main__":
    start_test_server()