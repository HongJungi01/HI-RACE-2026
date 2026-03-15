import socket
import struct
import main as main_module

# --- 설정 ---
SERVER_ADDR = ('0.0.0.0', 5555)
MAX_IMG_SIZE = 20 * 1024 * 1024  # 20MB 제한

def recvall(sock, n):
    """정해진 바이트만큼 데이터를 수신"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return data

def tcp_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # 지연 방지
    server.bind(SERVER_ADDR)
    server.listen(5)
    
    print(f"Vision AI Server Running on {SERVER_ADDR[1]}...")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Connected: {addr}")

            try:
                while True:
                    # 1. 헤더 24바이트 수신 (IMG! 4바이트 + Big-Endian U32 5개)
                    # 헤더 구성: [IMG!][img_len][min_area][min_span][max_rmse][poly_degree]
                    header = recvall(conn, 24)
                    if not header or header[:4] != b'IMG!': 
                        break
                    
                    # 2. 헤더 해석
                    params_raw = struct.unpack('>IIIII', header[4:])
                    img_len = params_raw[0]
                    current_params = params_raw[1:] # (min_area, min_span, max_rmse, poly_degree)
                    
                    if img_len <= 0 or img_len > MAX_IMG_SIZE:
                        print(f"Invalid image size: {img_len}")
                        continue

                    # 3. 이미지 데이터 수신
                    img_bytes = recvall(conn, img_len)
                    if not img_bytes: 
                        break

                    # 4. AI 연산 수행 (전달받은 파라미터 적용)
                    try:
                        # *current_params를 사용하여 인자로 펼쳐서 전달
                        result_list = main_module.main(bytes(img_bytes), *current_params)
                        
                        if result_list:
                            # 결과 리스트 직렬화 ("val1|val2|...")
                            res_str = "|".join(str(x) if x is not None else "None" for x in result_list)
                            res_bytes = res_str.encode('utf-8')
                            
                            # 헤더(결과 길이)와 함께 전송
                            conn.sendall(struct.pack('>I', len(res_bytes)) + res_bytes)
                            print(f"[{addr}] Params: {current_params} | Processed: {res_str}")
                    except Exception as e:
                        print(f"Processing Error: {e}")
            
            finally:
                conn.close()
                print(f"Disconnected: {addr}")

    except KeyboardInterrupt:
        print("\nServer Stopped.")
    finally:
        server.close()

if __name__ == "__main__":
    tcp_server()