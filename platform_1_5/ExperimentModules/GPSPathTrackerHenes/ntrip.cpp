#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>
#include <winsock2.h>
#include <windows.h>

#pragma comment(lib, "ws2_32.lib")

// --- [설정 부분] ---
const int BAUD_RATE = CBR_115200;              // F9P 속도
const int TCP_OUT_PORT = 5000;                 // 외부 프로그램용 포트

const std::string NTRIP_HOST = "RTS2.ngii.go.kr";
const int NTRIP_PORT = 2101;
const std::string NTRIP_MOUNT = "VRS-RTCM32";

// 추가: 서버를 깨우기 위한 초기 위치 (서울 시청 근처)
const std::string INITIAL_GGA = "$GPGGA,000001,3733.90,N,12658.50,E,1,12,1.0,0.0,M,0.0,M,,*60\r\n";

const std::string CONFIG_FILE = "ntrip_config.txt";
// ------------------

// 런타임 설정 (입력으로 결정)
std::string COM_PORT;
std::string NTRIP_USER;
std::string NTRIP_PW;
std::string comNumber;  // COM 포트 번호만 저장 (저장/표시용)

HANDLE hSerial;
SOCKET ntripSocket;
std::vector<SOCKET> tcpClients;
std::mutex clientsMutex;
std::mutex coutMutex;

// 설정 파일에서 저장된 자격증명 로드
void loadConfig(std::string& com, std::string& user, std::string& pw) {
    std::ifstream f(CONFIG_FILE);
    if (f.is_open()) {
        std::getline(f, com);
        std::getline(f, user);
        std::getline(f, pw);
    }
}

// 자격증명을 설정 파일에 저장
void saveConfig(const std::string& com, const std::string& user, const std::string& pw) {
    std::ofstream f(CONFIG_FILE);
    if (f.is_open()) {
        f << com << "\n" << user << "\n" << pw << "\n";
    }
}

// 프롬프트 표시 후 입력 받기 (빈 입력이면 기본값 사용)
std::string promptInput(const std::string& label, const std::string& defaultVal) {
    if (defaultVal.empty())
        std::cout << " " << label << ": ";
    else
        std::cout << " " << label << " [" << defaultVal << "]: ";
    std::string input;
    std::getline(std::cin, input);
    return input.empty() ? defaultVal : input;
}

// Base64 인코딩 헬퍼 함수 (NTRIP 인증용)
std::string base64_encode(const std::string& in) {
    static const char lookup[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(lookup[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(lookup[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

// 1. NTRIP -> F9P (RTCM 주입 스레드)
void ThreadNtripToSerial() {
    char buffer[4096];
    while (true) {
        int bytesRead = recv(ntripSocket, buffer, sizeof(buffer), 0);
        if (bytesRead > 0) {
            DWORD bytesWritten;
            WriteFile(hSerial, buffer, bytesRead, &bytesWritten, NULL);
            // 필요 시 주석 해제하여 RTCM 수신 확인
            // std::lock_guard<std::mutex> lock(coutMutex);
            // std::cout << " [>>>] RTCM " << bytesRead << " bytes -> F9P" << std::endl;
        }
        else if (bytesRead == 0 || bytesRead == SOCKET_ERROR) break;
    }
}

// 2. 외부 TCP 클라이언트 수락 스레드 (LabVIEW용)
void ThreadTcpServer() {
    SOCKET serverSock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serverAddr = { 0 };
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(TCP_OUT_PORT);

    bind(serverSock, (sockaddr*)&serverAddr, sizeof(serverAddr));
    listen(serverSock, 5);

    while (true) {
        SOCKET clientSock = accept(serverSock, NULL, NULL);
        if (clientSock != INVALID_SOCKET) {
            std::lock_guard<std::mutex> lock(coutMutex);
            std::cout << " [+] 외부 프로그램(LabVIEW 등) 접속됨" << std::endl;
            std::lock_guard<std::mutex> lockClients(clientsMutex);
            tcpClients.push_back(clientSock);
        }
    }
}

// 3. F9P -> NTRIP & 외부 TCP (NMEA 중계 스레드)
void ThreadSerialToAll() {
    char buffer[1];
    std::string line = "";
    DWORD bytesRead;

    while (true) {
        if (ReadFile(hSerial, buffer, 1, &bytesRead, NULL) && bytesRead > 0) {
            line += buffer[0];
            if (buffer[0] == '\n') { // 한 줄 완성
                if (line.find("$GNGGA") != std::string::npos || line.find("$GPGGA") != std::string::npos) {
                    // 콘솔 출력
                    std::lock_guard<std::mutex> lock(coutMutex);
                    std::cout << " GPS Data: " << line;

                    // NTRIP 서버 피드백
                    send(ntripSocket, line.c_str(), (int)line.length(), 0);
                }

                // 외부 TCP 클라이언트들에게 전송
                std::lock_guard<std::mutex> lock(clientsMutex);
                for (auto it = tcpClients.begin(); it != tcpClients.end(); ) {
                    if (send(*it, line.c_str(), (int)line.length(), 0) == SOCKET_ERROR) {
                        closesocket(*it);
                        it = tcpClients.erase(it);
                    } else { ++it; }
                }
                line = "";
            }
        }
    }
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    // --- 초기화 ---
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);

    // 0. 설정 입력
    std::string savedCom, savedUser, savedPw;
    loadConfig(savedCom, savedUser, savedPw);

    std::cout << " === NTRIP RTK 설정 ===" << std::endl;
    comNumber  = promptInput("COM 포트 번호", savedCom);
    NTRIP_USER = promptInput("NTRIP 유저네임", savedUser);
    NTRIP_PW   = promptInput("NTRIP 비밀번호", savedPw);
    COM_PORT = "\\\\.\\COM" + comNumber;
    std::cout << std::endl;

    // 1. 시리얼 포트 열기
    hSerial = CreateFileA(COM_PORT.c_str(), GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    if (hSerial == INVALID_HANDLE_VALUE) {
        std::cerr << " [!] COM 포트를 열 수 없습니다." << std::endl;
        return 1;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
    GetCommState(hSerial, &dcbSerialParams);
    dcbSerialParams.BaudRate = BAUD_RATE;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;
    SetCommState(hSerial, &dcbSerialParams);

    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    SetCommTimeouts(hSerial, &timeouts);
    std::cout << " [*] F9P 연결 성공: " << COM_PORT << std::endl;

    // 2. NTRIP 서버 연결
    struct hostent* host = gethostbyname(NTRIP_HOST.c_str());
    ntripSocket = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in ntripAddr = { 0 };
    ntripAddr.sin_family = AF_INET;
    ntripAddr.sin_port = htons(NTRIP_PORT);
    ntripAddr.sin_addr.s_addr = *((unsigned long*)host->h_addr);

    if (connect(ntripSocket, (sockaddr*)&ntripAddr, sizeof(ntripAddr)) == SOCKET_ERROR) {
        std::cerr << " [!] NTRIP 서버 연결 실패" << std::endl;
        return 1;
    }

    std::string auth = base64_encode(NTRIP_USER + ":" + NTRIP_PW);
    std::string request = "GET /" + NTRIP_MOUNT + " HTTP/1.0\r\n" +
                          "User-Agent: NTRIP C++ Client\r\n" +
                          "Authorization: Basic " + auth + "\r\n" +
                          "Connection: close\r\n\r\n";
    send(ntripSocket, request.c_str(), (int)request.length(), 0);

    char resp[1024] = { 0 };
    recv(ntripSocket, resp, sizeof(resp), 0);
    std::cout << " [*] 서버 응답: " << resp << std::endl;

    // 로그인 성공 시 자격증명 저장
    std::string respStr(resp);
    if (respStr.find("ICY 200 OK") != std::string::npos) {
        saveConfig(comNumber, NTRIP_USER, NTRIP_PW);
        std::cout << " [*] 로그인 성공 - 설정 저장됨" << std::endl;
    } else {
        std::cerr << " [!] 로그인 실패 - 설정이 저장되지 않았습니다" << std::endl;
    }

    // 추가: 서버에 초기 GGA 전송 (데이터 송신 유도)
    send(ntripSocket, INITIAL_GGA.c_str(), (int)INITIAL_GGA.length(), 0);

    // 3. 스레드 실행
    std::cout << " [*] RTK 가동 시작 (포트: " << TCP_OUT_PORT << ")" << std::endl;
    std::thread t1(ThreadNtripToSerial);
    std::thread t2(ThreadTcpServer);
    std::thread t3(ThreadSerialToAll);

    t1.join(); t2.join(); t3.join();

    // 정리
    CloseHandle(hSerial);
    closesocket(ntripSocket);
    WSACleanup();
    return 0;
}