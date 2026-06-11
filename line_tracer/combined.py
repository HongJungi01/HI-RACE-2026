#!/usr/bin/env python3
"""
구동 + 조향 + 브레이크 통합 제어 GUI

Arduino 프로토콜:
  송신: S<angle>\n  조향 (90~130, 중립 110)
        B<angle>\n  브레이크 (0~80, 기본 0, 제동 20~50)
        V<pwm>\n    구동 (-255~255)
  수신: E<delta>,C<cumulative>,S<steerAngle>,B<brakeAngle>\n
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

import serial
import serial.tools.list_ports


class CombinedController:
    BAUD_RATE = 115200
    SEND_INTERVAL_MS = 20

    # 조향 파라미터
    STEER_MIN = 90
    STEER_MAX = 130
    STEER_CENTER = 110

    # 브레이크 파라미터
    BRAKE_MIN = 0
    BRAKE_MAX = 80
    BRAKE_DEFAULT = 0
    BRAKE_INTERLOCK = 20  # 이 값 이상이면 PWM 강제 0

    def __init__(self):
        self.ser = None
        self.connected = False
        self.running = True

        # 제어값
        self.steer = self.STEER_CENTER
        self.brake = self.BRAKE_DEFAULT
        self.pwm = 0

        # 피드백값
        self.fb_enc_delta = 0
        self.fb_enc_cumulative = 0
        self.fb_steer = 0
        self.fb_brake = 0

        self.keys_pressed = set()

        self._build_gui()
        self._bind_keys()

    # ================================================================
    # 시리얼 I/O
    # ================================================================
    def get_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self):
        port = self.port_var.get()
        if not port:
            return
        try:
            self.ser = serial.Serial(port, self.BAUD_RATE, timeout=0.1)
            self.connected = True
            self.status_label.config(text="● 연결됨", foreground="green")
            self.connect_btn.config(text="Disconnect")
            threading.Thread(target=self._read_loop, daemon=True).start()
            self._send_loop()
        except serial.SerialException as ex:
            self.status_label.config(text=f"✗ {ex}", foreground="red")

    def disconnect(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b"V0\n")
                self.ser.write(f"S{self.STEER_CENTER}\n".encode('ascii'))
                self.ser.write(f"B{self.BRAKE_DEFAULT}\n".encode('ascii'))
            except (serial.SerialException, OSError):
                pass
        self.connected = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.status_label.config(text="○ 미연결", foreground="gray")
        self.connect_btn.config(text="Connect")

    def toggle_connect(self):
        if self.connected:
            self.disconnect()
        else:
            self.connect()

    def _read_loop(self):
        buf = ""
        while self.connected and self.running:
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting:
                    raw = self.ser.read(self.ser.in_waiting)
                    buf += raw.decode('ascii', errors='ignore')
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        self._parse_feedback(line)
                else:
                    time.sleep(0.005)
            except (serial.SerialException, OSError):
                break

    def _parse_feedback(self, line):
        """E<delta>,C<cumulative>,S<steer>,B<brake> 파싱"""
        try:
            if not line.startswith('E'):
                self._append_log(line)
                return
            parts = line.split(',')
            for part in parts:
                if part.startswith('E'):
                    self.fb_enc_delta = int(part[1:])
                elif part.startswith('C'):
                    self.fb_enc_cumulative = int(part[1:])
                elif part.startswith('S'):
                    self.fb_steer = int(part[1:])
                elif part.startswith('B'):
                    self.fb_brake = int(part[1:])
        except (ValueError, IndexError):
            self._append_log(line)

    def _send_loop(self):
        if not self.connected:
            return
        try:
            if self.ser and self.ser.is_open:
                # 브레이크 인터록 적용 (표시용)
                effective_pwm = 0 if self.brake >= self.BRAKE_INTERLOCK else self.pwm
                cmds = f"S{self.steer}\nB{self.brake}\nV{effective_pwm}\n"
                self.ser.write(cmds.encode('ascii'))
        except (serial.SerialException, OSError):
            self.disconnect()
            return

        self._update_display()
        self.root.after(self.SEND_INTERVAL_MS, self._send_loop)

    # ================================================================
    # GUI
    # ================================================================
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("통합 제어 — 조향 / 브레이크 / 구동")
        self.root.resizable(False, False)
        self.root.configure(padx=8, pady=8)

        style = ttk.Style()
        style.configure('TLabelframe.Label', font=('Arial', 10, 'bold'))

        # ---- 시리얼 연결 ----
        conn = ttk.LabelFrame(self.root, text="시리얼 연결", padding=8)
        conn.pack(fill='x', pady=(0, 5))

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var,
                                       width=15, state='readonly')
        self.port_combo.pack(side='left', padx=(0, 5))

        ttk.Button(conn, text="Refresh", width=8,
                   command=self._refresh_ports).pack(side='left', padx=(0, 5))

        self.connect_btn = ttk.Button(conn, text="Connect", width=12,
                                      command=self.toggle_connect)
        self.connect_btn.pack(side='left', padx=(0, 10))

        self.status_label = ttk.Label(conn, text="○ 미연결", foreground="gray")
        self.status_label.pack(side='left')

        # ---- 조향 제어 ----
        steer_frame = ttk.LabelFrame(
            self.root, text=f"조향 ({self.STEER_MIN}~{self.STEER_MAX}°, 중립 {self.STEER_CENTER}°)",
            padding=8)
        steer_frame.pack(fill='x', pady=5)

        sf = ttk.Frame(steer_frame)
        sf.pack(fill='x', pady=2)
        ttk.Label(sf, text="Steer:", width=7).pack(side='left')
        self.steer_var = tk.IntVar(value=self.STEER_CENTER)
        self.steer_scale = ttk.Scale(sf, from_=self.STEER_MIN, to=self.STEER_MAX,
                                     orient='horizontal', variable=self.steer_var,
                                     command=self._on_steer_slider)
        self.steer_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.steer_lbl = ttk.Label(sf, text=str(self.STEER_CENTER), width=5, anchor='e')
        self.steer_lbl.pack(side='left')
        ttk.Button(sf, text="Center", width=7,
                   command=self._center_steer).pack(side='left', padx=(5, 0))

        # ---- 브레이크 제어 ----
        brake_frame = ttk.LabelFrame(
            self.root, text=f"브레이크 (기본 {self.BRAKE_DEFAULT}°, 제동 90~150°)",
            padding=8)
        brake_frame.pack(fill='x', pady=5)

        bf = ttk.Frame(brake_frame)
        bf.pack(fill='x', pady=2)
        ttk.Label(bf, text="Brake:", width=7).pack(side='left')
        self.brake_var = tk.IntVar(value=self.BRAKE_DEFAULT)
        self.brake_scale = ttk.Scale(bf, from_=self.BRAKE_MIN, to=self.BRAKE_MAX,
                                     orient='horizontal', variable=self.brake_var,
                                     command=self._on_brake_slider)
        self.brake_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.brake_lbl = ttk.Label(bf, text=str(self.BRAKE_DEFAULT), width=5, anchor='e')
        self.brake_lbl.pack(side='left')
        ttk.Button(bf, text="Release", width=7,
                   command=self._release_brake).pack(side='left', padx=(5, 0))

        self.brake_warn_lbl = ttk.Label(brake_frame, text="", foreground="red")
        self.brake_warn_lbl.pack(anchor='w')

        # ---- PWM 구동 ----
        pwm_frame = ttk.LabelFrame(self.root, text="구동 PWM (-255~255)", padding=8)
        pwm_frame.pack(fill='x', pady=5)

        df = ttk.Frame(pwm_frame)
        df.pack(fill='x', pady=2)
        ttk.Label(df, text="PWM:", width=7).pack(side='left')
        self.pwm_var = tk.IntVar(value=0)
        self.pwm_scale = ttk.Scale(df, from_=-255, to=255, orient='horizontal',
                                   variable=self.pwm_var,
                                   command=self._on_pwm_slider)
        self.pwm_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.pwm_lbl = ttk.Label(df, text="0", width=5, anchor='e')
        self.pwm_lbl.pack(side='left')
        ttk.Button(df, text="Stop", width=5,
                   command=self._stop_pwm).pack(side='left', padx=(5, 0))

        self.pwm_lock_lbl = ttk.Label(pwm_frame, text="", foreground="orange")
        self.pwm_lock_lbl.pack(anchor='w')

        # ---- ESTOP ----
        self.estop_btn = tk.Button(
            self.root, text="E-STOP (Space)", font=('Arial', 14, 'bold'),
            bg='#ee3333', fg='white', activebackground='#cc0000',
            activeforeground='white', relief='raised', bd=3, height=2,
            command=self._estop)
        self.estop_btn.pack(fill='x', pady=5)

        # ---- 피드백 ----
        fb = ttk.LabelFrame(self.root, text="피드백 (Arduino → PC)", padding=8)
        fb.pack(fill='x', pady=5)

        grid = ttk.Frame(fb)
        grid.pack(fill='x')
        grid.columnconfigure((0, 1), weight=1)

        self.fb_delta_lbl = ttk.Label(grid, text="Delta: ---", anchor='w',
                                      font=('Consolas', 11))
        self.fb_delta_lbl.grid(row=0, column=0, sticky='w', padx=5)

        self.fb_cum_lbl = ttk.Label(grid, text="Total: ---", anchor='w',
                                    font=('Consolas', 11))
        self.fb_cum_lbl.grid(row=0, column=1, sticky='w', padx=5)

        self.fb_steer_lbl = ttk.Label(grid, text="Steer: ---", anchor='w',
                                      font=('Consolas', 11))
        self.fb_steer_lbl.grid(row=1, column=0, sticky='w', padx=5, pady=(3, 0))

        self.fb_brake_lbl = ttk.Label(grid, text="Brake: ---", anchor='w',
                                      font=('Consolas', 11))
        self.fb_brake_lbl.grid(row=1, column=1, sticky='w', padx=5, pady=(3, 0))

        # ---- 송신 미리보기 ----
        pv = ttk.LabelFrame(self.root, text="송신 명령", padding=8)
        pv.pack(fill='x', pady=5)
        self.cmd_lbl = ttk.Label(pv, text="---", font=('Consolas', 10))
        self.cmd_lbl.pack(anchor='w')

        # ---- 로그 ----
        lf = ttk.LabelFrame(self.root, text="로그", padding=5)
        lf.pack(fill='x', pady=5)
        self.log_text = tk.Text(lf, height=4, width=55, font=('Consolas', 9),
                                state='disabled')
        self.log_text.pack(fill='x')

        # ---- 키보드 도움말 ----
        hf = ttk.LabelFrame(self.root, text="키보드", padding=5)
        hf.pack(fill='x', pady=(5, 0))
        ttk.Label(hf, foreground="gray",
                  text="A/D 조향±5° | ←/→ 조향±1° | W/S PWM±20 | Q/E 브레이크±10 | Space 전체정지").pack()

        self._refresh_ports()

    def _append_log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.log_text.delete('1.0', f'{lines - 100}.0')
        self.log_text.config(state='disabled')

    def _refresh_ports(self):
        ports = self.get_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    # ---- 슬라이더 콜백 ----
    def _on_steer_slider(self, v):
        self.steer = int(float(v))
        self.steer_lbl.config(text=str(self.steer))

    def _on_brake_slider(self, v):
        self.brake = int(float(v))
        self.brake_lbl.config(text=str(self.brake))

    def _on_pwm_slider(self, v):
        self.pwm = int(float(v))
        self.pwm_lbl.config(text=str(self.pwm))

    # ---- 버튼 콜백 ----
    def _center_steer(self):
        self.steer = self.STEER_CENTER
        self.steer_var.set(self.STEER_CENTER)
        self.steer_lbl.config(text=str(self.STEER_CENTER))

    def _release_brake(self):
        self.brake = self.BRAKE_DEFAULT
        self.brake_var.set(self.BRAKE_DEFAULT)
        self.brake_lbl.config(text=str(self.BRAKE_DEFAULT))

    def _stop_pwm(self):
        self.pwm = 0
        self.pwm_var.set(0)
        self.pwm_lbl.config(text="0")

    def _estop(self):
        """비상 정지: PWM=0, 조향=중립, 브레이크=해제"""
        self._stop_pwm()
        self._center_steer()
        self._release_brake()

    # ---- 디스플레이 업데이트 ----
    def _update_display(self):
        self.fb_delta_lbl.config(text=f"Delta: {self.fb_enc_delta}")
        self.fb_cum_lbl.config(text=f"Total: {self.fb_enc_cumulative}")
        self.fb_steer_lbl.config(text=f"Steer: {self.fb_steer}°")
        self.fb_brake_lbl.config(text=f"Brake: {self.fb_brake}°")

        effective_pwm = 0 if self.brake >= self.BRAKE_INTERLOCK else self.pwm
        self.cmd_lbl.config(text=f"S{self.steer}  B{self.brake}  V{effective_pwm}")

        # 브레이크 인터록 경고
        if self.brake >= self.BRAKE_INTERLOCK:
            self.brake_warn_lbl.config(text=f"⚠ 제동 중 (≥{self.BRAKE_INTERLOCK}°)")
            self.pwm_lock_lbl.config(text="🔒 브레이크 인터록 — PWM 강제 0")
        else:
            self.brake_warn_lbl.config(text="")
            self.pwm_lock_lbl.config(text="")

    # ================================================================
    # 키보드 제어
    # ================================================================
    def _bind_keys(self):
        self.root.bind('<KeyPress>', self._on_key_press)
        self.root.bind('<KeyRelease>', self._on_key_release)

    def _on_key_press(self, event):
        key = event.keysym.lower()
        if key in self.keys_pressed:
            return
        self.keys_pressed.add(key)

        if key == 'space':
            self._estop()
            return

        # 조향
        if key == 'a':
            self.steer = max(self.STEER_MIN, self.steer - 5)
        elif key == 'd':
            self.steer = min(self.STEER_MAX, self.steer + 5)
        elif key == 'left':
            self.steer = max(self.STEER_MIN, self.steer - 1)
        elif key == 'right':
            self.steer = min(self.STEER_MAX, self.steer + 1)
        # PWM
        elif key in ('w', 'up'):
            self.pwm = min(255, self.pwm + 20)
        elif key in ('s', 'down'):
            self.pwm = max(-255, self.pwm - 20)
        # 브레이크
        elif key == 'q':
            self.brake = max(self.BRAKE_MIN, self.brake - 10)
        elif key == 'e':
            self.brake = min(self.BRAKE_MAX, self.brake + 10)
        else:
            return

        # GUI 동기화
        self.steer_var.set(self.steer)
        self.steer_lbl.config(text=str(self.steer))
        self.pwm_var.set(self.pwm)
        self.pwm_lbl.config(text=str(self.pwm))
        self.brake_var.set(self.brake)
        self.brake_lbl.config(text=str(self.brake))

    def _on_key_release(self, event):
        key = event.keysym.lower()
        self.keys_pressed.discard(key)

    # ================================================================
    # 실행
    # ================================================================
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        self.disconnect()
        self.root.destroy()


if __name__ == '__main__':
    app = CombinedController()
    app.run()
