#!/usr/bin/env python3
"""
테스트용 단순 PWM 구동 + 엔코더 모니터링 GUI

Arduino 프로토콜:
  송신: V<pwm>\n   (pwm: -255~255)
  수신: E<delta>,C<cumulative>\n
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

import serial
import serial.tools.list_ports


class DebugController:
    BAUD_RATE = 115200
    SEND_INTERVAL_MS = 50

    def __init__(self):
        self.ser = None
        self.connected = False
        self.running = True

        # 제어값
        self.pwm = 0  # -255 ~ 255

        # 피드백값
        self.enc_delta = 0
        self.enc_cumulative = 0

        # 키보드
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
        """E<delta>,C<cumulative> 형식 파싱"""
        try:
            if not line.startswith('E'):
                self._append_log(line)
                return
            parts = line.split(',')
            for part in parts:
                if part.startswith('E'):
                    self.enc_delta = int(part[1:])
                elif part.startswith('C'):
                    self.enc_cumulative = int(part[1:])
        except (ValueError, IndexError):
            pass

    def _send_loop(self):
        if not self.connected:
            return
        try:
            if self.ser and self.ser.is_open:
                cmd = f"V{self.pwm}\n"
                self.ser.write(cmd.encode('ascii'))
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
        self.root.title("테스트 — 단순 PWM 구동")
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

        # ---- PWM 제어 ----
        ctrl = ttk.LabelFrame(self.root, text="PWM 구동", padding=8)
        ctrl.pack(fill='x', pady=5)

        df = ttk.Frame(ctrl)
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
                   command=self._reset_pwm).pack(side='left', padx=(5, 0))

        # ---- ESTOP ----
        self.estop_btn = tk.Button(
            self.root, text="STOP", font=('Arial', 14, 'bold'),
            bg='#ee3333', fg='white', activebackground='#cc0000',
            activeforeground='white', relief='raised', bd=3, height=2,
            command=self._reset_pwm)
        self.estop_btn.pack(fill='x', pady=5)

        # ---- 피드백 ----
        fb = ttk.LabelFrame(self.root, text="엔코더 피드백", padding=8)
        fb.pack(fill='x', pady=5)

        grid = ttk.Frame(fb)
        grid.pack(fill='x')
        grid.columnconfigure((0, 1), weight=1)

        self.enc_delta_lbl = ttk.Label(grid, text="Delta: ---", anchor='w',
                                       font=('Consolas', 11))
        self.enc_delta_lbl.grid(row=0, column=0, sticky='w', padx=5)

        self.enc_cum_lbl = ttk.Label(grid, text="Total: ---", anchor='w',
                                     font=('Consolas', 11))
        self.enc_cum_lbl.grid(row=0, column=1, sticky='w', padx=5)

        # ---- 송신 미리보기 ----
        pv = ttk.LabelFrame(self.root, text="송신 명령", padding=8)
        pv.pack(fill='x', pady=5)
        self.cmd_lbl = ttk.Label(pv, text="---", font=('Consolas', 10))
        self.cmd_lbl.pack(anchor='w')

        # ---- 로그 ----
        lf = ttk.LabelFrame(self.root, text="로그", padding=5)
        lf.pack(fill='x', pady=5)
        self.log_text = tk.Text(lf, height=4, width=50, font=('Consolas', 9),
                                state='disabled')
        self.log_text.pack(fill='x')

        # ---- 키보드 도움말 ----
        hf = ttk.LabelFrame(self.root, text="키보드", padding=5)
        hf.pack(fill='x', pady=(5, 0))
        ttk.Label(hf, foreground="gray",
                  text="W/S PWM ±20 | Space 정지").pack()

        self._refresh_ports()

    def _append_log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 100:
            self.log_text.delete('1.0', f'{lines - 100}.0')
        self.log_text.config(state='disabled')

    # ---- GUI 콜백 ----
    def _refresh_ports(self):
        ports = self.get_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def _on_pwm_slider(self, v):
        self.pwm = int(float(v))
        self.pwm_lbl.config(text=str(self.pwm))

    def _reset_pwm(self):
        self.pwm = 0
        self.pwm_var.set(0)
        self.pwm_lbl.config(text="0")

    def _update_display(self):
        self.enc_delta_lbl.config(text=f"Delta: {self.enc_delta}")
        self.enc_cum_lbl.config(text=f"Total: {self.enc_cumulative}")
        self.cmd_lbl.config(text=f"V{self.pwm}")

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
            self._reset_pwm()
            return
        if key in ('w', 'up'):
            self.pwm = min(255, self.pwm + 20)
        elif key in ('s', 'down'):
            self.pwm = max(-255, self.pwm - 20)

        self.pwm_var.set(self.pwm)
        self.pwm_lbl.config(text=str(self.pwm))

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
    app = DebugController()
    app.run()
