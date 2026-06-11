#!/usr/bin/env python3
"""
HS-956MG 서보모터 제어 GUI

Arduino 프로토콜:
  송신: A<angle>\n   (angle: 0~180)
  수신: A<angle>\n   (에코백)
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

import serial
import serial.tools.list_ports


class ServoController:
    BAUD_RATE = 115200
    SEND_INTERVAL_MS = 20

    def __init__(self):
        self.ser = None
        self.connected = False
        self.running = True

        self.angle = 90  # 0 ~ 180
        self.fb_angle = 0

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
                self.ser.write(b"A90\n")
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
        try:
            if line.startswith('A'):
                self.fb_angle = int(line[1:])
            else:
                self._append_log(line)
        except (ValueError, IndexError):
            self._append_log(line)

    def _send_loop(self):
        if not self.connected:
            return
        try:
            if self.ser and self.ser.is_open:
                cmd = f"A{self.angle}\n"
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
        self.root.title("HS-956MG 서보모터 제어")
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

        # ---- 각도 제어 ----
        ctrl = ttk.LabelFrame(self.root, text="서보 각도 (0~180°)", padding=8)
        ctrl.pack(fill='x', pady=5)

        af = ttk.Frame(ctrl)
        af.pack(fill='x', pady=2)
        ttk.Label(af, text="Angle:", width=7).pack(side='left')
        self.angle_var = tk.IntVar(value=90)
        self.angle_scale = ttk.Scale(af, from_=0, to=180, orient='horizontal',
                                     variable=self.angle_var,
                                     command=self._on_angle_slider)
        self.angle_scale.pack(side='left', fill='x', expand=True, padx=5)
        self.angle_lbl = ttk.Label(af, text="90", width=5, anchor='e')
        self.angle_lbl.pack(side='left')
        ttk.Button(af, text="Center", width=7,
                   command=self._center).pack(side='left', padx=(5, 0))

        # ---- 피드백 ----
        fb = ttk.LabelFrame(self.root, text="피드백 (Arduino → PC)", padding=8)
        fb.pack(fill='x', pady=5)
        self.fb_lbl = ttk.Label(fb, text="Angle: ---", anchor='w',
                                font=('Consolas', 11))
        self.fb_lbl.pack(anchor='w')

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
                  text="A/D ±5° | ←/→ ±1° | Space 중앙(90°)").pack()

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

    def _on_angle_slider(self, v):
        self.angle = int(float(v))
        self.angle_lbl.config(text=str(self.angle))

    def _center(self):
        self.angle = 90
        self.angle_var.set(90)
        self.angle_lbl.config(text="90")

    def _update_display(self):
        self.fb_lbl.config(text=f"Angle: {self.fb_angle}°")
        self.cmd_lbl.config(text=f"A{self.angle}")

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
            self._center()
            return

        if key in ('a',):
            self.angle = max(0, self.angle - 5)
        elif key in ('d',):
            self.angle = min(180, self.angle + 5)
        elif key == 'left':
            self.angle = max(0, self.angle - 1)
        elif key == 'right':
            self.angle = min(180, self.angle + 1)
        else:
            return

        self.angle_var.set(self.angle)
        self.angle_lbl.config(text=str(self.angle))

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
    app = ServoController()
    app.run()
