# encoder_monitor.py
"""
엔코더 실시간 모니터링 GUI
아두이노 프로토콜:
  수신: E<delta>,C<cumulative>\n
  송신: Z\n  (카운터 리셋)
"""

import threading
import time
import tkinter as tk
from tkinter import ttk
from collections import deque

import serial
import serial.tools.list_ports


BAUD_RATE     = 115200
SEND_INTERVAL = 20       # ms
GRAPH_POINTS  = 100      # 그래프에 표시할 최대 샘플 수


class EncoderMonitor:

    def __init__(self):
        self.ser        = None
        self.connected  = False
        self.running    = True

        self.enc_delta      = 0
        self.enc_cumulative = 0

        # 그래프용 버퍼
        self.delta_buf = deque([0] * GRAPH_POINTS, maxlen=GRAPH_POINTS)

        self._build_gui()

    # ────────────────────────────────────────────
    # 시리얼
    # ────────────────────────────────────────────
    def get_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self):
        port = self.port_var.get()
        if not port:
            return
        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            self.connected = True
            self.status_lbl.config(text="● 연결됨", foreground="green")
            self.connect_btn.config(text="Disconnect")
            threading.Thread(target=self._read_loop, daemon=True).start()
            self._update_loop()
        except serial.SerialException as ex:
            self.status_lbl.config(text=f"✗ {ex}", foreground="red")

    def disconnect(self):
        self.connected = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.status_lbl.config(text="○ 미연결", foreground="gray")
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
                    buf += self.ser.read(self.ser.in_waiting).decode('ascii', errors='ignore')
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        self._parse(line.strip())
                else:
                    time.sleep(0.005)
            except (serial.SerialException, OSError):
                break

    def _parse(self, line):
        """E<delta>,C<cumulative> 파싱"""
        if not line.startswith('E'):
            self._log(line)
            return
        try:
            for part in line.split(','):
                if part.startswith('E'):
                    self.enc_delta = int(part[1:])
                elif part.startswith('C'):
                    self.enc_cumulative = int(part[1:])
            self.delta_buf.append(self.enc_delta)
        except (ValueError, IndexError):
            pass

    def reset_counter(self):
        """아두이노 카운터 리셋"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b'Z\n')
            except (serial.SerialException, OSError):
                pass

    # ────────────────────────────────────────────
    # GUI
    # ────────────────────────────────────────────
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("엔코더 모니터")
        self.root.resizable(False, False)
        self.root.configure(padx=8, pady=8)

        # ── 시리얼 연결 ──
        conn = ttk.LabelFrame(self.root, text="시리얼 연결", padding=8)
        conn.pack(fill='x', pady=(0, 5))

        self.port_var   = tk.StringVar()
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var,
                                       width=15, state='readonly')
        self.port_combo.pack(side='left', padx=(0, 5))

        ttk.Button(conn, text="Refresh", width=8,
                   command=self._refresh_ports).pack(side='left', padx=(0, 5))
        self.connect_btn = ttk.Button(conn, text="Connect", width=12,
                                      command=self.toggle_connect)
        self.connect_btn.pack(side='left', padx=(0, 10))
        self.status_lbl = ttk.Label(conn, text="○ 미연결", foreground="gray")
        self.status_lbl.pack(side='left')

        # ── 수치 표시 ──
        fb = ttk.LabelFrame(self.root, text="엔코더 값", padding=8)
        fb.pack(fill='x', pady=5)

        row = ttk.Frame(fb)
        row.pack(fill='x')
        row.columnconfigure((0, 1), weight=1)

        self.delta_lbl = ttk.Label(row, text="Delta:  ---",
                                   font=('Consolas', 18, 'bold'), anchor='center')
        self.delta_lbl.grid(row=0, column=0, padx=10, pady=4, sticky='ew')

        self.cum_lbl = ttk.Label(row, text="Total:  ---",
                                 font=('Consolas', 18, 'bold'), anchor='center')
        self.cum_lbl.grid(row=0, column=1, padx=10, pady=4, sticky='ew')

        ttk.Button(fb, text="카운터 리셋 (Z)", command=self.reset_counter).pack(pady=(4, 0))

        # ── 간이 바 그래프 (Canvas) ──
        gf = ttk.LabelFrame(self.root, text=f"Delta 히스토리 (최근 {GRAPH_POINTS}샘플)", padding=5)
        gf.pack(fill='x', pady=5)

        self.canvas_w = 460
        self.canvas_h = 120
        self.canvas = tk.Canvas(gf, width=self.canvas_w, height=self.canvas_h,
                                bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack()

        # ── 로그 ──
        lf = ttk.LabelFrame(self.root, text="로그", padding=5)
        lf.pack(fill='x', pady=5)
        self.log_text = tk.Text(lf, height=5, width=56,
                                font=('Consolas', 9), state='disabled')
        self.log_text.pack(fill='x')

        self._refresh_ports()

    def _refresh_ports(self):
        ports = self.get_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def _log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 200:
            self.log_text.delete('1.0', f'{lines-200}.0')
        self.log_text.config(state='disabled')

    # ────────────────────────────────────────────
    # 디스플레이 루프
    # ────────────────────────────────────────────
    def _update_loop(self):
        if not self.connected:
            return
        self._refresh_display()
        self.root.after(SEND_INTERVAL, self._update_loop)

    def _refresh_display(self):
        self.delta_lbl.config(text=f"Delta:  {self.enc_delta:+d}")
        self.cum_lbl.config(text=f"Total:  {self.enc_cumulative:d}")
        self._draw_graph()

    def _draw_graph(self):
        c = self.canvas
        c.delete('all')

        w, h = self.canvas_w, self.canvas_h
        mid   = h // 2
        n     = len(self.delta_buf)
        if n == 0:
            return

        # 중심선
        c.create_line(0, mid, w, mid, fill='#444444')

        # y 스케일 (최대 절댓값 기준, 최소 1)
        max_val = max(abs(v) for v in self.delta_buf) or 1
        bar_w   = w / GRAPH_POINTS

        for i, val in enumerate(self.delta_buf):
            x0 = i * bar_w
            x1 = x0 + bar_w - 1
            bar_h = int((abs(val) / max_val) * (h // 2 - 4))
            color = '#00cc66' if val >= 0 else '#ff4444'
            if val >= 0:
                c.create_rectangle(x0, mid - bar_h, x1, mid, fill=color, outline='')
            else:
                c.create_rectangle(x0, mid, x1, mid + bar_h, fill=color, outline='')

        # 현재값 텍스트
        c.create_text(w - 4, 4, text=f"max±{max_val}",
                      fill='#888888', anchor='ne', font=('Consolas', 8))

    # ────────────────────────────────────────────
    # 실행
    # ────────────────────────────────────────────
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.running = False
        self.disconnect()
        self.root.destroy()


if __name__ == '__main__':
    app = EncoderMonitor()
    app.run()