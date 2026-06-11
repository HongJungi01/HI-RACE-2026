"""TDMS->CSV 로그 시각화 + 클릭으로 해당 위치 로그 조회 (4탭 분할).

사용법:
    python log_viewer.py 20260523_2240_0.csv

조작:
    - 인덱스 기준으로 데이터가 4등분되어 4개의 탭에 표시됩니다.
    - 각 탭 좌측 지도(UTM 평면)에서 점을 클릭하면 가장 가까운 GPS/Driving
      샘플의 모든 채널 값이 우측 패널에 그룹별로 정리되어 표시됩니다.
    - 'r' 키: 선택 초기화
    - 'g' 키: GPS 경로 토글
    - 'd' 키: Driving 경로 토글
    - 마우스 휠/툴바로 줌/팬 가능

요구사항: pip install pandas matplotlib numpy
"""

from __future__ import annotations

import argparse
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 컬럼명 패턴: /'GroupName'/'ChannelName'
COL_RE = re.compile(r"^/'(?P<group>[^']+)'/'(?P<channel>[^']+)'$")

N_TABS = 3


def parse_columns(columns: list[str]) -> dict[str, dict[str, str]]:
    """원본 컬럼명을 {group: {channel: original_col_name}} 매핑으로 변환."""
    mapping: dict[str, dict[str, str]] = {}
    for col in columns:
        m = COL_RE.match(col)
        if not m:
            mapping.setdefault("(misc)", {})[col] = col
            continue
        mapping.setdefault(m["group"], {})[m["channel"]] = col
    return mapping


def find_col(groups: dict[str, dict[str, str]], group: str, channel: str) -> str | None:
    return groups.get(group, {}).get(channel)


def format_value(v: object) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        if abs(v) < 1e15 and float(v).is_integer():
            return f"{int(v)}"
        return f"{v:.6g}"
    return str(v)


class LogPanel:
    """한 탭의 내용: df 의 [start, end) 구간만 표시하는 지도 + 텍스트 패널."""

    def __init__(
        self,
        parent: tk.Widget,
        df: pd.DataFrame,
        start: int,
        end: int,
        groups: dict[str, dict[str, str]],
        gps_cols: tuple[str, str],
        drv_cols: tuple[str, str],
    ) -> None:
        self.df = df
        self.start = start
        self.end = end
        self.groups = groups

        gx, gy = gps_cols
        dx, dy = drv_cols
        gx_arr = df[gx].to_numpy()[start:end]
        gy_arr = df[gy].to_numpy()[start:end]
        dx_arr = df[dx].to_numpy()[start:end]
        dy_arr = df[dy].to_numpy()[start:end]

        g_valid = (~np.isnan(gx_arr)) & (~np.isnan(gy_arr)) & (gx_arr != 0) & (gy_arr != 0)
        d_valid = (~np.isnan(dx_arr)) & (~np.isnan(dy_arr)) & (dx_arr != 0) & (dy_arr != 0)
        self.gps_xy = {"x": gx_arr[g_valid], "y": gy_arr[g_valid], "idx": np.flatnonzero(g_valid) + start}
        self.drv_xy = {"x": dx_arr[d_valid], "y": dy_arr[d_valid], "idx": np.flatnonzero(d_valid) + start}

        # 라디오로 선택할 강조 조건들 — 라벨 -> (group, channel, predicate)
        overlay_specs: list[tuple[str, str, str, "callable"]] = [
            ("E-Stop != 0", "Driving Data", "E-Stop", lambda a: (~np.isnan(a)) & (a != 0)),
            ("HAP_ESTOP != 0", "Judgment Data", "HAP_ESTOP", lambda a: (~np.isnan(a)) & (a != 0)),
            ("Position_Type != 56", "GPS Data", "Position_Type", lambda a: (~np.isnan(a)) & (a != 56)),
        ]
        self.overlays: dict[str, dict] = {}
        for label, group, channel, predicate in overlay_specs:
            self.overlays[label] = self._build_overlay(
                df, groups, group, channel, predicate,
                start, end, g_valid, d_valid, gx_arr, gy_arr, dx_arr, dy_arr,
            )

        self.fig = Figure(figsize=(15, 9))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
        self.ax_map = self.fig.add_subplot(gs[0, 0])
        self.ax_text = self.fig.add_subplot(gs[0, 1])
        self.ax_text.axis("off")

        ax = self.ax_map
        ax.set_title(f"UTM Map (rows {start} ~ {end - 1})")
        ax.set_xlabel("UTM_X (m)")
        ax.set_ylabel("UTM_Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.5)

        # 경로(라인) + 점(스캐터) — 모두 검정색
        self.gps_line, = ax.plot(
            self.gps_xy["x"], self.gps_xy["y"],
            "-", color="black", linewidth=1.0, alpha=0.6,
            label=f"GPS ({len(self.gps_xy['idx'])} pts)",
        )
        self.gps_pts = ax.scatter(
            self.gps_xy["x"], self.gps_xy["y"],
            s=6, c="black", alpha=0.5,
        )
        self.drv_line, = ax.plot(
            self.drv_xy["x"], self.drv_xy["y"],
            "-", color="black", linewidth=1.0, alpha=0.6,
            label=f"Driving ({len(self.drv_xy['idx'])} pts)",
        )
        self.drv_pts = ax.scatter(
            self.drv_xy["x"], self.drv_xy["y"],
            s=6, c="black", alpha=0.5,
        )
        # 라디오로 선택된 강조 오버레이 (빨간색)
        initial_label = next(iter(self.overlays))
        initial_xy = self.overlays[initial_label]
        self.overlay_pts = ax.scatter(
            initial_xy["x"], initial_xy["y"],
            s=24, c="#d62728", alpha=0.9, zorder=6,
            label=f"{initial_label} ({len(initial_xy['x'])} pts)",
        )
        self.sel_marker, = ax.plot(
            [], [], "o", markersize=14, markerfacecolor="none",
            markeredgecolor="magenta", markeredgewidth=2.5, zorder=10,
        )
        ax.legend(loc="upper right")

        self.text_artist = self.ax_text.text(
            0.0, 1.0, self._placeholder_text(),
            transform=self.ax_text.transAxes,
            va="top", ha="left",
            family="monospace", fontsize=8,
        )

        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)

        # 조건 라디오버튼 (상단)
        radio_frame = ttk.Frame(container)
        radio_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(radio_frame, text="강조 조건:").pack(side=tk.LEFT, padx=(6, 8))
        self.condition_var = tk.StringVar(value=initial_label)
        for label, xy in self.overlays.items():
            ttk.Radiobutton(
                radio_frame,
                text=f"{label}  ({len(xy['x'])})",
                variable=self.condition_var,
                value=label,
                command=self._on_condition_change,
            ).pack(side=tk.LEFT, padx=4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.toolbar = NavigationToolbar2Tk(self.canvas, container, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # 키보드 이벤트를 받으려면 위젯에 포커스가 필요
        widget = self.canvas.get_tk_widget()
        widget.bind("<Enter>", lambda _e: widget.focus_set())

        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("key_press_event", self._on_key)

    @staticmethod
    def _build_overlay(
        df: pd.DataFrame,
        groups: dict[str, dict[str, str]],
        group: str,
        channel: str,
        predicate,
        start: int,
        end: int,
        g_valid: np.ndarray,
        d_valid: np.ndarray,
        gx_arr: np.ndarray,
        gy_arr: np.ndarray,
        dx_arr: np.ndarray,
        dy_arr: np.ndarray,
    ) -> dict:
        col = find_col(groups, group, channel)
        if col is None:
            return {"x": np.array([]), "y": np.array([])}
        arr = df[col].to_numpy()[start:end].astype(float, copy=False)
        mask = predicate(arr)
        g_m = g_valid & mask
        d_m = d_valid & mask
        return {
            "x": np.concatenate([gx_arr[g_m], dx_arr[d_m]]),
            "y": np.concatenate([gy_arr[g_m], dy_arr[d_m]]),
        }

    def _on_condition_change(self) -> None:
        label = self.condition_var.get()
        xy = self.overlays.get(label, {"x": np.array([]), "y": np.array([])})
        if xy["x"].size:
            self.overlay_pts.set_offsets(np.column_stack([xy["x"], xy["y"]]))
        else:
            self.overlay_pts.set_offsets(np.empty((0, 2)))
        self.overlay_pts.set_label(f"{label} ({len(xy['x'])} pts)")
        self.ax_map.legend(loc="upper right")
        self.canvas.draw_idle()

    @staticmethod
    def _placeholder_text() -> str:
        return (
            "지도에서 점을 클릭하세요.\n\n"
            "단축키:\n"
            "  r : 선택 해제\n"
            "  g : GPS 경로 표시 토글\n"
            "  d : Driving 경로 표시 토글\n"
        )

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax_map or event.xdata is None:
            return
        if self.toolbar.mode:
            return

        cands = []
        for source, xy in (("GPS", self.gps_xy), ("Driving", self.drv_xy)):
            if xy["x"].size == 0:
                continue
            d2 = (xy["x"] - event.xdata) ** 2 + (xy["y"] - event.ydata) ** 2
            k = int(np.argmin(d2))
            cands.append((d2[k], source, int(xy["idx"][k]), float(xy["x"][k]), float(xy["y"][k])))
        if not cands:
            return
        cands.sort(key=lambda t: t[0])
        _, source, row_idx, px, py = cands[0]

        self.sel_marker.set_data([px], [py])
        self.text_artist.set_text(self._format_row(row_idx, source, px, py))
        self.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "r":
            self.sel_marker.set_data([], [])
            self.text_artist.set_text(self._placeholder_text())
        elif event.key == "g":
            vis = not self.gps_line.get_visible()
            self.gps_line.set_visible(vis)
            self.gps_pts.set_visible(vis)
        elif event.key == "d":
            vis = not self.drv_line.get_visible()
            self.drv_line.set_visible(vis)
            self.drv_pts.set_visible(vis)
        else:
            return
        self.canvas.draw_idle()

    def _format_row(self, row_idx: int, clicked_source: str, px: float, py: float) -> str:
        row = self.df.iloc[row_idx]
        lines: list[str] = []
        lines.append(f"Clicked: {clicked_source}  (row #{row_idx})")
        lines.append(f"UTM: ({px:.3f}, {py:.3f})")
        lines.append("=" * 48)

        preferred = ["GPS Data", "IMU Data", "Driving Data", "Obs Data", "Judgment Data"]
        ordered = [g for g in preferred if g in self.groups]
        ordered += [g for g in self.groups if g not in ordered]

        for group in ordered:
            channels = self.groups[group]
            lines.append(f"\n[{group}]")
            width = min(max((len(c) for c in channels), default=10), 24)
            for ch, col in channels.items():
                val = row[col]
                lines.append(f"  {ch:<{width}} : {format_value(val)}")
                if group == "GPS Data" and ch == "GPS Millisecond":
                    lines.append(f"  {'Δ to next':<{width}} : {self._next_ms_diff(row_idx, col)}")
        return "\n".join(lines)

    def _next_ms_diff(self, row_idx: int, col: str) -> str:
        if row_idx + 1 >= len(self.df):
            return "N/A (last row)"
        try:
            diff = float(self.df.iloc[row_idx + 1][col]) - float(self.df.iloc[row_idx][col])
        except (TypeError, ValueError):
            return "N/A"
        if np.isnan(diff):
            return "NaN"
        return f"{format_value(diff)} ms"


class LogViewerApp:
    def __init__(self, df: pd.DataFrame, csv_path: Path) -> None:
        self.df = df
        self.csv_path = csv_path
        groups = parse_columns(list(df.columns))

        gps_x = find_col(groups, "GPS Data", "UTM_X")
        gps_y = find_col(groups, "GPS Data", "UTM_Y")
        drv_x = find_col(groups, "Driving Data", "UTM_X")
        drv_y = find_col(groups, "Driving Data", "UTM_Y")
        if not all([gps_x, gps_y, drv_x, drv_y]):
            raise ValueError("GPS Data 또는 Driving Data 의 UTM_X/UTM_Y 컬럼을 찾지 못했습니다.")

        self.root = tk.Tk()
        self.root.title(f"Log Viewer - {csv_path.name}")
        self.root.geometry("1500x900")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        n = len(df)
        chunk = n // N_TABS
        self.panels: list[LogPanel] = []
        for i in range(N_TABS):
            start = i * chunk
            end = (i + 1) * chunk if i < N_TABS - 1 else n
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"Part {i + 1}  ({start} ~ {end - 1})")
            panel = LogPanel(frame, df, start, end, groups, (gps_x, gps_y), (drv_x, drv_y))
            self.panels.append(panel)

    def show(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize GPS/Driving UTM paths and inspect log rows by clicking.")
    parser.add_argument("csv_path", type=Path, help="Path to CSV produced by tdms_to_csv.py")
    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        return 1

    print(f"Loading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path, encoding="utf-8-sig", low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    app = LogViewerApp(df, args.csv_path)
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
