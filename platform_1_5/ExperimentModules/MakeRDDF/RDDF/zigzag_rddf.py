import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"C:\Users\JungiHong\OneDrive\Hongik\HI-RACE\2026-01\platform_1_5\Competition\1_5_map\competition.txt")
COLUMNS = ["IDX", "UTM_X", "UTM_Y", "Velocity", "Evt"]
ext = os.path.splitext(csv_path)[1].lower()
if ext == ".csv":
    df = pd.read_csv(csv_path)
else:
    df = pd.read_csv(csv_path, sep="\t", header=None, names=COLUMNS)
df["IDX"] = df["IDX"].astype(int)
df["Evt"] = df["Evt"].astype(int)

# ── 파라미터 ──
AMPLITUDE = 2.0        # 가상직선으로부터의 수직 거리 (미터)
PEAK_SPACING = 11.0     # 연속 꼭지점 간격 (미터, 반파장)

# ── 상태 ──
state = "idle"         # "idle" → "start_selected"
start_idx = None
undo_stack = []        # [(indices_list, old_x_arr, old_y_arr), ...]
guide_lines = []       # 가이드 라인 아티스트 목록

fig, ax = plt.subplots(figsize=(10, 8))

# 플롯 요소
line, = ax.plot(df["UTM_X"], df["UTM_Y"], color="gray", linewidth=1, zorder=1)
scatter = ax.scatter(df["UTM_X"], df["UTM_Y"], color="steelblue", s=12, zorder=2)
highlight_start = ax.scatter([], [], color="lime", s=100, zorder=5,
                             edgecolors="black", linewidths=1.5, label="선택점")
guide_line, = ax.plot([], [], color="gray", linewidth=1.5,
                      linestyle="--", zorder=3, label="가상직선")
ax.scatter(df["UTM_X"].iloc[0], df["UTM_Y"].iloc[0],
           color="green", s=120, zorder=6, label="Start")
ax.scatter(df["UTM_X"].iloc[-1], df["UTM_Y"].iloc[-1],
           color="red", s=120, zorder=6, label="End")

ax.set_xlabel("UTM X")
ax.set_ylabel("UTM Y")
ax.set_title(f"지그재그 생성 (진폭={AMPLITUDE}m)  |  좌클릭: 시작→끝  |  스크롤: 줌  |  Ctrl+Z: 되돌리기  |  Enter: 저장")
ax.legend(fontsize=8)
ax.set_aspect("equal")
margin = 5
ax.set_xlim(df["UTM_X"].min() - margin, df["UTM_X"].max() + margin)
ax.set_ylim(df["UTM_Y"].min() - margin, df["UTM_Y"].max() + margin)


def refresh_plot():
    scatter.set_offsets(np.column_stack([df["UTM_X"], df["UTM_Y"]]))
    line.set_data(df["UTM_X"], df["UTM_Y"])
    if start_idx is not None:
        highlight_start.set_offsets(
            [[df.at[start_idx, "UTM_X"], df.at[start_idx, "UTM_Y"]]])
    else:
        highlight_start.set_offsets(np.empty((0, 2)))
    fig.canvas.draw_idle()


def find_nearest(mx, my):
    coords = ax.transData.transform(
        np.column_stack([df["UTM_X"], df["UTM_Y"]]))
    mx_d, my_d = ax.transData.transform((mx, my))
    dists = np.hypot(coords[:, 0] - mx_d, coords[:, 1] - my_d)
    nearest = int(np.argmin(dists))
    if dists[nearest] > 15:
        return None
    return nearest


def generate_zigzag(si, ei):
    lo, hi = min(si, ei), max(si, ei)
    indices = list(range(lo, hi + 1))
    n = len(indices)
    if n < 3:
        print("[오류] 포인트가 너무 적습니다 (최소 3개 필요)")
        return None

    x1, y1 = df.at[lo, "UTM_X"], df.at[lo, "UTM_Y"]
    x2, y2 = df.at[hi, "UTM_X"], df.at[hi, "UTM_Y"]
    dx, dy = x2 - x1, y2 - y1
    L = np.hypot(dx, dy)
    if L < 0.5:
        print("[오류] 선택 구간이 너무 짧습니다")
        return None

    ux, uy = dx / L, dy / L       # 직선 방향 단위벡터
    px, py = -uy, ux               # 수직 방향 (반시계 90°)

    # 원본 좌표 백업 (undo용)
    old_x = df.loc[indices, "UTM_X"].values.copy()
    old_y = df.loc[indices, "UTM_Y"].values.copy()

    # 항상 1사이클 (시작→꼭지점→끝)
    n_cycles = 1
    wavelength = L

    for i, idx in enumerate(indices):
        t = i / (n - 1)            # 0 → 1

        # 반사이클: t=0 → 0, t=0.5 → +A, t=1 → 0
        wave = 1.0 - abs(2.0 * t - 1.0)   # 0 → 1 → 0

        offset = -AMPLITUDE * wave

        base_x = x1 + t * dx
        base_y = y1 + t * dy
        df.at[idx, "UTM_X"] = base_x + offset * px
        df.at[idx, "UTM_Y"] = base_y + offset * py

    # 가이드 직선 표시
    guide_line.set_data([x1, x2], [y1, y2])

    print(f"[지그재그] IDX {df.at[lo, 'IDX']}~{df.at[hi, 'IDX']}  "
          f"({n}개 포인트, {n_cycles}사이클, 진폭={AMPLITUDE}m, 파장={wavelength:.1f}m)")
    return indices, old_x, old_y


def on_click(event):
    global start_idx, state
    if event.inaxes != ax or event.button != 1:
        return
    if ax.get_navigate_mode() is not None:
        return

    idx = find_nearest(event.xdata, event.ydata)
    if idx is None:
        return

    if state == "idle":
        start_idx = idx
        state = "start_selected"
        print(f"[시작점] IDX={df.at[idx, 'IDX']}  "
              f"({df.at[idx, 'UTM_X']:.6f}, {df.at[idx, 'UTM_Y']:.6f})")
        refresh_plot()

    elif state == "start_selected":
        if idx == start_idx:
            return
        result = generate_zigzag(start_idx, idx)
        if result:
            undo_stack.append(result)
        start_idx = None
        state = "idle"
        refresh_plot()


def on_scroll(event):
    if event.inaxes != ax:
        return
    scale = 0.8 if event.button == "up" else 1.25
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    cx, cy = event.xdata, event.ydata
    ax.set_xlim(cx + (xlim[0] - cx) * scale, cx + (xlim[1] - cx) * scale)
    ax.set_ylim(cy + (ylim[0] - cy) * scale, cy + (ylim[1] - cy) * scale)
    fig.canvas.draw_idle()


def on_key(event):
    global start_idx, state
    if event.key == "ctrl+z":
        if not undo_stack:
            print("[Undo] 되돌릴 내용이 없습니다")
            return
        indices, old_x, old_y = undo_stack.pop()
        for i, idx in enumerate(indices):
            df.at[idx, "UTM_X"] = old_x[i]
            df.at[idx, "UTM_Y"] = old_y[i]
        start_idx = None
        state = "idle"
        guide_line.set_data([], [])
        print(f"[Undo] {len(indices)}개 포인트 복원됨")
        refresh_plot()

    elif event.key == "escape":
        if state == "start_selected":
            start_idx = None
            state = "idle"
            print("[취소] 선택 해제")
            refresh_plot()

    elif event.key == "enter":
        if ext == ".csv":
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, sep="\t", index=False,
                      header=False, float_format="%.6f")
        print(f"[저장완료] {csv_path}")


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.tight_layout()
plt.show()
