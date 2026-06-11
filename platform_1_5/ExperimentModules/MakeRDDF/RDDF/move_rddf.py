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

# ── 상태 ──
selected_idx = None          # 선택된 포인트의 df 인덱스
undo_stack = []              # (df_index, old_x, old_y) 리스트

fig, ax = plt.subplots(figsize=(10, 8))

# 경로 선 + 산점도
line, = ax.plot(df["UTM_X"], df["UTM_Y"], color="gray", linewidth=1, zorder=1)
scatter = ax.scatter(df["UTM_X"], df["UTM_Y"], color="steelblue", s=12, zorder=2, picker=True)
highlight = ax.scatter([], [], color="red", s=80, zorder=5, edgecolors="black", linewidths=1.5)
ax.scatter(df["UTM_X"].iloc[0], df["UTM_Y"].iloc[0], color="green", s=120, zorder=6, label="Start")
ax.scatter(df["UTM_X"].iloc[-1], df["UTM_Y"].iloc[-1], color="red", s=120, zorder=6, label="End")

ax.set_xlabel("UTM X")
ax.set_ylabel("UTM Y")
ax.set_title("포인트 이동  |  좌클릭: 선택→이동  |  스크롤: 줌  |  Ctrl+Z: 되돌리기  |  Enter: 저장")
ax.legend()
ax.set_aspect("equal")
margin = 5
ax.set_xlim(df["UTM_X"].min() - margin, df["UTM_X"].max() + margin)
ax.set_ylim(df["UTM_Y"].min() - margin, df["UTM_Y"].max() + margin)


def refresh_plot():
    """산점도와 연결선을 갱신"""
    scatter.set_offsets(np.column_stack([df["UTM_X"], df["UTM_Y"]]))
    line.set_data(df["UTM_X"], df["UTM_Y"])
    if selected_idx is not None:
        highlight.set_offsets([[df.at[selected_idx, "UTM_X"], df.at[selected_idx, "UTM_Y"]]])
    else:
        highlight.set_offsets(np.empty((0, 2)))
    fig.canvas.draw_idle()


def find_nearest(mx, my):
    """마우스 좌표에서 가장 가까운 포인트의 df 인덱스를 반환"""
    # 화면 좌표 기준 거리 계산 (줌 레벨 독립)
    coords = ax.transData.transform(np.column_stack([df["UTM_X"], df["UTM_Y"]]))
    mx_disp, my_disp = ax.transData.transform((mx, my))
    dists = np.hypot(coords[:, 0] - mx_disp, coords[:, 1] - my_disp)
    nearest = int(np.argmin(dists))
    if dists[nearest] > 15:          # 15 픽셀 이상이면 무시
        return None
    return nearest


def on_click(event):
    global selected_idx
    if event.inaxes != ax or event.button != 1:
        return
    # 팬/줌 모드일 때는 무시
    if ax.get_navigate_mode() is not None:
        return

    if selected_idx is None:
        # ── 선택 단계 ──
        idx = find_nearest(event.xdata, event.ydata)
        if idx is None:
            return
        selected_idx = idx
        print(f"[선택] IDX={df.at[idx, 'IDX']}  ({df.at[idx, 'UTM_X']:.6f}, {df.at[idx, 'UTM_Y']:.6f})")
        refresh_plot()
    else:
        # ── 이동 단계 ──
        old_x = df.at[selected_idx, "UTM_X"]
        old_y = df.at[selected_idx, "UTM_Y"]
        undo_stack.append((selected_idx, old_x, old_y))
        df.at[selected_idx, "UTM_X"] = event.xdata
        df.at[selected_idx, "UTM_Y"] = event.ydata
        print(f"[이동] IDX={df.at[selected_idx, 'IDX']}  → ({event.xdata:.6f}, {event.ydata:.6f})")
        selected_idx = None
        refresh_plot()


def on_scroll(event):
    """마우스 스크롤로 줌인/줌아웃"""
    if event.inaxes != ax:
        return
    scale = 0.8 if event.button == "up" else 1.25
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx, cy = event.xdata, event.ydata
    ax.set_xlim(cx + (xlim[0] - cx) * scale, cx + (xlim[1] - cx) * scale)
    ax.set_ylim(cy + (ylim[0] - cy) * scale, cy + (ylim[1] - cy) * scale)
    fig.canvas.draw_idle()


def on_key(event):
    global selected_idx
    if event.key == "ctrl+z":
        if not undo_stack:
            print("[Undo] 되돌릴 내용이 없습니다")
            return
        idx, old_x, old_y = undo_stack.pop()
        df.at[idx, "UTM_X"] = old_x
        df.at[idx, "UTM_Y"] = old_y
        selected_idx = None
        print(f"[Undo] IDX={df.at[idx, 'IDX']}  → ({old_x:.6f}, {old_y:.6f})")
        refresh_plot()
    elif event.key == "escape":
        if selected_idx is not None:
            print(f"[취소] 선택 해제")
            selected_idx = None
            refresh_plot()
    elif event.key == "enter":
        if ext == ".csv":
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, sep="\t", index=False, header=False, float_format="%.6f")
        print(f"[저장완료] {csv_path}")


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("key_press_event", on_key)

plt.tight_layout()
plt.show()
