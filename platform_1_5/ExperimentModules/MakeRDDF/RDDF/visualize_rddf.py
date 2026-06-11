import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import os

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r"C:\Users\JungiHong\OneDrive\Hongik\HI-RACE\2026-01\platform_1_5\ExperimentModules\MakeRDDF\RDDF\충북대_Outline_UTM_corner_ban.txt")
COLUMNS = ["IDX", "UTM_X", "UTM_Y", "Velocity", "Evt"]
ext = os.path.splitext(csv_path)[1].lower()
if ext == ".csv":
    df = pd.read_csv(csv_path)
else:
    df = pd.read_csv(csv_path, sep="\t", header=None, names=COLUMNS)
df["IDX"] = df["IDX"].astype(int)
df["Evt"] = df["Evt"].astype(int)

# # 실행 시 Evt=2인 포인트를 모두 0으로 초기화
count_reset = (df["Evt"] == 2).sum()
if count_reset > 0:
    df.loc[df["Evt"] == 2, "Evt"] = 0
    print(f"[초기화] Evt=2 → Evt=0: {count_reset}개 포인트 변경됨")

fig, ax = plt.subplots(figsize=(10, 8))

# 기본 경로 (Evt=0: 회색, Evt=1: 주황색, Evt=2: 빨간색)
evt0 = df[df["Evt"] == 0]
evt1 = df[df["Evt"] == 1]
evt2 = df[df["Evt"] == 2]
line, = ax.plot(df["UTM_X"], df["UTM_Y"], color="gray", linewidth=1, zorder=1)
scatter_normal = ax.scatter(evt0["UTM_X"], evt0["UTM_Y"], color="gray", s=10, zorder=2)
scatter_evt = ax.scatter(evt1["UTM_X"], evt1["UTM_Y"], color="orange", s=20, zorder=3)
scatter_evt2 = ax.scatter(evt2["UTM_X"], evt2["UTM_Y"], color="magenta", s=20, zorder=4)
ax.scatter(df["UTM_X"].iloc[0], df["UTM_Y"].iloc[0], color="green", s=120, zorder=5, label="Start")
ax.scatter(df["UTM_X"].iloc[-1], df["UTM_Y"].iloc[-1], color="red", s=120, zorder=5, label="End")
ax.set_xlabel("UTM X")
ax.set_ylabel("UTM Y")
ax.set_title("RDDF Path  |  Drag → Evt=1  |  Right-drag → Evt=2  |  Enter to save")
ax.legend()
ax.set_aspect("equal")
# 아래 두 줄 추가
margin = 5
ax.set_xlim(df["UTM_X"].min() - margin, df["UTM_X"].max() + margin)
ax.set_ylim(df["UTM_Y"].min() - margin, df["UTM_Y"].max() + margin)


def refresh_plot():
    evt0 = df[df["Evt"] == 0]
    evt1 = df[df["Evt"] == 1]
    evt2 = df[df["Evt"] == 2]
    scatter_normal.set_offsets(np.column_stack([evt0["UTM_X"], evt0["UTM_Y"]]))
    scatter_evt.set_offsets(np.column_stack([evt1["UTM_X"], evt1["UTM_Y"]]) if len(evt1) > 0 else np.empty((0, 2)))
    scatter_evt2.set_offsets(np.column_stack([evt2["UTM_X"], evt2["UTM_Y"]]) if len(evt2) > 0 else np.empty((0, 2)))
    fig.canvas.draw_idle()


def on_select(eclick, erelease):
    x1, x2 = sorted([eclick.xdata, erelease.xdata])
    y1, y2 = sorted([eclick.ydata, erelease.ydata])
    mask = (df["UTM_X"] >= x1) & (df["UTM_X"] <= x2) & (df["UTM_Y"] >= y1) & (df["UTM_Y"] <= y2)
    count = mask.sum()
    if count == 0:
        return
    if eclick.button == 1:  # 좌클릭 드래그 → Evt=1
        df.loc[mask, "Evt"] = 1
        print(f"[Evt=1] {count}개 포인트 선택됨 (IDX: {df.loc[mask, 'IDX'].min()} ~ {df.loc[mask, 'IDX'].max()})")
    elif eclick.button == 3:  # 우클릭 드래그 → Evt=2
        df.loc[mask, "Evt"] = 2
        print(f"[Evt=2] {count}개 포인트 선택됨 (IDX: {df.loc[mask, 'IDX'].min()} ~ {df.loc[mask, 'IDX'].max()})")
    refresh_plot()


def on_key(event):
    if event.key == "enter":
        if ext == ".csv":
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, sep="\t", index=False, header=False, float_format="%.6f")


fig.canvas.mpl_connect("key_press_event", on_key)

selector = RectangleSelector(ax, on_select, useblit=True,
                             button=[1, 3], interactive=False,
                             props=dict(facecolor="blue", alpha=0.2))

plt.tight_layout()
plt.show()
