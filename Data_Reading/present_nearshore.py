#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
present_nearshore.py — 中国近海 Jason-3 回波波形呈现（成品图输出）
====================================================================

默认数据段（可用参数替换）：
    cycle 603 / pass 147，2025-07-22 03:25~03:32 UTC
    一条沿福建—浙江海岸线北上的上升弧段：
      南海深海盆(301 km 离岸) → 台湾海峡陆架 → 福建长乐沿岸(1.9 km)
      → 浙江沿岸(8.8 km) → 穿越台州陆地 → 杭州湾(0.4 km) → 东海陆架

输出（output/ 目录）：
    nearshore_track_overview.png   全弧段在中国海区的轨迹总览
    nearshore_track_zoom.png       闽浙沿岸放大轨迹图（含采样点着色）
    nearshore_heatmap.png          近海段沿轨波形瀑布图（含地表分类色带）
    nearshore_stack.png            近海段波形堆叠图
    nearshore_waveforms.png        五个典型位置的单点波形对比
    （另将每个典型波形单独存为 wf_idx*.png）

用法：
    python present_nearshore.py                     # 用默认数据段
    python present_nearshore.py FILE --i0 A --i1 B --spots 123,456,789
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jason3_reader import read_pass, SURFACE_FLAG_MEANINGS, format_time
from plot_waveform import plot_heatmap, plot_stack, plot_single, _point_info
from plot_track import plot_track
import plot_track as _pt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

DEFAULT_FILE = (
    r"H:\毕业设计\Jason3_Data_Download\cycle_603"
    r"\JA3_GPS_2PgP603_147_20250722_025134_20250722_034730.nc"
)
DEFAULT_I0, DEFAULT_I1 = 43200, 46000
DEFAULT_SPOTS = "43233,43839,44445,45051,45657"
SPOT_NAMES = {
    43233: "台湾海峡陆架（离岸62km）",
    43839: "福建长乐近岸（离岸1.9km）",
    44445: "浙江沿岸（离岸8.8km）",
    45051: "台州陆地",
    45657: "杭州湾口（离岸0.4km）",
}


def waveforms_panel(pd, spots, out_png):
    """五个典型位置的波形对比（2 列多行）"""
    n = len(spots)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.6 * nrow), dpi=120)
    axes = np.atleast_1d(axes).ravel()
    gates = np.arange(104)
    for k, (idx, name) in enumerate(spots):
        ax = axes[k]
        wf = pd.waveform[idx]
        s = int(pd.surface_class[idx])
        color = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}.get(s, "#7f7f7f")
        ax.plot(gates, wf, "-o", color=color, ms=2.8, lw=1.2)
        ax.set_title(f"#{idx}  {name}\n{_point_info(pd, idx)}", fontsize=8.5)
        ax.set_xlabel("距离门", fontsize=8)
        ax.set_ylabel("回波功率 (count)", fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)
        ax.set_xlim(-1, 104)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Jason-3 Ku 波段近海回波波形典型形态对比（沿轨迹由深水到近岸到陆地）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, bbox_inches="tight")
    print(f"已保存: {out_png}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", default=DEFAULT_FILE)
    ap.add_argument("--i0", type=int, default=DEFAULT_I0)
    ap.add_argument("--i1", type=int, default=DEFAULT_I1)
    ap.add_argument("--spots", default=DEFAULT_SPOTS)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    pd = read_pass(a.file, with_waveform=True)
    print(pd.summary())

    spots = []
    for tok in a.spots.split(","):
        idx = int(tok)
        spots.append((idx, SPOT_NAMES.get(idx, "")))

    # 1) 全弧段中国海区总览
    plot_track([a.file], out_png=os.path.join(OUT, "nearshore_track_overview.png"),
               bbox=[105, 145, 0, 42], tiles="esri", nearshore_km=30)

    # 2) 闽浙沿岸放大图
    _pt.TILE_CACHE_FORCE_ZOOM = None
    plot_track([a.file], out_png=os.path.join(OUT, "nearshore_track_zoom.png"),
               bbox=[117.5, 124.5, 24, 31.5], tiles="esri", zoom=7, nearshore_km=30)

    # 3) 近海段瀑布图
    plot_heatmap(pd, a.i0, a.i1, out_png=os.path.join(OUT, "nearshore_heatmap.png"))

    # 4) 近海段堆叠图
    plot_stack(pd, a.i0, a.i1, n_max=260, out_png=os.path.join(OUT, "nearshore_stack.png"))

    # 5) 典型波形对比 + 单点图
    waveforms_panel(pd, spots, os.path.join(OUT, "nearshore_waveforms.png"))
    for idx, name in spots:
        plot_single(pd, idx, out_png=os.path.join(OUT, f"wf_idx{idx}.png"))

    print("\n全部成品图已输出到 output/ 目录。")


if __name__ == "__main__":
    main()
