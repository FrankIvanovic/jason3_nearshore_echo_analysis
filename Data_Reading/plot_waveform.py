#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_waveform.py — Jason-3 Ku 波段 104 距离门回波波形可视化
============================================================

呈现 NetCDF 文件内各采样门的回波数据，四种视图：

  1. single   某个 20Hz 采样点的完整波形（功率 vs 距离门 0~103）
  2. overlay  沿轨多个采样点波形叠加对比
  3. heatmap  沿轨"瀑布图"（x=距离门, y=沿轨采样序号, 色=功率）
  4. stack    波形堆叠图（x=距离门, y=沿轨, 每条波形垂向平移）——
              近海过渡段海洋 Brown 波形 → 前沿后移/尖峰波形的经典呈现方式

用法示例
--------
    # 画第 1000 个采样点的波形
    python plot_waveform.py FILE --index 1000

    # 自动找开阔海洋点与近海点各画一张
    python plot_waveform.py FILE --mode ocean
    python plot_waveform.py FILE --mode coastal --nearshore-km 20

    # 沿轨瀑布图（索引 5000~8000 段）
    python plot_waveform.py FILE --heatmap --i0 5000 --i1 8000

    # 近海过渡段波形堆叠图
    python plot_waveform.py FILE --stack --i0 5000 --i1 8000 --stack-n 120
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from jason3_reader import (
    read_pass, read_waveforms, SURFACE_FLAG_MEANINGS, format_time, N_GATES,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GATES = np.arange(N_GATES)


def _point_info(pd, i: int) -> str:
    s = int(pd.surface_class[i])
    return (f"{format_time(pd.time[i])} UTC | "
            f"{pd.lat[i]:.4f}°N, {pd.lon_180[i]:.4f}°E | "
            f"{SURFACE_FLAG_MEANINGS.get(s, s)} | 离岸 {pd.dist_coast[i]/1e3:.1f} km")


def plot_single(pd, i: int, out_png=None):
    """单个采样点波形 + 局部噪声统计"""
    wf = pd.waveform[i]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)
    ax.plot(GATES, wf, "-o", color="#1f77b4", ms=3.5, lw=1.4)
    ax.axvspan(0, N_GATES - 1, color="none")

    # 噪声底（前沿前 10 门）与峰值标注
    noise = np.nanmedian(wf[:10])
    imax = int(np.nanargmax(wf))
    ax.axhline(noise, color="gray", ls=":", lw=1, label=f"噪声底≈{noise:.1f} count")
    ax.annotate(f"峰值门 {imax}\n{wf[imax]:.1f} count",
                xy=(imax, wf[imax]), xytext=(imax + 6, wf[imax] * 0.85),
                arrowprops=dict(arrowstyle="->", color="k", lw=0.8), fontsize=9)

    ax.set_xlabel("距离门序号（Ku 波段，约 0.47 m/门）")
    ax.set_ylabel("回波功率 (count)")
    ax.set_title(f"Jason-3 Ku 波段回波波形 — 采样点 #{i}\n{_point_info(pd, i)}", fontsize=10)
    ax.set_xlim(-1, N_GATES)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def plot_overlay(pd, i_center: int, half: int = 25, out_png=None):
    """中心点附近 2*half+1 个波形叠加"""
    i0, i1 = max(0, i_center - half), min(pd.n, i_center + half + 1)
    wf = pd.waveform[i0:i1]
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=120)
    cmap = plt.get_cmap("viridis")
    for k in range(wf.shape[0]):
        frac = k / max(1, wf.shape[0] - 1)
        ax.plot(GATES, wf[k], lw=0.8, color=cmap(frac), alpha=0.75)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(i0, i1 - 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("沿轨采样点索引")
    im = i_center - i0
    ax.plot(GATES, wf[im], color="red", lw=1.8, label=f"中心点 #{i_center}")
    ax.set_xlabel("距离门序号（Ku 波段，约 0.47 m/门）")
    ax.set_ylabel("回波功率 (count)")
    ax.set_title(f"Jason-3 Ku 波段波形叠加（沿轨 ±{half} 点）\n中心点: {_point_info(pd, i_center)}",
                 fontsize=10)
    ax.set_xlim(-1, N_GATES)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def _class_band(ax_top, surf, i0, i1):
    """heatmap 顶部画地表分类色带"""
    seg = surf[i0:i1]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#8c564b", "#7f7f7f", "#bcbd22", "#9467bd"]
    xs = np.arange(i0, i0 + len(seg))
    cs = [colors[k] if 0 <= k < len(colors) else "k" for k in seg]
    ax_top.bar(xs, 1, width=1.5, color=cs, align="edge", linewidth=0)
    ax_top.set_ylim(0, 1)


def plot_heatmap(pd, i0: int, i1: int, out_png=None, vmax_pct=99):
    """沿轨波形瀑布热图 + 地表分类色带"""
    wf = pd.waveform[i0:i1]
    fig = plt.figure(figsize=(12, 8.6), dpi=120)
    gs = GridSpec(2, 1, height_ratios=[1, 24], hspace=0.08)
    ax_band = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_band)

    vmax = np.nanpercentile(wf, vmax_pct)
    im = ax.imshow(wf.T, aspect="auto", cmap="jet", origin="lower",
                   extent=[i0, i1, -0.5, N_GATES - 0.5],
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("沿轨 20Hz 采样点索引")
    ax.set_ylabel("距离门序号")
    _class_band(ax_band, pd.surface_class, i0, i1)
    ax_band.set_xlim(i0, i1)
    ax_band.set_yticks([])
    ax_band.set_xticklabels([])
    ax_band.set_title("地表分类：蓝=海洋 红=陆地 绿=内陆水 棕=水生植被", fontsize=8.5, pad=2)
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("回波功率 (count)")

    fig.suptitle(f"Jason-3 Ku 波段沿轨波形瀑布图（索引 {i0}~{i1 - 1}）\n"
                 f"起点: {_point_info(pd, i0)}   |   终点: {_point_info(pd, i1 - 1)}",
                 fontsize=10, y=0.995)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def plot_stack(pd, i0: int, i1: int, n_max: int = 150, out_png=None):
    """波形堆叠图：近海过渡段可视化的经典画法"""
    n = i1 - i0
    step = max(1, n // n_max)
    idx = np.arange(i0, i1, step)
    wf = pd.waveform[idx]

    # 以各波形峰值门对齐显示，纵轴为沿轨序号
    offsets = np.arange(len(idx), dtype=float)
    ysc = _yscale(wf)
    fig, ax = plt.subplots(figsize=(11, 10), dpi=120)
    colors = np.where(pd.surface_class[idx] == 0, "#1f77b4",
                      np.where(pd.surface_class[idx] == 1, "#d62728", "#2ca02c"))
    for k in range(len(idx)):
        ax.plot(GATES, wf[k] + offsets[k] * ysc, color=colors[k], lw=0.8)
    ax.set_xlabel("距离门序号（Ku 波段，约 0.47 m/门）")
    ax.set_ylabel("沿轨 20Hz 采样点索引")
    ks = np.linspace(0, len(idx) - 1, 8).astype(int)
    ax.set_yticks(ks * ysc)
    ax.set_yticklabels([str(int(i)) for i in idx[ks]], fontsize=8)
    ax.set_ylim(-1.5 * ysc, (len(idx) - 1) * ysc + 2.0 * ysc)
    ax.set_xlim(-1, N_GATES)
    ax.set_title("Jason-3 Ku 波段波形堆叠图（蓝=海洋 红=陆地 绿=其他）\n"
                 f"索引 {i0}~{i1 - 1} | 起点离岸 {pd.dist_coast[i0]/1e3:.1f} km → "
                 f"终点离岸 {pd.dist_coast[i1-1]/1e3:.1f} km", fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def _yscale(wf) -> float:
    """堆叠垂向间距：按波形 98 分位数自适应"""
    v = np.nanpercentile(wf, 98)
    return (v * 1.1) if np.isfinite(v) and v > 0 else 1.0


def find_index(pd, mode: str, nearshore_km: float) -> int:
    """按模式自动选采样点：ocean=开阔海，coast=近海（离岸<阈值 且仍为海洋面）"""
    if mode == "ocean":
        cand = np.where((pd.surface_class == 0) & (pd.dist_coast > 200e3))[0]
    else:
        cand = np.where((pd.surface_class == 0) &
                        (pd.dist_coast < nearshore_km * 1e3) &
                        (pd.dist_coast > 1e3))[0]
        if len(cand) == 0:  # 放宽到非陆地点
            cand = np.where((pd.surface_class != 1) &
                            (pd.dist_coast < nearshore_km * 1e3))[0]
    if len(cand) == 0:
        raise SystemExit(f"该文件中找不到满足 {mode} 条件的采样点")
    return int(cand[len(cand) // 2])   # 取中位附近一个


def main():
    ap = argparse.ArgumentParser(description="Jason-3 Ku 波段波形可视化")
    ap.add_argument("file")
    ap.add_argument("--index", type=int, help="20Hz 采样点沿轨索引")
    ap.add_argument("--mode", choices=["ocean", "coastal", "index"], default=None,
                    help="自动选择采样点（开阔海/近海）")
    ap.add_argument("--nearshore-km", type=float, default=20.0)
    ap.add_argument("--overlay", type=int, default=0, metavar="HALF",
                    help="叠加中心点 ±HALF 的波形")
    ap.add_argument("--heatmap", action="store_true", help="沿轨瀑布热图")
    ap.add_argument("--stack", action="store_true", help="波形堆叠图")
    ap.add_argument("--stack-n", type=int, default=150)
    ap.add_argument("--i0", type=int, default=None)
    ap.add_argument("--i1", type=int, default=None)
    ap.add_argument("-o", "--out", default=None, help="输出 PNG（缺省弹窗）")
    a = ap.parse_args()

    pd = read_pass(a.file, with_waveform=True)
    n = pd.n
    i0 = a.i0 if a.i0 is not None else 0
    i1 = a.i1 if a.i1 is not None else n
    i0, i1 = max(0, i0), min(n, i1)

    idx = a.index if a.index is not None else (
        find_index(pd, a.mode, a.nearshore_km) if a.mode else None)

    if a.heatmap:
        plot_heatmap(pd, i0, i1, out_png=a.out)
    elif a.stack:
        plot_stack(pd, i0, i1, n_max=a.stack_n, out_png=a.out)
    elif a.overlay:
        plot_overlay(pd, idx if idx is not None else n // 2,
                     half=a.overlay, out_png=a.out)
    else:
        if idx is None:
            raise SystemExit("请用 --index / --mode 指定采样点，或加 --heatmap/--stack/--overlay")
        plot_single(pd, idx, out_png=a.out)


if __name__ == "__main__":
    main()
