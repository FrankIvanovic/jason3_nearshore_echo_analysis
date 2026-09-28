#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_track.py — Jason-3 卫星地面轨迹 + 20Hz 采样点地图
========================================================

在卫星底图（Esri World Imagery 在线瓦片，国内可直连）上绘制：
  - 卫星飞行轨迹（青色线）
  - 各采样门对应的 20Hz 采样点（彩色散点，可按时间或地表类型着色）
  - 近海采样点（离岸 < --nearshore-km）红色高亮

用法示例
--------
    # 单文件全局轨迹
    python plot_track.py "H:/毕业设计/Jason3_Data_Download/cycle_500/JA3_GPS_2PgP500_206_*.nc"

    # 只画中国近海范围，红点标出离岸 60km 内采样点
    python plot_track.py FILE --bbox 105 145 0 42 --nearshore-km 60

    # 离线模式（不联网，无底图海岸线）
    python plot_track.py FILE --tiles off
"""
from __future__ import annotations

import argparse
import glob
import io
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from PIL import Image

from jason3_reader import (
    read_pass, SURFACE_FLAG_MEANINGS, format_time,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TILE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tile_cache")

SURFACE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#8c564b",
                  "#7f7f7f", "#bcbd22", "#9467bd"]  # 与 SURFACE_FLAG_MEANINGS 对应


# ---------------------------------------------------------------------------
# Web Mercator 瓦片
# ---------------------------------------------------------------------------

def _lonlat_to_tile(lon_deg: float, lat_deg: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    x = (lon_deg + 180.0) / 360.0 * n
    lat = max(-85.05112878, min(85.05112878, lat_deg))
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _tile_url_esri(z: int, x: int, y: int) -> str:
    return (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
            f"World_Imagery/MapServer/tile/{z}/{y}/{x}")


def _tile_url_osm(z: int, x: int, y: int) -> str:
    return f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def _fetch_tile(url: str, session) -> Image.Image | None:
    os.makedirs(TILE_CACHE, exist_ok=True)
    safe = url.replace("://", "_").replace("/", "_")
    cache_fp = os.path.join(TILE_CACHE, safe)
    if os.path.isfile(cache_fp):
        try:
            return Image.open(cache_fp).convert("RGB")
        except Exception:
            pass
    try:
        r = session.get(url, timeout=15, headers={"User-Agent": "jason3-quicklook/1.0"})
        if r.status_code == 200 and len(r.content) > 500:
            with open(cache_fp, "wb") as fh:
                fh.write(r.content)
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    return None


def stitch_tiles(lon_min, lat_min, lon_max, lat_max, source="esri",
                 max_tiles=96, zoom=None):
    """
    下载并拼接覆盖 bbox 的瓦片图。
    返回 (PIL.Image, (px_left, px_right, px_top, px_bottom), zoom) —— 像素坐标
    为全局 Web Mercator 像素坐标（z 级），lat_min/lon_min 对应 left/top。
    """
    import requests
    session = requests.Session()

    if zoom is None:
        zoom = 17
        for z in range(2, 13):
            x0, y0 = _lonlat_to_tile(lon_min, lat_max, z)   # 左上角
            x1, y1 = _lonlat_to_tile(lon_max, lat_min, z)   # 右下角
            if (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1) <= max_tiles:
                zoom = z
    x0f, y0f = _lonlat_to_tile(lon_min, lat_max, zoom)
    x1f, y1f = _lonlat_to_tile(lon_max, lat_min, zoom)
    tx0, tx1 = int(math.floor(x0f)), int(math.floor(x1f))
    ty0, ty1 = int(math.floor(y0f)), int(math.floor(y1f))
    url_fn = _tile_url_esri if source == "esri" else _tile_url_osm

    W, H = 256, 256
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * W, (ty1 - ty0 + 1) * H), (40, 44, 52))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            if ty < 0 or ty >= 2 ** zoom:
                continue
            txx = tx % (2 ** zoom)
            img = _fetch_tile(url_fn(zoom, txx, ty), session)
            if img is not None:
                canvas.paste(img, ((tx - tx0) * W, (ty - ty0) * H))
    px_left = (x0f - tx0) * W
    px_right = (x1f - tx0) * W
    px_top = (y0f - ty0) * W
    px_bottom = (y1f - ty0) * W
    return canvas, (px_left, px_right, px_top, px_bottom), zoom


def _merc_px(lon_deg, lat_deg, zoom):
    """经纬度 -> z 级全局像素坐标"""
    n = 2.0 ** zoom * 256.0
    x = (lon_deg + 180.0) / 360.0 * n
    lat = max(-85.05112878, min(85.05112878, lat_deg))
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _make_degree_formatters(zoom):
    """把 Mercator 像素坐标轴刻度换算回经纬度显示"""
    from matplotlib.ticker import FuncFormatter
    n = 2.0 ** zoom * 256.0

    def lon_fmt(px, _):
        return f"{px / n * 360.0 - 180.0:.1f}°E"

    def lat_fmt(py, _):
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * py / n))))
        return f"{lat:.1f}°N"

    return FuncFormatter(lon_fmt), FuncFormatter(lat_fmt)


# ---------------------------------------------------------------------------
# 主绘图
# ---------------------------------------------------------------------------

def plot_track(files: list[str], out_png: str | None = None, bbox=None,
               tiles: str = "esri", nearshore_km: float = 60.0,
               color_by: str = "surface", point_step: int = 1, zoom: int | None = None):
    # 读取数据（多文件按时间拼接）
    passes = []
    for fp in files:
        for p in sorted(glob.glob(fp)) or [fp]:
            passes.append(read_pass(p, with_waveform=False))
    if not passes:
        print("没有可读文件")
        return

    lat = np.concatenate([p.lat for p in passes])
    lon = np.concatenate([p.lon for p in passes])            # 0~360
    lon180 = np.concatenate([p.lon_180 for p in passes])
    t_s = np.concatenate([p.time_s for p in passes])
    surf = np.concatenate([p.surface_class for p in passes])
    dtc = np.concatenate([p.dist_coast for p in passes])
    cyc = passes[0].cycle
    pas = passes[0].pass_no
    title_tag = f"cycle {cyc} / pass {pas}" if len(passes) == 1 else f"{len(passes)} 个 pass 文件"

    # 自动范围
    if bbox is None:
        lon_min, lon_max = np.nanmin(lon180) - 3, np.nanmax(lon180) + 3
        lat_min, lat_max = np.nanmin(lat) - 3, np.nanmax(lat) + 3
    else:
        lon_min, lon_max, lat_min, lat_max = bbox

    fig_w, fig_h = 14, 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    used_tiles = False
    if tiles != "off":
        try:
            print(f"正在下载 {tiles} 底图瓦片 ...")
            img, (pl, pr, pt, pb), z = stitch_tiles(
                lon_min, lat_min, lon_max, lat_max, source=tiles, zoom=zoom)
            x0, _ = _merc_px(lon_min, 0, z)
            _, y0 = _merc_px(0, lat_max, z)
            ax.imshow(np.asarray(img),
                      extent=[x0 + pl, x0 + pr, y0 + pb, y0 + pt],
                      interpolation="bilinear", zorder=0)
            ax.set_xlim(x0 + pl, x0 + pr)
            ax.set_ylim(y0 + pb, y0 + pt)
            used_tiles = True
            proj_note = f"Web Mercator (zoom={z}, {tiles})"
            lon_fmt, lat_fmt = _make_degree_formatters(z)
            ax.xaxis.set_major_formatter(lon_fmt)
            ax.yaxis.set_major_formatter(lat_fmt)
        except Exception as e:
            print(f"底图下载失败（{e}），退回经纬度平面图")

    if not used_tiles:
        if zoom is None:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("auto")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("经度 (°E, ±180)")
        ax.set_ylabel("纬度 (°N)")
        proj_note = "无底图经纬度平面图"
        # 数据点坐标转换
        px_x, px_y = lon180, lat
    else:
        z_used = z
        px = np.array([_merc_px(lo, la, z_used) for lo, la in zip(lon180, lat)])
        px_x, px_y = px[:, 0], px[:, 1]
        ax.set_xlabel("经度 (°E, ±180)")
        ax.set_ylabel("纬度 (°N)")

    # 轨迹线（跨日期变更线处断开）
    line_x, line_y = px_x.copy(), px_y.copy()
    jump = np.abs(np.diff(lon180)) > 180
    line_x[np.concatenate([jump, [False]])] = np.nan
    line_y[np.concatenate([jump, [False]])] = np.nan
    ax.plot(line_x, line_y, color="cyan" if used_tiles else "blue",
            lw=1.0, alpha=0.9, zorder=2)

    # 采样点散点
    idx = np.arange(0, len(lat), point_step)
    if color_by == "surface":
        cmap = ListedColormap(SURFACE_COLORS)
        norm = BoundaryNorm(np.arange(-0.5, 7.5, 1), cmap.N)
        sc = ax.scatter(px_x[idx], px_y[idx], c=surf[idx], cmap=cmap, norm=norm,
                        s=6, zorder=3, alpha=0.85)
        handles = [Line2D([0], [0], marker="o", ls="", color=SURFACE_COLORS[k],
                          label=f"{k} {SURFACE_FLAG_MEANINGS[k]}")
                   for k in sorted(np.unique(surf)) if k in SURFACE_FLAG_MEANINGS]
        leg1 = ax.legend(handles=handles, loc="upper left", fontsize=8, title="地表分类")
        ax.add_artist(leg1)
    else:
        sc = ax.scatter(px_x[idx], px_y[idx], c=t_s[idx], cmap="plasma",
                        s=6, zorder=3, alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01)
        cbar.set_label("时间 (s since 2000-01-01 UTC)")

    # 近海高亮
    near = dtc < nearshore_km * 1e3
    if np.any(near):
        ax.scatter(px_x[idx][near[idx]], px_y[idx][near[idx]], s=14,
                   facecolors="none", edgecolors="red", lw=0.8, zorder=4,
                   label=f"离岸<{nearshore_km:.0f}km ({near.sum()}点)")
        ax.legend(loc="upper right", fontsize=9)

    t0, t1 = format_time(np.datetime64("2000-01-01T00:00:00") + np.timedelta64(int(t_s.min() * 1e3), "ms")), \
        format_time(np.datetime64("2000-01-01T00:00:00") + np.timedelta64(int(t_s.max() * 1e3), "ms"))
    ax.set_title(f"Jason-3 卫星地面轨迹与 20Hz 采样点 — {title_tag}\n"
                 f"{t0} → {t1} UTC | 底图: {proj_note}", fontsize=11)

    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"已保存: {out_png}")
    else:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Jason-3 轨迹地图")
    ap.add_argument("files", nargs="+", help=".nc 文件（支持通配符）")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON0", "LON1", "LAT0", "LAT1"),
                    help="±180 经度范围")
    ap.add_argument("--tiles", choices=["esri", "osm", "off"], default="esri")
    ap.add_argument("--zoom", type=int, default=None)
    ap.add_argument("--nearshore-km", type=float, default=60.0)
    ap.add_argument("--color-by", choices=["surface", "time"], default="surface")
    ap.add_argument("--point-step", type=int, default=1)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    plot_track(a.files, out_png=a.out, bbox=a.bbox, tiles=a.tiles, zoom=a.zoom,
               nearshore_km=a.nearshore_km, color_by=a.color_by,
               point_step=a.point_step)


if __name__ == "__main__":
    main()
