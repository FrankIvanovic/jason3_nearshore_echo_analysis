#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oifd_reader.py — 中国全球海洋融合数据产品 OIFD 1.0（有效波高）读取库
====================================================================

数据说明
--------
目录：仓库根目录 参考文献/中国全球海洋融合数据1.0-有效波高/（已 .gitignore，
不随仓库分发），36 个月文件：

    OIFD_0p08_Significant_Wave_Height_YYYYMM.nc    201901 ~ 202112

每个文件为一个月平均有效波高（SWH）格点场：

    latitude   (2251,)  float32  -90 ~ 90，步长 0.08°（约 9 km）
    longitude  (4500,)  float32  -180 ~ 179.92，步长 0.08°
    swh        (2251, 4500) float32，单位 m，陆地/无数据为 NaN

为什么用 h5py 而不是 netCDF4：同 jason3_reader.py——netCDF4-python
在 Windows 上打不开含中文的路径，而本目录路径含中文；.nc 本质是
HDF5 文件，h5py 对 Unicode 路径支持完好。

在误差分析中的定位（论文第 5 章）
--------------------------------
OIFD 融合产品时间范围（2019-2021）与 Jason-3 实测数据（2025 年
cycle_500/600 系列）**不重叠**，因此只作"海况气候态背景 + 空间参照"，
不做逐点时间配准验证：

  1. 中国近海 SWH 季节气候态与近岸-开阔海梯度 → 为仿真海况档位
     （U19.5 = 8/12/16 m/s ↔ SWH ≈ 2/3/4 m）和中国近海典型海况
     提供独立于高度计的数据源佐证；
  2. 沿 Jason-3 c603p147 闽浙沿岸弧段采样 → 给出该弧段所处海况
     环境（近海组 vs 开阔海组逐月 SWH），支撑实测验证章节的
     海况背景描述。

对外接口
--------
    OIFDReader(data_dir=None)      # 默认指向仓库内参考文献目录
      .months()                    -> ['YYYYMM', ...] 排序月列表
      .read_month(ym, bbox=None)   -> (lat, lon, swh)
      .point_series(lat, lon)      -> (months, swh_values)
      .calendar_climatology(bbox)  -> (month 1~12, lat, lon, swh_mean)

    python oifd_reader.py --info 202007        # 单文件结构与数值范围
    python oifd_reader.py --point 26.5 121.5   # 单点 36 个月序列
    python oifd_reader.py --demo               # 生成论文用图/表（output/）
    python oifd_reader.py --demo --no-track    # 不叠加 Jason-3 轨迹
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import h5py
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(
    _REPO_ROOT, "参考文献", "中国全球海洋融合数据1.0-有效波高")

FILE_RE = re.compile(r"OIFD_0p08_Significant_Wave_Height_(\d{6})\.nc$")

# 中国近海展示范围（闽浙沿岸为主）
CHINA_BBOX = dict(lat_min=22.0, lat_max=34.0, lon_min=116.0, lon_max=132.0)


class OIFDReader:
    """OIFD 1.0 月平均 SWH 格点场读取器（中文路径安全，仅依赖 h5py）"""

    def __init__(self, data_dir: str | None = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self._files = {}  # 'YYYYMM' -> path
        for p in glob.glob(os.path.join(self.data_dir, "*.nc")):
            m = FILE_RE.search(os.path.basename(p))
            if m:
                self._files[m.group(1)] = p
        if not self._files:
            raise FileNotFoundError(
                f"在 {self.data_dir} 下没有找到 OIFD_0p08_Significant_Wave_Height_*.nc")

    # ------------------------------------------------------------------ #
    def months(self) -> list[str]:
        return sorted(self._files)

    def path_of(self, ym: str) -> str:
        if ym not in self._files:
            raise KeyError(f"没有 {ym} 月文件，可用：{self.months()}")
        return self._files[ym]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _bbox_slice(lat: np.ndarray, lon: np.ndarray, bbox: dict | None):
        """bbox -> (iy, ix) 索引切片；无 bbox 时取全球"""
        if bbox is None:
            return slice(None), slice(None)
        iy = np.where((lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]))[0]
        ix = np.where((lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"]))[0]
        return slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)

    def read_month(self, ym: str, bbox: dict | None = None):
        """读一个月的 SWH 场，返回 (lat, lon, swh)；陆地/缺测为 NaN"""
        with h5py.File(self.path_of(ym), "r") as f:
            lat = np.asarray(f["latitude"][...], dtype=np.float64)
            lon = np.asarray(f["longitude"][...], dtype=np.float64)
            iy, ix = self._bbox_slice(lat, lon, bbox)
            swh = np.asarray(f["swh"][iy, ix][...], dtype=np.float64)
        return lat[iy], lon[ix], swh

    # ------------------------------------------------------------------ #
    def point_series(self, lat0: float, lon0: float):
        """
        最近格点上的 36 个月 SWH 序列（0.08° 网格 ≈ 9 km 分辨率，
        点位取最近邻已足够；返回 (months, values)，缺测为 NaN）
        """
        months, values = [], []
        for ym in self.months():
            lat, lon, swh = self.read_month(ym)
            iy = int(np.argmin(np.abs(lat - lat0)))
            ix = int(np.argmin(np.abs(lon - lon0)))
            months.append(ym)
            values.append(float(swh[iy, ix]))
        return months, np.asarray(values)

    # ------------------------------------------------------------------ #
    def calendar_climatology(self, bbox: dict | None = None):
        """
        日历月气候态：同日历月（1~12）跨年求平均。
        返回 (cal_month 1~12 数组, lat, lon, swh_clim[12, nlat, nlon])
        """
        buckets: dict[int, list[np.ndarray]] = {m: [] for m in range(1, 13)}
        lat = lon = None
        for ym in self.months():
            lat, lon, swh = self.read_month(ym, bbox)
            buckets[int(ym[4:6])].append(swh)
        clim = np.full((12, lat.size, lon.size), np.nan)
        for m in range(1, 13):
            if buckets[m]:
                clim[m - 1] = np.nanmean(np.stack(buckets[m]), axis=0)
        return np.arange(1, 13), lat, lon, clim


# --------------------------------------------------------------------------- #
# CLI / 论文演示输出
# --------------------------------------------------------------------------- #

def _cmd_info(reader: OIFDReader, ym: str | None) -> None:
    ym = ym or reader.months()[0]
    lat, lon, swh = reader.read_month(ym)
    valid = np.isfinite(swh)
    print(f"{ym}: lat {lat[0]}~{lat[-1]} ({lat.size})  "
          f"lon {lon[0]}~{lon[-1]} ({lon.size})")
    print(f"  swh 有效 {valid.sum()}/{valid.size}，"
          f"范围 {np.nanmin(swh):.2f} ~ {np.nanmax(swh):.2f} m")


def _cmd_point(reader: OIFDReader, lat0: float, lon0: float) -> None:
    months, v = reader.point_series(lat0, lon0)
    print(f"点 ({lat0}, {lon0}) 的月平均 SWH 序列：")
    for ym, x in zip(months, v):
        bar = "#" * (0 if not np.isfinite(x) else int(max(x, 0) * 10))
        print(f"  {ym}  {x:5.2f} m  {bar}")


def _load_track(cycle: int, pass_no: int):
    """读取 Jason-3 轨迹（离岸距离用于近海/开阔海分组），失败返回 None"""
    try:
        from jason3_reader import list_cycles, list_passes, read_pass
    except ImportError:
        return None
    root = os.environ.get("JASON3_ROOT", r"H:\毕业设计\Jason3_Data_Download")
    if not os.path.isdir(root):
        return None
    for cyc, cdir in list_cycles(root):
        if cyc != cycle:
            continue
        for c, p, fpath in list_passes(cdir):
            if p == pass_no:
                pd = read_pass(fpath, with_waveform=False,
                               bbox=dict(lat_min=CHINA_BBOX["lat_min"] - 2,
                                         lat_max=CHINA_BBOX["lat_max"] + 2,
                                         lon_min=CHINA_BBOX["lon_min"] - 2,
                                         lon_max=CHINA_BBOX["lon_max"] + 2))
                return pd
    return None


def _cmd_demo(reader: OIFDReader, with_track: bool, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(out_dir, exist_ok=True)

    track = _load_track(603, 147) if with_track else None

    # ---------- 图 1：7 月气候态场 + c603p147 轨迹 ----------
    _, lat, lon, clim = reader.calendar_climatology(CHINA_BBOX)
    july = clim[6]  # 7 月（2019-2021 三年平均）

    fig, ax = plt.subplots(figsize=(9, 7))
    pcm = ax.pcolormesh(lon, lat, july, cmap="viridis",
                        shading="auto", vmin=0.5, vmax=3.0)
    plt.colorbar(pcm, ax=ax, label="SWH（m，2019—2021 年 7 月平均）")
    if track is not None:
        m = np.isfinite(track.lat) & np.isfinite(track.lon_180)
        ax.plot(track.lon_180[m], track.lat[m], "r-", lw=1.2,
                label="Jason-3 c603 p147（2025-07 实测弧段）")
        ax.legend(loc="upper left")
    ax.set_xlim(CHINA_BBOX["lon_min"], CHINA_BBOX["lon_max"])
    ax.set_ylim(CHINA_BBOX["lat_min"], CHINA_BBOX["lat_max"])
    ax.set_xlabel("经度（°E）")
    ax.set_ylabel("纬度（°N）")
    ax.set_title("中国近海月平均有效波高气候态（OIFD 1.0，7 月）\n"
                 "与 Jason-3 近海验证弧段位置")
    out1 = os.path.join(out_dir, "oifd_july_field.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已输出", out1)

    # ---------- 图 2 / CSV：沿弧段近海组 vs 开阔海组逐月 SWH ----------
    if track is not None:
        m = np.isfinite(track.lat) & np.isfinite(track.lon_180) & \
            (track.dist_coast > 0)  # dist_coast<0 在陆地一侧
        t_lat, t_lon, t_dtc = track.lat[m], track.lon_180[m], track.dist_coast[m]
        near = t_dtc <= 20e3
        far = t_dtc >= 50e3
        print(f"弧段点数：近海(≤20km) {near.sum()}，开阔海(≥50km) {far.sum()}")

        months = reader.months()
        series_near, series_far = [], []
        for ym in months:
            g_lat, g_lon, g_swh = reader.read_month(ym, CHINA_BBOX)
            iy = np.clip(np.searchsorted(g_lat, t_lat), 0, g_lat.size - 1)
            ix = np.clip(np.searchsorted(g_lon, t_lon), 0, g_lon.size - 1)
            vals = g_swh[iy, ix]
            series_near.append(np.nanmean(vals[near]) if near.sum() else np.nan)
            series_far.append(np.nanmean(vals[far]) if far.sum() else np.nan)
        series_near = np.asarray(series_near)
        series_far = np.asarray(series_far)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(months))
        ax.plot(x, series_far, "o-", ms=3, color="#1f77b4",
                label=f"开阔海组（离岸≥50 km，{far.sum()} 点均值）")
        ax.plot(x, series_near, "s-", ms=3, color="#d62728",
                label=f"近海组（离岸≤20 km，{near.sum()} 点均值）")
        step = 3
        ax.set_xticks(x[::step])
        ax.set_xticklabels(months[::step], rotation=45, fontsize=8)
        ax.set_ylabel("月平均 SWH（m）")
        ax.set_title("闽浙沿岸 Jason-3 c603 p147 弧段所处海况（OIFD 1.0 融合产品，"
                     "2019—2021）")
        ax.grid(alpha=0.3)
        ax.legend()
        out2 = os.path.join(out_dir, "oifd_climatology_track.png")
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("已输出", out2)

        csv_path = os.path.join(out_dir, "oifd_track_series.csv")
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write("month,swh_nearshore_m,swh_offshore_m\n")
            for ym, a, b in zip(months, series_near, series_far):
                f.write(f"{ym},{a:.4f},{b:.4f}\n")
        print("已输出", csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="OIFD 1.0 融合 SWH 产品读取")
    ap.add_argument("--data-dir", default=None, help="数据目录（默认仓库内参考文献目录）")
    ap.add_argument("--info", nargs="?", const="", metavar="YYYYMM",
                    help="打印某月文件信息（缺省取第一个月）")
    ap.add_argument("--point", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="打印单点 36 个月序列")
    ap.add_argument("--demo", action="store_true", help="生成论文用图（output/）")
    ap.add_argument("--no-track", action="store_true", help="demo 不叠加 Jason-3 轨迹")
    args = ap.parse_args()

    reader = OIFDReader(args.data_dir)
    if args.info is not None:
        _cmd_info(reader, args.info or None)
    if args.point:
        _cmd_point(reader, *args.point)
    if args.demo:
        _cmd_demo(reader, with_track=not args.no_track,
                  out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "output"))


if __name__ == "__main__":
    main()
