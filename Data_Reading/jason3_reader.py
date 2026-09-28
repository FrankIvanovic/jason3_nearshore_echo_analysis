#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jason3_reader.py — Jason-3 SGDR-T NetCDF 数据通用读取库
================================================================

适用范围
--------
H:\\毕业设计\\Jason3_Data_Download 下全部 cycle_XXX 目录（不限于 500~513，
600 段同样支持）中的 JA3_GPS_2PgP{cycle}_{pass}_*.nc 文件
（CNES/EUMETSAT "GDR - Expertise dataset"，Processing Baseline G，
CF-1.7 / NetCDF4-HDF5 格式）。

为什么用 h5py 而不是 netCDF4
----------------------------
netCDF4-python 在 Windows 上调用 C 库打开文件，无法处理含中文
（任何非 ANSI/ASCII）字符的路径 —— 对 H:\\毕业设计\\... 这类目录
会直接报 FileNotFoundError。本目录下的 .nc 本质是 HDF5 文件，
h5py 对 Windows Unicode 路径支持完好，且我们本来就需要手工做
_FillValue / scale_factor / add_offset 解码，因此统一只用 h5py。
（Panoply 打开文件可见同样的 group/variable 树，可交叉验证。）

文件内部结构（实测，可用 inspect_nc.py 复核）
----------------------------------------------
/data_01                     1 Hz 数据组（含 ku/c 子组）
/data_20                     20 Hz 数据组
    time                     float64, s, units="seconds since 2000-01-01 00:00:00.0" (UTC)
    latitude / longitude     int32 微度 (scale_factor=1e-6)，经度 0~360°E 约定
    surface_classification_flag  int8  0=open_ocean 1=land 2=continental_water
                                  3=aquatic_vegetation 4=continental_ice_snow
                                  5=floating_ice 6=salted_basin
    distance_to_coast        int32 米（可为负：位于陆地/内陆水体时）
    angle_of_approach_to_coast int16 scale=0.01 度
    altitude                 int32 off=1.3e6, scale=1e-4, m (椭球高)
/data_20/ku
    power_waveform           int32 (N, 104) scale=0.001 —— Ku 波段 104 个
                             距离门回波功率（count），本套代码的核心变量
    range_ocean / swh_ocean / sig0_ocean ...   MLE4 重定结果
    range_ocean_mle3 / swh_ocean_mle3 ...      MLE3 重定结果
    epoch_ocean / amplitude_ocean / noise_floor_ocean ...  波形拟合参数
    peakiness / mqe_ocean / wvf_main_class ...  波形分类参数
/data_20/c                   C 波段（结构与 ku 类似，无 power_waveform）

对外接口
--------
    list_cycles(root)          -> [(cycle_no, dir_path), ...]
    list_passes(cycle_dir)     -> [(cycle_no, pass_no, file_path), ...]
    parse_pass_filename(path)  -> (cycle_no, pass_no)
    read_pass(path, ...)       -> PassData
    read_waveforms(path, i0, i1) -> np.ndarray (n, 104)
    decode_variable(h5var)     -> np.ndarray（已解 fill/scale/offset）

时间基准：2000-01-01 00:00:00 UTC（tai_utc_difference=37 s 见文件属性）
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TIME_EPOCH_STR = "seconds since 2000-01-01 00:00:00.0"
TIME_EPOCH = np.datetime64("2000-01-01T00:00:00", "us")

SURFACE_FLAG_MEANINGS = {
    0: "open_ocean",
    1: "land",
    2: "continental_water",
    3: "aquatic_vegetation",
    4: "continental_ice_snow",
    5: "floating_ice",
    6: "salted_basin",
}

N_GATES = 104            # Ku 波段波形距离门数
GATE_DURATION_S = 3.125e-9   # Jason-3 Ku 波段距离门时间宽度
GATE_RANGE_M = 2.99792458e8 * GATE_DURATION_S / 2.0   # ≈0.4684 m/门（单向距离）

# 20Hz 波形变量 -> (data_20 子组, 变量名)
WAVEFORM_VARS = {
    "ku": "data_20/ku/power_waveform",
}

# 中国近海（含渤海/黄海/东海/南海北部）粗框，用于快速筛选
CHINA_SEAS_BBOX = dict(lat_min=0.0, lat_max=42.0, lon_min=100.0, lon_max=146.0)

_PASS_RE = re.compile(r"JA3_GPS_2PgP(\d+)_(\d+)_", re.IGNORECASE)


# ---------------------------------------------------------------------------
# 文件发现
# ---------------------------------------------------------------------------

def list_cycles(root: str) -> list[tuple[int, str]]:
    """返回 root 下所有 cycle_数字 目录，按 cycle 号排序。"""
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        m = re.fullmatch(r"cycle_(\d+)", name)
        if m and os.path.isdir(full):
            out.append((int(m.group(1)), full))
    out.sort()
    return out


def parse_pass_filename(path: str) -> tuple[int, int] | tuple[None, None]:
    """从文件名解析 (cycle, pass)，如 JA3_GPS_2PgP500_206_... -> (500, 206)"""
    m = _PASS_RE.search(os.path.basename(path))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def list_passes(cycle_dir: str) -> list[tuple[int, int, str]]:
    """列出 cycle 目录下全部 pass 文件，按 pass 号排序。"""
    out = []
    for fp in glob.glob(os.path.join(cycle_dir, "*.nc")):
        cyc, pas = parse_pass_filename(fp)
        if cyc is None:
            continue
        out.append((cyc, pas, fp))
    out.sort(key=lambda x: x[1])
    return out


def iter_all_passes(root: str):
    """遍历 root 下所有 cycle 的所有 pass 文件，产出 (cycle, pass, path)。"""
    for _, cdir in list_cycles(root):
        yield from list_passes(cdir)


# ---------------------------------------------------------------------------
# 变量解码
# ---------------------------------------------------------------------------

def _scalar_attr(attrs: h5py.AttributeManager, name: str) -> float | None:
    if name not in attrs:
        return None
    v = attrs[name]
    arr = np.ravel(np.asarray(v))
    return float(arr[0]) if arr.size else None


def decode_variable(h5var: h5py.Dataset) -> np.ndarray:
    """
    读取 HDF5 数据集并按 CF 约定解码：
      _FillValue -> NaN；data = raw * scale_factor + add_offset
    返回 float64（整型原始数据）或保持原 dtype（无定标的数据，如 time）。
    """
    raw = h5var[...]
    attrs = h5var.attrs
    fill = _scalar_attr(attrs, "_FillValue")
    scale = _scalar_attr(attrs, "scale_factor")
    offset = _scalar_attr(attrs, "add_offset")

    if scale is None and offset is None and fill is None:
        return np.asarray(raw)

    data = np.asarray(raw, dtype=np.float64)
    if fill is not None:
        data[data == fill] = np.nan
    if scale is not None:
        data *= scale
    if offset is not None:
        data += offset
    return data


def seconds_to_datetime64(sec: np.ndarray | float) -> np.ndarray:
    """seconds since 2000-01-01 UTC -> datetime64[us]（保留毫秒级精度）"""
    sec = np.atleast_1d(np.asarray(sec, dtype=np.float64))
    us = np.round(sec * 1e6).astype(np.int64)
    return TIME_EPOCH + us.astype("timedelta64[us]")


def format_time(dt64) -> str:
    return np.datetime_as_string(np.datetime64(dt64, "ms"), unit="ms")


def lon360_to_180(lon: np.ndarray) -> np.ndarray:
    """0~360°E 约定 -> -180~+180°E 约定"""
    lon = np.asarray(lon, dtype=np.float64)
    return np.where(lon > 180.0, lon - 360.0, lon)


# ---------------------------------------------------------------------------
# PassData 数据容器
# ---------------------------------------------------------------------------

@dataclass
class PassData:
    """一个 pass 文件的 20 Hz 数据（均为解码后的物理值）"""
    path: str
    cycle: int | None = None
    pass_no: int | None = None
    time_s: np.ndarray | None = None          # float64, seconds since 2000 UTC
    time: np.ndarray | None = None            # datetime64[us]
    lat: np.ndarray | None = None             # 度（北正）
    lon: np.ndarray | None = None             # 度（0~360°E）
    lon_180: np.ndarray | None = None         # 度（-180~+180）
    surface_class: np.ndarray | None = None   # int8
    dist_coast: np.ndarray | None = None      # 米
    waveform: np.ndarray | None = None        # (N, 104) float32, count
    meta: dict = field(default_factory=dict)  # 文件全局属性

    @property
    def n(self) -> int:
        return 0 if self.lat is None else len(self.lat)

    def china_seas_mask(self, bbox: dict | None = None) -> np.ndarray:
        b = bbox or CHINA_SEAS_BBOX
        return (
            (self.lat >= b["lat_min"]) & (self.lat <= b["lat_max"]) &
            (self.lon_180 >= b["lon_min"]) & (self.lon_180 <= b["lon_max"])
        )

    def summary(self) -> str:
        if self.n == 0:
            return f"{os.path.basename(self.path)}: 无有效数据"
        return (
            f"{os.path.basename(self.path)}\n"
            f"  采样点 {self.n} | 时间 {format_time(self.time[0])} ~ {format_time(self.time[-1])} UTC\n"
            f"  纬度 {self.lat.min():.3f}~{self.lat.max():.3f}° | "
            f"经度(±180) {self.lon_180.min():.3f}~{self.lon_180.max():.3f}°\n"
            f"  离岸距离 min/中位 {np.nanmin(self.dist_coast)/1e3:.1f}/"
            f"{np.nanmedian(self.dist_coast)/1e3:.1f} km | "
            f"波形 {None if self.waveform is None else self.waveform.shape}"
        )


# ---------------------------------------------------------------------------
# 读取入口
# ---------------------------------------------------------------------------

def read_pass(
    path: str,
    with_waveform: bool = True,
    bbox: dict | None = None,
) -> PassData:
    """
    读取一个 pass 文件。

    with_waveform=False 时跳过波形矩阵（用于大范围扫描，快很多）。
    bbox 给出时只保留框内采样点（lat/lon 20Hz 原始顺序被重排）。
    """
    cyc, pas = parse_pass_filename(path)
    pd = PassData(path=path, cycle=cyc, pass_no=pas)

    with h5py.File(path, "r") as f:
        pd.meta = {k: f.attrs[k] for k in f.attrs}
        d20 = f["data_20"]

        t = np.asarray(d20["time"][...], dtype=np.float64)
        lat = decode_variable(d20["latitude"])
        lon = decode_variable(d20["longitude"])
        surf = np.asarray(d20["surface_classification_flag"][...]).astype(np.int8)
        dtc = decode_variable(d20["distance_to_coast"])

        valid = np.isfinite(lat) & np.isfinite(lon)

        if with_waveform:
            wf = decode_variable(d20["ku/power_waveform"])
        else:
            wf = None

        if bbox is not None:
            lon180 = lon360_to_180(lon)
            keep = (
                (lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]) &
                (lon180 >= bbox["lon_min"]) & (lon180 <= bbox["lon_max"]) & valid
            )
        else:
            keep = valid

        pd.time_s = t[keep]
        pd.time = seconds_to_datetime64(pd.time_s)
        pd.lat = lat[keep]
        pd.lon = lon[keep]
        pd.lon_180 = lon360_to_180(pd.lon)
        pd.surface_class = surf[keep]
        pd.dist_coast = dtc[keep]
        if wf is not None:
            pd.waveform = wf[keep].astype(np.float32)

    return pd


def read_waveforms(path: str, i0: int, i1: int) -> np.ndarray:
    """只读取一段沿轨索引 [i0, i1) 的波形矩阵（省内存），返回 (n, 104) float32"""
    with h5py.File(path, "r") as f:
        var = f["data_20/ku/power_waveform"]
        raw = var[i0:i1, ...]
    return decode_variable(raw).astype(np.float32)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def self_test(path: str) -> bool:
    """用单个文件验证读取库的正确性，打印关键数值供人工比对 Panoply。"""
    pd = read_pass(path)
    print(pd.summary())
    assert pd.waveform is not None and pd.waveform.shape[1] == N_GATES
    wf0 = pd.waveform[0]
    print(f"\n第 1 个采样点：时间 {format_time(pd.time[0])}，"
          f"位置 ({pd.lat[0]:.6f}°N, {pd.lon_180[0]:.6f}°E)，"
          f"surface={SURFACE_FLAG_MEANINGS.get(int(pd.surface_class[0]), '?')}，"
          f"离岸 {pd.dist_coast[0]/1e3:.1f} km")
    print(f"  波形前 12 门: {np.round(wf0[:12], 3).tolist()}")
    print(f"  波形 46~59 门: {np.round(wf0[46:60], 3).tolist()}")
    print(f"  波形 min/max: {np.nanmin(wf0):.1f} / {np.nanmax(wf0):.1f} count")
    print("\n>>> 请用 Panoply 打开同一文件，在 data_20/ku/power_waveform "
          "查看第 1 行数值是否与上面一致（Panoply 显示的是已乘 scale_factor=0.001 的值）。")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Jason-3 读取库自检")
    ap.add_argument("file", nargs="?", help=".nc 文件路径；缺省取 cycle_500 第一个文件")
    ap.add_argument("--root", default=r"H:\毕业设计\Jason3_Data_Download")
    a = ap.parse_args()
    fp = a.file
    if fp is None:
        cyc, p, fp = next(iter_all_passes(a.root))
        print(f"未指定文件，自动选择: {fp}")
    self_test(fp)
