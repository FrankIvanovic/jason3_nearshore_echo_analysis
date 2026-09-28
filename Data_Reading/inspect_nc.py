#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jason-3 NetCDF 文件结构独立探查工具（ncdump -h 的 Python 等价物）

用途：
    不依赖任何既有代码的假设，直接打印 .nc 文件的真实结构——
    全局属性、每一个 group（含嵌套 group）、每一个变量的
    维度 / 形状 / 类型 / 属性，以及关键变量的少量真实数值采样。

用法：
    python inspect_nc.py <文件路径> [--sample 变量全路径 ...]
    python inspect_nc.py "H:/毕业设计/Jason3_Data_Download/cycle_500/JA3_GPS_2PgP500_206_20250130_162932_20250130_172545.nc"

    --depth N        限制 group 递归深度（默认无限）
    --values 变量路径  额外打印指定变量的数值采样（可多次给出）
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from netCDF4 import Dataset


def fmt_value(v):
    """把属性值压缩成短字符串"""
    s = str(v)
    if len(s) > 90:
        s = s[:90] + "..."
    return s


def dump_attr(attrs, indent: str) -> None:
    for name in attrs.ncattrs():
        print(f"{indent}{name} = {fmt_value(attrs.getncattr(name))}")


def sample_values(var, max_show: int = 6) -> str:
    """取变量前几个有效值（已按 scale_factor/offset 解码）"""
    try:
        data = var[...]
    except Exception as e:  # 稀疏变量或读取失败
        return f"<读取失败: {e}>"
    if np.ma.isMaskedArray(data):
        n_valid = int(np.count_nonzero(~np.ma.getmaskarray(data)))
        flat = data.compressed()
    else:
        n_valid = data.size
        flat = np.ravel(data)
    head = np.array2string(
        np.asarray(flat[:max_show]), max_line_width=200, precision=6, separator=", "
    )
    return f"有效值 {n_valid}/{data.size} | 头部样本: {head}"


def dump_group(grp, indent: str = "", depth: int | None = None) -> None:
    tag = "group" if indent else "root"
    print(f"{indent}== {tag}: {grp.path} ==")
    dump_attr(grp.attributes, indent + "  ")
    if grp.dimensions:
        print(f"{indent}  [dimensions]")
        for dname, dim in grp.dimensions.items():
            print(f"{indent}    {dname} = {dim.size}{' (unlimited)' if dim.isunlimited() else ''}")
    if grp.variables:
        print(f"{indent}  [variables]")
        for vname, var in grp.variables.items():
            dtype = var.dtype
            dims = "x".join(str(var.dimensions).split("'")) if False else ", ".join(var.dimensions)
            print(f"{indent}    {vname} : {dtype} ({dims})")
            dump_attr(var.attributes, indent + "      ")
    if depth is None or depth > 0:
        sub_depth = None if depth is None else depth - 1
        for child in grp.groups.values():
            print()
            dump_group(child, indent + "  ", sub_depth)


def main() -> None:
    ap = argparse.ArgumentParser(description="NetCDF 结构探查（ncdump -h 等价）")
    ap.add_argument("file", help=".nc 文件路径")
    ap.add_argument("--depth", type=int, default=None, help="group 递归深度")
    ap.add_argument("--values", action="append", default=[], metavar="VARPATH",
                    help="打印某变量的数值采样，如 data_20/ku/power_waveform")
    args = ap.parse_args()

    with Dataset(args.file, "r") as nc:
        dump_group(nc, depth=args.depth)
        for path in args.values:
            try:
                var = nc[path]
            except KeyError:
                print(f"\n!! 未找到变量: {path}")
                continue
            print(f"\n== 变量数值采样: {path} ==")
            print("   " + sample_values(var))


if __name__ == "__main__":
    main()
