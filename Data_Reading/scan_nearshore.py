#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_nearshore.py — 扫描全部 Jason-3 数据，检索中国近海采样区段
================================================================

遍历 H:\\毕业设计\\Jason3_Data_Download 下所有 cycle 的所有 pass 文件，
只读 20Hz 的经纬度/地表分类/离岸距离（不读波形，速度快），
统计每个 pass 与"中国近海"的重叠情况，输出 CSV 排名。

判定口径（可用参数调整）：
  中国海区框   : 纬度 0~42°N，经度(±180) 100~146°E（渤海/黄海/东海/南海全域）
  近海采样点   : surface_class == 0 (open_ocean) 且 离岸距离 < --coast-km
  陆地采样点   : surface_class == 1 (land) 且在中国海区框内

用法：
    python scan_nearshore.py                     # 全库扫描（默认 8 进程）
    python scan_nearshore.py --coast-km 30
    python scan_nearshore.py --cycles 500 601    # 只扫指定 cycle
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from jason3_reader import (
    iter_all_passes, read_pass, CHINA_SEAS_BBOX, format_time,
)

ROOT_DEFAULT = r"H:\毕业设计\Jason3_Data_Download"


def scan_one(task: tuple[str, float, float, float, float, float]) -> dict:
    path, coast_km, lat_min, lat_max, lon_min, lon_max = task
    bbox = dict(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
    try:
        pd = read_pass(path, with_waveform=False, bbox=bbox)
    except Exception as e:
        return dict(file=path, error=str(e))

    cyc, pas = pd.cycle, pd.pass_no
    n = pd.n
    ocean = pd.surface_class == 0
    land = pd.surface_class == 1
    coast_m = coast_km * 1e3
    near = ocean & (pd.dist_coast < coast_m) & (pd.dist_coast > 0)
    return dict(
        file=path, cycle=cyc, pass_no=pas,
        t_start=format_time(pd.time[0]) if n else "",
        t_end=format_time(pd.time[-1]) if n else "",
        n_pts=n,
        n_ocean=int(ocean.sum()),
        n_coast=int(near.sum()),
        min_dtc_km=float(np.nanmin(pd.dist_coast[ocean])) / 1e3 if ocean.any() else float("nan"),
        n_land=int(land.sum()),
        lat_min=float(pd.lat.min()) if n else float("nan"),
        lat_max=float(pd.lat.max()) if n else float("nan"),
        lon_min=float(pd.lon_180.min()) if n else float("nan"),
        lon_max=float(pd.lon_180.max()) if n else float("nan"),
        error="",
    )


def main():
    ap = argparse.ArgumentParser(description="中国近海 Jason-3 数据检索")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--coast-km", type=float, default=50.0)
    ap.add_argument("--cycles", type=int, nargs="*", default=None,
                    help="只扫描这些 cycle 号")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="nearshore_scan.csv")
    a = ap.parse_args()

    tasks = []
    for cyc, pas, path in iter_all_passes(a.root):
        if a.cycles and cyc not in a.cycles:
            continue
        tasks.append((path, a.coast_km,
                      CHINA_SEAS_BBOX["lat_min"], CHINA_SEAS_BBOX["lat_max"],
                      CHINA_SEAS_BBOX["lon_min"], CHINA_SEAS_BBOX["lon_max"]))

    print(f"共 {len(tasks)} 个 pass 文件待扫描（{a.workers} 进程并行）")
    t0 = time.time()
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(scan_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append(dict(file=futs[fut], error=repr(e)))
            done += 1
            if done % 50 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  进度 {done}/{len(tasks)}  用时 {el:.0f}s  预计剩余 {el/done*(len(tasks)-done):.0f}s")

    # 写 CSV（全部行，含无中国海区数据的）
    rows.sort(key=lambda r: (r.get("cycle") or 0, r.get("pass_no") or 0))
    fields = ["cycle", "pass_no", "file", "t_start", "t_end", "n_pts", "n_ocean",
              "n_coast", "min_dtc_km", "n_land", "lat_min", "lat_max",
              "lon_min", "lon_max", "error"]
    out_fp = os.path.join(os.path.dirname(os.path.abspath(__file__)), a.out)
    with open(out_fp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n明细已写入: {out_fp}")

    # 打印中国近海候选排名
    cand = [r for r in rows if r.get("n_coast", 0) and r.get("n_coast", 0) > 0]
    cand.sort(key=lambda r: r["n_coast"], reverse=True)
    print(f"\n=== 中国近海候选 pass（按近海采样点数排序，近海=海面且离岸<{a.coast_km:.0f}km）===")
    print(f"{'cycle':>6} {'pass':>5} {'近海点':>7} {'最浅/km':>8} {'陆地点':>7}  时间(UTC)              文件")
    for r in cand[:40]:
        print(f"{r['cycle']:>6} {r['pass_no']:>5} {r['n_coast']:>7} "
              f"{r['min_dtc_km']:>8.1f} {r['n_land']:>7}  {r['t_start']}  "
              f"{os.path.basename(r['file'])}")
    if not cand:
        print("（无）请检查海区框或数据。")
    total = sum(r.get("n_pts", 0) for r in rows)
    print(f"\n扫描完成：{len(rows)} 个文件，中国海区框内总采样点 {total}，"
          f"总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
