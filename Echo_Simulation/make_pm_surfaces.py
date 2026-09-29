# -*- coding: utf-8 -*-
"""重新生成 output/pm_surfaces.png（修复总标题与子图标题叠印）。

内容与 run_ocean_echo_sim.py 的图 1 完全一致（同样的种子与抽样），
仅修正排版：subplots_adjust 为 suptitle 留白，总标题不再引用式号。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from pm_surface import generate_pm_surface, pm_swh

WINDS_U195 = [8.0, 12.0, 16.0]
L_DOMAIN = 32e3
N_GRID = 2048
SEED0 = 2026
OUT = Path(__file__).resolve().parent / "output"

fig = plt.figure(figsize=(18, 6.6))
for i, u195 in enumerate(WINDS_U195):
    eta, meta = generate_pm_surface(u195, L_DOMAIN, N_GRID, 0.0, seed=SEED0)
    s = eta[::8, ::8]
    xs = np.arange(s.shape[1]) * (L_DOMAIN / N_GRID) * 8 / 1e3
    ys = np.arange(s.shape[0]) * (L_DOMAIN / N_GRID) * 8 / 1e3
    Xs, Ys = np.meshgrid(xs, ys)
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.plot_surface(Xs, Ys, s, cmap="jet", rstride=1, cstride=1,
                    linewidth=0, antialiased=True)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("η (m)")
    ax.set_title(f"U19.5={u195:.0f} m/s, SWH={pm_swh(u195):.2f} m")
    ax.view_init(elev=35, azim=-60)
    ax.ticklabel_format(style="plain")
fig.suptitle("二维 PM 方向谱随机海面（FFT 线性滤波法）", fontsize=14, y=0.97)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.82, wspace=0.22)
fig.savefig(OUT / "pm_surfaces.png", dpi=150)
plt.close(fig)
print("已重新生成", OUT / "pm_surfaces.png")
